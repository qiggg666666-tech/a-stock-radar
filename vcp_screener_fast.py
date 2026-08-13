#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCP 快速精简版：面向 GitHub Actions 的趋势收缩与突破候选筛选。

保留：趋势模板、相对强度、波动/成交量收缩、枢轴突破、检查点、Server 酱摘要。
移除：滚动回测、IsolationForest、逐只行业请求、Baostock 主取数、重型本地缓存。
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd


@dataclass
class Config:
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "420"))
    min_required: int = int(os.getenv("MIN_REQUIRED", "252"))
    scan_offset: int = int(os.getenv("SCAN_OFFSET", "0"))
    scan_limit: int = int(os.getenv("SCAN_LIMIT", "0"))
    workers: int = int(os.getenv("NUM_WORKERS", "4"))
    request_timeout: int = int(os.getenv("AK_TIMEOUT", "12"))
    max_runtime_seconds: int = int(os.getenv("MAX_RUNTIME_SECONDS", "19200"))
    max_failures: int = int(os.getenv("MAX_FAILURES", "200"))
    checkpoint_every: int = int(os.getenv("CHECKPOINT_EVERY", "25"))
    min_price: float = float(os.getenv("MIN_PRICE", "3"))
    min_amount: float = float(os.getenv("PRE_AMOUNT_MIN", "50000000"))
    min_turnover: float = float(os.getenv("PRE_TURNOVER_MIN", "0.3"))
    trend_score_min: int = int(os.getenv("TREND_SCORE_MIN", "7"))
    rs_percentile_min: float = float(os.getenv("RS_PERCENTILE_MIN", "70"))
    vcp_score_min: int = int(os.getenv("VCP_SCORE_MIN", "5"))
    contractions_min: int = int(os.getenv("VCP_MIN_CONTRACTIONS", "2"))
    volume_dry_ratio: float = float(os.getenv("VOL_DRY_RATIO", "0.70"))
    rvol_threshold: float = float(os.getenv("RVOL_THRESHOLD", "1.20"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    serverchan_key: str = os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY", "")

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, min(self.workers, 6))
        self.checkpoint_every = max(1, self.checkpoint_every)


CFG = Config()


def hard_call(fn: Any, *args: Any, timeout: int, **kwargs: Any) -> Any:
    """在超时后不等待后台线程，避免 ThreadPoolExecutor 上下文退出时伪超时。"""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"{getattr(fn, '__name__', 'request')} timed out after {timeout}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def normalize_history(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    data = raw.rename(columns=rename).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    if any(col not in data.columns for col in required):
        return None
    data = data[required]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return data if len(data) >= CFG.min_required else None


def fetch_universe() -> pd.DataFrame:
    raw = hard_call(ak.stock_info_a_code_name, timeout=CFG.request_timeout)
    if raw is None or raw.empty or "code" not in raw.columns:
        raise RuntimeError("AkShare 未返回有效股票列表")
    name_col = "name" if "name" in raw.columns else raw.columns[1]
    data = raw[["code", name_col]].copy()
    data.columns = ["code", "name"]
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data[data["code"].str.match(r"^(00|30|60|68)", na=False)]
    data = data[~data["name"].astype(str).str.contains(r"ST|退|\*ST|^N", na=False, regex=True)]
    return data.drop_duplicates("code").reset_index(drop=True)


def snapshot_prefilter(universe: pd.DataFrame) -> pd.DataFrame:
    try:
        raw = hard_call(ak.stock_zh_a_spot_em, timeout=CFG.request_timeout)
        if raw is None or raw.empty or "代码" not in raw.columns:
            return universe
        spot = raw.copy()
        spot["代码"] = spot["代码"].astype(str).str.zfill(6)
        for col in ("最新价", "成交额", "换手率"):
            if col in spot.columns:
                spot[col] = pd.to_numeric(spot[col], errors="coerce")
        mask = pd.Series(True, index=spot.index)
        if "最新价" in spot:
            mask &= spot["最新价"] >= CFG.min_price
        if "成交额" in spot:
            mask &= spot["成交额"] >= CFG.min_amount
        if "换手率" in spot:
            mask &= spot["换手率"] >= CFG.min_turnover
        eligible = set(spot.loc[mask, "代码"])
        filtered = universe[universe["code"].isin(eligible)].copy()
        return filtered if not filtered.empty else universe
    except Exception as exc:
        print(f"[WARN] snapshot prefilter skipped: {exc}")
        return universe


def fetch_one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
    start = (datetime.now() - timedelta(days=CFG.lookback_days)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    try:
        raw = hard_call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
            timeout=CFG.request_timeout,
        )
        data = normalize_history(raw)
        return code, data, None if data is not None else "invalid history"
    except Exception as exc:
        return code, None, str(exc)


def trend_and_vcp(code: str, name: str, data: pd.DataFrame) -> dict[str, Any] | None:
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    if len(close) < CFG.min_required:
        return None

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high252 = high.rolling(252).max()
    low252 = low.rolling(252).min()
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()

    latest = close.iloc[-1]
    trend_score = sum(
        [
            latest > sma50.iloc[-1],
            latest > sma150.iloc[-1],
            latest > sma200.iloc[-1],
            sma50.iloc[-1] > sma150.iloc[-1],
            sma150.iloc[-1] > sma200.iloc[-1],
            sma200.iloc[-1] > sma200.shift(20).iloc[-1],
            latest >= low252.iloc[-1] * 1.30,
            latest >= high252.iloc[-1] * 0.75,
        ]
    )

    # 三个滚动区间的价格幅度与 ATR 比例均收缩，近似衡量 VCP 的波动收缩。
    range20 = (high.rolling(20).max() - low.rolling(20).min()) / close.rolling(20).mean()
    range40 = (high.rolling(40).max() - low.rolling(40).min()) / close.rolling(40).mean()
    range60 = (high.rolling(60).max() - low.rolling(60).min()) / close.rolling(60).mean()
    contractions = int(range20.iloc[-1] < range40.iloc[-1] * 0.85) + int(range40.iloc[-1] < range60.iloc[-1] * 0.90)
    atr_tight = bool(atr14.iloc[-1] < atr14.rolling(60).mean().iloc[-1] * 0.85)
    vol_dry = bool(volume.tail(10).mean() < volume.tail(50).mean() * CFG.volume_dry_ratio)
    higher_lows = bool(low.tail(20).min() >= low.tail(60).min() * 0.97)
    vcp_score = contractions * 2 + int(atr_tight) + int(vol_dry) + int(higher_lows)

    pivot = high.iloc[-21:-1].max()
    rvol = volume.iloc[-1] / max(volume.tail(50).mean(), 1.0)
    breakout = latest >= pivot * 0.995 and rvol >= CFG.rvol_threshold
    return_63d = close.iloc[-1] / close.iloc[-64] - 1 if len(close) >= 64 else 0.0
    stop = latest - atr14.iloc[-1] * 2.5
    return {
        "代码": code,
        "名称": name,
        "最新价": round(float(latest), 2),
        "Trend_Score": int(trend_score),
        "VCP_Score": int(vcp_score),
        "Contraction_Count": contractions,
        "ATR_Tight": atr_tight,
        "Vol_Dry": vol_dry,
        "Higher_Lows": higher_lows,
        "Pivot": round(float(pivot), 2),
        "Breakout": bool(breakout),
        "RVOL": round(float(rvol), 2),
        "Return_63D": round(float(return_63d * 100), 2),
        "Stop_Price": round(float(stop), 2),
        "Last_Date": str(data["date"].iloc[-1].date()),
    }


def save_checkpoint(rows: list[dict[str, Any]], processed: int, total: int, reason: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["VCP_Score", "Trend_Score"], ascending=False)
    frame.to_csv(CFG.output_dir / f"vcp_fast_{stamp}_checkpoint_{processed}.csv", index=False, encoding="utf-8-sig")
    (CFG.output_dir / f"vcp_fast_{stamp}_progress.json").write_text(
        json.dumps({"processed": processed, "total": total, "rows": len(rows), "reason": reason, "saved_at": datetime.now().isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def send_serverchan(title: str, content: str) -> None:
    if not CFG.serverchan_key:
        return
    try:
        import requests
        response = requests.post(
            f"https://sctapi.ftqq.com/{CFG.serverchan_key}.send",
            data={"title": title, "desp": content[:3800]},
            timeout=12,
        )
        if response.status_code >= 400:
            print(f"[WARN] Server 酱 HTTP {response.status_code}")
    except Exception as exc:
        print(f"[WARN] Server 酱发送失败: {exc}")


def run() -> int:
    started = time.monotonic()
    deadline = started + CFG.max_runtime_seconds if CFG.max_runtime_seconds > 0 else None
    print(f"VCP fast | workers={CFG.workers} | timeout={CFG.request_timeout}s | offset={CFG.scan_offset} | limit={CFG.scan_limit or 'all'}")
    universe = snapshot_prefilter(fetch_universe()).sort_values("code").reset_index(drop=True)
    if CFG.scan_offset:
        universe = universe.iloc[CFG.scan_offset:]
    if CFG.scan_limit:
        universe = universe.iloc[:CFG.scan_limit]
    tasks = list(universe.itertuples(index=False, name=None))
    total = len(tasks)
    rows: list[dict[str, Any]] = []
    failures = 0
    processed = 0
    timed_out = False

    executor = ThreadPoolExecutor(max_workers=CFG.workers)
    futures = {executor.submit(fetch_one, code): (code, name) for code, name in tasks}
    try:
        for future in as_completed(futures):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                print("安全运行预算到达，停止尚未完成的请求")
                break
            code, name = futures[future]
            processed += 1
            try:
                _, history, error = future.result()
                if history is None:
                    failures += 1
                else:
                    item = trend_and_vcp(code, name, history)
                    if item is not None:
                        rows.append(item)
            except Exception:
                failures += 1
            if processed % CFG.checkpoint_every == 0:
                save_checkpoint(rows, processed, total, "定期保存")
                print(f"VCP 扫描: {processed}/{total} | 有效{len(rows)} | 失败{failures}")
            if failures >= CFG.max_failures:
                print("失败数量达到断路器阈值，停止全局扫描")
                break
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    save_checkpoint(rows, processed, total, "安全预算退出" if timed_out else "扫描结束")
    all_rows = pd.DataFrame(rows)
    tag = datetime.now().strftime("%Y%m%d")
    if all_rows.empty:
        final = pd.DataFrame(columns=["代码", "名称", "RS_Percentile", "Trend_Score", "VCP_Score", "Breakout"])
    else:
        all_rows["RS_Percentile"] = all_rows["Return_63D"].rank(pct=True) * 100
        final = all_rows[
            (all_rows["Trend_Score"] >= CFG.trend_score_min)
            & (all_rows["RS_Percentile"] >= CFG.rs_percentile_min)
            & (all_rows["VCP_Score"] >= CFG.vcp_score_min)
            & (all_rows["Contraction_Count"] >= CFG.contractions_min)
        ].sort_values(["Breakout", "VCP_Score", "RS_Percentile"], ascending=False)
    final.to_csv(CFG.output_dir / f"vcp_fast_{tag}.csv", index=False, encoding="utf-8-sig")
    (CFG.output_dir / f"vcp_fast_{tag}.json").write_text(
        json.dumps({"date": tag, "config": {k: str(v) for k, v in asdict(CFG).items()}, "processed": processed, "total": total, "failures": failures, "candidates": final.to_dict("records")}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"完成：处理{processed}/{total}，有效历史{len(rows)}，候选{len(final)}，失败{failures}")
    if not final.empty:
        lines = [f"- {row.名称}({row.代码}) VCP{row.VCP_Score} Trend{row.Trend_Score} RS{row.RS_Percentile:.0f} 突破{'✓' if row.Breakout else '—'}" for row in final.head(15).itertuples()]
        send_serverchan(f"📐 VCP 快速版 {len(final)}只", "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(run())
