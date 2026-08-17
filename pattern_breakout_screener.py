#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股形态突破筛选器（安全分片版）。

策略口径（研究用途，非交易指令）：
  1. 趋势背景：收盘价位于50日均线之上，50日均线不低于120日均线；
  2. 结构收缩：最近5日振幅小于此前20日振幅的指定比例，且成交量同步收缩；
  3. 突破确认：收盘站上此前20日平台高点附近，阳线且量比达到阈值；
  4. 位置控制：距离此前60日高点不超过指定比例，避免把深度超跌反弹误作突破。

运行设计：AkShare 主路径、BaoStock 单标回退；每一数据源调用在可终止子进程中执行，
父调度器使用四分片、每25只检查点、200次失败断路和320分钟主动退出。
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd
import requests


LOGGER = logging.getLogger("pattern_breakout_safe")
STAGE = "平台收缩放量突破"
OUTPUT_COLUMNS = [
    "代码", "名称", "信号日期", "阶段", "最新价", "评分", "数据源", "MA50", "MA120",
    "趋势差%", "平台振幅%", "收缩比", "量比", "突破幅度%", "距60日高点%", "RSI14",
]


@dataclass(frozen=True)
class Config:
    scan_offset: int = 0
    scan_limit: int = 1500
    shard_name: str = "a"
    workers: int = 2
    ak_timeout_seconds: int = 12
    bao_timeout_seconds: int = 8
    per_symbol_timeout_seconds: int = 36
    max_runtime_seconds: int = 19_200
    max_failures: int = 200
    checkpoint_every: int = 25
    resume: bool = True
    output_dir: Path = Path("output")
    history_days: int = 700
    min_history_bars: int = 160
    base_days: int = 20
    contraction_days: int = 5
    breakout_lookback: int = 20
    high_lookback: int = 60
    ma_fast: int = 50
    ma_slow: int = 120
    base_range_max: float = 0.18
    contraction_max: float = 0.78
    volume_contraction_max: float = 0.82
    volume_ratio_min: float = 1.20
    breakout_tolerance: float = 0.01
    high_distance_max: float = 0.12
    min_price: float = 3.0
    enable_notify: bool = False
    serverchan_key: str = ""


def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default


def env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def config_from_env() -> Config:
    return Config(
        scan_offset=max(0, env_int("SCAN_OFFSET", 0)), scan_limit=max(0, env_int("SCAN_LIMIT", 1500)),
        shard_name=os.getenv("SCAN_SHARD", "a").strip().lower() or "a", workers=max(1, min(4, env_int("NUM_WORKERS", 2))),
        ak_timeout_seconds=max(1, env_int("AK_TIMEOUT", 12)), bao_timeout_seconds=max(1, env_int("BAO_TIMEOUT", 8)),
        per_symbol_timeout_seconds=max(8, env_int("PER_SYMBOL_TIMEOUT", 36)), max_runtime_seconds=max(60, env_int("MAX_RUNTIME_SECONDS", 19_200)),
        max_failures=max(1, env_int("MAX_FAILURES", 200)), checkpoint_every=max(1, env_int("CHECKPOINT_EVERY", 25)),
        resume=env_bool("RESUME", True), output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        history_days=max(250, env_int("HISTORY_DAYS", 700)), min_history_bars=max(140, env_int("MIN_HISTORY_BARS", 160)),
        base_days=max(10, env_int("BASE_DAYS", 20)), contraction_days=max(3, env_int("CONTRACTION_DAYS", 5)),
        breakout_lookback=max(10, env_int("BREAKOUT_LOOKBACK", 20)), high_lookback=max(30, env_int("HIGH_LOOKBACK", 60)),
        base_range_max=env_float("BASE_RANGE_MAX", 0.18), contraction_max=env_float("CONTRACTION_MAX", 0.78),
        volume_contraction_max=env_float("VOLUME_CONTRACTION_MAX", 0.82), volume_ratio_min=env_float("VOLUME_RATIO_MIN", 1.20),
        breakout_tolerance=env_float("BREAKOUT_TOLERANCE", 0.01), high_distance_max=env_float("HIGH_DISTANCE_MAX", 0.12),
        min_price=env_float("MIN_PRICE", 3.0), enable_notify=env_bool("ENABLE_NOTIFY", False),
        serverchan_key=os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY", ""),
    )


class DataSourceError(RuntimeError):
    pass


def mp_context() -> Any:
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def normalize_history(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mapping = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    frame = raw.rename(columns=mapping).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing: raise DataSourceError("缺少字段:" + ",".join(missing))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    frame = frame[(frame["close"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["volume"] > 0)]
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if len(frame) < cfg.min_history_bars: raise DataSourceError(f"历史不足:{len(frame)}/{cfg.min_history_bars}")
    return frame[required]


def fetch_ak(symbol: str, cfg: Config) -> pd.DataFrame:
    end = date.today(); start = end - timedelta(days=cfg.history_days)
    raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
    if raw is None or raw.empty: raise DataSourceError("AkShare空日线")
    return normalize_history(raw, cfg)


def fetch_bao(symbol: str, cfg: Config) -> pd.DataFrame:
    market = "sh" if symbol.startswith(("6", "9")) else "sz"
    end = date.today(); start = end - timedelta(days=cfg.history_days)
    login = bs.login()
    if getattr(login, "error_code", "1") != "0": raise DataSourceError("BaoStock登录失败")
    try:
        query = bs.query_history_k_data_plus(f"{market}.{symbol}", "date,open,high,low,close,volume,tradestatus", start_date=start.isoformat(), end_date=end.isoformat(), frequency="d", adjustflag="2")
        if getattr(query, "error_code", "1") != "0": raise DataSourceError(f"BaoStock日线失败:{getattr(query, 'error_msg', '')}")
        rows: list[list[str]] = []
        while query.next(): rows.append(query.get_row_data())
        raw = pd.DataFrame(rows, columns=query.fields)
        if "tradestatus" in raw.columns: raw = raw[raw["tradestatus"].astype(str) == "1"]
        if raw.empty: raise DataSourceError("BaoStock空日线")
        return normalize_history(raw, cfg)
    finally:
        try: bs.logout()
        except Exception: pass


def source_worker(result_queue: Any, source: str, symbol: str, cfg_dict: dict[str, Any]) -> None:
    cfg = Config(**{**cfg_dict, "output_dir": Path(cfg_dict["output_dir"])})
    try:
        frame = fetch_ak(symbol, cfg) if source == "akshare" else fetch_bao(symbol, cfg)
        result_queue.put({"ok": True, "frame": frame})
    except Exception as error:
        result_queue.put({"ok": False, "reason": f"{type(error).__name__}:{str(error)[:150]}"})


def fetch_hard(source: str, symbol: str, cfg: Config) -> pd.DataFrame:
    timeout = cfg.ak_timeout_seconds if source == "akshare" else cfg.bao_timeout_seconds
    ctx = mp_context(); result_queue = ctx.Queue(maxsize=1); cfg_dict = asdict(cfg); cfg_dict["output_dir"] = str(cfg.output_dir)
    process = ctx.Process(target=source_worker, args=(result_queue, source, symbol, cfg_dict)); process.start()
    try:
        payload = result_queue.get(timeout=timeout)
    except queue.Empty as error:
        if process.is_alive(): process.terminate()
        process.join(5); result_queue.close()
        raise DataSourceError(f"{source}超时{timeout}秒") from error
    process.join(5)
    if process.is_alive(): process.terminate(); process.join(5)
    result_queue.close()
    if not payload.get("ok"): raise DataSourceError(str(payload.get("reason", f"{source}未知错误")))
    return payload["frame"]


def rsi14(close: pd.Series) -> float | None:
    delta = close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    value = (100 - 100 / (1 + gain / loss)).iloc[-1]
    return None if pd.isna(value) else float(value)


def evaluate(symbol: str, name: str, frame: pd.DataFrame, source: str, cfg: Config) -> dict[str, Any] | None:
    if len(frame) < max(cfg.min_history_bars, cfg.ma_slow + 1, cfg.high_lookback + 1): return None
    close, open_, high, low, volume = (frame[column].astype(float) for column in ("close", "open", "high", "low", "volume"))
    price = float(close.iloc[-1])
    if price < cfg.min_price: return None
    ma_fast, ma_slow = close.rolling(cfg.ma_fast).mean(), close.rolling(cfg.ma_slow).mean()
    if pd.isna(ma_fast.iloc[-1]) or pd.isna(ma_slow.iloc[-1]): return None
    trend_ok = price >= float(ma_fast.iloc[-1]) and float(ma_fast.iloc[-1]) >= float(ma_slow.iloc[-1])
    if not trend_ok: return None
    # 平台测量排除当日，避免把突破日的大波动计入收缩区间。
    base = frame.iloc[-(cfg.base_days + 1):-1]
    contraction = frame.iloc[-(cfg.contraction_days + 1):-1]
    if len(base) < cfg.base_days or len(contraction) < cfg.contraction_days: return None
    base_range = (float(base["high"].max()) - float(base["low"].min())) / max(float(base["low"].min()), 1e-9)
    contraction_range = (float(contraction["high"].max()) - float(contraction["low"].min())) / max(float(contraction["low"].min()), 1e-9)
    contraction_ratio = contraction_range / max(base_range, 1e-9)
    previous_volume = float(base["volume"].mean()); recent_volume = float(contraction["volume"].mean())
    volume_contraction = recent_volume / max(previous_volume, 1e-9)
    pivot = float(high.iloc[-(cfg.breakout_lookback + 1):-1].max())
    breakout_pct = (price / pivot - 1) * 100
    today_volume_ratio = float(volume.iloc[-1]) / max(recent_volume, 1e-9)
    recent_high = float(high.iloc[-(cfg.high_lookback + 1):-1].max())
    distance_to_high = (recent_high - price) / recent_high if recent_high > 0 else 1.0
    green = price >= float(open_.iloc[-1])
    if not (base_range <= cfg.base_range_max and contraction_ratio <= cfg.contraction_max and volume_contraction <= cfg.volume_contraction_max): return None
    if not (price >= pivot * (1 - cfg.breakout_tolerance) and green and today_volume_ratio >= cfg.volume_ratio_min and distance_to_high <= cfg.high_distance_max): return None
    trend_gap = (float(ma_fast.iloc[-1]) / float(ma_slow.iloc[-1]) - 1) * 100
    score = round(
        min(max(trend_gap, 0), 20) * 2 + (1 - contraction_ratio) * 25 + (1 - volume_contraction) * 15
        + min(today_volume_ratio, 3) / 3 * 25 + min(max(breakout_pct, 0), 10) * 3.5 + (1 - distance_to_high / cfg.high_distance_max) * 10, 1
    )
    return {
        "代码": symbol, "名称": name, "信号日期": frame["date"].iloc[-1].strftime("%Y-%m-%d"), "阶段": STAGE,
        "最新价": round(price, 2), "评分": score, "数据源": source, "MA50": round(float(ma_fast.iloc[-1]), 2), "MA120": round(float(ma_slow.iloc[-1]), 2),
        "趋势差%": round(trend_gap, 2), "平台振幅%": round(base_range * 100, 2), "收缩比": round(contraction_ratio, 2),
        "量比": round(today_volume_ratio, 2), "突破幅度%": round(breakout_pct, 2), "距60日高点%": round(distance_to_high * 100, 2), "RSI14": round(rsi14(close) or 0, 1),
    }


def evaluate_child(result_queue: Any, symbol: str, name: str, cfg_dict: dict[str, Any]) -> None:
    cfg = Config(**{**cfg_dict, "output_dir": Path(cfg_dict["output_dir"])})
    try:
        try:
            frame = fetch_hard("akshare", symbol, cfg); source = "akshare"
        except Exception as ak_error:
            try:
                frame = fetch_hard("baostock", symbol, cfg); source = "baostock"
            except Exception as bao_error:
                result_queue.put({"kind": "source_error", "symbol": symbol, "reason": f"AK:{str(ak_error)[:90]} | BAO:{str(bao_error)[:90]}"}); return
        result = evaluate(symbol, name, frame, source, cfg)
        result_queue.put({"kind": "candidate" if result else "no_signal", "symbol": symbol, "result": result})
    except Exception as error:
        result_queue.put({"kind": "logic_error", "symbol": symbol, "reason": f"{type(error).__name__}:{str(error)[:150]}"})


def baostock_cursor_to_frame(response: Any) -> pd.DataFrame:
    """逐行读取BaoStock游标，避免response.get_data()依赖DataFrame.append。"""
    fields = list(getattr(response, "fields", []) or [])
    if not fields:
        raise DataSourceError("BaoStock股票池字段为空")
    rows: list[list[str]] = []
    while response.next():
        row = list(response.get_row_data())
        if len(row) != len(fields):
            raise DataSourceError(f"BaoStock股票池字段数量异常:{len(row)}/{len(fields)}")
        rows.append(row)
    if not rows:
        raise DataSourceError("BaoStock股票池无数据")
    return pd.DataFrame(rows, columns=fields)


def universe_worker(result_queue: Any, source: str) -> None:
    try:
        if source == "akshare":
            raw = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
        else:
            login = bs.login()
            if getattr(login, "error_code", "1") != "0": raise DataSourceError("BaoStock股票池登录失败")
            try: raw = baostock_cursor_to_frame(bs.query_stock_basic()).rename(columns={"code": "代码", "code_name": "名称"})
            finally:
                try: bs.logout()
                except Exception: pass
        raw = raw[["代码", "名称"]].copy(); raw["代码"] = raw["代码"].astype(str).str.extract(r"(\d{6})", expand=False); raw["名称"] = raw["名称"].astype(str)
        selected = raw[raw["代码"].notna() & raw["代码"].str.startswith(("0", "3", "6")) & ~raw["名称"].str.contains("ST|退", na=False, regex=True)].drop_duplicates("代码").sort_values("代码")
        if selected.empty: raise DataSourceError("清洗后股票池为空")
        result_queue.put({"ok": True, "universe": selected.to_dict(orient="records")})
    except Exception as error:
        result_queue.put({"ok": False, "reason": f"{type(error).__name__}:{str(error)[:150]}"})


def get_universe(cfg: Config) -> list[dict[str, str]]:
    ctx = mp_context(); errors: list[str] = []
    for source, timeout in (("akshare", cfg.ak_timeout_seconds), ("baostock", cfg.bao_timeout_seconds)):
        result_queue = ctx.Queue(maxsize=1); process = ctx.Process(target=universe_worker, args=(result_queue, source)); process.start()
        try: payload = result_queue.get(timeout=timeout)
        except queue.Empty:
            if process.is_alive(): process.terminate()
            process.join(5); result_queue.close(); errors.append(f"{source}股票池超时{timeout}秒"); continue
        process.join(5)
        if process.is_alive(): process.terminate(); process.join(5)
        result_queue.close()
        if payload.get("ok"): return payload["universe"]
        errors.append(f"{source}:{payload.get('reason', '未知错误')}")
    raise DataSourceError("；".join(errors))


def paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    prefix = f"pattern_breakout_{cfg.shard_name}"
    return tuple(cfg.output_dir / f"{prefix}{suffix}" for suffix in (".checkpoint.json", ".csv", ".json", ".md"))  # type: ignore[return-value]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"); temp.replace(path)


def render_markdown(frame: pd.DataFrame) -> str:
    if frame.empty: return "本分片暂无符合收缩放量突破条件的标的。"
    columns = list(frame.columns); lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None): lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def load_checkpoint(path: Path, cfg: Config) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    defaults = {"source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}
    if not cfg.resume or not path.exists(): return set(), [], defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8")); return set(data.get("processed", [])), list(data.get("results", [])), {**defaults, **{key: int(value) for key, value in data.get("stats", {}).items()}}
    except Exception: return set(), [], defaults


def save_outputs(cfg: Config, universe_size: int, processed: set[str], results: list[dict[str, Any]], stats: dict[str, int], reason: str, started: float) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True); checkpoint, csv_path, json_path, md_path = paths(cfg)
    ordered = sorted(results, key=lambda row: -float(row.get("评分", 0)))
    frame = pd.DataFrame(ordered, columns=OUTPUT_COLUMNS); frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {"version": "safe-v1", "strategy": STAGE, "shard": cfg.shard_name, "scan_offset": cfg.scan_offset, "scan_limit": cfg.scan_limit, "universe_size": universe_size, "processed": len(processed), "candidates": len(ordered), "stats": stats, "stop_reason": reason, "elapsed_seconds": round(time.monotonic() - started, 1), "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}, "results": ordered}
    write_json_atomic(json_path, payload); write_json_atomic(checkpoint, {"processed": sorted(processed), "results": ordered, "stats": stats})
    md_path.write_text(f"# 形态突破筛选（分片 {cfg.shard_name.upper()}）\n\n- 运行状态：`{reason}`\n- 已处理：{len(processed)}/{universe_size}\n- 候选：{len(ordered)}\n- 异常统计：`{json.dumps(stats, ensure_ascii=False)}`\n\n## 候选\n\n{render_markdown(frame)}\n", encoding="utf-8")


def notify(cfg: Config, results: list[dict[str, Any]], reason: str) -> None:
    if not cfg.enable_notify or not cfg.serverchan_key: return
    lines = [f"- {row['名称']}({row['代码']}) 分{row['评分']} 量比{row['量比']} 突破{row['突破幅度%']}%" for row in sorted(results, key=lambda value: -float(value["评分"]))[:20]] or ["本分片无候选。"]
    try: requests.post(f"https://sctapi.ftqq.com/{cfg.serverchan_key}.send", data={"title": f"形态突破[{cfg.shard_name.upper()}] {len(results)}只", "desp": f"运行状态：{reason}\n\n" + "\n".join(lines)}, timeout=15).raise_for_status()
    except requests.RequestException as error: LOGGER.warning("Server酱推送失败：%s", error)


def run(cfg: Config) -> int:
    started = time.monotonic(); cfg.output_dir.mkdir(parents=True, exist_ok=True); checkpoint, _, _, _ = paths(cfg)
    try: universe = get_universe(cfg)
    except DataSourceError as error:
        LOGGER.warning("股票池不可用，安全结束：%s", error); save_outputs(cfg, 0, set(), [], {"source_error": 1, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}, "universe_unavailable", started); return 0
    shard = universe[cfg.scan_offset:cfg.scan_offset + cfg.scan_limit] if cfg.scan_limit else universe[cfg.scan_offset:]
    processed, results, stats = load_checkpoint(checkpoint, cfg); tasks = [item for item in shard if item["代码"] not in processed]
    LOGGER.info("分片%s：总池%s，本片%s，待处理%s，workers=%s", cfg.shard_name.upper(), len(universe), len(shard), len(tasks), cfg.workers)
    ctx = mp_context(); active: dict[str, tuple[Any, Any, float]] = {}; index = 0; stop_reason = "completed"; last_save = len(processed); cfg_dict = asdict(cfg); cfg_dict["output_dir"] = str(cfg.output_dir)
    def start(item: dict[str, str]) -> None:
        result_queue = ctx.Queue(maxsize=1); process = ctx.Process(target=evaluate_worker, args=(result_queue, item["代码"], item["名称"], cfg_dict)); process.start(); active[item["代码"]] = (process, result_queue, time.monotonic())
    while index < len(tasks) or active:
        if time.monotonic() - started >= cfg.max_runtime_seconds and stop_reason == "completed": stop_reason = "runtime_budget_reached"
        if stats["source_error"] >= cfg.max_failures and stop_reason == "completed": stop_reason = "circuit_breaker_open"
        while stop_reason == "completed" and len(active) < cfg.workers and index < len(tasks): start(tasks[index]); index += 1
        for symbol, (process, result_queue, launched) in list(active.items()):
            try: payload = result_queue.get_nowait()
            except queue.Empty: payload = None
            if payload is not None:
                process.join(2); result_queue.close(); active.pop(symbol, None); processed.add(symbol); kind = payload.get("kind", "worker_crash")
                if kind == "candidate" and payload.get("result"): results.append(payload["result"])
                elif kind in stats: stats[kind] += 1
                else: stats["worker_crash"] += 1
            elif time.monotonic() - launched >= cfg.per_symbol_timeout_seconds:
                if process.is_alive(): process.terminate()
                process.join(5); result_queue.close(); active.pop(symbol, None); processed.add(symbol); stats["timeout"] += 1
            elif process.exitcode is not None:
                result_queue.close(); active.pop(symbol, None); processed.add(symbol); stats["worker_crash"] += 1
        if len(processed) - last_save >= cfg.checkpoint_every:
            save_outputs(cfg, len(shard), processed, results, stats, "in_progress" if stop_reason == "completed" else stop_reason, started); last_save = len(processed)
        if stop_reason != "completed":
            for process, result_queue, _ in active.values():
                if process.is_alive(): process.terminate()
                process.join(5); result_queue.close()
            active.clear(); break
        time.sleep(0.1)
    save_outputs(cfg, len(shard), processed, results, stats, stop_reason, started); notify(cfg, results, stop_reason)
    LOGGER.info("分片%s完成：%s，已处理%s/%s，候选%s，统计%s", cfg.shard_name.upper(), stop_reason, len(processed), len(shard), len(results), stats)
    return 0


def self_test() -> int:
    # 创建趋势向上、末端收缩并在最后一天放量突破的离线序列。
    dates = pd.date_range("2023-01-02", periods=180, freq="B"); close = np.linspace(10, 18, 180); close[-21:-1] = np.linspace(17.2, 17.5, 20); close[-6:-1] = np.linspace(17.38, 17.48, 5); close[-1] = 17.85
    frame = pd.DataFrame({"date": dates, "open": close * 0.995, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000})
    frame.loc[frame.index[-6:-1], "volume"] = 500_000; frame.loc[frame.index[-1], "volume"] = 1_500_000
    cfg = Config(min_history_bars=160, history_days=700, ma_slow=120)
    result = evaluate("600000", "测试", frame, "test", cfg)
    assert result is not None and result["阶段"] == STAGE
    print("SELF_TEST_OK"); return 0


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    return self_test() if "--self-test" in sys.argv else run(config_from_env())


if __name__ == "__main__": raise SystemExit(main())
