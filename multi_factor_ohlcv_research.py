# -*- coding: utf-8 -*-
"""Q01：A股多因子OHLCV研究筛选器。

边界：signed-flow是由日线收盘位置推导的成交量代理，不是逐笔Delta/CVD或主力净流入。
生产端应传入共同股票池、固定signal-date和分片参数；指标仅使用信号日及之前数据。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None
try:
    import baostock as bs
except ImportError:  # pragma: no cover
    bs = None

LOG = logging.getLogger("multi_factor_ohlcv")
COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class Config:
    history_calendar_days: int = 420
    min_history_rows: int = 130
    min_price: float = 3.0
    max_price: float = 100.0
    min_avg_turnover_yuan: float = 20_000_000.0
    min_score: int = 65
    min_volume_ratio: float = 1.05
    max_20d_return_pct: float = 28.0
    max_daily_return_pct: float = 9.5
    top: int = 80
    request_pause_seconds: float = 0.20
    retries: int = 2
    live_only: bool = True
    include_sector_labels: bool = False


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    aliases = {"日期": "date", "date": "date", "开盘": "open", "open": "open", "最高": "high", "high": "high", "最低": "low", "low": "low", "收盘": "close", "close": "close", "成交量": "volume", "volume": "volume", "成交额": "amount", "amount": "amount"}
    frame = raw.rename(columns={key: value for key, value in aliases.items() if key in raw.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=COLUMNS)
    if "amount" not in frame:
        frame["amount"] = np.nan
    frame = frame[COLUMNS]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for key in COLUMNS[1:]:
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame.dropna(subset=required).drop_duplicates("date").sort_values("date").query("close > 0 and high >= low and volume >= 0").reset_index(drop=True)


class DailyDataClient:
    """AkShare主路径、BaoStock备路径；源切换会被记录到候选记录。"""
    def __init__(self, cfg: Config) -> None:
        self.cfg, self._bao_logged_in = cfg, False

    def close(self) -> None:
        if self._bao_logged_in and bs is not None:
            try:
                bs.logout()
            except Exception:
                pass

    def _ak(self, code: str, start: str, end: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("AkShare unavailable")
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")

    def _bao(self, code: str, start: str, end: str) -> pd.DataFrame:
        if bs is None:
            raise RuntimeError("BaoStock unavailable")
        if not self._bao_logged_in:
            response = bs.login()
            if response.error_code != "0":
                raise RuntimeError(f"BaoStock login: {response.error_msg}")
            self._bao_logged_in = True
        exchange = "sh." if code.startswith("6") else "sz."
        response = bs.query_history_k_data_plus(f"{exchange}{code}", "date,open,high,low,close,volume,amount", start_date=datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d"), end_date=datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d"), frequency="d", adjustflag="2")
        if response.error_code != "0":
            raise RuntimeError(f"BaoStock query: {response.error_msg}")
        rows: list[list[str]] = []
        while response.next():
            rows.append(response.get_row_data())
        return pd.DataFrame(rows, columns=response.fields)

    def history(self, code: str, signal_date: str | None) -> tuple[pd.DataFrame, str]:
        end = pd.Timestamp(signal_date) if signal_date else pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=self.cfg.history_calendar_days)
        start_text, end_text = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        errors: list[str] = []
        for label, getter in (("akshare", self._ak), ("baostock", self._bao)):
            for attempt in range(self.cfg.retries + 1):
                try:
                    frame = normalize_ohlcv(getter(code, start_text, end_text))
                    if len(frame) >= self.cfg.min_history_rows:
                        return frame, label
                    errors.append(f"{label}:rows={len(frame)}")
                except Exception as exc:
                    errors.append(f"{label}:{type(exc).__name__}")
                time.sleep(self.cfg.request_pause_seconds * (attempt + 1))
        raise RuntimeError(";".join(errors[-4:]))


def completed_periods(frame: pd.DataFrame, rule: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    indexed = frame.set_index("date")[["open", "high", "low", "close", "volume", "amount"]]
    bars = indexed.resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}).dropna(subset=["open", "high", "low", "close"])
    return bars[bars.index < signal_date.normalize()]


def enrich_daily(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = frame.copy()
    close, high, low, volume = (df[key] for key in ("close", "high", "low", "volume"))
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    candle_range = (high - low).replace(0, np.nan)
    df["close_location"] = ((2 * close - high - low) / candle_range).clip(-1, 1).fillna(0.0)
    df["signed_flow_proxy"] = volume * df["close_location"]
    df["signed_flow_ratio_5"] = df["signed_flow_proxy"].rolling(5, min_periods=5).sum() / volume.rolling(5, min_periods=5).sum().replace(0, np.nan)
    df["signed_flow_ratio_10"] = df["signed_flow_proxy"].rolling(10, min_periods=10).sum() / volume.rolling(10, min_periods=10).sum().replace(0, np.nan)
    df["ma5"], df["ma10"], df["ma20"] = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean()
    df["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    df["ret_1"], df["ret_3"], df["ret_20"] = close.pct_change(1) * 100, close.pct_change(3) * 100, close.pct_change(20) * 100
    df["atr_pct"] = tr.rolling(14).mean() / close * 100
    low120, high120 = low.rolling(120).min(), high.rolling(120).max()
    df["position_120"] = ((close - low120) / (high120 - low120).replace(0, np.nan)).clip(0, 1)
    df["turnover_ma20"] = df["amount"].rolling(20).mean()
    if df["turnover_ma20"].isna().all():
        df["turnover_ma20"] = close * volume
    score = pd.Series(0, index=df.index, dtype="int64")
    score += ((df["signed_flow_ratio_5"] > .12) & (df["signed_flow_ratio_10"] > .03)).astype(int) * 25
    score += ((close > df["ma10"]) & (df["ma5"] >= df["ma10"] * .995)).astype(int) * 20
    score += ((df["volume_ratio"] >= cfg.min_volume_ratio) & (df["ret_1"] > 0)).astype(int) * 15
    score += ((df["ret_3"] > 0) & (df["close_location"] > .15)).astype(int) * 15
    score += ((df["position_120"] >= .08) & (df["position_120"] <= .75)).astype(int) * 15
    score += ((df["turnover_ma20"] >= cfg.min_avg_turnover_yuan) & (df["atr_pct"] <= 9.0)).astype(int) * 10
    df["research_score"] = score
    return df


def multi_period_flags(frame: pd.DataFrame, signal_date: pd.Timestamp) -> dict[str, bool]:
    weekly, monthly = completed_periods(frame, "W-FRI", signal_date), completed_periods(frame, "ME", signal_date)
    if len(weekly) < 12 or len(monthly) < 6:
        return {"weekly_repair": False, "monthly_position_ok": False}
    w5, w10 = weekly.close.rolling(5).mean(), weekly.close.rolling(10).mean()
    return {"weekly_repair": bool(weekly.close.iloc[-1] > w5.iloc[-1] and w5.iloc[-1] >= w10.iloc[-1] * .99), "monthly_position_ok": bool(-25 <= monthly.close.pct_change(3).iloc[-1] * 100 <= 45)}


def evaluate_one(code: str, name: str, history: pd.DataFrame, source: str, cfg: Config, signal_date: str | None) -> dict[str, Any] | None:
    frame = enrich_daily(history, cfg)
    if frame.empty:
        return None
    selected = frame[frame.date == pd.Timestamp(signal_date)] if signal_date else frame.tail(1)
    if selected.empty:
        return None
    row = selected.iloc[-1]
    required = ["ma10", "ma20", "signed_flow_ratio_5", "volume_ratio", "research_score", "position_120"]
    if any(pd.isna(row[key]) for key in required) or (cfg.live_only and signal_date and row.date.date() != pd.Timestamp(signal_date).date()):
        return None
    price, ret1 = safe_number(row.close), safe_number(row.ret_1)
    flags = {"score": int(row.research_score) >= cfg.min_score, "price": cfg.min_price <= price <= cfg.max_price, "liquid": safe_number(row.turnover_ma20) >= cfg.min_avg_turnover_yuan, "not_overheated": safe_number(row.ret_20) <= cfg.max_20d_return_pct and ret1 < cfg.max_daily_return_pct}
    flags.update(multi_period_flags(history[history.date <= row.date], pd.Timestamp(row.date)))
    if not all(flags.values()):
        return None
    return {"code": code, "name": name, "signal_date": pd.Timestamp(row.date).strftime("%Y-%m-%d"), "data_source": source, "close": round(price, 2), "ret_1_pct": round(ret1, 2), "ret_20_pct": round(safe_number(row.ret_20), 2), "research_score": int(row.research_score), "signed_flow_ratio_5": round(safe_number(row.signed_flow_ratio_5), 4), "signed_flow_ratio_10": round(safe_number(row.signed_flow_ratio_10), 4), "volume_ratio": round(safe_number(row.volume_ratio), 2), "position_120_pct": round(safe_number(row.position_120) * 100, 1), "turnover_ma20_yuan": round(safe_number(row.turnover_ma20), 0), "weekly_repair": flags["weekly_repair"], "monthly_position_ok": flags["monthly_position_ok"]}


def load_universe(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        raise RuntimeError("生产扫描必须提供 --universe-file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = (payload.get("universe") or payload.get("stocks") or payload.get("codes") or payload) if isinstance(payload, dict) else payload
    rows = [(str(item.get("code") or item.get("代码") or "").zfill(6), str(item.get("name") or item.get("名称") or "")) for item in raw if isinstance(item, dict)]
    return sorted({(code, name or code) for code, name in rows if code and code != "000000"})


def write_outputs(rows: list[dict[str, Any]], errors: list[dict[str, str]], output: Path, args: argparse.Namespace, cfg: Config, processed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["research_score", "signed_flow_ratio_5", "volume_ratio"], ascending=False).head(cfg.top)
    result.to_csv(output / "multi_factor_ohlcv_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(output / "multi_factor_ohlcv_errors.csv", index=False, encoding="utf-8-sig")
    state = {"state": "completed", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "signal_date": args.signal_date, "shard_index": args.shard_index, "shard_count": args.shard_count, "processed": processed, "candidates": len(result), "errors": len(errors), "config": asdict(cfg), "disclosure": "signed_flow_ratio为日线OHLCV成交量代理，不等同逐笔Delta/CVD或主力净流入。"}
    (output / "multi_factor_ohlcv_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))


def self_test() -> None:
    dates = pd.bdate_range("2025-01-02", periods=180)
    close = pd.Series(np.linspace(10, 16, len(dates)))
    sample = pd.DataFrame({"date": dates, "open": close - .1, "high": close + .25, "low": close - .35, "close": close, "volume": np.linspace(1e6, 2.2e6, len(dates)), "amount": close * np.linspace(1e6, 2.2e6, len(dates))})
    base = enrich_daily(sample, Config()).iloc[-1]
    extended = pd.concat([sample, pd.DataFrame([{"date": dates[-1] + pd.offsets.BDay(1), "open": 99, "high": 110, "low": 90, "close": 105, "volume": 9e6, "amount": 945e6}])], ignore_index=True)
    future = enrich_daily(extended, Config()).iloc[len(sample) - 1]
    assert base.research_score == future.research_score and base.signed_flow_ratio_5 == future.signed_flow_ratio_5 and 0 <= base.research_score <= 100
    print("SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q01 A股多因子OHLCV研究筛选器")
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--signal-date")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--min-score", type=int, default=65)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--max-price", type=float, default=100.0)
    parser.add_argument("--min-turnover-20d-wan", type=float, default=20.0)
    parser.add_argument("--max-20d-return-pct", type=float, default=28.0)
    parser.add_argument("--max-daily-return-pct", type=float, default=9.5)
    parser.add_argument("--request-pause-seconds", type=float, default=.20)
    parser.add_argument("--live-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        self_test(); return 0
    if not args.universe_file or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("需要共同股票池，且shard-index必须位于[0, shard-count)")
    cfg = Config(top=args.top, min_score=args.min_score, min_price=args.min_price, max_price=args.max_price, min_avg_turnover_yuan=args.min_turnover_20d_wan * 10_000, max_20d_return_pct=args.max_20d_return_pct, max_daily_return_pct=args.max_daily_return_pct, request_pause_seconds=max(0, args.request_pause_seconds), live_only=args.live_only)
    subset = load_universe(args.universe_file)[args.shard_index::args.shard_count]
    client, rows, errors = DailyDataClient(cfg), [], []
    try:
        for index, (code, name) in enumerate(subset, 1):
            try:
                history, source = client.history(code, args.signal_date)
                candidate = evaluate_one(code, name, history, source, cfg, args.signal_date)
                if candidate: rows.append(candidate)
            except Exception as exc:
                errors.append({"code": code, "name": name, "error": f"{type(exc).__name__}: {exc}"})
            if index % 25 == 0:
                LOG.info("shard %s/%s: %s/%s processed, %s candidates, %s errors", args.shard_index + 1, args.shard_count, index, len(subset), len(rows), len(errors))
            time.sleep(cfg.request_pause_seconds)
    finally:
        client.close()
    write_outputs(rows, errors, Path(args.output_dir), args, cfg, len(subset))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        LOG.exception("research scan failed: %s", exc)
        raise SystemExit(1)
