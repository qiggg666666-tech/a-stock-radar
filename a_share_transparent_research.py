#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股全市场透明量价研究引擎。

只读取信号日及以前的真实日线。输出为横截面研究排序和数据质量记录，
不输出买卖、持仓、收益或因果结论。
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import queue
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"
VALID_CODE = re.compile(r"^(?:00|30|60|68)\d{4}$")
FACTOR_WEIGHTS = {
    "trend_20": 0.25,
    "momentum_20": 0.20,
    "drawdown_30": 0.20,
    "volatility_20": 0.15,
    "liquidity_20": 0.10,
    "volume_ratio_20": 0.10,
}


@dataclass(frozen=True)
class FetchAttempt:
    source: str
    ok: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class FetchResult:
    frame: Optional[pd.DataFrame]
    source: Optional[str]
    attempts: list[FetchAttempt]


def normalize_code(value: Any) -> str:
    code = str(value).strip().replace(".0", "")
    if not VALID_CODE.fullmatch(code):
        raise ValueError(f"unsupported_a_share_code:{code}")
    return code


def _bs_code(code: str) -> str:
    return f"sh.{code}" if code.startswith(("600", "601", "603", "605", "688")) else f"sz.{code}"


def _normalize_daily(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    frame = raw.rename(columns=aliases).copy()
    columns = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}_missing_columns:{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=columns).sort_values("date").drop_duplicates("date", keep="last")
    frame = frame[(frame["close"] > 0) & (frame["high"] >= frame["low"]) & (frame["volume"] >= 0)]
    if frame.empty:
        raise ValueError(f"{source}_no_valid_daily_rows")
    return frame.set_index("date")[["open", "high", "low", "close", "volume"]]


def _akshare_payload(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    import akshare as ak

    raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust="qfq")
    if raw is None or raw.empty:
        raise ValueError("akshare_empty_response")
    frame = _normalize_daily(raw, "akshare").reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


def _baostock_payload(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_history_k_data_plus(_bs_code(code), "date,open,high,low,close,volume", start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        if raw.empty:
            raise ValueError("baostock_empty_response")
        frame = _normalize_daily(raw, "baostock").reset_index()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame.to_dict(orient="records")
    finally:
        bs.logout()


def _worker(fetcher: Callable[[str, str, str], list[dict[str, Any]]], code: str, start_date: str, end_date: str, result_queue: mp.Queue) -> None:
    try:
        result_queue.put({"ok": True, "records": fetcher(code, start_date, end_date)})
    except BaseException as exc:
        result_queue.put({"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)[:500], "traceback": traceback.format_exc(limit=3)})


def _call_with_timeout(fetcher: Callable[[str, str, str], list[dict[str, Any]]], code: str, start_date: str, end_date: str, timeout_seconds: int) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(fetcher, code, start_date, end_date, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        return None, "TimeoutError", f"request_exceeded_{timeout_seconds}_seconds"
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return None, "ChildProcessError", f"child_exit_{process.exitcode}_without_payload"
    finally:
        result_queue.close()
        result_queue.join_thread()
    if not payload.get("ok"):
        return None, str(payload.get("error_type", "SourceError")), str(payload.get("error_message", "unknown_error"))
    try:
        return _normalize_daily(pd.DataFrame(payload["records"]), "child_payload"), None, None
    except Exception as exc:
        return None, type(exc).__name__, str(exc)


def fetch_real_daily(code: str, start_date: str, end_date: str, timeout_seconds: int, retries: int) -> FetchResult:
    code = normalize_code(code)
    attempts: list[FetchAttempt] = []
    for source, fetcher in (("akshare", _akshare_payload), ("baostock", _baostock_payload)):
        for attempt in range(1, max(1, retries) + 1):
            frame, error_type, error_message = _call_with_timeout(fetcher, code, start_date, end_date, timeout_seconds)
            if frame is not None:
                attempts.append(FetchAttempt(f"{source}:{attempt}", True))
                return FetchResult(frame, source, attempts)
            attempts.append(FetchAttempt(f"{source}:{attempt}", False, error_type, error_message))
    return FetchResult(None, None, attempts)


def parse_signal_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value).normalize()
    if pd.isna(parsed) or parsed.date() > date.today():
        raise ValueError("invalid_or_future_signal_date")
    return parsed


def calculate_raw_factors(frame: pd.DataFrame, signal_date: pd.Timestamp, max_stale_days: int) -> dict[str, float | str]:
    visible = frame.loc[frame.index <= signal_date].copy()
    if len(visible) < 61:
        raise ValueError(f"insufficient_history:{len(visible)}_rows")
    last_date = visible.index.max().normalize()
    stale_days = int((signal_date - last_date).days)
    if stale_days > max_stale_days:
        raise ValueError(f"stale_daily_data:{stale_days}_days")
    close = visible["close"]
    amount = close * visible["volume"]
    max_30 = close.rolling(30).max().iloc[-1]
    values: dict[str, float | str] = {
        "data_last_date": last_date.strftime("%Y-%m-%d"),
        "staleness_days": float(stale_days),
        "close": float(close.iloc[-1]),
        "trend_20": float(close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1.0),
        "momentum_20": float(close.iloc[-1] / close.iloc[-21] - 1.0),
        "drawdown_30": float(close.iloc[-1] / max_30 - 1.0),
        "volatility_20": float(close.pct_change().tail(20).std(ddof=0) * math.sqrt(252)),
        "liquidity_20": float(math.log1p(amount.tail(20).mean())),
        "volume_ratio_20": float(visible["volume"].iloc[-1] / (visible["volume"].tail(20).mean() + 1e-12)),
    }
    if not all(np.isfinite(float(value)) for key, value in values.items() if key != "data_last_date"):
        raise ValueError("non_finite_factor")
    return values


def _robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    scale = 1.4826 * (series - median).abs().median()
    if not np.isfinite(scale) or scale < 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((series - median) / scale).clip(-3.0, 3.0)


def score_cross_section(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    missing = set(FACTOR_WEIGHTS) - set(raw.columns)
    if missing:
        raise ValueError(f"missing_factor_columns:{','.join(sorted(missing))}")
    direction = {"trend_20": 1.0, "momentum_20": 1.0, "drawdown_30": 1.0, "volatility_20": -1.0, "liquidity_20": 1.0, "volume_ratio_20": 1.0}
    scored = raw.copy()
    total = pd.Series(0.0, index=scored.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        scored[f"z_{factor}"] = _robust_zscore(scored[factor]) * direction[factor]
        total += weight * scored[f"z_{factor}"]
    scored["transparent_research_score"] = (50.0 + 10.0 * total).clip(0.0, 100.0)
    scored["rank"] = scored["transparent_research_score"].rank(method="first", ascending=False).astype(int)
    return scored.sort_values(["rank", "code"]).reset_index(drop=True)


def attempts_json(result: FetchResult) -> str:
    return json.dumps([asdict(item) for item in result.attempts], ensure_ascii=False)


def _self_test() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    close = 10.0 + np.arange(100) * 0.03
    frame = pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000.0}, index=dates)
    signal = dates[79]
    baseline = calculate_raw_factors(frame, signal, 0)
    future = frame.copy()
    future.loc[future.index > signal, ["open", "high", "low", "close", "volume"]] *= 50
    assert baseline == calculate_raw_factors(future, signal, 0), "future_rows_changed_factors"
    assert normalize_code("302132") == "302132"
    try:
        normalize_code("830001")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported_code_accepted")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A股全市场透明量价研究引擎")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("SELF_TEST_OK")
