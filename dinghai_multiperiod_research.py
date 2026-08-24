#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定海神针独立多周期研究引擎。

仅使用真实A股日线数据。日线触发为下影线与近期低点的透明观察条件；周/月
仅使用在信号日之前已经完成的周期条，用于结构标签而非收益预测。输出不构成
买卖建议、收益承诺或仓位建议。
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

SCRIPT_VERSION = "1.0.1"
VALID_CODE = re.compile(r"^(?:00|30|60|68)\d{4}$")
# 仅容忍OHLC小数减法的机器精度误差，保持观察阈值本身不变。
THRESHOLD_EPS = 1e-9


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


def baostock_code(code: str) -> str:
    return f"sh.{code}" if code.startswith(("600", "601", "603", "605", "688")) else f"sz.{code}"


def _normalize_daily(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    frame = raw.rename(columns=aliases).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}_missing_columns:{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).sort_values("date").drop_duplicates("date", keep="last")
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
        result = bs.query_history_k_data_plus(baostock_code(code), "date,open,high,low,close,volume", start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
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


def _fetch_worker(fetcher: Callable[[str, str, str], list[dict[str, Any]]], code: str, start_date: str, end_date: str, result_queue: mp.Queue) -> None:
    try:
        result_queue.put({"ok": True, "records": fetcher(code, start_date, end_date)})
    except BaseException as exc:  # The parent records every source failure.
        result_queue.put({"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)[:500], "traceback": traceback.format_exc(limit=3)})


def call_with_timeout(fetcher: Callable[[str, str, str], list[dict[str, Any]]], code: str, start_date: str, end_date: str, timeout_seconds: int) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(target=_fetch_worker, args=(fetcher, code, start_date, end_date, result_queue))
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
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__, str(exc)


def fetch_real_daily(code: str, start_date: str, end_date: str, timeout_seconds: int, retries: int) -> FetchResult:
    code = normalize_code(code)
    attempts: list[FetchAttempt] = []
    for source, fetcher in (("akshare", _akshare_payload), ("baostock", _baostock_payload)):
        for attempt in range(1, max(1, retries) + 1):
            frame, error_type, error_message = call_with_timeout(fetcher, code, start_date, end_date, timeout_seconds)
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


def _completed_periods(frame: pd.DataFrame, rule: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    """Resample only periods whose label is not after the signal date.

    This deliberately excludes the in-progress week/month. A Friday can be a completed
    weekly period; a calendar month is only included after its period end is known.
    """
    aggregated = frame.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return aggregated.loc[aggregated.index <= signal_date]


def evaluate_multiperiod(frame: pd.DataFrame, signal_date: pd.Timestamp, max_stale_days: int = 3) -> dict[str, Any]:
    visible = frame.loc[frame.index <= signal_date].copy()
    if len(visible) < 26:
        raise ValueError(f"insufficient_daily_history:{len(visible)}_rows")
    data_last_date = visible.index.max().normalize()
    staleness_days = int((signal_date - data_last_date).days)
    if staleness_days > max_stale_days:
        raise ValueError(f"stale_daily_data:{staleness_days}_days")
    latest, previous = visible.iloc[-1], visible.iloc[-2]
    body = abs(float(latest.close - latest.open))
    lower_shadow = float(min(latest.close, latest.open) - latest.low)
    upper_shadow = float(latest.high - max(latest.close, latest.open))
    close = float(latest.close)
    previous_close = float(previous.close)
    if close <= 0 or previous_close <= 0 or lower_shadow < 0 or upper_shadow < 0:
        raise ValueError("invalid_latest_ohlc")
    body_safe, upper_safe = max(body, 1e-9), max(upper_shadow, 1e-9)
    amplitude_pct = float((latest.high - latest.low) / previous_close * 100.0)
    recent_low = float(visible["low"].tail(25).min())
    lower_to_body = lower_shadow / body_safe
    lower_to_upper = lower_shadow / upper_safe
    daily_observation = bool(
        latest.close > latest.open
        and lower_to_body >= 2.0 - THRESHOLD_EPS
        and lower_to_upper >= 2.0 - THRESHOLD_EPS
        and body / close <= 0.018 + THRESHOLD_EPS
        and amplitude_pct >= 2.0 - THRESHOLD_EPS
        and abs(float(latest.low) - recent_low) <= 1e-8
    )
    weekly = _completed_periods(visible, "W-FRI", signal_date)
    monthly = _completed_periods(visible, "ME", signal_date)
    weekly_support = bool(len(weekly) >= 10 and weekly["close"].iloc[-1] >= weekly["close"].rolling(10).mean().iloc[-1] and weekly["close"].iloc[-1] / weekly["close"].iloc[-5] - 1.0 >= -0.08)
    monthly_support = bool(len(monthly) >= 6 and monthly["close"].iloc[-1] >= monthly["close"].rolling(6).mean().iloc[-1])
    support_count = int(weekly_support) + int(monthly_support)
    observation_score = (60 if daily_observation else 0) + 20 * support_count
    values = {
        "data_last_date": data_last_date.strftime("%Y-%m-%d"), "staleness_days": float(staleness_days), "close": round(close, 4),
        "low": round(float(latest.low), 4), "amplitude_pct": round(amplitude_pct, 4), "lower_shadow_to_body": round(lower_to_body, 4),
        "lower_shadow_to_upper": round(lower_to_upper, 4), "body_to_close": round(body / close, 6), "recent_low_25": round(recent_low, 4),
        "daily_dinghai_observation": daily_observation, "completed_week_count": int(len(weekly)), "completed_month_count": int(len(monthly)),
        "weekly_structure_support": weekly_support, "monthly_structure_support": monthly_support, "structure_support_count": support_count,
        "dinghai_research_score": observation_score,
    }
    if not all(np.isfinite(float(values[key])) for key in ("staleness_days", "close", "low", "amplitude_pct", "lower_shadow_to_body", "lower_shadow_to_upper", "body_to_close", "recent_low_25")):
        raise ValueError("non_finite_indicator")
    return values


def attempts_json(result: FetchResult) -> str:
    return json.dumps([asdict(item) for item in result.attempts], ensure_ascii=False)


def _self_test() -> None:
    dates = pd.bdate_range("2025-07-01", periods=320)
    close = np.linspace(12.0, 10.0, len(dates))
    frame = pd.DataFrame({"open": close - 0.08, "high": close + 0.12, "low": close - 0.2, "close": close, "volume": 100000.0}, index=dates)
    frame.iloc[-1] = [10.0, 10.25, 8.0, 10.1, 120000.0]
    result = evaluate_multiperiod(frame, dates[-1])
    assert result["daily_dinghai_observation"] is True
    assert result["completed_month_count"] >= 6
    assert normalize_code("302132") == "302132"
    try:
        normalize_code("830001")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid_code_was_accepted")
    # 边界回归：恰好达到下影/实体、下影/上影和振幅阈值的K线不能因二进制小数误差被误排除。
    boundary_dates = pd.bdate_range("2023-06-01", periods=300)
    boundary_close = np.full(300, 10.0)
    boundary_frame = pd.DataFrame(
        {"open": boundary_close - 0.03, "high": boundary_close + 0.06, "low": boundary_close - 0.10, "close": boundary_close, "volume": 100000.0},
        index=boundary_dates,
    )
    boundary_frame.iloc[-1] = [10.00, 10.10, 9.90, 10.05, 100000.0]
    boundary_result = evaluate_multiperiod(boundary_frame, boundary_dates[-1])
    assert boundary_result["daily_dinghai_observation"] is True, "float_boundary_regression"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="定海神针独立多周期研究引擎")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("SELF_TEST_OK")
