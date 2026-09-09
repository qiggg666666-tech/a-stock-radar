#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定海神针独立多周期研究引擎（v1.3.0）。

支持：
- 日线长上影 / 长下影观察（含原定海神针严格长下影）
- 周线 MACD 常规底背离 / 顶背离
- 周 / 月 / 季完成周期结构支持
仅使用真实A股日线数据，无未来函数。输出不构成买卖建议、收益承诺或仓位建议。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import re
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.3.0"
VALID_CODE = re.compile(r"^(?:00|30|60|68)\d{4}$")
# 仅容忍OHLC小数减法的机器精度误差，保持观察阈值本身不变。
THRESHOLD_EPS = 1e-9
# 与workflow中"TZ=Asia/Shanghai date +%F"生成signal_date的时区口径保持一致，
# 避免GitHub Actions runner（UTC）与北京时间的日期错位导致合法的当日日期被误判为未来。
SIGNAL_DATE_TZ = ZoneInfo("Asia/Shanghai")


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
    aliases = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume",
        "date": "date", "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    frame = raw.rename(columns=aliases).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{source}_missing_columns:{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in required[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=required).sort_values("date").drop_duplicates("date", keep="last")
    frame = frame[(frame["close"] > 0) & (frame["high"] >= frame["low"]) & (frame["volume"] >= 0)]
    if frame.empty:
        raise ValueError(f"{source}_no_valid_daily_rows")
    return frame.set_index("date")[["open", "high", "low", "close", "volume"]]


def _akshare_payload(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    import akshare as ak
    raw = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq"
    )
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
        result = bs.query_history_k_data_plus(
            baostock_code(code),
            "date,open,high,low,close,volume",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
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


def _fetch_worker(fetcher: Callable[[str, str, str], list[dict[str, Any]]],
                  code: str, start_date: str, end_date: str,
                  result_queue: mp.Queue) -> None:
    try:
        result_queue.put({"ok": True, "records": fetcher(code, start_date, end_date)})
    except BaseException as exc:  # The parent records every source failure.
        result_queue.put({
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "traceback": traceback.format_exc(limit=3)
        })


def call_with_timeout(fetcher: Callable[[str, str, str], list[dict[str, Any]]],
                      code: str, start_date: str, end_date: str,
                      timeout_seconds: int) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
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


def fetch_real_daily(code: str, start_date: str, end_date: str,
                     timeout_seconds: int, retries: int) -> FetchResult:
    code = normalize_code(code)
    attempts: list[FetchAttempt] = []
    for source, fetcher in (("akshare", _akshare_payload), ("baostock", _baostock_payload)):
        for attempt in range(1, max(1, retries) + 1):
            frame, error_type, error_message = call_with_timeout(
                fetcher, code, start_date, end_date, timeout_seconds
            )
            if frame is not None:
                attempts.append(FetchAttempt(f"{source}:{attempt}", True))
                return FetchResult(frame, source, attempts)
            attempts.append(FetchAttempt(f"{source}:{attempt}", False, error_type, error_message))
    return FetchResult(None, None, attempts)


def parse_signal_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value).normalize()
    # 用北京时间计算"今天"，与workflow里`TZ=Asia/Shanghai date +%F`生成signal_date的口径一致；
    # 若改用系统默认（GitHub Actions runner为UTC），北京时间0:00-7:59触发时会把合法的当日日期误判为未来。
    today_beijing = pd.Timestamp.now(tz=SIGNAL_DATE_TZ).normalize().tz_localize(None)
    if pd.isna(parsed) or parsed > today_beijing:
        raise ValueError("invalid_or_future_signal_date")
    return parsed


def _completed_periods(frame: pd.DataFrame, rule: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    """只使用信号日之前已经完成的周期条，杜绝未来函数。

    这会刻意排除进行中的周/月/季。周五可以是已完成的周线；日历月只有在其周期
    结束已知之后才会被纳入。
    """
    aggregated = frame.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    return aggregated.loc[aggregated.index <= signal_date]


def _detect_weekly_macd_divergence(weekly: pd.DataFrame) -> dict[str, Any]:
    """
    在已完成的周线数据上检测 MACD 常规底背离 / 顶背离。
    返回字典，字段始终存在，便于后续评分与汇总。

    注意：这里用"最近极值点 vs 排除最近3根之后的最早极值点"做简化实现，
    不是严格的局部极值（swing point）识别。在周线噪声较大时，可能漏报
    真实背离，也可能把非真实摆动点误判为背离。
    """
    result = {
        "weekly_macd_bottom_div": False,
        "weekly_macd_top_div": False,
        "weekly_dif": 0.0,
        "weekly_dea": 0.0,
        "weekly_macd": 0.0,
    }
    if len(weekly) < 40:
        return result

    close = weekly["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    result["weekly_dif"] = round(float(dif.iloc[-1]), 4)
    result["weekly_dea"] = round(float(dea.iloc[-1]), 4)
    result["weekly_macd"] = round(float(macd.iloc[-1]), 4)

    # 金叉 / 死叉
    cross_up = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    cross_down = (dif < dea) & (dif.shift(1) >= dea.shift(1))

    # ---------- 底背离（价格创新低，DIF 未创新低 + 金叉） ----------
    lookback = min(60, len(weekly))
    recent = weekly.tail(lookback).copy()
    recent_dif = dif.tail(lookback)

    low_series = recent["low"]
    try:
        idx2 = low_series.idxmin()
        earlier = low_series.loc[:idx2].iloc[:-3] if len(low_series.loc[:idx2]) > 5 else low_series.iloc[:-3]
        if len(earlier) > 0:
            idx1 = earlier.idxmin()
            price_ll = recent.loc[idx2, "low"] < recent.loc[idx1, "low"]
            dif_hl = recent_dif.loc[idx2] > recent_dif.loc[idx1]
            recent_cross_up = cross_up.tail(3).any()
            result["weekly_macd_bottom_div"] = bool(price_ll and dif_hl and recent_cross_up)
    except Exception:
        pass

    # ---------- 顶背离（价格创新高，DIF 未创新高 + 死叉） ----------
    try:
        high_series = recent["high"]
        idx2 = high_series.idxmax()
        earlier = high_series.loc[:idx2].iloc[:-3] if len(high_series.loc[:idx2]) > 5 else high_series.iloc[:-3]
        if len(earlier) > 0:
            idx1 = earlier.idxmax()
            price_hh = recent.loc[idx2, "high"] > recent.loc[idx1, "high"]
            dif_lh = recent_dif.loc[idx2] < recent_dif.loc[idx1]
            recent_cross_down = cross_down.tail(3).any()
            result["weekly_macd_top_div"] = bool(price_hh and dif_lh and recent_cross_down)
    except Exception:
        pass

    return result


def evaluate_multiperiod(frame: pd.DataFrame, signal_date: pd.Timestamp,
                         max_stale_days: int = 3) -> dict[str, Any]:
    visible = frame.loc[frame.index <= signal_date].copy()
    if len(visible) < 30:
        raise ValueError(f"insufficient_daily_history:{len(visible)}_rows")

    data_last_date = visible.index.max().normalize()
    staleness_days = int((signal_date - data_last_date).days)
    if staleness_days > max_stale_days:
        raise ValueError(f"stale_daily_data:{staleness_days}_days")

    latest = visible.iloc[-1]
    previous = visible.iloc[-2]

    body = abs(float(latest.close - latest.open))
    lower_shadow = float(min(latest.close, latest.open) - latest.low)
    upper_shadow = float(latest.high - max(latest.close, latest.open))
    total_range = float(latest.high - latest.low)
    close = float(latest.close)
    previous_close = float(previous.close)

    if close <= 0 or previous_close <= 0 or lower_shadow < 0 or upper_shadow < 0 or total_range <= 0:
        raise ValueError("invalid_latest_ohlc")

    body_safe = max(body, 1e-9)
    upper_safe = max(upper_shadow, 1e-9)
    range_safe = max(total_range, 1e-9)

    amplitude_pct = float(total_range / previous_close * 100.0)
    recent_low = float(visible["low"].tail(25).min())
    recent_high = float(visible["high"].tail(25).max())

    # ---------- 通用长影线判定 ----------
    lower_ratio = lower_shadow / range_safe
    upper_ratio = upper_shadow / range_safe
    lower_to_body = lower_shadow / body_safe
    upper_to_body = upper_shadow / body_safe

    long_lower_classic = lower_ratio >= 0.667 - THRESHOLD_EPS
    long_upper_classic = upper_ratio >= 0.667 - THRESHOLD_EPS
    long_lower_body = lower_to_body >= 2.0 - THRESHOLD_EPS
    long_upper_body = upper_to_body >= 2.0 - THRESHOLD_EPS

    is_long_lower = (long_lower_classic or long_lower_body) and amplitude_pct >= 1.5 - THRESHOLD_EPS
    is_long_upper = (long_upper_classic or long_upper_body) and amplitude_pct >= 1.5 - THRESHOLD_EPS

    # 原「定海神针」严格长下影
    daily_dinghai_observation = bool(
        latest.close > latest.open
        and lower_to_body >= 2.0 - THRESHOLD_EPS
        and (lower_shadow / upper_safe) >= 2.0 - THRESHOLD_EPS
        and (body / close) <= 0.018 + THRESHOLD_EPS
        and amplitude_pct >= 2.0 - THRESHOLD_EPS
        and abs(float(latest.low) - recent_low) <= 1e-8
    )

    # ---------- 完成周期 ----------
    weekly = _completed_periods(visible, "W-FRI", signal_date)
    monthly = _completed_periods(visible, "ME", signal_date)
    quarterly = _completed_periods(visible, "QE", signal_date)

    # 周线 MACD 背离
    weekly_div = _detect_weekly_macd_divergence(weekly)

    # 结构支持
    weekly_support = False
    if len(weekly) >= 10:
        w_ma = weekly["close"].rolling(10).mean().iloc[-1]
        weekly_support = bool(
            weekly["close"].iloc[-1] >= w_ma
            and weekly["close"].iloc[-1] / weekly["close"].iloc[-5] - 1.0 >= -0.08
        )

    monthly_support = False
    if len(monthly) >= 6:
        m_ma = monthly["close"].rolling(6).mean().iloc[-1]
        monthly_support = bool(monthly["close"].iloc[-1] >= m_ma)

    quarterly_support = False
    if len(quarterly) >= 4:
        q_ma = quarterly["close"].rolling(4).mean().iloc[-1]
        quarterly_support = bool(quarterly["close"].iloc[-1] >= q_ma)

    support_count = int(weekly_support) + int(monthly_support) + int(quarterly_support)

    # ---------- 研究评分（最高约 120） ----------
    observation_score = 0
    if daily_dinghai_observation:
        observation_score += 50
    if is_long_lower:
        observation_score += 25
    if is_long_upper:
        observation_score += 15
    if weekly_div["weekly_macd_bottom_div"]:
        observation_score += 25          # 周线底背离权重较高
    if weekly_div["weekly_macd_top_div"]:
        observation_score += 15
    observation_score += 8 * support_count

    values = {
        "data_last_date": data_last_date.strftime("%Y-%m-%d"),
        "staleness_days": float(staleness_days),
        "close": round(close, 4),
        "low": round(float(latest.low), 4),
        "high": round(float(latest.high), 4),
        "amplitude_pct": round(amplitude_pct, 4),
        "lower_shadow_to_body": round(lower_to_body, 4),
        "upper_shadow_to_body": round(upper_to_body, 4),
        "lower_shadow_ratio": round(lower_ratio, 4),
        "upper_shadow_ratio": round(upper_ratio, 4),
        "body_to_close": round(body / close, 6),
        "recent_low_25": round(recent_low, 4),
        "recent_high_25": round(recent_high, 4),

        # 兼容原字段
        "daily_dinghai_observation": daily_dinghai_observation,
        "daily_long_lower": bool(is_long_lower),
        "daily_long_upper": bool(is_long_upper),
        "has_long_shadow": bool(is_long_lower or is_long_upper),

        # 周线 MACD 背离
        "weekly_macd_bottom_div": weekly_div["weekly_macd_bottom_div"],
        "weekly_macd_top_div": weekly_div["weekly_macd_top_div"],
        "weekly_dif": weekly_div["weekly_dif"],
        "weekly_dea": weekly_div["weekly_dea"],
        "weekly_macd": weekly_div["weekly_macd"],

        # 结构支持
        "completed_week_count": int(len(weekly)),
        "completed_month_count": int(len(monthly)),
        "completed_quarter_count": int(len(quarterly)),
        "weekly_structure_support": weekly_support,
        "monthly_structure_support": monthly_support,
        "quarterly_structure_support": quarterly_support,
        "structure_support_count": support_count,

        "dinghai_research_score": min(120, observation_score),
    }

    numeric_keys = (
        "staleness_days", "close", "low", "high", "amplitude_pct",
        "lower_shadow_to_body", "upper_shadow_to_body",
        "lower_shadow_ratio", "upper_shadow_ratio", "body_to_close",
        "recent_low_25", "recent_high_25", "weekly_dif", "weekly_dea", "weekly_macd"
    )
    if not all(np.isfinite(float(values[k])) for k in numeric_keys):
        raise ValueError("non_finite_indicator")
    return values


def attempts_json(result: FetchResult) -> str:
    return json.dumps([asdict(item) for item in result.attempts], ensure_ascii=False)


def _self_test() -> None:
    dates = pd.bdate_range("2024-01-01", periods=400)
    close = np.linspace(15.0, 10.0, len(dates))
    frame = pd.DataFrame({
        "open": close - 0.08, "high": close + 0.15,
        "low": close - 0.30, "close": close, "volume": 120000.0
    }, index=dates)
    # 人为制造长下影
    frame.iloc[-1] = [10.0, 10.20, 8.60, 10.05, 180000.0]
    result = evaluate_multiperiod(frame, dates[-1])
    assert "weekly_macd_bottom_div" in result
    assert "weekly_dif" in result
    assert result["daily_long_lower"] is True
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
        {"open": boundary_close - 0.03, "high": boundary_close + 0.06, "low": boundary_close - 0.10,
         "close": boundary_close, "volume": 100000.0},
        index=boundary_dates,
    )
    boundary_frame.iloc[-1] = [10.00, 10.10, 9.90, 10.05, 100000.0]
    boundary_result = evaluate_multiperiod(boundary_frame, boundary_dates[-1])
    assert boundary_result["daily_dinghai_observation"] is True, "float_boundary_regression"

    # 时区回归：用北京"今天"作为signal_date必须始终被接受，不受运行环境系统时区影响。
    today_beijing_str = pd.Timestamp.now(tz=SIGNAL_DATE_TZ).normalize().tz_localize(None).strftime("%Y-%m-%d")
    parse_signal_date(today_beijing_str)
    try:
        parse_signal_date((pd.Timestamp.now(tz=SIGNAL_DATE_TZ).normalize().tz_localize(None) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    except ValueError:
        pass
    else:
        raise AssertionError("future_signal_date_was_accepted")
    print("SELF_TEST_OK")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="定海神针独立多周期研究引擎 v1.3.0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
