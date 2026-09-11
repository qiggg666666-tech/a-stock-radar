#!/usr/bin/env python3
"""FINAL Chip 日线筹码峰、集中度、突破、五维评分 + 市盈率(PE) + 多均线系统。

新逻辑（相对原版）：
- 增加 peTTM 抓取与 pe_score（低PE高分，亏损/极高PE低分）
- WEIGHTS 加入 pe=0.12，其余维度按比例微调，总和仍为 1.0
- analyze() 返回 pe_ttm / pe_score，total_score 含价值维度
- 排除北交所；OLD_CHIP_DECAY_CAP=0.005 保留
- 盘中首红确认改为新浪优先、东财兜底双源，且套 provider_call 硬超时保护

用法：
  python final_chip_research.py          # self_test
  from final_chip_research import fetch_ohlcv, analyze
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

DECAY, STEP, LOOKBACK = 0.5, 0.01, 100
OLD_CHIP_AGE_DAYS = 30
OLD_CHIP_DECAY_CAP = 0.005
NARROW_DAY_REL_TOL = 0.003
BELOW_SPIKE_BAND_MIN, BELOW_SPIKE_GAP_MIN = 0.12, 3.0
BELOW_SPIKE_RATIO_MIN = 0.008
WIDE_ZONE_MASS_TARGET = 0.55
WIDE_ZONE_MIN_WIDTH_PCT = 0.05
WIDE_ZONE_MAX_WIDTH_PCT = 0.35
APPROACH_RATIO, BREAK_RATIO, VOL_MULTIPLIER, CONC_THRESHOLD = 0.97, 1.01, 1.5, 0.20

# 筹码主峰尖峰判定：与下方长红柱(below-spike)沿用同一套已在本文件验证过的阈值，
# 保持"尖峰"定义在同一文件内一致，不为主峰另起一套新参数。
SPIKE_BAND_MIN, SPIKE_GAP_MIN, SPIKE_RATIO_MIN = BELOW_SPIKE_BAND_MIN, BELOW_SPIKE_GAP_MIN, BELOW_SPIKE_RATIO_MIN

# 长上影/长下影判定：直接沿用 dinghai_multiperiod_research.py 里已经跑在生产上的同一套
# 阈值口径（经典占比0.667、影线/实体比2.0、日振幅下限1.5%），两个项目对"长影线"的定义
# 保持一致，避免同一概念在不同脚本里有不同数值标准。
LONG_SHADOW_EPS = 1e-9
LONG_SHADOW_RATIO_CLASSIC = 0.667
LONG_SHADOW_BODY_RATIO = 2.0
LONG_SHADOW_MIN_AMPLITUDE_PCT = 1.5

# OBV三均线粘合：向量化移植自用户提供的参考脚本，原版用逐行.apply()判断
# 5/10/20日OBV均线是否收敛在一个窄区间内，代表资金流进入"横盘但内部平衡"状态。
# sticky_threshold沿用参考脚本自身给出的默认值，未在本项目真实数据上验证过，先作为起始值。
OBV_MA_PERIODS = (5, 10, 20)
OBV_STICKY_THRESHOLD = 0.02

# 主力资金流入二次确认：移植自用户提供的参考脚本 check_main_fund_flow()。
# 只对已通过其它初筛(洗盘/尖峰筹码柱/长影线等)的候选调用，不在analyze()内部对全市场
# 逐一调用——ak.stock_individual_fund_flow是逐股票单独请求的接口，比日线更容易被限流。
# 阈值沿用参考脚本自身给出的建议值，未在本项目真实数据上验证过，先作为起始值。
MAIN_FUND_MIN_NET_TODAY = 0.0
MAIN_FUND_MIN_NET_3DAYS = 5_000_000.0

# 首红+盘中MA5拐头
INTRADAY_MA_PERIOD = 5
INTRADAY_TURN_UP_THRESHOLD = 0.0008

# 新逻辑：加入市盈率后权重（总和=1.0）
WEIGHTS = {
    "line": 0.10,
    "conc": 0.10,
    "peak": 0.08,
    "break": 0.33,
    "profit": 0.10,
    "ma": 0.17,
    "pe": 0.12,
}


def is_beijing_stock(code: str) -> bool:
    """排除北交所股票。"""
    c = str(code).zfill(6)
    return c.startswith(("83", "87", "88", "82", "920", "4"))


def _call(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send((True, function()))
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}:{str(exc)[:300]}"))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, function: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return function()
    parent, child = mp.get_context("fork").Pipe(duplex=False)
    process = mp.get_context("fork").Process(target=_call, args=(child, function), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=2)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        ok, value = parent.recv()
        process.join(timeout=2)
        if not ok:
            raise RuntimeError(f"provider_error:{label}:{value}")
        return value
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        parent.close()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover", "turn": "turnover", "peTTM": "pe_ttm",
    }
    data = frame.rename(columns={k: v for k, v in columns.items() if k in frame.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    if data.empty or any(item not in data.columns for item in required):
        raise ValueError("invalid_ohlcv_schema")
    for item in required[1:] + ["amount", "turnover", "pe_ttm"]:
        if item not in data:
            data[item] = 0.0 if item != "pe_ttm" else np.nan
        data[item] = pd.to_numeric(data[item], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=required).sort_values("date").drop_duplicates("date")
    # baostock/akshare 换手率多为百分比，统一转 0~1
    turnover = data["turnover"].fillna(0.0).astype(float)
    data["turnover"] = turnover / 100.0
    data = data[(data["close"] > 0) & (data["high"] >= data["low"]) & (data["volume"] > 0)]
    if len(data) < 40:
        raise ValueError(f"insufficient_history:{len(data)}")
    return data.tail(LOOKBACK).reset_index(drop=True)


def _ak_history(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    return normalize_frame(
        ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            adjust="qfq",
        )
    )


def _bs_history(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        exchange = "sh" if code.startswith(("60", "68")) else "sz"
        # 含 peTTM，供价值评分
        result = bs.query_history_k_data_plus(
            f"{exchange}.{code}",
            "date,open,high,low,close,volume,amount,turn,peTTM",
            start_date=start, end_date=end, frequency="d", adjustflag="2",
        )
        if result.error_code != "0":
            raise RuntimeError(f"baostock_history:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return normalize_frame(pd.DataFrame(rows, columns=result.fields))
    finally:
        bs.logout()


def fetch_ohlcv(code: str, timeout_seconds: float = 35, retries: int = 2) -> tuple[pd.DataFrame, str, list[str]]:
    if is_beijing_stock(code):
        raise RuntimeError(f"beijing_stock_excluded:{code}")
    end = datetime.now().date()
    start = end - timedelta(days=LOOKBACK + 110)
    errors: list[str] = []
    # 优先 baostock（带 peTTM），失败再试 akshare
    for label, request in (("baostock", _bs_history), ("akshare", _ak_history)):
        for attempt in range(1, max(retries, 1) + 1):
            try:
                df = provider_call(
                    f"{label}:{code}", timeout_seconds,
                    lambda fn=request: fn(code, start.isoformat(), end.isoformat()),
                )
                return df, label, errors
            except Exception as exc:
                errors.append(f"{label}:{attempt}:{type(exc).__name__}:{str(exc)[:220]}")
    raise RuntimeError("ohlcv_unavailable:" + " | ".join(errors))


# ====================== 改进版盘中确认（新浪优先 + 东财兜底） ======================
def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一不同源的字段名为 datetime / close"""
    if df is None or df.empty:
        raise ValueError("empty_frame")
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("时间", "datetime", "day", "date", "time"):
            col_map[c] = "datetime"
        elif cl in ("收盘", "close"):
            col_map[c] = "close"
    df = df.rename(columns=col_map)
    if "datetime" not in df.columns or "close" not in df.columns:
        if len(df.columns) >= 5:
            df = df.copy()
            df.columns = ["datetime", "open", "high", "low", "close"] + list(df.columns[5:])
        else:
            raise ValueError(f"cannot_normalize_columns:{list(df.columns)}")
    df = df[["datetime", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
    if df.empty:
        raise ValueError("empty_after_normalize")
    return df


def _fetch_sina_5min(code: str) -> pd.DataFrame:
    import akshare as ak
    code6 = str(code).zfill(6)
    symbol = f"sh{code6}" if code6.startswith(("5", "6", "9")) else f"sz{code6}"
    df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
    return _normalize_minute_df(df)


def _fetch_em_5min(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    code6 = str(code).zfill(6)
    df = ak.stock_zh_a_hist_min_em(
        symbol=code6,
        period="5",
        adjust="qfq",
        start_date=start,
        end_date=end,
    )
    return _normalize_minute_df(df)


def fetch_intraday_ma5_turnup(
    code: str,
    timeout_seconds: float = 25.0,
    retries: int = 3,
) -> tuple[bool, float, str, list[str]]:
    """改进版：新浪优先 → 东财兜底，每次实际网络调用都套 provider_call 硬超时，
    带限速和轻量指数退避，避免打得太密被单一数据源限流/拉黑。

    这是一次额外的接口调用（分钟级数据），成本明显高于日线，设计上不在 analyze() 内部
    对全市场逐一调用，而是由分片脚本只对通过 is_first_red_daily 等日线初筛的候选股票调用。
    失败时返回 (False, 0.0, 说明, 错误列表)，从不假装有信号。
    """
    import random

    errors: list[str] = []
    start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d 09:30:00")
    end = datetime.now().strftime("%Y-%m-%d 15:00:00")

    sources = [
        ("sina", lambda: _fetch_sina_5min(code)),
        ("em", lambda: _fetch_em_5min(code, start, end)),
    ]

    for source_name, fetcher in sources:
        for attempt in range(1, max(retries, 1) + 1):
            sleep_t = 1.6 + random.uniform(0.3, 0.9)
            if attempt > 1:
                sleep_t += (2 ** (attempt - 2)) * 0.9
            time.sleep(sleep_t)

            try:
                # 修复：原始改进版直接调 fetcher()，没有硬超时保护，请求卡住会无限挂起。
                # 套 provider_call，超时了会被 SIGTERM 强制打断，跟 fetch_ohlcv 的保护级别一致。
                frame = provider_call(f"{source_name}5min:{code}", timeout_seconds, fetcher)
                frame = frame.set_index("datetime").sort_index()
                close_10 = (
                    pd.to_numeric(frame["close"], errors="coerce")
                    .resample("10min")
                    .last()
                    .dropna()
                )
                if len(close_10) < INTRADAY_MA_PERIOD + 3:
                    errors.append(f"{source_name}:{attempt}:10min_data_insufficient:{len(close_10)}")
                    continue

                ma5 = close_10.rolling(INTRADAY_MA_PERIOD).mean()
                recent = ma5.iloc[-3:].values
                if np.any(np.isnan(recent)):
                    errors.append(f"{source_name}:{attempt}:ma5_nan")
                    continue

                slope_now = float(recent[-1] - recent[-2])
                slope_prev = float(recent[-2] - recent[-3])
                is_up = bool(
                    slope_prev <= INTRADAY_TURN_UP_THRESHOLD
                    and slope_now > INTRADAY_TURN_UP_THRESHOLD
                )
                note = f"{source_name}|前={slope_prev:.5f} 现={slope_now:.5f}"
                return is_up, slope_now, note, errors

            except Exception as exc:
                errors.append(
                    f"{source_name}:{attempt}:{type(exc).__name__}:{str(exc)[:180]}"
                )
                continue

    return False, 0.0, "盘中数据获取失败(多源)", errors


# ====================== 主力资金流入二次确认（移植自用户提供的参考脚本） ======================
def _main_fund_flow_market(code: str) -> str:
    """final_chip里北交所股票在is_beijing_stock就已经被排除，不会走到这里，
    所以只需要区分沪/深，不用像参考脚本那样再判一个bj分支。"""
    return "sh" if str(code).zfill(6).startswith(("6", "9")) else "sz"


def _fetch_main_fund_flow_raw(code: str, market: str) -> pd.DataFrame:
    import akshare as ak
    return ak.stock_individual_fund_flow(stock=code, market=market)


def fetch_main_fund_flow(
    code: str,
    timeout_seconds: float = 20.0,
    retries: int = 2,
) -> dict[str, Any]:
    """主力资金净流入二次确认：今日净流入>0 且 近3日累计净流入>500万(默认阈值)。

    这是逐股票单独请求的接口，比日线更容易被限流，设计上不在 analyze() 内部对全市场
    逐一调用，而是由分片脚本只对已经命中其它初筛信号(洗盘/尖峰筹码柱/长影线等)的候选
    调用。失败时 main_fund_ok=False，不假装有信号。
    """
    market = _main_fund_flow_market(code)
    errors: list[str] = []
    for attempt in range(1, max(retries, 1) + 1):
        if attempt > 1:
            time.sleep(1.0 + 0.5 * (attempt - 1))
        try:
            frame = provider_call(
                f"fund_flow:{code}", timeout_seconds,
                lambda: _fetch_main_fund_flow_raw(code, market),
            )
            if frame is None or len(frame) < 3:
                errors.append(f"attempt{attempt}:insufficient_rows:{0 if frame is None else len(frame)}")
                continue
            frame = frame.sort_values("日期").reset_index(drop=True)
            today_net = float(frame.iloc[-1]["主力净流入-净额"])
            today_ratio = float(frame.iloc[-1]["主力净流入-净占比"])
            days3_net = float(frame.iloc[-3:]["主力净流入-净额"].sum())
            ok = bool(today_net > MAIN_FUND_MIN_NET_TODAY and days3_net > MAIN_FUND_MIN_NET_3DAYS)
            return {
                "main_fund_ok": ok,
                "main_fund_today_net": round(today_net, 2),
                "main_fund_days3_net": round(days3_net, 2),
                "main_fund_today_ratio_pct": round(today_ratio, 2),
                "main_fund_errors": " | ".join(errors),
            }
        except Exception as exc:
            errors.append(f"attempt{attempt}:{type(exc).__name__}:{str(exc)[:200]}")
    return {
        "main_fund_ok": False,
        "main_fund_today_net": 0.0,
        "main_fund_days3_net": 0.0,
        "main_fund_today_ratio_pct": 0.0,
        "main_fund_errors": " | ".join(errors),
    }


def average_price(row: pd.Series) -> float:
    if float(row["amount"]) > 0 and float(row["volume"]) > 0:
        return float(row["amount"]) / float(row["volume"])
    return float((row["open"] + row["high"] + row["low"] + row["close"]) / 4)


def daily_distribution(low: float, high: float, average: float, volume: float) -> dict[float, float]:
    if high < low or volume <= 0:
        return {}
    if math.isclose(high, low) or (high - low) <= max(average, 1e-8) * NARROW_DAY_REL_TOL:
        return {round(average, 2): volume}
    prices = np.unique(np.round(np.linspace(low, high, max(int(round((high - low) / STEP)) + 1, 2)), 2))
    values = np.array([
        max((price - low) / max(average - low, 1e-8), 0.0) if price <= average
        else max((high - price) / max(high - average, 1e-8), 0.0)
        for price in prices
    ])
    values = values / values.sum() if values.sum() > 0 else np.ones(len(prices)) / len(prices)
    return {float(price): float(weight * volume) for price, weight in zip(prices, values)}


def update_chip(chip: dict[float, float], row: pd.Series, decay_cap: float = 1.0) -> dict[float, float]:
    absorb = min(max(float(row["turnover"]) * DECAY, 0.0), 1.0)
    erode = min(absorb, decay_cap)
    result = {price: weight * (1 - erode) for price, weight in chip.items() if weight * (1 - erode) > 1e-8}
    for price, weight in daily_distribution(
        float(row["low"]), float(row["high"]), average_price(row), float(row["volume"])
    ).items():
        result[price] = result.get(price, 0.0) + weight * absorb
    return result


def find_wide_zone(
    prices_b: np.ndarray, weights_b: np.ndarray, below_total: float, mass_target: float
) -> tuple[float, float, float]:
    n = len(prices_b)
    if n == 0 or below_total <= 1e-8:
        return 0.0, 0.0, 0.0
    target = below_total * mass_target
    left = 0
    window_sum = 0.0
    best_width = float("inf")
    best_low, best_high, best_ratio = float(prices_b[0]), float(prices_b[-1]), 0.0
    for right in range(n):
        window_sum += float(weights_b[right])
        while window_sum - float(weights_b[left]) >= target and left < right:
            window_sum -= float(weights_b[left])
            left += 1
        if window_sum >= target:
            width = float(prices_b[right] - prices_b[left])
            if width < best_width:
                best_width = width
                best_low, best_high = float(prices_b[left]), float(prices_b[right])
                best_ratio = window_sum / below_total
    return best_low, best_high, best_ratio


def compute_ma_features(data: pd.DataFrame) -> dict[str, Any]:
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    n = len(close)
    if n < 25:
        return {
            "ma5": None, "ma10": None, "ma20": None, "ma60": None,
            "ma_alignment": False, "ma5_above_ma10": False, "ma10_above_ma20": False,
            "golden_cross_5_10": False, "golden_cross_10_20": False,
            "price_above_ma5": False, "price_above_ma10": False, "price_above_ma20": False,
            "ma_slope_up": False, "vol_confirm": False, "ma_score": 0.0, "ma_signal": "无",
        }

    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean() if n >= 60 else pd.Series([np.nan] * n)

    last_close = float(close.iloc[-1])
    last_ma5 = float(ma5.iloc[-1]) if pd.notna(ma5.iloc[-1]) else None
    last_ma10 = float(ma10.iloc[-1]) if pd.notna(ma10.iloc[-1]) else None
    last_ma20 = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None
    last_ma60 = float(ma60.iloc[-1]) if n >= 60 and pd.notna(ma60.iloc[-1]) else None

    ma_alignment = bool(
        last_ma5 is not None and last_ma10 is not None and last_ma20 is not None
        and last_ma5 > last_ma10 > last_ma20
    )
    ma5_above_ma10 = bool(last_ma5 is not None and last_ma10 is not None and last_ma5 > last_ma10)
    ma10_above_ma20 = bool(last_ma10 is not None and last_ma20 is not None and last_ma10 > last_ma20)

    golden_cross_5_10 = False
    golden_cross_10_20 = False
    if n >= 12:
        for i in range(1, min(4, n)):
            prev_ma5, prev_ma10 = ma5.iloc[-1 - i], ma10.iloc[-1 - i]
            curr_ma5, curr_ma10 = ma5.iloc[-i], ma10.iloc[-i]
            if all(pd.notna(x) for x in (prev_ma5, prev_ma10, curr_ma5, curr_ma10)):
                if prev_ma5 <= prev_ma10 and curr_ma5 > curr_ma10:
                    golden_cross_5_10 = True
                    break
    if n >= 22:
        for i in range(1, min(4, n)):
            prev_ma10, prev_ma20 = ma10.iloc[-1 - i], ma20.iloc[-1 - i]
            curr_ma10, curr_ma20 = ma10.iloc[-i], ma20.iloc[-i]
            if all(pd.notna(x) for x in (prev_ma10, prev_ma20, curr_ma10, curr_ma20)):
                if prev_ma10 <= prev_ma20 and curr_ma10 > curr_ma20:
                    golden_cross_10_20 = True
                    break

    price_above_ma5 = bool(last_ma5 is not None and last_close > last_ma5)
    price_above_ma10 = bool(last_ma10 is not None and last_close > last_ma10)
    price_above_ma20 = bool(last_ma20 is not None and last_close > last_ma20)

    ma_slope_up = False
    if n >= 8 and last_ma5 is not None and last_ma10 is not None:
        ma5_3ago = ma5.iloc[-4] if pd.notna(ma5.iloc[-4]) else None
        ma10_3ago = ma10.iloc[-4] if pd.notna(ma10.iloc[-4]) else None
        if ma5_3ago is not None and ma10_3ago is not None:
            ma_slope_up = (last_ma5 > ma5_3ago) and (last_ma10 > ma10_3ago)

    vol_ma5 = volume.tail(5).mean()
    vol_ma20 = volume.tail(20).mean() if n >= 20 else volume.mean()
    last_vol = float(volume.iloc[-1])
    recent3_vol = float(volume.tail(3).mean())
    vol_confirm = bool(
        (vol_ma5 > 0 and last_vol >= vol_ma5 * 1.2)
        or (vol_ma20 > 0 and recent3_vol >= vol_ma20 * 1.2)
    )

    s_align = 1.0 if ma_alignment else (0.6 if ma5_above_ma10 and ma10_above_ma20 else 0.3 if ma5_above_ma10 else 0.0)
    s_cross = 1.0 if (golden_cross_5_10 or golden_cross_10_20) else 0.4 if ma5_above_ma10 else 0.0
    s_price = 1.0 if (price_above_ma5 and price_above_ma10 and price_above_ma20) else (
        0.6 if price_above_ma5 and price_above_ma10 else 0.3 if price_above_ma5 else 0.0
    )
    s_slope = 1.0 if ma_slope_up else 0.4
    s_vol = 1.0 if vol_confirm else 0.35
    ma_score = float(np.clip(
        100 * (0.30 * s_align + 0.25 * s_cross + 0.20 * s_price + 0.15 * s_slope + 0.10 * s_vol), 0, 100
    ))

    if ma_alignment and (golden_cross_5_10 or golden_cross_10_20) and vol_confirm and price_above_ma5:
        ma_signal = "强多·多头排列+金叉+量能确认"
    elif ma_alignment and price_above_ma5:
        ma_signal = "多头排列·趋势健康"
    elif golden_cross_5_10 or golden_cross_10_20:
        ma_signal = "金叉出现·关注确认"
    elif ma5_above_ma10 and price_above_ma5:
        ma_signal = "短期均线支撑"
    else:
        ma_signal = "无"

    return {
        "ma5": round(last_ma5, 2) if last_ma5 is not None else None,
        "ma10": round(last_ma10, 2) if last_ma10 is not None else None,
        "ma20": round(last_ma20, 2) if last_ma20 is not None else None,
        "ma60": round(last_ma60, 2) if last_ma60 is not None else None,
        "ma_alignment": ma_alignment,
        "ma5_above_ma10": ma5_above_ma10,
        "ma10_above_ma20": ma10_above_ma20,
        "golden_cross_5_10": golden_cross_5_10,
        "golden_cross_10_20": golden_cross_10_20,
        "price_above_ma5": price_above_ma5,
        "price_above_ma10": price_above_ma10,
        "price_above_ma20": price_above_ma20,
        "ma_slope_up": ma_slope_up,
        "vol_confirm": vol_confirm,
        "ma_score": round(ma_score, 1),
        "ma_signal": ma_signal,
    }


def concentration_band(chip: dict[float, float], pct: float) -> tuple[float, float, float]:
    """移植自筹码尖峰蒸馏引擎：pct总仓位对应的价格区间宽度(百分比口径，供洗盘阶段判定用)。"""
    items, total = sorted(chip.items()), sum(chip.values())
    if not items or total <= 0:
        return 100.0, 0.0, 0.0
    lower_target, upper_target = total * (1 - pct) / 2, total * (1 + pct) / 2
    cumulative, lower, upper = 0.0, items[0][0], items[-1][0]
    for price, weight in items:
        cumulative += weight
        if cumulative >= lower_target:
            lower = price
            break
    cumulative = 0.0
    for price, weight in items:
        cumulative += weight
        if cumulative >= upper_target:
            upper = price
            break
    width = (upper - lower) / (upper + lower) * 100 if upper + lower > 0 else 100.0
    return float(width), float(lower), float(upper)


def _stage_winner(chip: dict[float, float], price: float) -> float:
    """移植：现价下方筹码占比(严格小于)，专供阶段判定使用。"""
    total = sum(chip.values())
    return float(sum(weight for chip_price, weight in chip.items() if chip_price < price) / total) if total > 0 else 0.0


def _cross_metrics(previous: dict[float, float], open_price: float, high: float, low: float, close: float, turnover: float) -> tuple[float, float, float, float]:
    """移植：今日K线穿透昨日筹码的比例；仅阳线(close>open)才有效，否则视为无穿透。"""
    total = sum(previous.values())
    if total <= 0 or close <= open_price:
        return 0.0, 0.0, 0.0, 0.0
    body_low, body_high = min(open_price, close), max(open_price, close)
    bar_low, bar_high = min(low, open_price, close), max(high, open_price, close)
    cross = sum(weight for price, weight in previous.items() if body_low <= price <= body_high) / total
    profit = sum(weight for price, weight in previous.items() if bar_low <= price <= bar_high and price < close) / total
    locked = sum(weight for price, weight in previous.items() if bar_low <= price <= bar_high and price > close) / total
    return float(cross), float(profit), float(locked), float(cross / turnover if turnover > 1e-8 else 0.0)


def classify_stage(close: float, average_cost: float, winner: float, conc90: float, recent_high: float, recent_low: float, cross_ratio: float) -> tuple[str, str]:
    """移植自筹码尖峰蒸馏引擎：吸筹/洗盘/拉升/出货/震荡五阶段判定，取代原 below/wide 洗盘启发式。"""
    if recent_high <= recent_low or average_cost <= 0:
        return "震荡", "数据不足"
    position = (close - recent_low) / (recent_high - recent_low + 1e-8)
    premium = (close - average_cost) / average_cost
    tight, very_tight = conc90 < 15.0, conc90 < 10.0
    if position < 0.35 and winner < 0.45 and tight and abs(premium) < 0.12:
        return "吸筹", "低位较集中，获利盘少"
    if 0.25 < position < 0.55 and 0.35 < winner < 0.65 and conc90 < 22.0 and cross_ratio > 0.08:
        return "洗盘", "中低位洗盘后穿透"
    if position > 0.55 and winner > 0.55 and 0.08 < premium < 0.35:
        return "拉升", "脱离成本区，顺势"
    if position > 0.75 and winner > 0.70 and (very_tight or premium > 0.25):
        return "出货", "高位高获利，警惕"
    if position > 0.80 and winner > 0.75:
        return "出货", "高位风险偏大"
    return "震荡", "筹码分散或多空平衡"


def calc_washout_score(cross_ratio: float, profit_cross: float, locked_cross: float, penetrate: float, winner: float, pct_change: float) -> float:
    """移植自筹码尖峰蒸馏引擎 calc_score：给洗盘候选打分，供排序用。"""
    cross_score = min(100.0, max(0.0, (cross_ratio - 0.03) / 0.17 * 100))
    side_min, side_sum = min(profit_cross, locked_cross), profit_cross + locked_cross + 1e-8
    balance_score = min(100.0, side_min / 0.03 * 50 + (1 - abs(profit_cross - locked_cross) / side_sum) * 50)
    penetration_score = min(100.0, penetrate / 3.0 * 100)
    winner_score = 100.0 if 0.4 <= winner <= 0.7 else (winner / 0.4 * 80 if winner < 0.4 else max(0.0, 100 - (winner - 0.7) / 0.3 * 100))
    pct_score = 100.0 if 0.02 <= pct_change <= 0.07 else (pct_change / 0.02 * 60 if pct_change < 0.02 else max(0.0, 100 - (pct_change - 0.07) / 0.08 * 80))
    return round(float(cross_score * 0.35 + balance_score * 0.25 + penetration_score * 0.20 + winner_score * 0.15 + pct_score * 0.05), 1)


def compute_shadow_features(data: pd.DataFrame) -> dict[str, Any]:
    """当日K线长上影/长下影判定，口径与 dinghai_multiperiod_research.py 保持一致（见文件顶部常量注释）。"""
    last = data.iloc[-1]
    open_, high, low, close = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    prior_close = float(data.iloc[-2]["close"]) if len(data) >= 2 else close
    body = abs(close - open_)
    lower_shadow = min(close, open_) - low
    upper_shadow = high - max(close, open_)
    total_range = high - low
    if total_range <= 0 or prior_close <= 0:
        return {
            "daily_long_lower": False, "daily_long_upper": False, "has_long_shadow": False,
            "lower_shadow_ratio": 0.0, "upper_shadow_ratio": 0.0,
            "lower_shadow_to_body": 0.0, "upper_shadow_to_body": 0.0, "shadow_amplitude_pct": 0.0,
        }
    body_safe = max(body, LONG_SHADOW_EPS)
    range_safe = max(total_range, LONG_SHADOW_EPS)
    lower_ratio = lower_shadow / range_safe
    upper_ratio = upper_shadow / range_safe
    lower_to_body = lower_shadow / body_safe
    upper_to_body = upper_shadow / body_safe
    amplitude_pct = total_range / prior_close * 100.0
    long_lower = bool(
        (lower_ratio >= LONG_SHADOW_RATIO_CLASSIC - LONG_SHADOW_EPS
         or lower_to_body >= LONG_SHADOW_BODY_RATIO - LONG_SHADOW_EPS)
        and amplitude_pct >= LONG_SHADOW_MIN_AMPLITUDE_PCT - LONG_SHADOW_EPS
    )
    long_upper = bool(
        (upper_ratio >= LONG_SHADOW_RATIO_CLASSIC - LONG_SHADOW_EPS
         or upper_to_body >= LONG_SHADOW_BODY_RATIO - LONG_SHADOW_EPS)
        and amplitude_pct >= LONG_SHADOW_MIN_AMPLITUDE_PCT - LONG_SHADOW_EPS
    )
    return {
        "daily_long_lower": long_lower,
        "daily_long_upper": long_upper,
        "has_long_shadow": bool(long_lower or long_upper),
        "lower_shadow_ratio": round(float(lower_ratio), 4),
        "upper_shadow_ratio": round(float(upper_ratio), 4),
        "lower_shadow_to_body": round(float(lower_to_body), 4),
        "upper_shadow_to_body": round(float(upper_to_body), 4),
        "shadow_amplitude_pct": round(float(amplitude_pct), 4),
    }


def compute_obv_features(data: pd.DataFrame) -> dict[str, Any]:
    """向量化OBV + 三均线粘合判定，逻辑等价于参考脚本的calculate_obv()+is_sticky()逐行版本，
    但用np.diff/np.sign/cumsum一次性算完，避免全市场分片扫描时逐行.apply()的开销。"""
    close = data["close"].astype(float).to_numpy()
    volume = data["volume"].astype(float).to_numpy()
    if len(close) < 2:
        return {"obv_sticky": False, "obv_ma_spread_pct": None}
    diff = np.diff(close, prepend=close[0])
    direction = np.sign(diff)  # diff[0]恒为0 → direction[0]=0，与参考脚本obv[0]=0.0基线一致
    obv = pd.Series(np.cumsum(direction * volume), index=data.index)
    last_mas = []
    for period in OBV_MA_PERIODS:
        ma = obv.rolling(period).mean().iloc[-1]
        if pd.isna(ma):
            return {"obv_sticky": False, "obv_ma_spread_pct": None}
        last_mas.append(float(ma))
    max_v, min_v = max(last_mas), min(last_mas)
    if abs(max_v) < 1e-6:
        return {"obv_sticky": False, "obv_ma_spread_pct": None}
    spread_pct = (max_v - min_v) / abs(max_v) * 100.0
    return {
        "obv_sticky": bool(spread_pct <= OBV_STICKY_THRESHOLD * 100.0),
        "obv_ma_spread_pct": round(float(spread_pct), 3),
    }


def pe_score_from_raw(pe_raw: float | None) -> float:
    """市盈率评分：越低越好；亏损或无效给较低分。"""
    if pe_raw is None or (isinstance(pe_raw, float) and (math.isnan(pe_raw) or pe_raw <= 0)):
        return 25.0
    if pe_raw <= 10:
        return 100.0
    if pe_raw <= 15:
        return 90.0
    if pe_raw <= 20:
        return 75.0
    if pe_raw <= 30:
        return 55.0
    if pe_raw <= 50:
        return 35.0
    if pe_raw <= 80:
        return 20.0
    return 8.0


def build_short_term_note(
    close: float,
    total_score: float,
    ma_signal: str,
    is_tradeable: bool,
    stage: str,
    main_peak: float,
    avg_cost: float,
    ma5: float | None,
    ma10: float | None,
    ma20: float | None,
    wide_zone_low: float | None,
    wide_zone_high: float | None,
    recent_low: float,
    recent_high: float,
) -> str:
    """生成统一的短期视角标注：偏弱/中性/偏强 + 支撑/压力区间。

    修复：支撑区间原先按升序取最小的两个候选价位，等于每次都选中离现价最远的支撑，
    而不是最贴近现价、最有参考意义的那一档。现在改成取最贴近现价（数值最大）的两个。
    压力区间本来就是升序取最小两个 = 离现价最近的两档，逻辑是对的，没有改。
    """
    # 1. 判定短期强弱
    if total_score >= 60 and is_tradeable and (ma_signal.startswith("强多") or ma_signal.startswith("多头")):
        bias = "短期偏强"
    elif total_score <= 40 or ma_signal == "无" or stage in {"震荡", "出货"}:
        bias = "短期偏弱"
    else:
        bias = "短期中性"

    # 2. 支撑区间：优先宽幅区下沿 / 近期低点 / 现价下方5%~8%，取离现价最近的两档
    support_candidates = []
    if wide_zone_low is not None and wide_zone_low > 0:
        support_candidates.append(wide_zone_low)
    support_candidates.append(recent_low)
    support_candidates.append(close * 0.92)
    support_candidates.append(close * 0.95)
    support_raw = sorted(set(round(x, 2) for x in support_candidates if x < close))
    if len(support_raw) >= 2:
        support_low, support_high = support_raw[-2], support_raw[-1]
    elif support_raw:
        support_high = support_raw[-1]
        support_low = round(support_high * 0.98, 2)
    else:
        support_low = round(close * 0.92, 2)
        support_high = round(close * 0.95, 2)

    # 3. 压力区间：优先主峰 / MA10/MA20 / 宽幅区上沿 / 近期高点，取离现价最近的两档
    resist_candidates = []
    if main_peak > close:
        resist_candidates.append(main_peak)
    for ma in (ma5, ma10, ma20):
        if ma is not None and ma > close:
            resist_candidates.append(ma)
    if wide_zone_high is not None and wide_zone_high > close:
        resist_candidates.append(wide_zone_high)
    resist_candidates.append(recent_high)
    resist_candidates.append(close * 1.05)
    resist_raw = sorted(set(round(x, 2) for x in resist_candidates if x > close))
    if len(resist_raw) >= 2:
        resist_low, resist_high = resist_raw[0], resist_raw[min(1, len(resist_raw)-1)]
    elif resist_raw:
        resist_low = resist_raw[0]
        resist_high = round(resist_low * 1.03, 2)
    else:
        resist_low = round(close * 1.03, 2)
        resist_high = round(close * 1.06, 2)

    # 统一输出格式
    return f"{bias}，下方{support_low}-{support_high}元支撑，上方压力{resist_low}-{resist_high}元"


def features(chip: dict[float, float], close: float) -> dict[str, float]:
    items = sorted(chip.items())
    if not items:
        raise ValueError("empty_chip")
    prices = np.array([item[0] for item in items])
    weights = np.array([item[1] for item in items])
    total = weights.sum()
    peak_index = int(np.argmax(weights))
    main_peak, main_weight = float(prices[peak_index]), float(weights[peak_index])
    local = [
        index for index in range(len(weights))
        if (index == 0 or weights[index] >= weights[index - 1])
        and (index == len(weights) - 1 or weights[index] >= weights[index + 1])
    ]
    second = float(sorted((weights[index] for index in local), reverse=True)[1]) if len(local) > 1 else main_weight * 0.01
    cumulative = np.cumsum(weights) / total
    p5 = float(prices[min(np.searchsorted(cumulative, 0.05), len(prices) - 1)])
    p95 = float(prices[min(np.searchsorted(cumulative, 0.95), len(prices) - 1)])
    band = max(close * 0.01, STEP * 5)

    below_indices = [index for index in local if prices[index] < close]
    below_mask = prices < close
    below_total = float(weights[below_mask].sum()) if below_mask.any() else 0.0
    if below_indices and below_total > 1e-8:
        below_index = max(below_indices, key=lambda index: weights[index])
        below_peak, below_weight = float(prices[below_index]), float(weights[below_index])
        other_below_peaks = sorted(
            (weights[index] for index in below_indices if index != below_index), reverse=True
        )
        below_second = float(other_below_peaks[0]) if other_below_peaks else below_weight * 0.01
        below_gap = below_weight / max(below_second, 1e-8)
        below_band_ratio = float(
            weights[below_mask & (prices >= below_peak - band) & (prices <= below_peak + band)].sum()
            / below_total
        )
        below_peak_ratio = below_weight / below_total
        wide_low, wide_high, wide_ratio = find_wide_zone(
            prices[below_mask], weights[below_mask], below_total, WIDE_ZONE_MASS_TARGET
        )
    else:
        below_peak, below_gap, below_band_ratio, below_peak_ratio = None, 1.0, 0.0, 0.0
        wide_low, wide_high, wide_ratio = 0.0, 0.0, 0.0

    return {
        "main_peak": main_peak,
        "peak_ratio": main_weight / total,
        "peak_gap": main_weight / max(second, 1e-8),
        "band_ratio": float(weights[(prices >= main_peak - band) & (prices <= main_peak + band)].sum() / total),
        "conc90": (p95 - p5) / (p95 + p5) if p95 + p5 > 0 else 1.0,
        "p5": p5, "p95": p95,
        "avg_cost": float((prices * weights).sum() / total),
        "profit": float(weights[prices <= close].sum() / total),
        "below_peak": below_peak,
        "below_peak_gap": below_gap,
        "below_band_ratio": below_band_ratio,
        "below_peak_ratio": below_peak_ratio,
        "wide_zone_low": wide_low,
        "wide_zone_high": wide_high,
        "wide_zone_ratio": wide_ratio,
    }


def analyze(code: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    if is_beijing_stock(code):
        raise ValueError(f"beijing_stock_excluded:{code}")
    data = normalize_frame(frame)
    chip: dict[float, float] = {}
    previous_chip: dict[float, float] = {}
    n = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        if i == n - 1:
            previous_chip = dict(chip)  # 最后一天更新前的快照，供洗盘阶段判定的穿透计算使用
        cap = 1.0 if i >= n - OLD_CHIP_AGE_DAYS else OLD_CHIP_DECAY_CAP
        chip = update_chip(chip, row, decay_cap=cap)
    last = data.iloc[-1]
    close = float(last["close"])
    feat = features(chip, close)
    ma_feat = compute_ma_features(data)
    shadow_feat = compute_shadow_features(data)
    obv_feat = compute_obv_features(data)

    # ===== 洗盘阶段判定（移植自筹码尖峰蒸馏引擎 classify_stage，取代原 below/wide 洗盘启发式） =====
    cross_ratio, profit_cross, locked_cross, penetrate = _cross_metrics(
        previous_chip, float(last["open"]), float(last["high"]), float(last["low"]), close, float(last["turnover"])
    )
    conc90_width_pct, cost90_low, cost90_high = concentration_band(chip, 0.90)
    stage_winner = _stage_winner(chip, close)
    lookback_window = data.iloc[max(0, n - 60):n]
    recent_high = float(lookback_window["high"].max())
    recent_low = float(lookback_window["low"].min())
    stage, stage_note = classify_stage(
        close, feat["avg_cost"], stage_winner, conc90_width_pct, recent_high, recent_low, cross_ratio
    )
    is_washout = bool(stage == "洗盘")
    prior_close = float(data.iloc[-2]["close"]) if n >= 2 else float(last["open"])
    pct_change = (close - prior_close) / prior_close if prior_close > 0 else 0.0

    # ===== 首红判定（仅用日线，成本低） =====
    # 昨天阴/平线（收<=开），今天收阳（收>开）——"第一天翻红"。
    # 盘中MA5拐头的确认需要分钟级数据，见 fetch_intraday_ma5_turnup()，
    # 由分片脚本对通过本判定的候选股票单独调用，不在此处直接发起网络请求。
    is_first_red_daily = False
    if n >= 2:
        prior_row = data.iloc[-2]
        prior_was_down_or_flat = float(prior_row["close"]) <= float(prior_row["open"])
        today_is_red = close > float(last["open"])
        is_first_red_daily = bool(prior_was_down_or_flat and today_is_red)
    washout_score = (
        calc_washout_score(cross_ratio, profit_cross, locked_cross, penetrate, stage_winner, pct_change)
        if is_washout else 0.0
    )

    ma5 = data["close"].tail(5).mean()
    ma10 = data["close"].tail(10).mean()
    trend = bool(close > ma5 and ma5 > ma10) or ma_feat["ma_alignment"]
    volume_ma = data["volume"].iloc[-6:-1].mean()
    volume_ratio = float(last["volume"] / volume_ma) if volume_ma > 0 else 0.0
    in_zone = feat["main_peak"] * APPROACH_RATIO <= close < feat["main_peak"] * BREAK_RATIO
    confirmed = volume_ratio >= VOL_MULTIPLIER and feat["conc90"] <= CONC_THRESHOLD
    tradeable = bool(in_zone and trend and confirmed)
    amplitude = (float(last["high"]) - float(last["low"])) / close if close else 1.0

    line = 100 * (
        0.30 * np.clip(1 - amplitude / 0.05, 0, 1)
        + 0.25 * np.clip(float(last["turnover"]) / 0.03, 0, 1)
        + 0.30 * np.clip(feat["band_ratio"] / 0.25, 0, 1)
        + 0.15 * (1 if amplitude < 0.012 else 0.5 if amplitude < 0.025 else 0)
    )
    conc = 100 * np.clip(1 - (feat["conc90"] - 0.05) / 0.35, 0, 1)
    peak = 100 * (
        0.55 * np.clip(feat["band_ratio"] / 0.30, 0, 1)
        + 0.25 * np.clip(np.log1p(feat["peak_gap"]) / 4, 0, 1)
        + 0.20 * np.clip(feat["peak_ratio"] / 0.02, 0, 1)
    )
    position = (
        np.clip(
            (close - feat["main_peak"] * APPROACH_RATIO)
            / max(feat["main_peak"] * (BREAK_RATIO - APPROACH_RATIO), 1e-8),
            0, 1,
        )
        if in_zone
        else (
            0.35 if close >= feat["main_peak"] * BREAK_RATIO
            else max(0, 1 + (close - feat["main_peak"]) / feat["main_peak"] / 0.15)
        )
    )
    breakout = 100 * (
        0.35 * position
        + 0.25 * (1 if trend else 0.25)
        + 0.20 * np.clip(volume_ratio / VOL_MULTIPLIER, 0, 1.2) / 1.2
        + 0.10 * (1 if confirmed else 0.4)
        + 0.10 * (1 if tradeable else 0.45)
    )
    profit_pct = feat["profit"] * 100
    profit = (
        100 if 20 <= profit_pct <= 55
        else max(5, 100 - profit_pct) if profit_pct > 70
        else max(10, profit_pct * 2) if profit_pct < 10
        else 60
    )
    ma_score = float(ma_feat["ma_score"])

    # ===== 市盈率价值评分 =====
    pe_raw = None
    if "pe_ttm" in last.index and pd.notna(last.get("pe_ttm")):
        try:
            pe_raw = float(last["pe_ttm"])
        except (TypeError, ValueError):
            pe_raw = None
    pe_score = pe_score_from_raw(pe_raw)

    total = (
        WEIGHTS["line"] * line
        + WEIGHTS["conc"] * conc
        + WEIGHTS["peak"] * peak
        + WEIGHTS["break"] * breakout
        + WEIGHTS["profit"] * profit
        + WEIGHTS["ma"] * ma_score
        + WEIGHTS["pe"] * pe_score
    )

    is_below_spike = bool(
        feat["below_peak"] is not None
        and feat["below_band_ratio"] >= BELOW_SPIKE_BAND_MIN
        and feat["below_peak_gap"] >= BELOW_SPIKE_GAP_MIN
        and feat["below_peak_ratio"] >= BELOW_SPIKE_RATIO_MIN
    )

    # 筹码主峰尖峰（不区分现价上/下方，只判定该主峰本身是否窄而突出）
    is_chip_spike = bool(
        feat["band_ratio"] >= SPIKE_BAND_MIN
        and feat["peak_gap"] >= SPIKE_GAP_MIN
        and feat["peak_ratio"] >= SPIKE_RATIO_MIN
    )
    main_peak_above_close = bool(feat["main_peak"] > close)

    # 宽幅堆积区
    wide_width_pct = 0.0
    is_wide_zone = False
    wide_score = 0.0
    wide_state = "无"
    wide_dist_pct = None
    if feat["wide_zone_high"] > 0 and close > 0:
        wide_width_pct = (feat["wide_zone_high"] - feat["wide_zone_low"]) / close * 100
        is_wide_zone = bool(
            feat["wide_zone_ratio"] >= WIDE_ZONE_MASS_TARGET
            and WIDE_ZONE_MIN_WIDTH_PCT * 100 <= wide_width_pct <= WIDE_ZONE_MAX_WIDTH_PCT * 100
        )
        if is_wide_zone:
            wz_top = feat["wide_zone_high"]
            wide_dist_pct = (close - wz_top) / wz_top * 100
            wide_in_zone = wz_top * APPROACH_RATIO <= close < wz_top * BREAK_RATIO
            wide_confirmed = bool(volume_ratio >= VOL_MULTIPLIER and trend)
            s_ratio = np.clip((feat["wide_zone_ratio"] - WIDE_ZONE_MASS_TARGET) / (1 - WIDE_ZONE_MASS_TARGET), 0, 1)
            s_narrow = np.clip(1 - wide_width_pct / (WIDE_ZONE_MAX_WIDTH_PCT * 100), 0, 1)
            s_pos = 1.0 if wide_in_zone else float(np.clip(1 - abs(wide_dist_pct) / 30, 0, 1))
            s_confirm = 1.0 if wide_confirmed else (0.6 if trend else 0.3)
            wide_score = float(np.clip(100 * (0.30 * s_ratio + 0.25 * s_narrow + 0.25 * s_pos + 0.20 * s_confirm), 0, 100))
            # 洗盘文案统一：是否用"洗盘"这个词，交给顶层 classify_stage 的 is_washout 说了算，
            # 这里只负责描述相对宽幅堆积区上沿的位置，不再自己独立判断"算不算洗盘"，
            # 避免和 stage 字段在同一条记录上给出矛盾结论。
            wide_position_desc = (
                "贴近宽幅堆积区上沿未确认量能趋势" if wide_in_zone
                else "宽幅堆积区蓄势中" if close < wz_top * BREAK_RATIO
                else "已远离宽幅堆积区"
            )
            if wide_in_zone and wide_confirmed:
                wide_state = "买入·贴近宽幅堆积区上沿+量能趋势确认"
            elif is_washout:
                wide_state = f"洗盘·{wide_position_desc}"
            else:
                wide_state = f"观察·{wide_position_desc}"

    signal = (
        "可交易·接近尖峰+趋势确认" if tradeable
        else "尖峰关注·现价下方长红柱" if is_below_spike
        else "尖峰关注·宽幅堆积区" if is_wide_zone
        else "尖峰关注·筹码主峰" if is_chip_spike
        else "长影线关注" if shadow_feat["has_long_shadow"]
        else "观察·接近尖峰未确认" if in_zone
        else "无"
    )
    if ma_feat["ma_signal"].startswith("强多") and (
        tradeable or is_below_spike or is_wide_zone or is_chip_spike
        or shadow_feat["has_long_shadow"] or in_zone
    ):
        signal = "高潜力·筹码+均线共振"
    elif ma_feat["ma_signal"].startswith("强多"):
        signal = "均线强多·次日潜力关注"

    # 下方长红柱专属
    below_score = 0.0
    below_state = "无"
    below_dist_pct = None
    if feat["below_peak"] is not None and feat["below_peak"] > 0:
        bp = feat["below_peak"]
        below_dist_pct = (close - bp) / bp * 100
        below_in_zone = bp * APPROACH_RATIO <= close < bp * BREAK_RATIO
        below_confirmed = bool(volume_ratio >= VOL_MULTIPLIER and trend)
        s_band = np.clip(feat["below_band_ratio"] / 0.95, 0, 1)
        s_gap = np.clip(np.log1p(feat["below_peak_gap"]) / np.log1p(50), 0, 1)
        s_ratio = np.clip(feat["below_peak_ratio"] / 0.05, 0, 1)
        s_pos = 1.0 if below_in_zone else float(np.clip(1 - abs(below_dist_pct) / 30, 0, 1))
        s_confirm = 1.0 if below_confirmed else (0.6 if trend else 0.3)
        below_score = float(np.clip(
            100 * (0.30 * s_band + 0.15 * s_gap + 0.15 * s_ratio + 0.25 * s_pos + 0.15 * s_confirm), 0, 100
        ))
        # 洗盘文案统一：同 wide_state，"洗盘"这个词是否出现由 is_washout 决定，
        # 这里只描述相对下方长红柱的位置。
        below_position_desc = (
            "贴近下方长红柱未确认量能趋势" if below_in_zone
            else "下方长红柱蓄势中" if close < bp * BREAK_RATIO
            else "已远离下方长红柱"
        )
        if not is_below_spike:
            below_state = "无"
        elif below_in_zone and below_confirmed:
            below_state = "买入·贴近下方长红柱+量能趋势确认"
        elif is_washout:
            below_state = f"洗盘·{below_position_desc}"
        else:
            below_state = f"观察·{below_position_desc}"

    # ===== 短期视角标注（偏弱/中性/偏强 + 支撑压力） =====
    short_term_note = build_short_term_note(
        close=close,
        total_score=float(total),
        ma_signal=ma_feat["ma_signal"],
        is_tradeable=tradeable,
        stage=stage,
        main_peak=feat["main_peak"],
        avg_cost=feat["avg_cost"],
        ma5=ma_feat["ma5"],
        ma10=ma_feat["ma10"],
        ma20=ma_feat["ma20"],
        wide_zone_low=feat["wide_zone_low"] if is_wide_zone else None,
        wide_zone_high=feat["wide_zone_high"] if is_wide_zone else None,
        recent_low=recent_low,
        recent_high=recent_high,
    )

    return {
        "code": str(code).zfill(6),
        "name": name,
        "date": str(last["date"].date()),
        "close": round(close, 2),
        "main_peak": round(feat["main_peak"], 2),
        "avg_cost": round(feat["avg_cost"], 2),
        "dist_to_peak_pct": round((close - feat["main_peak"]) / feat["main_peak"] * 100, 2),
        "band_ratio_pct": round(feat["band_ratio"] * 100, 2),
        "conc90_pct": round(feat["conc90"] * 100, 2),
        "profit_pct": round(profit_pct, 2),
        "p5": round(feat["p5"], 2),
        "p95": round(feat["p95"], 2),
        "short_term_note": short_term_note,
        "turnover_pct": round(float(last["turnover"]) * 100, 2),
        "volume_ratio": round(volume_ratio, 2),
        "line_score": round(float(line), 1),
        "conc_score": round(float(conc), 1),
        "peak_score": round(float(peak), 1),
        "break_score": round(float(breakout), 1),
        "profit_score": round(float(profit), 1),
        "ma_score": ma_score,
        "pe_ttm": round(pe_raw, 2) if pe_raw is not None else None,
        "pe_score": round(pe_score, 1),
        "total_score": round(float(total), 1),
        "is_approaching": in_zone,
        "is_tradeable": tradeable,
        "below_peak": round(feat["below_peak"], 2) if feat["below_peak"] is not None else None,
        "below_dist_pct": round(below_dist_pct, 2) if below_dist_pct is not None else None,
        "below_band_ratio_pct": round(feat["below_band_ratio"] * 100, 2),
        "below_peak_ratio_pct": round(feat["below_peak_ratio"] * 100, 2),
        "below_peak_gap": round(feat["below_peak_gap"], 2),
        "below_score": round(below_score, 1),
        "below_state": below_state,
        "is_below_spike": is_below_spike,
        "is_chip_spike": is_chip_spike,
        "main_peak_above_close": main_peak_above_close,
        "daily_long_lower": shadow_feat["daily_long_lower"],
        "daily_long_upper": shadow_feat["daily_long_upper"],
        "has_long_shadow": shadow_feat["has_long_shadow"],
        "lower_shadow_ratio": shadow_feat["lower_shadow_ratio"],
        "upper_shadow_ratio": shadow_feat["upper_shadow_ratio"],
        "lower_shadow_to_body": shadow_feat["lower_shadow_to_body"],
        "upper_shadow_to_body": shadow_feat["upper_shadow_to_body"],
        "shadow_amplitude_pct": shadow_feat["shadow_amplitude_pct"],
        "obv_sticky": obv_feat["obv_sticky"],
        "obv_ma_spread_pct": obv_feat["obv_ma_spread_pct"],
        "wide_zone_low": round(feat["wide_zone_low"], 2) if feat["wide_zone_high"] > 0 else None,
        "wide_zone_high": round(feat["wide_zone_high"], 2) if feat["wide_zone_high"] > 0 else None,
        "wide_zone_ratio_pct": round(feat["wide_zone_ratio"] * 100, 2),
        "wide_width_pct": round(wide_width_pct, 2),
        "wide_dist_pct": round(wide_dist_pct, 2) if wide_dist_pct is not None else None,
        "wide_score": round(wide_score, 1),
        "wide_state": wide_state,
        "is_wide_zone": is_wide_zone,
        "signal": signal,
        "ma5": ma_feat["ma5"],
        "ma10": ma_feat["ma10"],
        "ma20": ma_feat["ma20"],
        "ma60": ma_feat["ma60"],
        "ma_alignment": ma_feat["ma_alignment"],
        "golden_cross_5_10": ma_feat["golden_cross_5_10"],
        "golden_cross_10_20": ma_feat["golden_cross_10_20"],
        "ma_signal": ma_feat["ma_signal"],
        "stage": stage,
        "stage_note": stage_note,
        "is_washout": is_washout,
        "washout_score": washout_score,
        "cross_ratio_pct": round(cross_ratio * 100, 2),
        "conc90_width_pct": round(conc90_width_pct, 2),
        "cost90_low": round(cost90_low, 2),
        "cost90_high": round(cost90_high, 2),
        "is_first_red_daily": is_first_red_daily,
        "profile": "winrate",
        "confirm_mode": "and",
    }


def self_test() -> None:
    dates = pd.date_range("2025-01-01", periods=110, freq="B")
    close = np.linspace(10, 12, 110)
    frame = pd.DataFrame({
        "date": dates,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.linspace(1000, 2000, 110),
        "amount": close * np.linspace(1000, 2000, 110),
        "turnover": [0.02] * 110,
        "pe_ttm": [15.0] * 110,
    })
    result = analyze("000001", "样本", frame)
    assert result["code"] == "000001" and 0 <= result["total_score"] <= 100
    assert "conc90_pct" in result and "ma_score" in result and "ma_signal" in result
    assert "pe_score" in result and "pe_ttm" in result
    assert result["pe_score"] == 90.0  # PE=15 → 90
    assert {"stage", "stage_note", "is_washout", "washout_score", "cross_ratio_pct", "conc90_width_pct"}.issubset(result)
    assert result["stage"] in {"吸筹", "洗盘", "拉升", "出货", "震荡"}
    assert "is_first_red_daily" in result and isinstance(result["is_first_red_daily"], bool)
    assert callable(fetch_intraday_ma5_turnup)
    assert callable(fetch_main_fund_flow)
    assert {"is_chip_spike", "main_peak_above_close", "daily_long_lower", "daily_long_upper",
            "has_long_shadow", "lower_shadow_ratio", "upper_shadow_ratio",
            "obv_sticky", "obv_ma_spread_pct"}.issubset(result)
    assert "short_term_note" in result and isinstance(result["short_term_note"], str)
    assert any(x in result["short_term_note"] for x in ("短期偏弱", "短期中性", "短期偏强"))
    assert "支撑" in result["short_term_note"] and "压力" in result["short_term_note"]
    # 支撑区间方向回归：解析出的支撑区间必须都在现价下方，且上沿(support_high)不能小于下沿。
    import re
    match = re.search(r"下方([\d.]+)-([\d.]+)元支撑", result["short_term_note"])
    assert match is not None, "short_term_note缺少支撑区间"
    s_low, s_high = float(match.group(1)), float(match.group(2))
    assert s_low <= s_high <= result["close"], f"支撑区间方向仍有问题: {s_low}-{s_high}, close={result['close']}"
    # 洗盘文案统一回归：below_state/wide_state 里只要出现"洗盘"，就必须和顶层 is_washout 一致，
    # 不能各说各话。
    if result["below_state"].startswith("洗盘"):
        assert result["is_washout"] is True, "below_state洗盘文案与is_washout矛盾"
    if result["wide_state"].startswith("洗盘"):
        assert result["is_washout"] is True, "wide_state洗盘文案与is_washout矛盾"
    try:
        analyze("830001", "北交测试", frame)
        raise AssertionError("should have excluded beijing stock")
    except ValueError as e:
        assert "beijing_stock_excluded" in str(e)
    print("self_test passed")


if __name__ == "__main__":
    self_test()
