#!/usr/bin/env python3
"""FINAL Chip 日线筹码峰、集中度、突破、五维评分 + 市盈率(PE) + 多均线系统。

新逻辑（相对原版）：
- 增加 peTTM 抓取与 pe_score（低PE高分，亏损/极高PE低分）
- WEIGHTS 加入 pe=0.12，其余维度按比例微调，总和仍为 1.0
- analyze() 返回 pe_ttm / pe_score，total_score 含价值维度
- 排除北交所；OLD_CHIP_DECAY_CAP=0.005 保留

用法：
  python final_chip_research.py          # self_test
  from final_chip_research import fetch_ohlcv, analyze
"""
from __future__ import annotations

import math
import multiprocessing as mp
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
# 筹码集中度确认：conc90 ≤ 0.30（约 30% 相对宽度内）
APPROACH_RATIO, BREAK_RATIO, VOL_MULTIPLIER, CONC_THRESHOLD = 0.97, 1.01, 1.5, 0.30

# 周线 MA5/MA10「即将粘合」阈值
WEEKLY_MA_GLUE_GAP_PCT = 0.03          # |MA5-MA10|/close ≤ 3% → 已粘合
WEEKLY_MA_GLUE_NEAR_PCT = 0.06         # ≤ 6% → 即将粘合
WEEKLY_MA_MIN_BARS = 12

# 权重含周线粘合分 glue（总和=1.0）
WEIGHTS = {
    "line": 0.09,
    "conc": 0.09,
    "peak": 0.07,
    "break": 0.30,
    "profit": 0.09,
    "ma": 0.15,
    "pe": 0.11,
    "glue": 0.10,
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


def compute_weekly_ma_glue(data: pd.DataFrame) -> dict[str, Any]:
    """周线五日线与十日线即将粘合检测。

    用日线 OHLCV 重采样为周K（周五为周期末），计算周 MA5 / MA10：
    - weekly_ma5 / weekly_ma10：最新一周的均线值
    - weekly_ma_gap_pct：|MA5-MA10|/close * 100
    - weekly_ma_glue：间距 ≤ 3%（粘合）
    - weekly_ma_glue_near：间距 ≤ 5%（即将粘合）
    - weekly_ma_converging：近 2~3 周间距在收窄
    - weekly_ma_glue_signal：文案状态
    """
    empty = {
        "weekly_ma5": None,
        "weekly_ma10": None,
        "weekly_ma_gap_pct": None,
        "weekly_ma_glue": False,
        "weekly_ma_glue_near": False,
        "weekly_ma_converging": False,
        "weekly_ma_glue_score": 0.0,
        "weekly_ma_glue_signal": "无",
    }
    if data is None or len(data) < 30:
        return empty

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    # 周K：取每周最后一个交易日收盘（与常见行情软件一致）
    weekly = df["close"].resample("W-FRI").last().dropna()
    if len(weekly) < WEEKLY_MA_MIN_BARS:
        return empty

    w_ma5 = weekly.rolling(5, min_periods=5).mean()
    w_ma10 = weekly.rolling(10, min_periods=10).mean()
    if pd.isna(w_ma5.iloc[-1]) or pd.isna(w_ma10.iloc[-1]):
        return empty

    last_close = float(weekly.iloc[-1])
    last_ma5 = float(w_ma5.iloc[-1])
    last_ma10 = float(w_ma10.iloc[-1])
    if last_close <= 0:
        return empty

    gap_pct = abs(last_ma5 - last_ma10) / last_close * 100.0
    glue = gap_pct <= WEEKLY_MA_GLUE_GAP_PCT * 100
    glue_near = gap_pct <= WEEKLY_MA_GLUE_NEAR_PCT * 100

    # 近 3 周间距是否收窄
    converging = False
    if len(w_ma5) >= 3 and len(w_ma10) >= 3:
        gaps = []
        for i in range(-3, 0):
            c = float(weekly.iloc[i])
            if c > 0 and pd.notna(w_ma5.iloc[i]) and pd.notna(w_ma10.iloc[i]):
                gaps.append(abs(float(w_ma5.iloc[i]) - float(w_ma10.iloc[i])) / c)
        if len(gaps) >= 2:
            converging = gaps[-1] < gaps[0]

    # 评分 0~100：间距越近越高，收窄加分
    # gap 0% → 100，gap 5% → 约 40，gap 10% → 0
    gap_score = float(np.clip(100 * (1 - gap_pct / 10.0), 0, 100))
    if converging:
        gap_score = float(np.clip(gap_score + 15, 0, 100))
    if glue:
        gap_score = float(np.clip(gap_score + 10, 0, 100))

    if glue and converging:
        signal = "周线5/10粘合·收窄确认"
    elif glue:
        signal = "周线5/10已粘合"
    elif glue_near and converging:
        signal = "周线5/10即将粘合·收窄中"
    elif glue_near:
        signal = "周线5/10接近粘合"
    elif converging:
        signal = "周线5/10间距收窄"
    else:
        signal = "无"

    return {
        "weekly_ma5": round(last_ma5, 2),
        "weekly_ma10": round(last_ma10, 2),
        "weekly_ma_gap_pct": round(gap_pct, 2),
        "weekly_ma_glue": bool(glue),
        "weekly_ma_glue_near": bool(glue_near),
        "weekly_ma_converging": bool(converging),
        "weekly_ma_glue_score": round(gap_score, 1),
        "weekly_ma_glue_signal": signal,
    }


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
    n = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        cap = 1.0 if i >= n - OLD_CHIP_AGE_DAYS else OLD_CHIP_DECAY_CAP
        chip = update_chip(chip, row, decay_cap=cap)
    last = data.iloc[-1]
    close = float(last["close"])
    feat = features(chip, close)
    ma_feat = compute_ma_features(data)
    weekly_glue = compute_weekly_ma_glue(data)
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
    glue_score = float(weekly_glue.get("weekly_ma_glue_score") or 0.0)

    total = (
        WEIGHTS["line"] * line
        + WEIGHTS["conc"] * conc
        + WEIGHTS["peak"] * peak
        + WEIGHTS["break"] * breakout
        + WEIGHTS["profit"] * profit
        + WEIGHTS["ma"] * ma_score
        + WEIGHTS["pe"] * pe_score
        + WEIGHTS["glue"] * glue_score
    )

    is_below_spike = bool(
        feat["below_peak"] is not None
        and feat["below_band_ratio"] >= BELOW_SPIKE_BAND_MIN
        and feat["below_peak_gap"] >= BELOW_SPIKE_GAP_MIN
        and feat["below_peak_ratio"] >= BELOW_SPIKE_RATIO_MIN
    )

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
            if wide_in_zone and wide_confirmed:
                wide_state = "买入·贴近宽幅堆积区上沿+量能趋势确认"
            elif wide_in_zone:
                wide_state = "洗盘·贴近宽幅堆积区上沿未确认量能趋势"
            elif close < wz_top * BREAK_RATIO:
                wide_state = "洗盘·宽幅堆积区蓄势中"
            else:
                wide_state = "观察·已远离宽幅堆积区"

    signal = (
        "可交易·接近尖峰+趋势确认" if tradeable
        else "尖峰关注·现价下方长红柱" if is_below_spike
        else "观察·接近尖峰未确认" if in_zone
        else "无"
    )
    if ma_feat["ma_signal"].startswith("强多") and (tradeable or is_below_spike or in_zone):
        signal = "高潜力·筹码+均线共振"
    elif ma_feat["ma_signal"].startswith("强多"):
        signal = "均线强多·次日潜力关注"
    # 周线5/10粘合增强：与强多或可交易叠加时提升标记
    if weekly_glue.get("weekly_ma_glue_near") and signal in (
        "高潜力·筹码+均线共振", "均线强多·次日潜力关注", "可交易·接近尖峰+趋势确认"
    ):
        signal = signal + "·周线粘合"
    elif weekly_glue.get("weekly_ma_glue_near") and signal == "无":
        signal = "周线5/10即将粘合·关注"

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
        if not is_below_spike:
            below_state = "无"
        elif below_in_zone and below_confirmed:
            below_state = "买入·贴近下方长红柱+量能趋势确认"
        elif below_in_zone:
            below_state = "洗盘·贴近下方长红柱未确认量能趋势"
        elif close < bp * BREAK_RATIO:
            below_state = "洗盘·下方长红柱蓄势中"
        else:
            below_state = "观察·已远离下方长红柱"

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
        # 周线五日/十日即将粘合
        "weekly_ma5": weekly_glue["weekly_ma5"],
        "weekly_ma10": weekly_glue["weekly_ma10"],
        "weekly_ma_gap_pct": weekly_glue["weekly_ma_gap_pct"],
        "weekly_ma_glue": weekly_glue["weekly_ma_glue"],
        "weekly_ma_glue_near": weekly_glue["weekly_ma_glue_near"],
        "weekly_ma_converging": weekly_glue["weekly_ma_converging"],
        "weekly_ma_glue_score": weekly_glue["weekly_ma_glue_score"],
        "weekly_ma_glue_signal": weekly_glue["weekly_ma_glue_signal"],
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
    assert "weekly_ma_glue_signal" in result
    assert "weekly_ma_gap_pct" in result
    try:
        analyze("830001", "北交测试", frame)
        raise AssertionError("should have excluded beijing stock")
    except ValueError as e:
        assert "beijing_stock_excluded" in str(e)
    print("self_test passed")


if __name__ == "__main__":
    self_test()
