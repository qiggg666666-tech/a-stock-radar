#!/usr/bin/env python3
"""FINAL Chip 的日线筹码峰、集中度、突破和五维评分研究内核。
修改版：红色尖峰专门定义为「现价下方」的又长又窄尖峰柱，
不再使用传统「套牢区（现价上方）」定义。
针对图中 11.30\~15 一带那种超长红柱进行参数优化。
"""
from __future__ import annotations

import math
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

# ====================== 核心参数（已按「下方长窄红尖峰」优化）======================
DECAY, STEP, LOOKBACK = 1.0, 0.01, 100

# 窄幅但非零区间（涨停/一字板/单日巨量窄幅拉升）：区间宽度相对现价趋近于零时，
# 三角分布公式里的分母(high-low)理论上该让密度趋于无穷、图形收成一根尖峰。
# 相对容差再收紧一点，让真正的窄日更容易变成单一尖峰。
NARROW_DAY_REL_TOL = 0.0025

# 接近主峰 / 突破主峰 比例（围绕「下方主峰」使用）
APPROACH_RATIO, BREAK_RATIO = 0.96, 1.015

# 量能确认倍数 & 集中度阈值（下方尖峰场景下适当放宽一点集中度）
VOL_MULTIPLIER, CONC_THRESHOLD = 1.4, 0.22

# 五维权重：略微提高 peak 权重，因为本版核心就是「下方长窄尖峰」
WEIGHTS = {
    "line": 0.12,
    "conc": 0.13,
    "peak": 0.18,      # 提高尖峰权重
    "break": 0.42,
    "profit": 0.15,
}


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
        "换手率": "turnover", "turn": "turnover",
    }
    data = frame.rename(columns={key: value for key, value in columns.items() if key in frame.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    if data.empty or any(item not in data.columns for item in required):
        raise ValueError("invalid_ohlcv_schema")
    for item in required[1:] + ["amount", "turnover"]:
        if item not in data:
            data[item] = 0.0
        data[item] = pd.to_numeric(data[item], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=required).sort_values("date").drop_duplicates("date")

    # 修复：baostock/akshare 的换手率本来就是百分比数字（如 1.85 代表1.85%），
    # 统一 /100 转成 0\~1 小数。
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
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
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
        result = bs.query_history_k_data_plus(
            f"{exchange}.{code}",
            "date,open,high,low,close,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
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
    end = datetime.now().date()
    start = end - timedelta(days=LOOKBACK + 110)
    errors: list[str] = []
    for label, request in (("akshare", _ak_history), ("baostock", _bs_history)):
        for attempt in range(1, max(retries, 1) + 1):
            try:
                return (
                    provider_call(
                        f"{label}:{code}",
                        timeout_seconds,
                        lambda fn=request: fn(code, start.isoformat(), end.isoformat()),
                    ),
                    label,
                    errors,
                )
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
    # 更严格的窄日判定 → 更容易形成真正的尖峰
    if math.isclose(high, low) or (high - low) <= max(average, 1e-8) * NARROW_DAY_REL_TOL:
        return {round(average, 2): volume}
    prices = np.unique(
        np.round(np.linspace(low, high, max(int(round((high - low) / STEP)) + 1, 2)), 2)
    )
    values = np.array([
        max((price - low) / max(average - low, 1e-8), 0.0)
        if price <= average
        else max((high - price) / max(high - average, 1e-8), 0.0)
        for price in prices
    ])
    values = values / values.sum() if values.sum() > 0 else np.ones(len(prices)) / len(prices)
    return {float(price): float(weight * volume) for price, weight in zip(prices, values)}


def update_chip(chip: dict[float, float], row: pd.Series) -> dict[float, float]:
    moved = min(max(float(row["turnover"]) * DECAY, 0.0), 1.0)
    result = {
        price: weight * (1 - moved)
        for price, weight in chip.items()
        if weight * (1 - moved) > 1e-8
    }
    for price, weight in daily_distribution(
        float(row["low"]), float(row["high"]), average_price(row), float(row["volume"])
    ).items():
        result[price] = result.get(price, 0.0) + weight * moved
    return result


def features(chip: dict[float, float], close: float) -> dict[str, float]:
    """核心改动：主峰、峰间隙、带宽占比全部只在现价下方计算。
    专门针对「又长又窄的红色尖峰柱」（如图中 11.30\~15 一带）。
    """
    items = sorted(chip.items())
    if not items:
        raise ValueError("empty_chip")

    prices = np.array([item[0] for item in items])
    weights = np.array([item[1] for item in items])
    total = weights.sum()

    # ========== 只在现价下方寻找红色尖峰 ==========
    below_mask = prices < close
    if below_mask.any() and weights[below_mask].sum() > 1e-8:
        prices_b = prices[below_mask]
        weights_b = weights[below_mask]
        total_b = weights_b.sum()
        is_below = True
    else:
        # 极端保护：下方没有筹码时回退全量
        prices_b, weights_b, total_b = prices, weights, total
        is_below = False

    peak_index = int(np.argmax(weights_b))
    main_peak = float(prices_b[peak_index])
    main_weight = float(weights_b[peak_index])

    # 局部峰（用于计算峰间隙 → 衡量「又窄」）
    local = [
        i for i in range(len(weights_b))
        if (i == 0 or weights_b[i] >= weights_b[i - 1])
        and (i == len(weights_b) - 1 or weights_b[i] >= weights_b[i + 1])
    ]
    second = (
        float(sorted((weights_b[i] for i in local), reverse=True)[1])
        if len(local) > 1
        else main_weight * 0.008   # 更严格一点，让真正尖峰的 gap 更大
    )

    # 90%集中度仍用全量（更稳定）
    cumulative = np.cumsum(weights) / total
    p5 = float(prices[min(np.searchsorted(cumulative, 0.05), len(prices) - 1)])
    p95 = float(prices[min(np.searchsorted(cumulative, 0.95), len(prices) - 1)])

    # 带宽：围绕主峰的窄带（衡量「又窄」）
    band = max(close * 0.008, STEP * 4)   # 比原来更窄的带宽，突出尖峰
    band_mask = (prices_b >= main_peak - band) & (prices_b <= main_peak + band)

    return {
        "main_peak": main_peak,
        "peak_ratio": main_weight / total_b,                    # 下方尖峰占比（越长越高）
        "peak_gap": main_weight / max(second, 1e-8),            # 尖峰锐度（越大越窄）
        "band_ratio": float(weights_b[band_mask].sum() / total_b),  # 尖峰带宽占比
        "conc90": (p95 - p5) / (p95 + p5) if p95 + p5 > 0 else 1.0,
        "p5": p5,
        "p95": p95,
        "avg_cost": float((prices * weights).sum() / total),
        "profit": float(weights[prices <= close].sum() / total),
        "is_below_peak": is_below,
    }


def analyze(code: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    data = normalize_frame(frame)
    chip: dict[float, float] = {}
    for _, row in data.iterrows():
        chip = update_chip(chip, row)

    last = data.iloc[-1]
    close = float(last["close"])
    feat = features(chip, close)

    ma5 = data["close"].tail(5).mean()
    ma10 = data["close"].tail(10).mean()
    trend = bool(close > ma5 and ma5 > ma10)

    volume_ma = data["volume"].iloc[-6:-1].mean()
    volume_ratio = float(last["volume"] / volume_ma) if volume_ma > 0 else 0.0

    # 围绕「下方主峰」判断接近/突破
    in_zone = feat["main_peak"] * APPROACH_RATIO <= close < feat["main_peak"] * BREAK_RATIO
    confirmed = volume_ratio >= VOL_MULTIPLIER and feat["conc90"] <= CONC_THRESHOLD
    tradeable = bool(in_zone and trend and confirmed and feat.get("is_below_peak", False))

    amplitude = (float(last["high"]) - float(last["low"])) / close if close else 1.0

    # 线型分（保持原逻辑）
    line = 100 * (
        0.30 * np.clip(1 - amplitude / 0.05, 0, 1)
        + 0.25 * np.clip(float(last["turnover"]) / 0.03, 0, 1)
        + 0.30 * np.clip(feat["band_ratio"] / 0.25, 0, 1)
        + 0.15 * (1 if amplitude < 0.012 else 0.5 if amplitude < 0.025 else 0)
    )

    # 集中度分
    conc = 100 * np.clip(1 - (feat["conc90"] - 0.05) / 0.35, 0, 1)

    # ========== 尖峰分：专门奖励「又长又窄」的下方红柱 ==========
    # band_ratio 高 → 窄；peak_gap 高 → 长且锐；peak_ratio 高 → 占比大
    peak = 100 * (
        0.42 * np.clip(feat["band_ratio"] / 0.22, 0, 1)          # 窄（带宽集中）
        + 0.38 * np.clip(np.log1p(feat["peak_gap"]) / 3.2, 0, 1) # 长且锐（峰间隙）
        + 0.20 * np.clip(feat["peak_ratio"] / 0.012, 0, 1)       # 占比
    )

    # 突破分
    position = (
        np.clip(
            (close - feat["main_peak"] * APPROACH_RATIO)
            / max(feat["main_peak"] * (BREAK_RATIO - APPROACH_RATIO), 1e-8),
            0,
            1,
        )
        if in_zone
        else (
            0.35
            if close >= feat["main_peak"] * BREAK_RATIO
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

    # 获利盘分
    profit_pct = feat["profit"] * 100
    if 20 <= profit_pct <= 55:
        profit = 100
    elif profit_pct > 70:
        profit = max(5, 100 - profit_pct)
    elif profit_pct < 10:
        profit = max(10, profit_pct * 2)
    else:
        profit = 60

    total = (
        WEIGHTS["line"] * line
        + WEIGHTS["conc"] * conc
        + WEIGHTS["peak"] * peak
        + WEIGHTS["break"] * breakout
        + WEIGHTS["profit"] * profit
    )

    signal = (
        "可交易·下方尖峰+趋势确认"
        if tradeable
        else "观察·接近下方尖峰未确认"
        if in_zone
        else "无"
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
        "turnover_pct": round(float(last["turnover"]) * 100, 2),
        "volume_ratio": round(volume_ratio, 2),
        "line_score": round(float(line), 1),
        "conc_score": round(float(conc), 1),
        "peak_score": round(float(peak), 1),
        "break_score": round(float(breakout), 1),
        "profit_score": round(float(profit), 1),
        "total_score": round(float(total), 1),
        "is_approaching": in_zone,
        "is_tradeable": tradeable,
        "is_below_peak": feat.get("is_below_peak", False),
        "signal": signal,
        "profile": "winrate_below_peak",
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
    })
    result = analyze("000001", "样本", frame)
    assert result["code"] == "000001"
    assert 0 <= result["total_score"] <= 100
    assert "conc90_pct" in result
    assert "is_below_peak" in result
    print("self_test passed:", result["signal"], "total_score=", result["total_score"])


if __name__ == "__main__":
    self_test()
