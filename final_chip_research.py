#!/usr/bin/env python3
"""FINAL Chip的日线筹码峰、集中度、突破和五维评分研究内核。"""
from __future__ import annotations

import math
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

DECAY, STEP, LOOKBACK = 1.0, 0.01, 100
APPROACH_RATIO, BREAK_RATIO, VOL_MULTIPLIER, CONC_THRESHOLD = 0.97, 1.01, 1.5, 0.20
WEIGHTS = {"line": 0.15, "conc": 0.15, "peak": 0.10, "break": 0.45, "profit": 0.15}


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
    process.start(); child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate(); process.join(timeout=2)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        ok, value = parent.recv(); process.join(timeout=2)
        if not ok:
            raise RuntimeError(f"provider_error:{label}:{value}")
        return value
    finally:
        if process.is_alive():
            process.terminate(); process.join(timeout=2)
        parent.close()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover", "turn": "turnover"}
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
    turnover = data["turnover"].fillna(0.0).astype(float)
    data["turnover"] = np.where(turnover > 1.5, turnover / 100.0, turnover)
    data = data[(data["close"] > 0) & (data["high"] >= data["low"]) & (data["volume"] > 0)]
    if len(data) < 40:
        raise ValueError(f"insufficient_history:{len(data)}")
    return data.tail(LOOKBACK).reset_index(drop=True)


def _ak_history(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    return normalize_frame(ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq"))


def _bs_history(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        exchange = "sh" if code.startswith(("60", "68")) else "sz"
        result = bs.query_history_k_data_plus(f"{exchange}.{code}", "date,open,high,low,close,volume,amount,turn", start_date=start, end_date=end, frequency="d", adjustflag="2")
        if result.error_code != "0":
            raise RuntimeError(f"baostock_history:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return normalize_frame(pd.DataFrame(rows, columns=result.fields))
    finally:
        bs.logout()


def fetch_ohlcv(code: str, timeout_seconds: float = 35, retries: int = 2) -> tuple[pd.DataFrame, str, list[str]]:
    end, start = datetime.now().date(), datetime.now().date() - timedelta(days=LOOKBACK + 110)
    errors: list[str] = []
    for label, request in (("akshare", _ak_history), ("baostock", _bs_history)):
        for attempt in range(1, max(retries, 1) + 1):
            try:
                return provider_call(f"{label}:{code}", timeout_seconds, lambda fn=request: fn(code, start.isoformat(), end.isoformat())), label, errors
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
    if math.isclose(high, low):
        return {round(high, 2): volume}
    prices = np.unique(np.round(np.linspace(low, high, max(int(round((high - low) / STEP)) + 1, 2)), 2))
    values = np.array([max((price - low) / max(average - low, 1e-8), 0.0) if price <= average else max((high - price) / max(high - average, 1e-8), 0.0) for price in prices])
    values = values / values.sum() if values.sum() > 0 else np.ones(len(prices)) / len(prices)
    return {float(price): float(weight * volume) for price, weight in zip(prices, values)}


def update_chip(chip: dict[float, float], row: pd.Series) -> dict[float, float]:
    moved = min(max(float(row["turnover"]) * DECAY, 0.0), 1.0)
    result = {price: weight * (1 - moved) for price, weight in chip.items() if weight * (1 - moved) > 1e-8}
    for price, weight in daily_distribution(float(row["low"]), float(row["high"]), average_price(row), float(row["volume"])).items():
        result[price] = result.get(price, 0.0) + weight * moved
    return result


def features(chip: dict[float, float], close: float) -> dict[str, float]:
    items = sorted(chip.items())
    if not items:
        raise ValueError("empty_chip")
    prices = np.array([item[0] for item in items]); weights = np.array([item[1] for item in items]); total = weights.sum()
    peak_index = int(np.argmax(weights)); main_peak, main_weight = float(prices[peak_index]), float(weights[peak_index])
    local = [index for index in range(len(weights)) if (index == 0 or weights[index] >= weights[index - 1]) and (index == len(weights) - 1 or weights[index] >= weights[index + 1])]
    second = float(sorted((weights[index] for index in local), reverse=True)[1]) if len(local) > 1 else main_weight * 0.01
    cumulative = np.cumsum(weights) / total
    p5, p95 = float(prices[min(np.searchsorted(cumulative, .05), len(prices) - 1)]), float(prices[min(np.searchsorted(cumulative, .95), len(prices) - 1)])
    band = max(close * 0.01, STEP * 5)
    return {"main_peak": main_peak, "peak_ratio": main_weight / total, "peak_gap": main_weight / max(second, 1e-8), "band_ratio": float(weights[(prices >= main_peak - band) & (prices <= main_peak + band)].sum() / total), "conc90": (p95 - p5) / (p95 + p5) if p95 + p5 > 0 else 1.0, "p5": p5, "p95": p95, "avg_cost": float((prices * weights).sum() / total), "profit": float(weights[prices <= close].sum() / total)}


def analyze(code: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    data = normalize_frame(frame); chip: dict[float, float] = {}
    for _, row in data.iterrows(): chip = update_chip(chip, row)
    last, close, feat = data.iloc[-1], float(data.iloc[-1]["close"]), features(chip, float(data.iloc[-1]["close"]))
    ma5, ma10 = data["close"].tail(5).mean(), data["close"].tail(10).mean(); trend = bool(close > ma5 and ma5 > ma10)
    volume_ma = data["volume"].iloc[-6:-1].mean(); volume_ratio = float(last["volume"] / volume_ma) if volume_ma > 0 else 0.0
    in_zone = feat["main_peak"] * APPROACH_RATIO <= close < feat["main_peak"] * BREAK_RATIO
    confirmed = volume_ratio >= VOL_MULTIPLIER and feat["conc90"] <= CONC_THRESHOLD
    tradeable = bool(in_zone and trend and confirmed)
    amplitude = (float(last["high"]) - float(last["low"])) / close if close else 1.0
    line = 100 * (0.30 * np.clip(1 - amplitude / .05, 0, 1) + .25 * np.clip(float(last["turnover"]) / .03, 0, 1) + .30 * np.clip(feat["band_ratio"] / .25, 0, 1) + .15 * (1 if amplitude < .012 else .5 if amplitude < .025 else 0))
    conc = 100 * np.clip(1 - (feat["conc90"] - .05) / .35, 0, 1); peak = 100 * (.55 * np.clip(feat["band_ratio"] / .30, 0, 1) + .25 * np.clip(np.log1p(feat["peak_gap"]) / 4, 0, 1) + .20 * np.clip(feat["peak_ratio"] / .02, 0, 1))
    position = np.clip((close - feat["main_peak"] * APPROACH_RATIO) / max(feat["main_peak"] * (BREAK_RATIO - APPROACH_RATIO), 1e-8), 0, 1) if in_zone else (0.35 if close >= feat["main_peak"] * BREAK_RATIO else max(0, 1 + (close - feat["main_peak"]) / feat["main_peak"] / .15))
    breakout = 100 * (.35 * position + .25 * (1 if trend else .25) + .20 * np.clip(volume_ratio / VOL_MULTIPLIER, 0, 1.2) / 1.2 + .10 * (1 if confirmed else .4) + .10 * (1 if tradeable else .45))
    profit_pct = feat["profit"] * 100; profit = 100 if 20 <= profit_pct <= 55 else max(5, 100 - profit_pct) if profit_pct > 70 else max(10, profit_pct * 2) if profit_pct < 10 else 60
    total = WEIGHTS["line"] * line + WEIGHTS["conc"] * conc + WEIGHTS["peak"] * peak + WEIGHTS["break"] * breakout + WEIGHTS["profit"] * profit
    signal = "可交易·接近尖峰+趋势确认" if tradeable else "观察·接近尖峰未确认" if in_zone else "无"
    return {"code": str(code).zfill(6), "name": name, "date": str(last["date"].date()), "close": round(close, 2), "main_peak": round(feat["main_peak"], 2), "avg_cost": round(feat["avg_cost"], 2), "dist_to_peak_pct": round((close - feat["main_peak"]) / feat["main_peak"] * 100, 2), "band_ratio_pct": round(feat["band_ratio"] * 100, 2), "conc90_pct": round(feat["conc90"] * 100, 2), "profit_pct": round(profit_pct, 2), "p5": round(feat["p5"], 2), "p95": round(feat["p95"], 2), "turnover_pct": round(float(last["turnover"]) * 100, 2), "volume_ratio": round(volume_ratio, 2), "line_score": round(float(line), 1), "conc_score": round(float(conc), 1), "peak_score": round(float(peak), 1), "break_score": round(float(breakout), 1), "profit_score": round(float(profit), 1), "total_score": round(float(total), 1), "is_approaching": in_zone, "is_tradeable": tradeable, "signal": signal, "profile": "winrate", "confirm_mode": "and"}


def self_test() -> None:
    dates = pd.date_range("2025-01-01", periods=110, freq="B"); close = np.linspace(10, 12, 110)
    frame = pd.DataFrame({"date": dates, "open": close-.1, "high": close+.2, "low": close-.2, "close": close, "volume": np.linspace(1000, 2000, 110), "amount": close*np.linspace(1000, 2000, 110), "turnover": [.02]*110})
    result = analyze("000001", "样本", frame)
    assert result["code"] == "000001" and 0 <= result["total_score"] <= 100 and "conc90_pct" in result

