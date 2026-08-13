"""A 股底部吸筹多周期共振筛选核心。

数据口径：AkShare 为主、BaoStock 仅在单标的主路径失败时回退；同一标的单次判定只采用一个来源。
结果口径：日线后复权；周/月/季线由同一份日线重采样，且仅使用已完成高周期。
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd
import requests


LOGGER = logging.getLogger("mtf_accumulation")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
HISTORY_CALENDAR_DAYS = int(os.getenv("HISTORY_CALENDAR_DAYS", "1000"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.20"))
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "2"))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "0"))


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    max_range_position: float
    range_compress_max: float
    quiet_vol_max: float
    atr_compress_max: float
    spring_vol_max: float
    test_vol_max: float
    demand_vol_min: float
    breakout_vol_min: float
    setup_threshold: int
    min_resonance_score: int
    min_passed_frames: int
    require_week_month: bool
    use_chan_center_filter: bool


PROFILES = {
    "aggressive": Profile(
        key="aggressive", label="激进档", max_range_position=0.50,
        range_compress_max=1.05, quiet_vol_max=0.98, atr_compress_max=1.02,
        spring_vol_max=1.60, test_vol_max=0.95, demand_vol_min=1.10,
        breakout_vol_min=1.20, setup_threshold=50, min_resonance_score=50,
        min_passed_frames=2, require_week_month=False, use_chan_center_filter=False,
    ),
    "robust": Profile(
        key="robust", label="稳健档", max_range_position=0.30,
        range_compress_max=0.82, quiet_vol_max=0.78, atr_compress_max=0.82,
        spring_vol_max=1.10, test_vol_max=0.70, demand_vol_min=1.45,
        breakout_vol_min=1.60, setup_threshold=72, min_resonance_score=90,
        min_passed_frames=4, require_week_month=True, use_chan_center_filter=False,
    ),
    "chan": Profile(
        key="chan", label="缠论中枢优化档", max_range_position=0.38,
        range_compress_max=0.92, quiet_vol_max=0.88, atr_compress_max=0.92,
        spring_vol_max=1.35, test_vol_max=0.80, demand_vol_min=1.20,
        breakout_vol_min=1.35, setup_threshold=60, min_resonance_score=70,
        min_passed_frames=3, require_week_month=True, use_chan_center_filter=True,
    ),
}


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _slope(series: pd.Series, length: int) -> float:
    values = series.dropna().tail(length).to_numpy(dtype=float)
    if len(values) < length or not np.isfinite(values).all():
        return float("nan")
    return float(np.polyfit(np.arange(length), values, 1)[0])


def _last_completed_resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """重采样后删除当前未完成的自然周期，等价于 Pine 高周期表达式的 [1]。"""
    agg = frame.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return agg.iloc[:-1].copy() if len(agg) > 1 else agg.iloc[0:0].copy()


def _timeframe_passes(frame: pd.DataFrame) -> dict[str, Any]:
    daily_completed = frame.iloc[:-1].copy()
    if len(daily_completed) < 260:
        raise ValueError("完整日线不足 260 根，不能稳定计算年线")

    weekly = _last_completed_resample(frame, "W-FRI")
    monthly = _last_completed_resample(frame, "ME")
    quarterly = _last_completed_resample(frame, "QE-DEC")
    if len(weekly) < 22 or len(monthly) < 12 or len(quarterly) < 6:
        raise ValueError("已完成的周/月/季线历史不足")

    weekly_fast = _ema(weekly["close"], 10)
    weekly_slow = _ema(weekly["close"], 20)
    week_pass = bool(weekly["close"].iloc[-1] >= weekly_fast.iloc[-1] and (weekly_fast.iloc[-1] >= weekly_slow.iloc[-1] or weekly_fast.iloc[-1] >= weekly_fast.iloc[-2]))

    monthly_ema = _ema(monthly["close"], 10)
    month_pass = bool(monthly["close"].iloc[-1] >= monthly_ema.iloc[-1] or monthly_ema.iloc[-1] >= monthly_ema.iloc[-2])

    quarterly_ema = _ema(quarterly["close"], 4)
    quarter_pass = bool(quarterly["close"].iloc[-1] >= quarterly_ema.iloc[-1] or quarterly_ema.iloc[-1] >= quarterly_ema.iloc[-2])

    yearline = daily_completed["close"].rolling(250, min_periods=250).mean()
    year_pass = bool(daily_completed["close"].iloc[-1] >= yearline.iloc[-1] or yearline.iloc[-1] >= yearline.iloc[-2])
    return {"week_pass": week_pass, "month_pass": month_pass, "quarter_pass": quarter_pass, "year_pass": year_pass, "yearline": float(yearline.iloc[-1])}


def _confirmed_pivots(frame: pd.DataFrame, span: int = 3, min_gap: int = 5) -> list[dict[str, Any]]:
    """仅保留左右各 span 根 K 线确认后的局部极值；同向相邻分型只保留更极端者。"""
    pivots: list[dict[str, Any]] = []
    for i in range(span, len(frame) - span):
        high_window = frame["high"].iloc[i - span : i + span + 1]
        low_window = frame["low"].iloc[i - span : i + span + 1]
        is_top = frame["high"].iloc[i] == high_window.max() and int((high_window == high_window.max()).sum()) == 1
        is_bottom = frame["low"].iloc[i] == low_window.min() and int((low_window == low_window.min()).sum()) == 1
        if not (is_top or is_bottom):
            continue
        kind = "top" if is_top else "bottom"
        price = float(frame["high"].iloc[i] if kind == "top" else frame["low"].iloc[i])
        if not pivots:
            pivots.append({"index": i, "kind": kind, "price": price})
            continue
        previous = pivots[-1]
        if kind == previous["kind"]:
            more_extreme = (kind == "top" and price > previous["price"]) or (kind == "bottom" and price < previous["price"])
            if more_extreme:
                pivots[-1] = {"index": i, "kind": kind, "price": price}
        elif i - previous["index"] >= min_gap:
            pivots.append({"index": i, "kind": kind, "price": price})
    return pivots


def _chan_center_filter(frame: pd.DataFrame) -> dict[str, Any]:
    """三段近似笔重叠中枢 + 上沿离开/回抽未回中枢的三类买点近似。"""
    pivots = _confirmed_pivots(frame)
    if len(pivots) < 4:
        return {"valid": False, "shape_pass": False, "reason": "已确认分型不足"}
    last = pivots[-4:]
    strokes = [{"low": min(last[i]["price"], last[i + 1]["price"]), "high": max(last[i]["price"], last[i + 1]["price"])} for i in range(3)]
    zd = max(stroke["low"] for stroke in strokes)
    zg = min(stroke["high"] for stroke in strokes)
    atr = _atr(frame)
    atr_now = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else float("nan")
    if not np.isfinite(atr_now) or zd >= zg:
        return {"valid": False, "shape_pass": False, "reason": "三段无重叠"}
    width_atr = (zg - zd) / max(atr_now, 1e-9)
    valid = width_atr <= 8.0
    post_center = frame.iloc[last[-1]["index"] + 1 :]
    volume_ma = frame["volume"].rolling(20, min_periods=20).mean()
    breakout_mask = (post_center["close"] > zg) & (post_center["volume"] >= volume_ma.loc[post_center.index] * 1.20)
    breakout_dates = post_center.index[breakout_mask]
    breakout = len(breakout_dates) > 0 and len(post_center) <= 15
    third_buy = False
    if breakout:
        after_breakout = frame.loc[breakout_dates[0] :]
        third_buy = bool(after_breakout["low"].min() >= zg - atr_now * 0.35 and frame["close"].iloc[-1] > zg)
    return {"valid": valid, "shape_pass": bool(valid and (breakout or third_buy)), "zg": float(zg), "zd": float(zd), "width_atr": float(width_atr), "pivot_count": len(pivots), "breakout": bool(breakout), "third_buy": bool(third_buy), "reason": "ok" if valid else "中枢宽度过大"}


def _daily_setup(frame: pd.DataFrame, index: int, profile: Profile) -> dict[str, Any] | None:
    """在指定日线位置计算 V3 当前周期底部结构；箱体不包含当天 K 线。"""
    if index < 160:
        return None
    data = frame.iloc[: index + 1].copy()
    row = data.iloc[-1]
    base = data.iloc[-41:-1]
    long_window = data.iloc[-120:]
    if len(base) < 40 or len(long_window) < 120:
        return None

    base_high, base_low = float(base["high"].max()), float(base["low"].min())
    base_width = (base_high - base_low) / max(base_low, 1e-9)
    widths = (data["high"].rolling(40).max().shift(1) - data["low"].rolling(40).min().shift(1)) / data["low"].rolling(40).min().shift(1)
    width_average = float(widths.iloc[-1])
    atr = _atr(data)
    atr_ratio = atr / data["close"]
    atr_ratio_average = atr_ratio.rolling(40, min_periods=40).mean()
    vol_ma = data["volume"].rolling(20, min_periods=20).mean()
    vol_ratio = data["volume"] / vol_ma

    if any(pd.isna(v) for v in [width_average, atr.iloc[-1], atr_ratio_average.iloc[-1], vol_ratio.iloc[-1]]):
        return None

    long_high, long_low = float(long_window["high"].max()), float(long_window["low"].min())
    position = (float(row.close) - long_low) / max(long_high - long_low, 1e-9)
    in_bottom_zone = position <= profile.max_range_position
    range_compressed = base_width <= width_average * profile.range_compress_max
    atr_compressed = float(atr_ratio.iloc[-1]) <= float(atr_ratio_average.iloc[-1]) * profile.atr_compress_max
    quiet_volume = float(vol_ratio.tail(10).mean()) <= profile.quiet_vol_max

    spread = (data["high"] - data["low"]).replace(0, np.nan)
    mf_multiplier = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / spread
    adl = (mf_multiplier.fillna(0) * data["volume"]).cumsum()
    obv = (np.sign(data["close"].diff()).fillna(0) * data["volume"]).cumsum()
    adl_divergence = _slope(adl, 20) > 0 and _slope(data["close"], 20) <= 0
    obv_divergence = _slope(obv, 20) > 0 and _slope(data["close"], 20) <= 0
    flow_support = adl_divergence or obv_divergence
    flow_strong = adl_divergence and obv_divergence

    prior_support = float(data["low"].iloc[-21:-1].min())
    bar_range = max(float(row.high - row.low), 1e-9)
    close_position = (float(row.close) - float(row.low)) / bar_range
    spring = bool(float(row.low) < prior_support - float(atr.iloc[-1]) * 0.25 and float(row.close) > prior_support and close_position >= 0.60 and float(vol_ratio.iloc[-1]) <= profile.spring_vol_max)
    test_support = float(data["low"].iloc[-13:-1].min())
    demand_test = bool(float(row.low) <= test_support + float(atr.iloc[-1]) * 0.25 and float(row.close) >= float(row.open) and close_position >= 0.55 and float(vol_ratio.iloc[-1]) <= profile.test_vol_max)
    demand_bar = bool(float(row.close) > float(row.open) and close_position >= 0.70 and float(vol_ratio.iloc[-1]) >= profile.demand_vol_min)
    breakout = bool(float(row.close) > base_high and close_position >= 0.65 and float(vol_ratio.iloc[-1]) >= profile.breakout_vol_min)

    score = sum([
        16 if in_bottom_zone else 0, 12 if range_compressed else 0, 10 if atr_compressed else 0,
        10 if quiet_volume else 0, 10 if adl_divergence else 0, 10 if obv_divergence else 0,
        16 if spring else 0, 8 if demand_test else 0, 8 if demand_bar else 0, 5 if flow_strong else 0,
    ])
    setup_event = bool(in_bottom_zone and range_compressed and atr_compressed and quiet_volume and flow_support and (spring or demand_test or demand_bar))
    return {"setup_event": setup_event, "candidate": setup_event and score >= profile.setup_threshold, "breakout": breakout, "score": int(score), "base_high": base_high, "base_low": base_low, "atr": float(atr.iloc[-1]), "position": position, "spring": spring, "demand_test": demand_test, "demand_bar": demand_bar}


def evaluate_symbol(frame: pd.DataFrame, profile: Profile) -> dict[str, Any] | None:
    if len(frame) < 300:
        return None
    mtf = _timeframe_passes(frame)
    latest = _daily_setup(frame, len(frame) - 1, profile)
    if latest is None:
        return None
    passed_frames = sum(bool(mtf[name]) for name in ("week_pass", "month_pass", "quarter_pass", "year_pass"))
    resonance_score = (30 if mtf["week_pass"] else 0) + (30 if mtf["month_pass"] else 0) + (20 if mtf["quarter_pass"] else 0) + (20 if mtf["year_pass"] else 0)
    mandatory_pass = (mtf["week_pass"] and mtf["month_pass"]) if profile.require_week_month else True
    resonance = passed_frames >= profile.min_passed_frames and resonance_score >= profile.min_resonance_score and mandatory_pass
    chan = _chan_center_filter(frame) if profile.use_chan_center_filter else {"valid": True, "shape_pass": True}

    start = max(160, len(frame) - 26)
    recent_setup = any((_daily_setup(frame, i, profile) or {}).get("candidate", False) for i in range(start, len(frame) - 1))
    confirmed = bool(recent_setup and latest["breakout"] and resonance and chan["shape_pass"])
    candidate = bool(latest["candidate"] and resonance and chan["shape_pass"])
    invalidated = bool(float(frame.close.iloc[-1]) < latest["base_low"] - latest["atr"] * 0.80)
    if not candidate and not confirmed:
        return None
    return {**latest, **mtf, **chan, "passed_frames": passed_frames, "resonance_score": resonance_score, "candidate": candidate, "confirmed": confirmed, "invalidated": invalidated, "yearline": mtf["yearline"]}


def _akshare_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="hfq")
    renamed = raw.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
    return renamed[["date", "open", "high", "low", "close", "volume"]]


def _baostock_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    exchange = "sh" if symbol.startswith(("6", "9")) else "sz"
    response = bs.query_history_k_data_plus(f"{exchange}.{symbol}", "date,open,high,low,close,volume,tradestatus,isST", start_date=start, end_date=end, frequency="d", adjustflag="1")
    if response.error_code != "0":
        raise RuntimeError(response.error_msg)
    rows: list[list[str]] = []
    while response.next():
        rows.append(response.get_row_data())
    data = pd.DataFrame(rows, columns=response.fields)
    if not data.empty:
        data = data[(data["tradestatus"] == "1") & (data["isST"] != "1")]
    return data[["date", "open", "high", "low", "close", "volume"]]


def fetch_history(symbol: str) -> tuple[pd.DataFrame, str]:
    end = date.today()
    start = end - timedelta(days=HISTORY_CALENDAR_DAYS)
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            data = _akshare_history(symbol, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
            return normalize_history(data), "akshare"
        except Exception as exc:  # 网络数据源异常仅回退当前标的
            last_error = exc
            time.sleep(1 + attempt)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"AkShare 失败且 BaoStock 登录失败：{last_error}; {login.error_msg}")
    try:
        return normalize_history(_baostock_history(symbol, start.isoformat(), end.isoformat())), "baostock"
    finally:
        bs.logout()


def normalize_history(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = _to_number(frame[column])
    frame = frame.dropna().drop_duplicates("date").sort_values("date")
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame["volume"] > 0)]
    return frame.set_index("date")


def fetch_universe() -> pd.DataFrame:
    spot = ak.stock_zh_a_spot_em()[["代码", "名称"]].rename(columns={"代码": "symbol", "名称": "name"})
    spot["symbol"] = spot["symbol"].astype(str).str.zfill(6)
    universe = spot[~spot["name"].astype(str).str.contains("ST|退", case=False, na=False)].drop_duplicates("symbol")
    return universe.head(SCAN_LIMIT) if SCAN_LIMIT > 0 else universe


def send_serverchan(profile: Profile, results: pd.DataFrame) -> None:
    key = os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY")
    if not key:
        LOGGER.info("未设置 SERVERCHAN_KEY/SENDKEY，跳过通知")
        return
    title = f"A股底部吸筹{profile.label}：{len(results)}只候选"
    body = "无候选。" if results.empty else "\n".join(f"- {r.symbol} {r.name}｜{'确认' if r.confirmed else '候选'}｜结构{r.base_score}｜共振{r.resonance_score}" for r in results.head(20).itertuples())
    try:
        requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body}, timeout=15).raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Server 酱通知失败：%s", exc)


def run_profile(profile_key: str) -> int:
    profile = PROFILES[profile_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = fetch_universe()
    findings: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for ordinal, stock in enumerate(universe.itertuples(index=False), start=1):
        try:
            history, source = fetch_history(stock.symbol)
            result = evaluate_symbol(history, profile)
            if result:
                findings.append({"symbol": stock.symbol, "name": stock.name, "source": source, "as_of": str(history.index[-1].date()), "base_score": result["score"], "resonance_score": result["resonance_score"], "passed_frames": result["passed_frames"], "confirmed": result["confirmed"], "candidate": result["candidate"], "week_pass": result["week_pass"], "month_pass": result["month_pass"], "quarter_pass": result["quarter_pass"], "year_pass": result["year_pass"], "yearline": round(result["yearline"], 4), "close": round(float(history.close.iloc[-1]), 4), "chan_zd": round(result.get("zd", float("nan")), 4), "chan_zg": round(result.get("zg", float("nan")), 4), "chan_width_atr": round(result.get("width_atr", float("nan")), 2), "chan_breakout": result.get("breakout", False), "chan_third_buy": result.get("third_buy", False), "chan_pivots": result.get("pivot_count", 0)})
        except Exception as exc:
            skipped.append({"symbol": stock.symbol, "name": stock.name, "reason": str(exc)[:200]})
        if ordinal % 100 == 0:
            LOGGER.info("%s：已处理 %s/%s", profile.label, ordinal, len(universe))
        time.sleep(REQUEST_DELAY_SECONDS)

    result_df = pd.DataFrame(findings).sort_values(["confirmed", "resonance_score", "base_score"], ascending=[False, False, False]) if findings else pd.DataFrame(columns=["symbol", "name", "confirmed", "candidate", "base_score", "resonance_score"])
    prefix = f"mtf_accumulation_{profile.key}"
    result_df.to_csv(OUTPUT_DIR / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    payload = {"profile": profile.label, "as_of": str(date.today()), "universe": len(universe), "candidates": len(result_df), "skipped": len(skipped), "results": result_df.to_dict(orient="records"), "skipped_samples": skipped[:50]}
    (OUTPUT_DIR / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = f"# {profile.label}多周期共振筛选\n\n- 股票池：{len(universe)}\n- 候选：{len(result_df)}\n- 跳过：{len(skipped)}\n\n" + (result_df.to_markdown(index=False) if not result_df.empty else "无候选。")
    (OUTPUT_DIR / f"{prefix}.md").write_text(markdown, encoding="utf-8")
    send_serverchan(profile, result_df)
    LOGGER.info("%s 完成：%s 个候选，%s 个跳过", profile.label, len(result_df), len(skipped))
    return 0
