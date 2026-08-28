#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL Chip Parameter Comparison base scanner (private copy).

This copy is used only by the chip-param-* experiment. It writes local shard
JSON for the comparison driver and never sends notifications. No performance
or win-rate claim is embedded in the experiment code.
"""


from __future__ import annotations

import json
import os
import random
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import baostock as bs
    HAVE_BS = True
except ImportError:
    HAVE_BS = False

try:
    import akshare as ak
    HAVE_AK = True
except ImportError:
    HAVE_AK = False

if not HAVE_BS and not HAVE_AK:
    raise ImportError("需要 baostock 或 akshare: pip install baostock akshare pandas numpy requests")

# ==================== 运行参数 ====================
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_TOTAL = int(os.environ.get("SHARD_TOTAL", "1"))
NUM_PROCESSES = int(os.environ.get("NUM_PROCESSES", "2"))
DECAY = float(os.environ.get("CHIP_DECAY", "1.0"))
STEP = float(os.environ.get("CHIP_STEP", "0.01"))
LOOKBACK = int(os.environ.get("CHIP_LOOKBACK", "100"))
PROFILE = os.environ.get("CHIP_PROFILE", "winrate").strip().lower()  # 回测推荐日常档
# Notifications are disabled in this private parameter experiment.
PUSH_ON_SHARD = False
SENDKEY = ""

_PROFILES = {
    "legacy": {"approach": 0.97, "break": 1.005, "vol": 1.2, "conc": 0.30},
    "winrate": {"approach": 0.97, "break": 1.01, "vol": 1.5, "conc": 0.20},
    "strict": {"approach": 0.96, "break": 1.02, "vol": 2.0, "conc": 0.15},
}
_P = _PROFILES.get(PROFILE, _PROFILES["winrate"])
APPROACH_RATIO = float(os.environ.get("CHIP_APPROACH_RATIO", _P["approach"]))
BREAK_RATIO = float(os.environ.get("CHIP_BREAK_RATIO", _P["break"]))
VOL_MULTIPLIER = float(os.environ.get("CHIP_VOL_MULTIPLIER", _P["vol"]))
CONC_THRESHOLD = float(os.environ.get("CHIP_CONC_THRESHOLD", _P["conc"]))

# 回测：突破信号质量最高 → 提高 break 权重
WEIGHTS = {"line": 0.15, "conc": 0.15, "peak": 0.10, "break": 0.45, "profit": 0.15}
# 确认模式：and=量且集中(回测更优) | or=量或集中(信号更多)
CONFIRM_MODE = os.environ.get("CHIP_CONFIRM_MODE", "and").strip().lower()
MAX_RETRIES = 3


# ==================== Server酱 ====================
def sc_send(*_args, **_kwargs) -> dict:
    """Notifications are disabled for this parameter experiment."""
    return {"code": -1, "message": "notifications_disabled_for_chip_param_experiment"}


def push_shard_summary(*_args, **_kwargs) -> None:
    """Per-shard notifications are disabled for this experiment."""
    return None


# ==================== 代码 / 板块 ====================
def to_bs(symbol: str) -> str:
    s = symbol.zfill(6)
    return f"sh.{s}" if s.startswith(("60", "68")) else f"sz.{s}"


def is_a_share(code: str) -> bool:
    s = code.replace("sh.", "").replace("sz.", "").zfill(6)
    if s.startswith(("90", "20", "43", "83", "87")):
        return False
    return s.startswith(("00", "30", "60", "68"))


def fetch_universe() -> List[str]:
    codes: List[str] = []
    if HAVE_BS:
        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_basic()
                while rs.error_code == "0" and rs.next():
                    r = rs.get_row_data()
                    if not r or len(r) < 6:
                        continue
                    code, _n, _ipo, out, typ, status = r[0], r[1], r[2], r[3], r[4], r[5]
                    if typ != "1" or status != "1":
                        continue
                    if out and out not in ("", "None"):
                        continue
                    pure = code.replace("sh.", "").replace("sz.", "")
                    if is_a_share(pure):
                        codes.append(pure.zfill(6))
            finally:
                bs.logout()
    if not codes and HAVE_AK:
        try:
            spot = ak.stock_info_a_code_name()
            for _, row in spot.iterrows():
                c = str(row.iloc[0]).zfill(6)
                if is_a_share(c):
                    codes.append(c)
        except Exception as e:
            print(f"akshare 股票列表失败: {e}")
    return sorted(set(codes))


_INDUSTRY_MAP: Dict[str, str] = {}


def load_industry_map() -> Dict[str, str]:
    global _INDUSTRY_MAP
    if _INDUSTRY_MAP:
        return _INDUSTRY_MAP
    m: Dict[str, str] = {}
    if HAVE_BS:
        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_industry()
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if not row or len(row) < 3:
                        continue
                    code = row[0].replace("sh.", "").replace("sz.", "").zfill(6)
                    m[code] = (row[2] or "").strip() or "未知"
            finally:
                bs.logout()
    _INDUSTRY_MAP = m
    return m


def shard_slice(universe: List[str], index: int, total: int) -> List[str]:
    if total <= 1:
        return universe
    size = (len(universe) + total - 1) // total
    start = index * size
    return universe[start: start + size]


# ==================== 双数据源 K 线 ====================
def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    colmap = {
        "日期": "date", "date": "date",
        "开盘": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "收盘": "close", "close": "close",
        "成交量": "volume", "volume": "volume",
        "成交额": "amount", "amount": "amount",
        "换手率": "turn", "turn": "turn", "turnover": "turn",
    }
    rename = {}
    for c in df.columns:
        if c in colmap:
            rename[c] = colmap[c]
    df = df.rename(columns=rename)
    need = ["date", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in need):
        return None
    for c in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    if "turn" in df.columns:
        # baostock turn 为百分比数值；akshare 换手率也可能是百分比
        t = df["turn"].fillna(0).astype(float)
        df["turnover"] = np.where(t > 1.5, t / 100.0, t)  # >1.5 视为百分比
    else:
        df["turnover"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df = df.sort_values("date").reset_index(drop=True)
    if float(df.iloc[-1]["volume"]) <= 0:
        return None
    if len(df) < 40:
        return None
    return df.tail(LOOKBACK).reset_index(drop=True)


def fetch_kline_baostock(symbol: str) -> Optional[pd.DataFrame]:
    if not HAVE_BS:
        return None
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=LOOKBACK + 90)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        to_bs(symbol),
        "date,open,high,low,close,volume,amount,turn",
        start_date=start, end_date=end, frequency="d", adjustflag="2",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    return _normalize_df(pd.DataFrame(rows, columns=rs.fields))


def fetch_kline_akshare(symbol: str) -> Optional[pd.DataFrame]:
    if not HAVE_AK:
        return None
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=LOOKBACK + 90)).strftime("%Y%m%d")
    # 前复权
    for fn in (
        lambda: ak.stock_zh_a_hist(symbol=symbol.zfill(6), period="daily",
                                   start_date=start, end_date=end, adjust="qfq"),
    ):
        try:
            raw = fn()
            if raw is None or raw.empty:
                return None
            return _normalize_df(raw)
        except Exception:
            return None
    return None


def fetch_kline(symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    """返回 (df, source_name)。baostock 优先，失败用 akshare。"""
    for attempt in range(MAX_RETRIES):
        try:
            if HAVE_BS:
                df = fetch_kline_baostock(symbol)
                if df is not None:
                    return df, "baostock"
        except Exception:
            pass
        try:
            if HAVE_AK:
                df = fetch_kline_akshare(symbol)
                if df is not None:
                    return df, "akshare"
        except Exception:
            pass
        time.sleep(0.2 * (attempt + 1) + random.uniform(0, 0.2))
    return None, ""


# ==================== 筹码 ====================
def avg_price(row) -> float:
    amt, vol = float(row.get("amount", 0) or 0), float(row.get("volume", 0) or 0)
    if amt > 0 and vol > 0:
        return amt / vol
    return (float(row["open"]) + float(row["high"]) + float(row["low"]) + float(row["close"])) / 4


def daily_B(low, high, avg, vol, step) -> Dict[float, float]:
    if vol <= 0 or high < low:
        return {}
    if high == low or abs(high - low) < 1e-12:
        return {float(round(high, 2)): float(vol)}
    n = max(int(round((high - low) / step)) + 1, 2)
    prices = np.unique(np.round(np.linspace(low, high, n), 2))
    if len(prices) == 0:
        return {}
    h = 2.0 / (high - low)
    w = np.zeros(len(prices))
    for i, p in enumerate(prices):
        if p <= avg:
            w[i] = h / max(avg - low, 1e-6) * (p - low)
        else:
            w[i] = h / max(high - avg, 1e-6) * (high - p)
        w[i] = max(w[i], 0.0)
    s = w.sum()
    if s <= 0:
        return {float(p): vol / len(prices) for p in prices}
    w = w / s * vol
    return {float(p): float(x) for p, x in zip(prices, w)}


def update_chip(chip: Dict[float, float], row, decay: float, step: float) -> Dict[float, float]:
    to = float(row["turnover"])
    m = min(max(to * decay, 0.0), 1.0)
    stay = 1.0 - m
    new_chip = {p: w * stay for p, w in chip.items() if w * stay > 1e-8}
    vol = float(row["volume"])
    if vol > 0 and m > 0:
        B = daily_B(float(row["low"]), float(row["high"]), avg_price(row), vol, step)
        for p, w in B.items():
            new_chip[p] = new_chip.get(p, 0.0) + w * m
    return new_chip


def chip_features(chip: Dict[float, float], close: float) -> dict:
    if not chip:
        return {}
    items = sorted(chip.items())
    total = sum(w for _, w in items) or 1.0
    prices = np.array([p for p, _ in items], dtype=float)
    weights = np.array([w for _, w in items], dtype=float)
    max_w = float(weights.max())
    eff_min = min(0.02, max(max_w / total * 0.5, 0.003))
    peaks = []
    for i in range(1, len(items) - 1):
        p, w = items[i]
        if w >= items[i - 1][1] and w >= items[i + 1][1] and w / total >= eff_min:
            peaks.append((p, w))
    if len(items) >= 2:
        if items[0][1] >= items[1][1] and items[0][1] / total >= eff_min:
            peaks.append(items[0])
        if items[-1][1] >= items[-2][1] and items[-1][1] / total >= eff_min:
            peaks.append(items[-1])
    if not peaks:
        peaks = [max(items, key=lambda x: x[1])]
    peaks.sort(key=lambda x: x[1], reverse=True)
    main_p, main_w = peaks[0]
    peak_ratio = main_w / total
    sec_w = peaks[1][1] if len(peaks) > 1 else main_w * 0.01
    peak_gap = main_w / max(sec_w, 1e-12)
    band = max(close * 0.01, STEP * 5)
    band_ratio = float(weights[(prices >= main_p - band) & (prices <= main_p + band)].sum() / total)
    cum = np.cumsum(weights) / total
    i5 = min(max(int(np.searchsorted(cum, 0.05)), 0), len(prices) - 1)
    i95 = min(max(int(np.searchsorted(cum, 0.95)), 0), len(prices) - 1)
    p5, p95 = float(prices[i5]), float(prices[i95])
    conc90 = (p95 - p5) / (p95 + p5) if (p95 + p5) > 0 else 1.0
    avg_cost = float(np.sum(prices * weights) / total)
    profit = float(weights[prices <= close].sum() / total)
    return {
        "main_peak": main_p, "peak_ratio": peak_ratio, "band_ratio": band_ratio,
        "peak_gap": peak_gap, "conc90": conc90, "p5": p5, "p95": p95,
        "avg_cost": avg_cost, "profit": profit,
    }


def score_line(feat, row, close) -> float:
    amp = (float(row["high"]) - float(row["low"])) / close if close else 1.0
    to = float(row["turnover"])
    s_amp = float(np.clip(1.0 - amp / 0.05, 0, 1))
    s_to = float(np.clip(to / 0.03, 0, 1)) * float(np.clip((0.30 - to) / 0.10, 0, 1)) if to < 0.30 else 0.3
    s_to = float(np.clip(s_to, 0, 1))
    s_band = float(np.clip(feat["band_ratio"] / 0.25, 0, 1))
    is_limit = 1.0 if amp < 0.012 else (0.5 if amp < 0.025 else 0.0)
    return round(100 * (0.30 * s_amp + 0.25 * s_to + 0.30 * s_band + 0.15 * is_limit), 1)


def score_conc(feat) -> float:
    return round(100 * float(np.clip(1.0 - (feat["conc90"] - 0.05) / 0.35, 0, 1)), 1)


def score_peak(feat) -> float:
    s_band = float(np.clip(feat["band_ratio"] / 0.30, 0, 1))
    s_gap = float(np.clip(np.log1p(feat["peak_gap"]) / 4.0, 0, 1))
    s_pr = float(np.clip(feat["peak_ratio"] / 0.02, 0, 1))
    return round(100 * (0.55 * s_band + 0.25 * s_gap + 0.20 * s_pr), 1)


def score_break(feat, df) -> Tuple[float, bool, bool]:
    close = float(df.iloc[-1]["close"])
    mp = feat["main_peak"]
    if mp <= 0:
        return 0.0, False, False
    in_zone = mp * APPROACH_RATIO <= close < mp * BREAK_RATIO
    if close >= mp * BREAK_RATIO:
        s_pos = 0.35
    elif in_zone:
        s_pos = 0.7 + 0.3 * float(np.clip(
            (close - mp * APPROACH_RATIO) / max(mp * (BREAK_RATIO - APPROACH_RATIO), 1e-6), 0, 1))
    else:
        dist = (close - mp) / mp
        s_pos = float(np.clip(1.0 + dist / 0.15, 0, 0.5)) if close < mp * APPROACH_RATIO else 0.4
    closes = df["close"].values
    ma5 = float(closes[-5:].mean()) if len(closes) >= 5 else close
    ma10 = float(closes[-10:].mean()) if len(closes) >= 10 else close
    trend = bool(close > ma5 and ma5 > ma10)
    vols = df["volume"].values
    vma = float(vols[-6:-1].mean()) if len(vols) >= 6 else float(vols[-1])
    vr = float(vols[-1]) / vma if vma > 0 else 0.0
    s_vol = float(np.clip(vr / VOL_MULTIPLIER, 0, 1.2)) / 1.2
    s_conc = 1.0 if feat["conc90"] <= CONC_THRESHOLD else float(
        np.clip(1.0 - (feat["conc90"] - CONC_THRESHOLD) / 0.3, 0, 1))
    if CONFIRM_MODE == "and":
        # 回测最优：量比与集中同时满足
        confirm_ok = vr >= VOL_MULTIPLIER and feat["conc90"] <= CONC_THRESHOLD
    else:
        confirm_ok = vr >= VOL_MULTIPLIER or feat["conc90"] <= CONC_THRESHOLD
    tradeable = bool(in_zone and trend and confirm_ok)
    confirm = 1.0 if tradeable else (0.55 if trend else 0.25)
    sc = 100 * (0.35 * s_pos + 0.25 * (1.0 if trend else 0.25) + 0.20 * s_vol + 0.10 * s_conc + 0.10 * confirm)
    return round(float(np.clip(sc, 0, 100)), 1), in_zone, tradeable


def score_profit(feat) -> float:
    p = feat["profit"] * 100
    if 20 <= p <= 55:
        score = 80 + 20 * (1 - abs(p - 37.5) / 17.5)
    elif 10 <= p < 20 or 55 < p <= 70:
        score = 50 + 30 * (1 - min(abs(p - 20), abs(p - 55)) / 15)
    elif p < 10:
        score = max(10, p * 2)
    else:
        score = max(5, 100 - p)
    return round(float(np.clip(score, 0, 100)), 1)


def analyze_symbol(symbol: str, industry: str) -> dict:
    df, source = fetch_kline(symbol)
    if df is None:
        return {"symbol": symbol, "industry": industry, "error": "no_data"}
    chip: Dict[float, float] = {}
    for _, row in df.iterrows():
        chip = update_chip(chip, row, DECAY, STEP)
    last = df.iloc[-1]
    close = float(last["close"])
    feat = chip_features(chip, close)
    if not feat:
        return {"symbol": symbol, "industry": industry, "error": "empty_chip", "source": source}

    line_s = score_line(feat, last, close)
    conc_s = score_conc(feat)
    peak_s = score_peak(feat)
    break_s, is_approaching, is_tradeable = score_break(feat, df)
    closes = df["close"].to_numpy()
    ma5 = float(closes[-5:].mean()) if len(closes) >= 5 else close
    ma10 = float(closes[-10:].mean()) if len(closes) >= 10 else close
    trend = bool(close > ma5 and ma5 > ma10)
    volumes = df["volume"].to_numpy()
    volume_baseline = float(volumes[-6:-1].mean()) if len(volumes) >= 6 else float(volumes[-1])
    volume_ratio = float(volumes[-1]) / volume_baseline if volume_baseline > 0 else 0.0
    profit_s = score_profit(feat)
    total = (
        WEIGHTS["line"] * line_s + WEIGHTS["conc"] * conc_s + WEIGHTS["peak"] * peak_s
        + WEIGHTS["break"] * break_s + WEIGHTS["profit"] * profit_s
    )
    dims = {"直线尖峰": line_s, "高集中": conc_s, "单峰": peak_s, "突破预警": break_s, "获利结构": profit_s}
    best_tag = max(dims, key=dims.get)
    if is_tradeable:
        signal = "可交易·接近尖峰+趋势确认"
    elif is_approaching:
        signal = "观察·接近尖峰未确认"
    else:
        signal = "无"
    amp = (float(last["high"]) - float(last["low"])) / close * 100 if close else 0
    return {
        "symbol": symbol,
        "industry": industry or "未知",
        "date": str(last["date"].date()),
        "close": round(close, 2),
        "main_peak": round(feat["main_peak"], 2),
        "avg_cost": round(feat["avg_cost"], 2),
        "dist_to_peak_pct": round((close - feat["main_peak"]) / feat["main_peak"] * 100, 2) if feat["main_peak"] else None,
        "band_ratio_pct": round(feat["band_ratio"] * 100, 2),
        "conc90_pct": round(feat["conc90"] * 100, 2),
        "profit_pct": round(feat["profit"] * 100, 2),
        "p5": round(feat["p5"], 2),
        "p95": round(feat["p95"], 2),
        "turnover_pct": round(float(last["turnover"]) * 100, 2),
        "amp_pct": round(amp, 2),
        "line_score": line_s,
        "conc_score": conc_s,
        "peak_score": peak_s,
        "break_score": break_s,
        "profit_score": profit_s,
        "total_score": round(total, 1),
        "best_tag": best_tag,
        "is_approaching": is_approaching,
        "is_tradeable": is_tradeable,
        "is_spike_tradeable": is_tradeable,
        "trend": trend,
        "volume_ratio": round(volume_ratio, 4),
        "signal": signal,
        "source": source,
        "profile": PROFILE,
    }


def _worker_init():
    if HAVE_BS:
        try:
            bs.login()
        except Exception:
            pass


def _worker_scan_chunk(symbols: List[str]) -> List[dict]:
    ind_map = load_industry_map()
    out = []
    for sym in symbols:
        try:
            out.append(analyze_symbol(sym, ind_map.get(sym, "未知")))
        except Exception as e:
            out.append({"symbol": sym, "industry": ind_map.get(sym, "未知"), "error": str(e)[:120]})
        time.sleep(random.uniform(0.02, 0.08))
    return out


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [lst]
    size = (len(lst) + n - 1) // n
    return [lst[i: i + size] for i in range(0, len(lst), size)]


def generate_report(df: pd.DataFrame, shard: int, total: int, scanned: int, err: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        f"# 筹码全市场扫描 · 分片 {shard}/{total}",
        f"> {now} | profile={PROFILE} | 扫描 {scanned} 失败 {err} | 双数据源 baostock→akshare",
        "",
    ]
    if df.empty:
        md.append("本分片无有效结果。")
        return "\n".join(md)
    if "is_tradeable" in df.columns:
        tdf = df[df["is_tradeable"] == True].sort_values("total_score", ascending=False)
        md.append(f"## 可交易预警（{len(tdf)} 只）\n")
        md.append("| 代码 | 板块 | 源 | 收盘 | 主峰 | 综合 | 突破 | 信号 |")
        md.append("|------|------|-----|------|------|------|------|------|")
        for _, r in tdf.head(40).iterrows():
            md.append(
                f"| {r['symbol']} | {r.get('industry','')} | {r.get('source','')} | {r['close']} | "
                f"{r['main_peak']} | {r['total_score']} | {r['break_score']} | {r.get('signal','')} |"
            )
        md.append("")
    top = df.sort_values("total_score", ascending=False).head(25)
    md.append("## 综合分 TOP25\n")
    md.append("| 代码 | 板块 | 收盘 | 线 | 集 | 峰 | 突 | 综合 | 标签 |")
    md.append("|------|------|------|----|----|----|----|------|------|")
    for _, r in top.iterrows():
        md.append(
            f"| {r['symbol']} | {r.get('industry','')} | {r['close']} | {r.get('line_score','')} | "
            f"{r.get('conc_score','')} | {r.get('peak_score','')} | {r.get('break_score','')} | "
            f"{r.get('total_score','')} | {r.get('best_tag','')} |"
        )
    return "\n".join(md)


def main():
    print("=" * 64)
    print(f"全市场筹码扫描 | shard {SHARD_INDEX}/{SHARD_TOTAL} | 双数据源+Server酱")
    print(f"BS={HAVE_BS} AK={HAVE_AK} profile={PROFILE} SENDKEY={'有' if SENDKEY else '无'}")
    print("=" * 64)

    universe = fetch_universe()
    shard_symbols = shard_slice(universe, SHARD_INDEX, SHARD_TOTAL)
    print(f"全市场 {len(universe)} 只，本分片 {len(shard_symbols)} 只")
    load_industry_map()

    chunks = _chunk(shard_symbols, NUM_PROCESSES)
    all_results: List[dict] = []
    if NUM_PROCESSES <= 1 or len(shard_symbols) < 8:
        if HAVE_BS:
            bs.login()
        try:
            all_results = _worker_scan_chunk(shard_symbols)
        finally:
            if HAVE_BS:
                try:
                    bs.logout()
                except Exception:
                    pass
    else:
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=_worker_init) as ex:
            for fut in [ex.submit(_worker_scan_chunk, c) for c in chunks]:
                try:
                    all_results.extend(fut.result())
                except Exception as e:
                    print(f"worker: {e}")

    ok_rows = [r for r in all_results if "error" not in r]
    err_count = len(all_results) - len(ok_rows)
    df = pd.DataFrame(ok_rows)
    print(f"成功 {len(df)} 失败 {err_count}")

    os.makedirs("results", exist_ok=True)
    json_path = f"results/shard_{SHARD_INDEX}_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    report_path = f"results/shard_{SHARD_INDEX}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_report(df, SHARD_INDEX, SHARD_TOTAL, len(all_results), err_count))

    alert_count = int((df["is_tradeable"] == True).sum()) if not df.empty and "is_tradeable" in df.columns else 0
    print(f"可交易: {alert_count}")
    push_shard_summary(df, alert_count, err_count, len(all_results))

    if os.environ.get("GITHUB_ACTIONS") == "true":
        out = os.environ.get("GITHUB_OUTPUT", "")
        if out:
            with open(out, "a", encoding="utf-8") as f:
                f.write(f"alert_count={alert_count}\n")
                f.write(f"success_count={len(df)}\n")
                f.write(f"error_count={err_count}\n")


if __name__ == "__main__":
    main()
