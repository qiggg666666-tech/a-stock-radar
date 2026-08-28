#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场筹码扫描 · GitHub Actions（双数据源 + Server酱）
【回测最优默认 + 红/绿尖峰区】

回测结论（2024–2025，持有5日）：
  · 严突破(区内+趋势+量且集中): 胜率~56–61%（profile=winrate 时）
  · CONFIRM_MODE=and
  · 默认找红色套牢尖峰 CHIP_PEAK_ZONE=red（绿=green，全局=all）

2026-08-27 全市场实测（4997只）复核：接近区里集中度(conc90)本来就有91%达标，
真正卡信号数量的是量比(VOL_MULTIPLIER)。默认档改为 balanced（量比1.5→1.3，
其余不变），预计信号量从 ~57只/日提升到 ~90只/日量级，但未经胜率回测验证。
要用胜率已验证的原档，设 CHIP_PROFILE=winrate。

  CHIP_PEAK_ZONE=red|green|all
  CHIP_CONFIRM_MODE=and|or
  CHIP_PROFILE=winrate|strict|legacy
  SERVERCHAN_SENDKEY / PUSH_ON_SHARD

依赖: pip install baostock akshare pandas numpy requests serverchan-sdk
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
PUSH_ON_SHARD = os.environ.get("PUSH_ON_SHARD", "0") == "1"
# Server酱 SendKey：兼容多种环境变量名
SENDKEY = (
    os.environ.get("SERVERCHAN_SENDKEY")
    or os.environ.get("SCT_KEY")
    or os.environ.get("SENDKEY")
    or ""
).strip()

_PROFILES = {
    "legacy": {"approach": 0.97, "break": 1.005, "vol": 1.2, "conc": 0.30},
    "winrate": {"approach": 0.97, "break": 1.01, "vol": 1.5, "conc": 0.20},
    "strict": {"approach": 0.96, "break": 1.02, "vol": 2.0, "conc": 0.15},
    # balanced: 基于 2026-08-27 全市场实测数据（4997只）调参 —— winrate 档下接近区
    # 里只有 7.5% 能同时过量比(1.5)+集中度(0.20)，其中集中度本身 91% 已达标，
    # 真正的瓶颈是量比。vol 从 1.5 降到 1.3，预计信号数从 57 只提升到 ~90 只量级，
    # 未经胜率回测验证，是数量优先的折中档；要严格按回测胜率跑用 CHIP_PROFILE=winrate。
    "balanced": {"approach": 0.97, "break": 1.01, "vol": 1.3, "conc": 0.20},
}
PROFILE = os.environ.get("CHIP_PROFILE", "balanced").strip().lower()  # 数据驱动折中档，回测档见 winrate
_P = _PROFILES.get(PROFILE, _PROFILES["winrate"])
APPROACH_RATIO = float(os.environ.get("CHIP_APPROACH_RATIO", _P["approach"]))
BREAK_RATIO = float(os.environ.get("CHIP_BREAK_RATIO", _P["break"]))
VOL_MULTIPLIER = float(os.environ.get("CHIP_VOL_MULTIPLIER", _P["vol"]))
CONC_THRESHOLD = float(os.environ.get("CHIP_CONC_THRESHOLD", _P["conc"]))

# 回测：突破信号质量最高 → 提高 break 权重
WEIGHTS = {"line": 0.15, "conc": 0.15, "peak": 0.10, "break": 0.45, "profit": 0.15}
# 确认模式：and=量且集中(回测更优) | or=量或集中(信号更多)
CONFIRM_MODE = os.environ.get("CHIP_CONFIRM_MODE", "and").strip().lower()
# 尖峰颜色区：red=套牢峰(p>现价) green=获利峰(p<=现价) all=全局主峰
PEAK_ZONE = os.environ.get("CHIP_PEAK_ZONE", "red").strip().lower()
if PEAK_ZONE not in ("red", "green", "all"):
    PEAK_ZONE = "red"
MAX_RETRIES = 3

# 尖峰柱专用阈值：独立于 CONC_THRESHOLD，专抓"单日巨量堆出的窄幅高峰"
# （典型场景：涨停/大涨当日换手率极高，价格波动窄，筹码瞬间在窄带内堆尖）
SPIKE_BAND_MIN = float(os.environ.get("CHIP_SPIKE_BAND_MIN", "0.12"))  # 峰值±1%价格带集中占比下限
SPIKE_GAP_MIN = float(os.environ.get("CHIP_SPIKE_GAP_MIN", "3.0"))     # 主峰权重/次峰权重 下限


# ==================== Server酱 ====================
def sc_send(sendkey: str, title: str, desp: str = "", options: Optional[dict] = None) -> dict:
    """兼容 SCT / Server酱³，不强制依赖 serverchan-sdk。"""
    if not sendkey:
        return {"code": -1, "message": "no sendkey"}
    options = options or {}
    try:
        from serverchan_sdk import sc_send as _sdk_send  # type: ignore
        return _sdk_send(sendkey, title, desp, options)
    except Exception:
        pass
    if sendkey.startswith("sctp"):
        m = re.match(r"sctp(\d+)t", sendkey)
        if not m:
            return {"code": -1, "message": "invalid sctp key"}
        url = f"https://{m.group(1)}.push.ft07.com/send/{sendkey}.send"
    else:
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
    payload = {"title": title.replace("\n", " ")[:100], "desp": desp, **options}
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json;charset=utf-8"}, timeout=15)
        return r.json()
    except Exception as e:
        return {"code": -1, "message": str(e)}


def push_shard_summary(df: pd.DataFrame, alert_n: int, err_n: int, total_n: int) -> None:
    if not SENDKEY or not PUSH_ON_SHARD:
        return
    title = f"筹码扫描 分片{SHARD_INDEX}/{SHARD_TOTAL} 可交易{alert_n}"
    lines = [
        f"- 成功 {total_n - err_n} / 失败 {err_n}",
        f"- profile={PROFILE}",
        "",
    ]
    if not df.empty and "is_tradeable" in df.columns:
        tdf = df[df["is_tradeable"] == True].sort_values(
            "total_score" if "total_score" in df.columns else "break_score", ascending=False
        ).head(15)
        if tdf.empty:
            lines.append("本分片无可交易标的")
        else:
            lines.append("| 代码 | 板块 | 收盘 | 综合 | 突破 |")
            lines.append("|------|------|------|------|------|")
            for _, r in tdf.iterrows():
                lines.append(
                    f"| {r.get('symbol','')} | {r.get('industry','')} | {r.get('close','')} | "
                    f"{r.get('total_score','')} | {r.get('break_score','')} |"
                )
    desp = "\n".join(lines)
    ret = sc_send(SENDKEY, title, desp)
    print(f"[Server酱] 分片推送: {ret}")


# ==================== 代码 / 板块 ====================
def to_bs(symbol: str) -> str:
    s = symbol.zfill(6)
    return f"sh.{s}" if s.startswith(("60", "68")) else f"sz.{s}"


def is_a_share(code: str) -> bool:
    s = code.replace("sh.", "").replace("sz.", "").zfill(6)
    if s.startswith(("90", "20", "43", "83", "87")):
        return False
    return s.startswith(("00", "30", "60", "68"))


def is_st_name(name: str) -> bool:
    """过滤 ST / *ST / S*ST / SST 等风险警示股。"""
    if not name:
        return False
    n = str(name).strip().upper().replace(" ", "")
    # 名称含 ST 即排除（含 *ST、ST、S*ST 等）
    return "ST" in n


def fetch_universe() -> List[str]:
    """全市场 A 股，默认去掉 ST/*ST。EXCLUDE_ST=0 可保留。"""
    exclude_st = os.environ.get("EXCLUDE_ST", "1") != "0"
    codes: List[str] = []
    st_skip = 0
    if HAVE_BS:
        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_basic()
                while rs.error_code == "0" and rs.next():
                    r = rs.get_row_data()
                    if not r or len(r) < 6:
                        continue
                    code, name, _ipo, out, typ, status = r[0], r[1], r[2], r[3], r[4], r[5]
                    if typ != "1" or status != "1":
                        continue
                    if out and out not in ("", "None"):
                        continue
                    pure = code.replace("sh.", "").replace("sz.", "")
                    if not is_a_share(pure):
                        continue
                    if exclude_st and is_st_name(name):
                        st_skip += 1
                        continue
                    codes.append(pure.zfill(6))
            finally:
                bs.logout()
    if not codes and HAVE_AK:
        try:
            spot = ak.stock_info_a_code_name()
            # 常见列: code / name 或 前两列
            cols = list(spot.columns)
            code_col = cols[0]
            name_col = cols[1] if len(cols) > 1 else None
            for _, row in spot.iterrows():
                c = str(row[code_col]).zfill(6)
                if not is_a_share(c):
                    continue
                nm = str(row[name_col]) if name_col is not None else ""
                if exclude_st and is_st_name(nm):
                    st_skip += 1
                    continue
                codes.append(c)
        except Exception as e:
            print(f"akshare 股票列表失败: {e}")
    codes = sorted(set(codes))
    if exclude_st:
        print(f"已过滤 ST/*ST 约 {st_skip} 只，剩余 {len(codes)} 只")
    return codes


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



def _local_peaks(items, total, eff_min):
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
    if not peaks and items:
        peaks = [max(items, key=lambda x: x[1])]
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks


def chip_features(chip: Dict[float, float], close: float, peak_zone: str = "red") -> dict:
    """
    peak_zone:
      red   — 主峰取套牢区 p>close 内最大峰（红柱尖峰）
      green — 主峰取获利区 p<=close 内最大峰（绿柱尖峰）
      all   — 全价位最大峰（旧行为）
    同时返回 red_peak / green_peak / all_peak 便于对照 App。
    """
    if not chip:
        return {}
    items = sorted(chip.items())
    total = sum(w for _, w in items) or 1.0
    prices = np.array([p for p, _ in items], dtype=float)
    weights = np.array([w for _, w in items], dtype=float)
    max_w = float(weights.max())
    eff_min = min(0.02, max(max_w / total * 0.5, 0.003))

    red_items = [(p, w) for p, w in items if p > close]
    green_items = [(p, w) for p, w in items if p <= close]
    red_w = sum(w for _, w in red_items)
    green_w = sum(w for _, w in green_items)

    peaks_all = _local_peaks(items, total, eff_min)
    peaks_red = _local_peaks(red_items, total, eff_min) if red_items else []
    peaks_green = _local_peaks(green_items, total, eff_min) if green_items else []

    def pack(peaks, fallback_items):
        if peaks:
            mp, mw = peaks[0]
            sec = peaks[1][1] if len(peaks) > 1 else mw * 0.01
        elif fallback_items:
            mp, mw = max(fallback_items, key=lambda x: x[1])
            sec = mw * 0.01
        else:
            return None, 0.0, 1.0
        return float(mp), float(mw / total), float(mw / max(sec, 1e-12))

    all_p, all_pr, all_gap = pack(peaks_all, items)
    red_p, red_pr, red_gap = pack(peaks_red, red_items) if red_items else (None, 0.0, 1.0)
    green_p, green_pr, green_gap = pack(peaks_green, green_items) if green_items else (None, 0.0, 1.0)

    zone = peak_zone if peak_zone in ("red", "green", "all") else "red"
    if zone == "red" and red_p is not None:
        main_p, peak_ratio, peak_gap = red_p, red_pr, red_gap
    elif zone == "green" and green_p is not None:
        main_p, peak_ratio, peak_gap = green_p, green_pr, green_gap
    else:
        # 指定色区无筹码时回退全局峰，并标记实际 zone
        main_p, peak_ratio, peak_gap = all_p, all_pr, all_gap
        if zone in ("red", "green"):
            zone = "all_fallback"
        if main_p is None:
            return {}

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
        "main_peak": main_p,
        "peak_ratio": peak_ratio,
        "band_ratio": band_ratio,
        "peak_gap": peak_gap,
        "conc90": conc90,
        "p5": p5,
        "p95": p95,
        "avg_cost": avg_cost,
        "profit": profit,
        "peak_zone": zone,
        "all_peak": all_p,
        "red_peak": red_p,
        "green_peak": green_p,
        "red_peak_ratio": red_pr,
        "green_peak_ratio": green_pr,
        "red_chip_pct": red_w / total * 100,
        "green_chip_pct": green_w / total * 100,
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


def score_break(feat, df) -> Tuple[float, bool, bool, float, float, bool]:
    """返回 (突破分, in_zone, tradeable, ma5, ma10, trend) —— ma5/ma10/trend 单独暴露，
    方便外层落盘和事后核对，而不只是埋在 tradeable 这一个布尔结果里。"""
    close = float(df.iloc[-1]["close"])
    mp = feat["main_peak"]
    if mp <= 0:
        return 0.0, False, False, close, close, False
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
    return round(float(np.clip(sc, 0, 100)), 1), in_zone, tradeable, round(ma5, 4), round(ma10, 4), trend


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
    feat = chip_features(chip, close, peak_zone=PEAK_ZONE)
    if not feat:
        return {"symbol": symbol, "industry": industry, "error": "empty_chip", "source": source}

    line_s = score_line(feat, last, close)
    conc_s = score_conc(feat)
    peak_s = score_peak(feat)
    break_s, is_approaching, is_tradeable, ma5, ma10, trend = score_break(feat, df)
    profit_s = score_profit(feat)

    # 尖峰柱判定：不看 conc90（全窗口离散度），只看局部峰形是否又高又窄
    is_spike = bool(feat["band_ratio"] >= SPIKE_BAND_MIN and feat["peak_gap"] >= SPIKE_GAP_MIN)
    is_spike_tradeable = bool(is_approaching and is_spike and not is_tradeable)
    total = (
        WEIGHTS["line"] * line_s + WEIGHTS["conc"] * conc_s + WEIGHTS["peak"] * peak_s
        + WEIGHTS["break"] * break_s + WEIGHTS["profit"] * profit_s
    )
    dims = {"直线尖峰": line_s, "高集中": conc_s, "单峰": peak_s, "突破预警": break_s, "获利结构": profit_s}
    best_tag = max(dims, key=dims.get)
    z = feat.get("peak_zone", PEAK_ZONE)
    zlabel = {"red": "红套牢峰", "green": "绿获利峰", "all": "全局峰"}.get(z, z)
    if is_tradeable:
        signal = f"可交易·接近{zlabel}+趋势确认"
    elif is_spike_tradeable:
        signal = f"尖峰关注·接近{zlabel}+窄幅高集中(忽略conc90)"
    elif is_approaching:
        signal = f"观察·接近{zlabel}未确认"
    else:
        signal = "无"
    amp = (float(last["high"]) - float(last["low"])) / close * 100 if close else 0

    def _r(x):
        return None if x is None else round(float(x), 2)

    return {
        "symbol": symbol,
        "industry": industry or "未知",
        "date": str(last["date"].date()),
        "close": round(close, 2),
        "peak_zone": z,
        "main_peak": _r(feat["main_peak"]),
        "all_peak": _r(feat.get("all_peak")),
        "red_peak": _r(feat.get("red_peak")),
        "green_peak": _r(feat.get("green_peak")),
        "red_chip_pct": round(feat.get("red_chip_pct") or 0, 2),
        "green_chip_pct": round(feat.get("green_chip_pct") or 0, 2),
        "avg_cost": round(feat["avg_cost"], 2),
        "dist_to_peak_pct": round((close - feat["main_peak"]) / feat["main_peak"] * 100, 2) if feat["main_peak"] else None,
        "band_ratio_pct": round(feat["band_ratio"] * 100, 2),
        "peak_gap": round(feat["peak_gap"], 2),
        "is_spike": is_spike,
        "is_spike_tradeable": is_spike_tradeable,
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
        "ma5": ma5,
        "ma10": ma10,
        "trend": trend,
        "is_approaching": is_approaching,
        "is_tradeable": is_tradeable,
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
    if "is_spike_tradeable" in df.columns:
        sdf = df[df["is_spike_tradeable"] == True].sort_values("band_ratio_pct", ascending=False)
        md.append(f"## 尖峰关注（{len(sdf)} 只，窄幅高集中·忽略conc90）\n")
        md.append("| 代码 | 板块 | 收盘 | 主峰 | 峰带占比% | 峰隙比 | 信号 |")
        md.append("|------|------|------|------|-----------|--------|------|")
        for _, r in sdf.head(40).iterrows():
            md.append(
                f"| {r['symbol']} | {r.get('industry','')} | {r['close']} | {r['main_peak']} | "
                f"{r.get('band_ratio_pct','')} | {r.get('peak_gap','')} | {r.get('signal','')} |"
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
    print(f"BS={HAVE_BS} AK={HAVE_AK} profile={PROFILE} peak_zone={PEAK_ZONE} SENDKEY={'有' if SENDKEY else '无'}")
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
