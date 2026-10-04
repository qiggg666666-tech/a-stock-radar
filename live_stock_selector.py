#!/usr/bin/env python3
"""
实战选股完整版 —— 两路分打 + 分开推送 + 多指数RSRS + 名称补全
============================================================
硬过滤：红筹码 + 获利盘40%~85% + MA20溢价≤18%
A路 宽幅启动 / B路 红筹尖峰 → 各自CSV、各自Server酱
择时：上证/深成/创业板/沪深300
多周期：日线重采样周/月线 → mtf_label / mtf_risk / 小幅加减分
  --mtf-filter off|soft|strict|resonate（默认 off）
  resonate = 周线MA20上 + 日线溢价>=-3% + 月线未极端高位破位

指标架构：
  核心：MA20 / EMA120 / EMA200 + 聚类买卖标注（只加减分，不硬剔）
  补全：TA-Lib 可用则 RSI/MACD/布林，否则 pandas 近似
  研究：--tsfresh-research 仅 merge 对 Top 抽 Minimal 特征（可选依赖）

用法：
  python live_stock_selector.py --shard 1
  python live_stock_selector.py --shard all --mtf-filter soft
  python live_stock_selector.py --shard all --exclude-st      # 跳过 ST/退市整理股
  python live_stock_selector.py --merge --top 15 --notify --mtf-filter resonate
  python live_stock_selector.py --merge --top 15 --tsfresh-research

两段式（推荐）：
  python mtf_resonate_universe.py --shard all
  python live_stock_selector.py --shard all --universe mtf_resonate_universe.csv
  python live_stock_selector.py --merge --top 15 --notify
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import akshare as ak
import numpy as np
import pandas as pd
import requests

try:
    from final_chip_research import analyze, fetch_ohlcv, is_beijing_stock
    FULL_CHIP_AVAILABLE = True
    print("已加载完整 final_chip_research 模块")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：未找到 final_chip_research.py，筹码过滤将失效")

# TA-Lib 可选：有则用 C 实现，无则 pandas 近似（不阻断主流程）
try:
    import talib  # type: ignore

    TALIB_AVAILABLE = True
    print("已加载 TA-Lib（指标补全）")
except ImportError:
    TALIB_AVAILABLE = False

# tsfresh 可选：仅 --tsfresh-research 时使用
try:
    from tsfresh import extract_features  # type: ignore
    from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore
    from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute  # type: ignore

    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False

CACHE_DIR = Path("./live_selector_cache")
CACHE_DIR.mkdir(exist_ok=True)

SHARD_MAP = {
    1: ["000", "001", "003"],
    2: ["002"],
    3: ["300", "301"],
    4: ["600"],
    5: ["601"],
    6: ["603"],
    7: ["605", "688"],
    8: ["8", "4", "9"],
}

TOP_N_DEFAULT = 15
PROFIT_MIN = 40.0
PROFIT_MAX = 85.0
TURN_OVER_CAP = 0.15
MIN_BARS = 100
HIST_DAYS = 800   # akshare 回退取数窗口（日历日，约2年），保证 MA250/年线样本
STALE_DAYS = 15   # 最后一根K线距今超过该天数视为停牌/退市，直接跳过
MA20_MAX_PREMIUM = 0.18
INDEX_CRASH_THRESHOLD = -0.03
PEAK_PROFIT_MIN = 70.0
# 多周期硬过滤：
# off=只标注加减分; soft=剔走坏; strict=周线偏多;
# resonate=多周期共振（周线多+日线未深破+月线未极端高位破位）
MTF_FILTER_DEFAULT = "off"

WEIGHTS_WIDE = {
    "wide_score": 0.35,
    "year_first_red": 0.20,
    "VwapClose": 0.15,
    "HighOpen": 0.10,
    "CloseLow": 0.10,
    "Turnover": 0.10,
}

WEIGHTS_PEAK = {
    "profit_pct": 0.30,
    "inv_wide": 0.25,
    "HighOpen": 0.15,
    "VwapClose": 0.15,
    "CloseLow": 0.10,
    "Turnover": 0.05,
}

RSRS_N = 18
RSRS_M = 600
RSRS_BUY_THRESHOLD = 0.7

INDEX_LIST = [
    ("上证综指", "sh000001"),
    ("深证成指", "sz399001"),
    ("创业板指", "sz399006"),
    ("沪深300", "sh000300"),
]

_NAME_MAP: dict[str, str] | None = None


def get_name_map() -> dict[str, str]:
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP
    _NAME_MAP = {}
    try:
        info = ak.stock_info_a_code_name()
        if info is not None and not info.empty:
            codes = info["code"].astype(str).str.zfill(6)
            names = info["name"].astype(str)
            _NAME_MAP = dict(zip(codes, names))
            print(f"名称表加载 {len(_NAME_MAP)} 只")
    except Exception as e:
        print(f"名称表加载失败: {e}")
    return _NAME_MAP


def fill_names(df: pd.DataFrame) -> pd.DataFrame:
    """把 name 为空白/nan 的行用 akshare 名称表补全。"""
    if df.empty:
        return df
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 先统一成普通 str，避免 pandas StringDtype 拒绝 float NaN
    if "name" not in df.columns:
        df["name"] = ""
    df["name"] = df["name"].map(
        lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    )
    df["name"] = df["name"].replace({"nan": "", "None": "", "NaN": ""})
    need = df["name"].str.strip() == ""
    if need.any():
        name_map = get_name_map()
        mapped = df.loc[need, "code"].map(
            lambda c: str(name_map.get(str(c).zfill(6), "") or "")
        )
        df.loc[need, "name"] = mapped.values
    df["name"] = df["name"].fillna("").astype(str)
    return df


_ALL_CODES: list[str] | None = None


def get_all_a_stocks() -> list[str]:
    """全市场代码（进程内缓存，--shard all 时只请求一次）。"""
    global _ALL_CODES
    if _ALL_CODES is not None:
        return list(_ALL_CODES)

    codes: list[str] = []
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_basic()
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    code = row[0]
                    stock_type = row[4] if len(row) > 4 else "1"
                    status = row[5] if len(row) > 5 else "1"  # 1=上市 0=退市
                    if stock_type == "1" and status == "1":
                        pure_code = code.split(".")[-1] if "." in code else code
                        codes.append(pure_code.zfill(6))
            finally:
                bs.logout()
            if codes:
                codes = sorted(set(codes))
                print(f"baostock 获取到 {len(codes)} 只股票")
                _ALL_CODES = codes
                return list(codes)
    except Exception as e:
        print(f"baostock 获取股票列表失败: {e}")

    try:
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        if codes:
            print(f"akshare 获取到 {len(codes)} 只股票")
            _ALL_CODES = codes
            return list(codes)
    except Exception as e:
        print(f"akshare 获取股票列表失败: {e}")

    print("错误：无法获取全市场股票列表")
    return []


def report_uncovered(codes: list[str]) -> None:
    """提示未被任何分片前缀覆盖的代码（防止像 301/003 那样被悄悄漏掉）。"""
    covered = tuple(p for ps in SHARD_MAP.values() for p in ps)
    miss = [c for c in codes if not c.startswith(covered)]
    if miss:
        print(f"警告：{len(miss)} 只代码未被任何分片覆盖，例如 {miss[:15]}")


def is_st_name(name: str) -> bool:
    n = str(name or "").upper().replace(" ", "")
    return bool(re.match(r"^(\*?ST|S\*ST|SST)", n)) or "退" in n


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def load_universe(path: str | Path) -> list[str]:
    """从共振池 CSV 读取代码列表（需含 code 列，或单列代码）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"universe 文件不存在: {p}")
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        df = pd.read_csv(p, header=None, dtype=str)
    if df.empty:
        return []
    if "code" in df.columns:
        codes = df["code"].astype(str)
    else:
        codes = df.iloc[:, 0].astype(str)
    codes = (
        codes.str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )
    codes = sorted({c for c in codes.tolist() if c and c != "000000"})
    return codes


def _normalize_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    """统一：date 为 datetime、升序、去重、OHLC 无空值。缺关键列则返回空表。"""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    if any(c not in df.columns for c in ("open", "high", "low", "close")):
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def _fetch_akshare_hist(symbol: str, retries: int = 2) -> pd.DataFrame:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=HIST_DAYS)).strftime("%Y%m%d")  # 不用 replace(year=)，避开 2/29
    for attempt in range(retries):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            raw = raw.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "换手率": "turnover",
            })
            raw["symbol"] = symbol
            time.sleep(0.2)
            return _normalize_ohlcv(raw)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def _is_stale(df: pd.DataFrame) -> bool:
    try:
        last = pd.to_datetime(df["date"]).max()
        return (datetime.now() - last.to_pydatetime()).days > STALE_DAYS
    except Exception:
        return False


def get_stock_data(symbol: str) -> pd.DataFrame:
    best = pd.DataFrame()
    if FULL_CHIP_AVAILABLE:
        try:
            tmp, _source, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if tmp is not None and not tmp.empty:
                tmp = tmp.copy()
                tmp["symbol"] = symbol
                best = _normalize_ohlcv(tmp)
        except Exception:
            pass

    if best.empty or len(best) < 200:
        fb = _fetch_akshare_hist(symbol)
        if len(fb) > len(best):  # 取更长的那份，避免回退数据反而覆盖掉主源
            best = fb

    return best if not best.empty else pd.DataFrame()


def calc_launch_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["CloseLow"] = df["close"] / df["low"] - 1

    nan_s = pd.Series(np.nan, index=df.index)
    amount = pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns else nan_s
    volume = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else nan_s
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap_raw = amount / (volume * 100 + 1e-8)
    # 单位异常（落在 [low*0.9, high*1.1] 之外）时退回典型价，防止 VwapClose 被脏数据拉飞
    vwap_ok = (volume > 0) & (amount > 0) & (vwap_raw >= df["low"] * 0.9) & (vwap_raw <= df["high"] * 1.1)
    df["vwap"] = np.where(vwap_ok, vwap_raw, typical)
    df["VwapClose"] = df["vwap"] / df["close"] - 1
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    turn_raw = pd.to_numeric(df["turnover"], errors="coerce") if "turnover" in df.columns else nan_s
    df["Turnover"] = turn_raw.fillna(0.0) / 100.0

    # 均线组 MA/EMA：10 / 20 / 30 / 60 / 120 / 200
    for n in (10, 20, 30, 60, 120, 200):
        mp = max(5, n // 2)
        df[f"ma{n}"] = df["close"].rolling(n, min_periods=mp).mean()
        df[f"ema{n}"] = df["close"].ewm(span=n, adjust=False, min_periods=mp).mean()
    df["ma20_premium"] = df["close"] / df["ma20"] - 1
    df["ema20_premium"] = df["close"] / df["ema20"] - 1
    df["ema60_premium"] = df["close"] / df["ema60"] - 1
    df["ema120_premium"] = df["close"] / df["ema120"] - 1
    df["ema200_premium"] = df["close"] / df["ema200"] - 1

    df["ma250"] = df["close"].rolling(250, min_periods=180).mean()
    df["above_ma250"] = df["close"] > df["ma250"]
    prev_above = df["above_ma250"].shift(1).fillna(False).astype(bool)
    five_days_ago_below = (df["close"].shift(5) < df["ma250"].shift(5)).fillna(False)
    df["year_first_red"] = (
        df["above_ma250"] & (~prev_above) & five_days_ago_below
    ).astype(float)
    return df


def calc_ta_supplement(df: pd.DataFrame) -> dict[str, Any]:
    """
    TA 指标补全（非核心）：RSI / MACD / 布林带位置。
    优先 TA-Lib；不可用时用 pandas 近似，保证主流程可跑。
    仅标注 + 小幅加减分，不硬过滤。
    """
    out: dict[str, Any] = {
        "rsi14": None,
        "macd": None,
        "macd_signal": None,
        "macd_hist": None,
        "bb_pos": None,  # 0=下轨 1=上轨
        "ta_label": "",
        "ta_bonus": 0.0,
        "ta_source": "none",
    }
    if df is None or len(df) < 35:
        return out
    try:
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values if "high" in df.columns else close
        low = df["low"].astype(float).values if "low" in df.columns else close

        if TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=14)
            macd, sig, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            out["ta_source"] = "talib"
            rsi_v = float(rsi[-1]) if np.isfinite(rsi[-1]) else None
            macd_v = float(macd[-1]) if np.isfinite(macd[-1]) else None
            sig_v = float(sig[-1]) if np.isfinite(sig[-1]) else None
            hist_v = float(hist[-1]) if np.isfinite(hist[-1]) else None
            up_v, lo_v = float(upper[-1]), float(lower[-1])
            c = float(close[-1])
            bb_pos = (c - lo_v) / (up_v - lo_v + 1e-12) if up_v > lo_v else None
        else:
            s = pd.Series(close)
            # RSI 近似
            delta = s.diff()
            # Wilder 平滑，与 TA-Lib 口径一致（原简单均值会让 RSI 偏离）
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            rs = gain / (loss + 1e-12)
            rsi_s = 100 - (100 / (1 + rs))
            # MACD 近似
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            macd_s = ema12 - ema26
            sig_s = macd_s.ewm(span=9, adjust=False).mean()
            hist_s = macd_s - sig_s
            # 布林
            mid = s.rolling(20, min_periods=15).mean()
            std = s.rolling(20, min_periods=15).std()
            up = mid + 2 * std
            lo = mid - 2 * std
            out["ta_source"] = "pandas"
            rsi_v = float(rsi_s.iloc[-1]) if np.isfinite(rsi_s.iloc[-1]) else None
            macd_v = float(macd_s.iloc[-1]) if np.isfinite(macd_s.iloc[-1]) else None
            sig_v = float(sig_s.iloc[-1]) if np.isfinite(sig_s.iloc[-1]) else None
            hist_v = float(hist_s.iloc[-1]) if np.isfinite(hist_s.iloc[-1]) else None
            c = float(close[-1])
            up_v, lo_v = float(up.iloc[-1]), float(lo.iloc[-1])
            bb_pos = (c - lo_v) / (up_v - lo_v + 1e-12) if up_v > lo_v else None

        out["rsi14"] = round(rsi_v, 2) if rsi_v is not None else None
        out["macd"] = round(macd_v, 4) if macd_v is not None else None
        out["macd_signal"] = round(sig_v, 4) if sig_v is not None else None
        out["macd_hist"] = round(hist_v, 4) if hist_v is not None else None
        out["bb_pos"] = round(bb_pos, 4) if bb_pos is not None else None

        tags: list[str] = []
        bonus = 0.0
        if rsi_v is not None:
            if rsi_v < 30:
                tags.append("RSI超卖")
                bonus += 0.06
            elif rsi_v > 70:
                tags.append("RSI超买")
                bonus -= 0.06
        if hist_v is not None and macd_v is not None and sig_v is not None:
            if hist_v > 0 and macd_v > sig_v:
                tags.append("MACD多头")
                bonus += 0.04
            elif hist_v < 0 and macd_v < sig_v:
                tags.append("MACD空头")
                bonus -= 0.04
        if bb_pos is not None:
            if bb_pos <= 0.15:
                tags.append("近布林下轨")
                bonus += 0.03
            elif bb_pos >= 0.85:
                tags.append("近布林上轨")
                bonus -= 0.03

        out["ta_label"] = "+".join(tags) if tags else ""
        out["ta_bonus"] = round(float(np.clip(bonus, -0.12, 0.12)), 3)
    except Exception:
        pass
    return out


def calc_ema_cluster(df: pd.DataFrame) -> dict[str, Any]:
    """
    【核心】MA/EMA 10·20·30·60·120·200 + 聚类买卖（默认只标注加减分）。
    多头：短期均线在上、价在关键线上方、上穿/回踩。
    空头：相反。
    """
    spans = (10, 20, 30, 60, 120, 200)
    out: dict[str, Any] = {
        "ema10": None,
        "ema20": None,
        "ema30": None,
        "ema60": None,
        "ema120": None,
        "ema200": None,
        "ma20": None,
        "ma60": None,
        "ema20_premium": None,
        "ema60_premium": None,
        "ema120_premium": None,
        "ema200_premium": None,
        "ema_stack": "",
        "above_ema_n": 0,
        "cluster_buy_n": 0,
        "cluster_sell_n": 0,
        "cluster_signal": "中性",
        "cluster_label": "",
        "cluster_bonus": 0.0,
    }
    if df is None or len(df) < 60:
        return out
    try:
        d = df.sort_values("date").copy() if "date" in df.columns else df.copy()
        for n in spans:
            if f"ema{n}" not in d.columns:
                mp = max(5, n // 2)
                d[f"ema{n}"] = d["close"].ewm(span=n, adjust=False, min_periods=mp).mean()
            if f"ma{n}" not in d.columns:
                mp = max(5, n // 2)
                d[f"ma{n}"] = d["close"].rolling(n, min_periods=mp).mean()

        close = float(d["close"].iloc[-1])
        emas = {}
        for n in spans:
            v = d[f"ema{n}"].iloc[-1]
            emas[n] = float(v) if pd.notna(v) and np.isfinite(v) else None
            out[f"ema{n}"] = round(emas[n], 2) if emas[n] else None

        if emas.get(20):
            out["ma20"] = round(float(d["ma20"].iloc[-1]), 2) if pd.notna(d["ma20"].iloc[-1]) else None
            out["ema20_premium"] = round(close / emas[20] - 1.0, 4)
        if emas.get(60):
            out["ma60"] = round(float(d["ma60"].iloc[-1]), 2) if "ma60" in d.columns and pd.notna(d["ma60"].iloc[-1]) else None
            out["ema60_premium"] = round(close / emas[60] - 1.0, 4)
        if emas.get(120):
            out["ema120_premium"] = round(close / emas[120] - 1.0, 4)
        if emas.get(200):
            out["ema200_premium"] = round(close / emas[200] - 1.0, 4)

        # 价在几条 EMA 上方
        above_n = sum(1 for n in spans if emas.get(n) and close > emas[n])
        out["above_ema_n"] = above_n

        # 排列：EMA10>20>30>60 为短多；120>200 为长多
        def _chain_up(keys: list[int]) -> bool:
            vals = [emas.get(k) for k in keys]
            if any(v is None for v in vals):
                return False
            return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

        def _chain_dn(keys: list[int]) -> bool:
            vals = [emas.get(k) for k in keys]
            if any(v is None for v in vals):
                return False
            return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

        short_bull = _chain_up([10, 20, 30, 60])
        short_bear = _chain_dn([10, 20, 30, 60])
        long_bull = (
            emas.get(60) and emas.get(120) and emas.get(200)
            and emas[60] > emas[120] > emas[200]
        )
        long_bear = (
            emas.get(60) and emas.get(120) and emas.get(200)
            and emas[60] < emas[120] < emas[200]
        )
        if short_bull and long_bull and above_n >= 5:
            out["ema_stack"] = "全多头排列"
        elif short_bull and above_n >= 4:
            out["ema_stack"] = "短多排列"
        elif short_bear and long_bear:
            out["ema_stack"] = "空头排列"
        elif short_bear:
            out["ema_stack"] = "短空排列"
        else:
            out["ema_stack"] = "纠缠"

        # 上穿/下穿 EMA20（敏感）与 EMA60（波段）
        cross_up20 = cross_dn20 = cross_up60 = False
        if len(d) >= 6 and emas.get(20):
            for i in range(-5, 0):
                c0, c1 = float(d["close"].iloc[i]), float(d["close"].iloc[i - 1])
                a0, a1 = float(d["ema20"].iloc[i]), float(d["ema20"].iloc[i - 1])
                if c1 <= a1 and c0 > a0:
                    cross_up20 = True
                if c1 >= a1 and c0 < a0:
                    cross_dn20 = True
                if emas.get(60):
                    b0, b1 = float(d["ema60"].iloc[i]), float(d["ema60"].iloc[i - 1])
                    if c1 <= b1 and c0 > b0:
                        cross_up60 = True

        buy_n = sell_n = 0
        buy_tags: list[str] = []
        sell_tags: list[str] = []

        if above_n >= 4:
            buy_n += 1
            buy_tags.append(f"站上{above_n}条EMA")
        elif above_n <= 2:
            sell_n += 1
            sell_tags.append(f"仅上{above_n}条EMA")

        if short_bull:
            buy_n += 1
            buy_tags.append("短多10>20>30>60")
        if short_bear:
            sell_n += 1
            sell_tags.append("短空排列")
        if long_bull:
            buy_n += 1
            buy_tags.append("长多60>120>200")
        if long_bear:
            sell_n += 1
            sell_tags.append("长空排列")

        if cross_up20:
            buy_n += 1
            buy_tags.append("上穿EMA20")
        if cross_dn20:
            sell_n += 1
            sell_tags.append("下穿EMA20")
        if cross_up60:
            buy_n += 1
            buy_tags.append("上穿EMA60")

        prem20 = out.get("ema20_premium")
        if prem20 is not None and 0 <= prem20 <= 0.02:
            buy_n += 1
            buy_tags.append("回踩EMA20")
        if prem20 is not None and -0.02 <= prem20 < 0:
            sell_n += 1
            sell_tags.append("EMA20受压")

        out["cluster_buy_n"] = buy_n
        out["cluster_sell_n"] = sell_n

        if buy_n >= 3 and buy_n > sell_n:
            out["cluster_signal"] = "聚类买入"
            out["cluster_label"] = "买:" + "+".join(buy_tags[:4])
            out["cluster_bonus"] = min(0.22, 0.04 * buy_n)
        elif sell_n >= 3 and sell_n > buy_n:
            out["cluster_signal"] = "聚类卖出"
            out["cluster_label"] = "卖:" + "+".join(sell_tags[:4])
            out["cluster_bonus"] = -min(0.22, 0.04 * sell_n)
        elif buy_n >= 2 and buy_n > sell_n:
            out["cluster_signal"] = "偏多观察"
            out["cluster_label"] = "买:" + "+".join(buy_tags[:3])
            out["cluster_bonus"] = 0.05
        elif sell_n >= 2 and sell_n > buy_n:
            out["cluster_signal"] = "偏空观察"
            out["cluster_label"] = "卖:" + "+".join(sell_tags[:3])
            out["cluster_bonus"] = -0.05
        else:
            out["cluster_signal"] = "中性"
            out["cluster_label"] = out["ema_stack"] or ""
            out["cluster_bonus"] = 0.0
    except Exception:
        pass
    return out


_FREQ_FALLBACK = {"ME": "M", "QE": "Q", "YE": "Y"}


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日线 → 周/月/季/年线。pandas>=2.2 用 ME/QE/YE，旧版自动回退 M/Q/Y。"""
    x = df.set_index("date").sort_index()

    def _do(r: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": x["open"].resample(r).first(),
                "high": x["high"].resample(r).max(),
                "low": x["low"].resample(r).min(),
                "close": x["close"].resample(r).last(),
                "volume": x["volume"].resample(r).sum() if "volume" in x.columns else 0,
            }
        ).dropna(subset=["close"])

    try:
        return _do(rule)
    except ValueError:
        if rule in _FREQ_FALLBACK:
            return _do(_FREQ_FALLBACK[rule])
        raise


def calc_multi_timeframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    多周期结构（研究用，默认不硬过滤，只输出标签与加分）。
    - 周线 MA20：波段方向
    - 月线相对近 12 月高低位：是否冲高过猛
    - 日线 vs 周线：是否日线弱于周线（回撤）
    """
    out: dict[str, Any] = {
        "w_ma20": None,
        "w_ma20_premium": None,
        "w_above_ma20": None,
        "m_close": None,
        "m_pos_12": None,  # 近12月区间位置 0~1
        "m_stretch": None,  # 相对近12月低点涨幅
        "mtf_label": "数据不足",
        "mtf_bonus": 0.0,
        "mtf_risk": "",
    }
    if df is None or len(df) < 60:
        return out

    try:
        daily = df.sort_values("date").copy()
        daily["date"] = pd.to_datetime(daily["date"])
        close = float(daily["close"].iloc[-1])

        # ----- 周线 -----
        weekly = _resample_ohlc(daily, "W-FRI")
        if len(weekly) >= 20:
            weekly["ma20"] = weekly["close"].rolling(20, min_periods=15).mean()
            w_ma20 = float(weekly["ma20"].iloc[-1])
            w_close = float(weekly["close"].iloc[-1])
            if np.isfinite(w_ma20) and w_ma20 > 0:
                out["w_ma20"] = round(w_ma20, 2)
                out["w_ma20_premium"] = round(w_close / w_ma20 - 1, 4)
                out["w_above_ma20"] = bool(w_close > w_ma20)

        # ----- 月线 -----
        monthly = _resample_ohlc(daily, "ME")
        if len(monthly) >= 6:
            m_close = float(monthly["close"].iloc[-1])
            out["m_close"] = round(m_close, 2)
            tail = monthly.tail(12)
            lo = float(tail["low"].min())
            hi = float(tail["high"].max())
            if hi > lo > 0:
                out["m_pos_12"] = round((m_close - lo) / (hi - lo), 4)
            if lo > 0:
                out["m_stretch"] = round(m_close / lo - 1, 4)

        # ----- 综合标签 -----
        _ma20_d = daily["close"].rolling(20, min_periods=15).mean().iloc[-1]
        d_prem = float(close / _ma20_d - 1) if (len(daily) >= 20 and pd.notna(_ma20_d) and _ma20_d > 0) else 0.0
        w_up = out.get("w_above_ma20")
        m_pos = out.get("m_pos_12")
        m_stretch = out.get("m_stretch")

        # 风险：月线近一年位置很高 + 日线已跌破日MA20 → 冲高回撤
        risk_bits = []
        if m_pos is not None and m_pos >= 0.75 and d_prem < 0:
            risk_bits.append("月线高位回撤")
        if m_stretch is not None and m_stretch >= 1.0 and d_prem < 0:
            risk_bits.append("较年内低点已翻倍级回落中")
        if w_up is False and d_prem < 0:
            risk_bits.append("周线日线均在MA20下")

        bonus = 0.0
        if w_up is True and d_prem >= -0.03:
            # 周线偏多且日线未深度跌破
            bonus += 0.15
        if w_up is True and -0.12 < d_prem < -0.03:  # 与上一档(>=-3%)互斥，原先 -3%~-2% 会重复加分
            # 周线仍强、日线小回踩 —— 常见观察结构
            bonus += 0.10
        if m_pos is not None and m_pos >= 0.85 and d_prem < -0.05:
            # 月线过高且日线已弱，降分
            bonus -= 0.15

        if w_up is True and (m_pos is None or m_pos < 0.85) and d_prem >= -0.05:
            label = "周线偏多·日线未过弱"
        elif w_up is True and d_prem < -0.05:
            label = "周线偏多·日线回踩"
        elif w_up is False and d_prem < 0:
            label = "周月偏弱·日线调整"
        elif risk_bits:
            label = "冲高回撤观察"
        else:
            label = "多周期中性"

        out["mtf_label"] = label
        out["mtf_bonus"] = round(bonus, 3)
        out["mtf_risk"] = "；".join(risk_bits)
    except Exception:
        pass
    return out


def calc_long_timeframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    年线 / 季线二次标注（研究用，默认不硬剔除，只标签 + 小幅加减分）。
    日线样本有限时年线统计不稳定，标签偏保守。
    """
    out: dict[str, Any] = {
        "q_pos_8": None,       # 近 8 季高低位置 0~1
        "q_above_ma4": None,   # 季线是否在 4 季均线上方
        "y_pos_5": None,       # 近 5 年高低位置 0~1
        "y_above_ma3": None,   # 年线是否在 3 年均线上方
        "long_label": "长周期数据不足",
        "long_bonus": 0.0,
    }
    if df is None or len(df) < 120:
        return out
    try:
        daily = df.sort_values("date").copy()
        daily["date"] = pd.to_datetime(daily["date"])
        close = float(daily["close"].iloc[-1])

        # ----- 季线 -----
        quarterly = _resample_ohlc(daily, "QE")
        if len(quarterly) >= 4:
            q_close = float(quarterly["close"].iloc[-1])
            q_ma4 = float(quarterly["close"].rolling(4, min_periods=3).mean().iloc[-1])
            if np.isfinite(q_ma4) and q_ma4 > 0:
                out["q_above_ma4"] = bool(q_close > q_ma4)
            tail_q = quarterly.tail(min(8, len(quarterly)))
            lo_q = float(tail_q["low"].min())
            hi_q = float(tail_q["high"].max())
            if hi_q > lo_q > 0:
                out["q_pos_8"] = round((q_close - lo_q) / (hi_q - lo_q), 4)

        # ----- 年线 -----
        yearly = _resample_ohlc(daily, "YE")
        if len(yearly) >= 2:
            y_close = float(yearly["close"].iloc[-1])
            if len(yearly) >= 3:
                y_ma3 = float(yearly["close"].rolling(3, min_periods=2).mean().iloc[-1])
                if np.isfinite(y_ma3) and y_ma3 > 0:
                    out["y_above_ma3"] = bool(y_close > y_ma3)
            tail_y = yearly.tail(min(5, len(yearly)))
            lo_y = float(tail_y["low"].min())
            hi_y = float(tail_y["high"].max())
            if hi_y > lo_y > 0:
                out["y_pos_5"] = round((y_close - lo_y) / (hi_y - lo_y), 4)

        q_up = out.get("q_above_ma4")
        y_up = out.get("y_above_ma3")
        q_pos = out.get("q_pos_8")
        y_pos = out.get("y_pos_5")

        # 标签
        if q_up is True and (y_up is True or y_up is None):
            if (q_pos is not None and q_pos >= 0.9) or (y_pos is not None and y_pos >= 0.9):
                label = "长周期偏多·高位区"
            else:
                label = "长周期偏多"
        elif q_up is False and (y_up is False or y_up is None):
            label = "长周期仍弱"
        elif q_up is True and y_up is False:
            label = "季线转强·年线仍弱"
        elif q_up is False and y_up is True:
            label = "年线偏多·季线调整"
        else:
            label = "长周期中性"

        # 小幅加减分：偏多加、仍弱减；高位区略减防追高
        bonus = 0.0
        if label == "长周期偏多":
            bonus += 0.08
        elif label == "长周期偏多·高位区":
            bonus += 0.02
        elif label == "季线转强·年线仍弱":
            bonus += 0.03
        elif label == "长周期仍弱":
            bonus -= 0.10

        out["long_label"] = label
        out["long_bonus"] = round(bonus, 3)
    except Exception:
        pass
    return out


def calc_limitup_setup(df: pd.DataFrame) -> dict[str, Any]:
    """
    冲板/强势结构观察（独立研究模块，不是「即将涨停」预测）。
    仅用日线收盘数据。

    命中 near_limitup_setup（score>=2）：
      A) 当日涨幅≥5% 且收盘位≥0.85 且换手相对5日放大
      B) 近2日累计≥9% 且收盘位≥0.75 且距20日高点≤4%
      C) 昨日涨幅≥9.5% 且今日>-3% 且收盘位≥0.6
      附加) 当日≥7% 且收盘位≥0.9 → 逼近涨停区
    """
    out: dict[str, Any] = {
        "day_ret": None,
        "ret_2d": None,
        "close_pos_lu": None,
        "turn_ratio_5": None,
        "near_limitup_setup": False,
        "limitup_label": "",
        "limitup_bonus": 0.0,
        "limitup_score": 0.0,
    }
    if df is None or len(df) < 15:
        return out
    try:
        d = df.sort_values("date").reset_index(drop=True)
        c0 = float(d["close"].iloc[-1])
        c1 = float(d["close"].iloc[-2])
        c2 = float(d["close"].iloc[-3]) if len(d) >= 3 else np.nan
        h0 = float(d["high"].iloc[-1])
        l0 = float(d["low"].iloc[-1])
        if c0 <= 0 or c1 <= 0:
            return out

        day_ret = c0 / c1 - 1.0
        prev_ret = c1 / c2 - 1.0 if np.isfinite(c2) and c2 > 0 else None
        ret_2d = c0 / c2 - 1.0 if np.isfinite(c2) and c2 > 0 else None
        close_pos = (c0 - l0) / (h0 - l0 + 1e-8)

        turn = pd.to_numeric(d["turnover"], errors="coerce").fillna(0.0) if "turnover" in d.columns else pd.Series([0.0] * len(d))
        if float(turn.median(skipna=True) or 0) > 1.5:
            turn = turn / 100.0
        t0 = float(turn.iloc[-1]) if len(turn) else 0.0
        t5 = float(turn.tail(6).iloc[:-1].mean()) if len(turn) >= 6 else t0
        turn_ratio = t0 / (t5 + 1e-8)

        high20 = float(d["high"].tail(20).max())
        dist20 = high20 / c0 - 1.0 if c0 > 0 else 9.0

        out["day_ret"] = round(day_ret, 4)
        out["ret_2d"] = round(ret_2d, 4) if ret_2d is not None else None
        out["close_pos_lu"] = round(float(close_pos), 4)
        out["turn_ratio_5"] = round(turn_ratio, 3)

        score = 0.0
        tags: list[str] = []

        if day_ret >= 0.05 and close_pos >= 0.85 and turn_ratio >= 1.3:
            score += 2.0
            tags.append("大阳收上沿")
        if ret_2d is not None and ret_2d >= 0.09 and close_pos >= 0.75 and 0 <= dist20 <= 0.04:
            score += 1.5
            tags.append("两日强势贴高")
        if prev_ret is not None and prev_ret >= 0.095 and day_ret > -0.03 and close_pos >= 0.6:
            score += 1.5
            tags.append("昨强今未崩")
        if day_ret >= 0.07 and close_pos >= 0.9:
            score += 1.0
            tags.append("逼近涨停区")

        near = score >= 2.0 and len(tags) >= 1
        out["near_limitup_setup"] = near
        out["limitup_score"] = round(score, 2)
        if near:
            out["limitup_label"] = "冲板结构·" + "+".join(tags[:3])
            out["limitup_bonus"] = min(0.25, 0.08 * score)
        elif tags:
            out["limitup_label"] = "强势观察·" + tags[0]
            out["limitup_bonus"] = 0.05
        else:
            out["limitup_label"] = ""
            out["limitup_bonus"] = 0.0
    except Exception:
        pass
    return out


def calc_breakout_setup(df: pd.DataFrame) -> dict[str, Any]:
    """
    「近端涨幅≥10% + 接近突破」研究标注（不是涨停预测）。
    条件：
      1) 相对近 20 日低点涨幅 >= 10%，或近 10 日涨幅 >= 8%
      2) 收盘接近近 20/60 日高点（距高点 ≤ 3%），尚未明显远离
    """
    out: dict[str, Any] = {
        "ret_from_low20": None,
        "ret_10d": None,
        "dist_to_high20": None,
        "dist_to_high60": None,
        "near_breakout": False,
        "breakout_label": "",
        "breakout_bonus": 0.0,
    }
    if df is None or len(df) < 25:
        return out
    try:
        d = df.sort_values("date").copy()
        close = float(d["close"].iloc[-1])
        if close <= 0:
            return out

        low20 = float(d["low"].tail(20).min())
        high20 = float(d["high"].tail(20).max())
        high60 = float(d["high"].tail(min(60, len(d))).max())

        ret_low20 = close / low20 - 1.0 if low20 > 0 else None
        ret_10d = None
        if len(d) >= 11:
            c10 = float(d["close"].iloc[-11])
            if c10 > 0:
                ret_10d = close / c10 - 1.0

        dist20 = high20 / close - 1.0 if close > 0 else None
        dist60 = high60 / close - 1.0 if close > 0 else None

        out["ret_from_low20"] = round(ret_low20, 4) if ret_low20 is not None else None
        out["ret_10d"] = round(ret_10d, 4) if ret_10d is not None else None
        out["dist_to_high20"] = round(dist20, 4) if dist20 is not None else None
        out["dist_to_high60"] = round(dist60, 4) if dist60 is not None else None

        strength = (ret_low20 is not None and ret_low20 >= 0.10) or (
            ret_10d is not None and ret_10d >= 0.08
        )
        near_h = (dist20 is not None and 0 <= dist20 <= 0.03) or (
            dist60 is not None and 0 <= dist60 <= 0.03
        )
        # 已明显突破并远离高点不算「即将」
        already_extended = dist20 is not None and dist20 < -0.02

        near = bool(strength and near_h and not already_extended)
        out["near_breakout"] = near

        if near:
            out["breakout_label"] = "近端强势·临近突破"
            out["breakout_bonus"] = 0.12
        elif strength and not near_h:
            out["breakout_label"] = "近端已有涨幅·未近高点"
            out["breakout_bonus"] = 0.0
        elif near_h and not strength:
            out["breakout_label"] = "贴近高点·涨幅不足"
            out["breakout_bonus"] = 0.0
        else:
            out["breakout_label"] = ""
            out["breakout_bonus"] = 0.0
    except Exception:
        pass
    return out


def pass_mtf_filter(mtf: dict[str, Any], daily_premium: float, mode: str) -> bool:
    """多周期硬过滤。mode: off / soft / strict / resonate。"""
    mode = (mode or "off").strip().lower()
    if mode in ("", "off", "none", "0"):
        return True

    w_up = mtf.get("w_above_ma20")
    m_pos = mtf.get("m_pos_12")
    m_stretch = mtf.get("m_stretch")

    if mode == "soft":
        # 周线在MA20下且日线也明显偏弱
        if w_up is False and daily_premium < -0.05:
            return False
        # 月线近一年高位 + 日线已破位
        if m_pos is not None and m_pos >= 0.85 and daily_premium < -0.05:
            return False
        return True

    if mode == "strict":
        # 必须周线在MA20上（数据不足则不通过，避免误放行）
        if w_up is not True:
            return False
        if m_pos is not None and m_pos >= 0.85 and daily_premium < -0.03:
            return False
        if m_stretch is not None and m_stretch >= 1.5 and daily_premium < -0.05:
            return False
        return True

    if mode in ("resonate", "resonance", "共振"):
        # 多周期共振：
        # 1) 周线收盘在周MA20上
        # 2) 日线MA20溢价 >= -3%（未深度跌破）
        # 3) 月线未处于「近一年极高位 + 日线已弱」
        if w_up is not True:
            return False
        if daily_premium < -0.03:
            return False
        if m_pos is not None and m_pos >= 0.85 and daily_premium < 0:
            return False
        if m_stretch is not None and m_stretch >= 1.5 and daily_premium < 0:
            return False
        return True

    # 未知模式当作 off
    return True


def _bump(stats: dict[str, Any] | None, key: str) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + 1


def apply_launch_filter(
    df: pd.DataFrame,
    code: str,
    mtf_filter: str = MTF_FILTER_DEFAULT,
    breakout_only: bool = False,
    limitup_setup: bool = False,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """analyze() 只调用一次；stats 用于累计各级过滤的通过数（原 run_shard 会再调一遍 analyze）。"""
    if not FULL_CHIP_AVAILABLE or df.empty or len(df) < MIN_BARS:
        return {}
    try:
        df = calc_launch_factors(df)
        result = analyze(code, "", df)

        if not result.get("is_red_heavy_chip"):
            return {}
        _bump(stats, "red")

        profit = float(result.get("profit_pct") or 0)
        if not (PROFIT_MIN <= profit <= PROFIT_MAX):
            return {}
        _bump(stats, "profit")

        latest = df.iloc[-1]
        premium = float(latest.get("ma20_premium") or 0)
        if not np.isfinite(premium) or premium > MA20_MAX_PREMIUM:
            return {}
        _bump(stats, "ma20")

        ma_signal = str(result.get("ma_signal") or "")
        ma_bonus = 0.0
        if "多头" in ma_signal or "金叉" in ma_signal:
            ma_bonus = 1.0
        elif "趋势健康" in ma_signal:
            ma_bonus = 0.6

        name = str(result.get("name") or "").strip()
        if not name or name.lower() == "nan":
            name = str(get_name_map().get(str(code).zfill(6), "") or "")

        wide_state = result.get("wide_state")
        if wide_state is None or (isinstance(wide_state, float) and np.isnan(wide_state)):
            wide_state = ""
        else:
            wide_state = str(wide_state)

        # 多周期：周线/月线结构
        mtf = calc_multi_timeframe(df)
        if not pass_mtf_filter(mtf, premium, mtf_filter):
            return {}

        # 年线/季线二次标注（不硬剔）
        long_tf = calc_long_timeframe(df)

        # 近端强势 + 临近突破（默认只标注；--breakout-only 时硬过滤）
        bo = calc_breakout_setup(df)
        if breakout_only and not bo.get("near_breakout"):
            return {}

        # 冲板结构模块（默认只标注；--limitup-setup 时硬过滤）
        lu = calc_limitup_setup(df)
        if limitup_setup and not lu.get("near_limitup_setup"):
            return {}

        # 【核心】EMA120/200 聚类买卖信号（默认只标注加减分）
        ema_c = calc_ema_cluster(df)
        # 【补全】RSI/MACD/布林（TA-Lib 或 pandas）
        ta = calc_ta_supplement(df)

        return {
            "code": str(code).zfill(6),
            "name": str(name or ""),
            "date": str(
                result.get(
                    "date",
                    latest["date"].date() if hasattr(latest["date"], "date") else latest["date"],
                )
            ),
            "close": round(float(latest["close"]), 2),
            # 与 short_term_note 支撑候选一致：现价下方 8% / 5% 参考位
            "support_092": round(float(latest["close"]) * 0.92, 2),
            "support_095": round(float(latest["close"]) * 0.95, 2),
            "HighOpen": round(float(latest["HighOpen"]), 4),
            "CloseLow": round(float(latest["CloseLow"]), 4),
            "VwapClose": round(float(latest["VwapClose"]), 4),
            "close_pos": round(float(latest["close_pos"]), 4),
            "Turnover": round(float(latest.get("Turnover", 0)), 4),
            "profit_pct": round(profit, 2),
            "ma20_premium": round(premium, 4),
            "wide_score": float(result.get("wide_score") or 0),
            "wide_state": wide_state,
            "is_wide_zone": bool(result.get("is_wide_zone")),
            "ma_signal": str(ma_signal or ""),
            "ma_bonus": ma_bonus,
            "year_first_red": float(latest.get("year_first_red", 0)),
            "ma250": round(float(latest["ma250"]), 2) if pd.notna(latest.get("ma250")) else None,
            "is_red_heavy_chip": True,
            # ----- 多周期 -----
            "w_ma20": mtf.get("w_ma20"),
            "w_ma20_premium": mtf.get("w_ma20_premium"),
            "w_above_ma20": mtf.get("w_above_ma20"),
            "m_pos_12": mtf.get("m_pos_12"),
            "m_stretch": mtf.get("m_stretch"),
            "mtf_label": mtf.get("mtf_label") or "",
            "mtf_bonus": float(mtf.get("mtf_bonus") or 0),
            "mtf_risk": mtf.get("mtf_risk") or "",
            # ----- 年/季线 -----
            "q_pos_8": long_tf.get("q_pos_8"),
            "q_above_ma4": long_tf.get("q_above_ma4"),
            "y_pos_5": long_tf.get("y_pos_5"),
            "y_above_ma3": long_tf.get("y_above_ma3"),
            "long_label": long_tf.get("long_label") or "",
            "long_bonus": float(long_tf.get("long_bonus") or 0),
            # ----- 突破结构 -----
            "ret_from_low20": bo.get("ret_from_low20"),
            "ret_10d": bo.get("ret_10d"),
            "dist_to_high20": bo.get("dist_to_high20"),
            "dist_to_high60": bo.get("dist_to_high60"),
            "near_breakout": bool(bo.get("near_breakout")),
            "breakout_label": bo.get("breakout_label") or "",
            "breakout_bonus": float(bo.get("breakout_bonus") or 0),
            # ----- 冲板结构 -----
            "day_ret": lu.get("day_ret"),
            "ret_2d": lu.get("ret_2d"),
            "turn_ratio_5": lu.get("turn_ratio_5"),
            "near_limitup_setup": bool(lu.get("near_limitup_setup")),
            "limitup_label": lu.get("limitup_label") or "",
            "limitup_bonus": float(lu.get("limitup_bonus") or 0),
            "limitup_score": float(lu.get("limitup_score") or 0),
            # ----- MA/EMA 10·20·30·60·120·200 聚类 -----
            "ema10": ema_c.get("ema10"),
            "ema20": ema_c.get("ema20"),
            "ema30": ema_c.get("ema30"),
            "ema60": ema_c.get("ema60"),
            "ema120": ema_c.get("ema120"),
            "ema200": ema_c.get("ema200"),
            "ema20_premium": ema_c.get("ema20_premium"),
            "ema60_premium": ema_c.get("ema60_premium"),
            "ema120_premium": ema_c.get("ema120_premium"),
            "ema200_premium": ema_c.get("ema200_premium"),
            "ema_stack": ema_c.get("ema_stack") or "",
            "above_ema_n": int(ema_c.get("above_ema_n") or 0),
            "cluster_buy_n": int(ema_c.get("cluster_buy_n") or 0),
            "cluster_sell_n": int(ema_c.get("cluster_sell_n") or 0),
            "cluster_signal": ema_c.get("cluster_signal") or "中性",
            "cluster_label": ema_c.get("cluster_label") or "",
            "cluster_bonus": float(ema_c.get("cluster_bonus") or 0),
            # ----- TA 补全 -----
            "rsi14": ta.get("rsi14"),
            "macd_hist": ta.get("macd_hist"),
            "bb_pos": ta.get("bb_pos"),
            "ta_label": ta.get("ta_label") or "",
            "ta_bonus": float(ta.get("ta_bonus") or 0),
            "ta_source": ta.get("ta_source") or "none",
        }
    except Exception as e:
        _bump(stats, "error")
        if stats is not None:
            stats["last_error"] = f"{code}: {type(e).__name__}: {e}"
        return {}


def notify_serverchan(title: str, content: str, retries: int = 2) -> dict[str, Any]:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "missing_SENDKEY"}
    last: dict[str, Any] = {}
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"https://sctapi.ftqq.com/{key}.send",
                data={"title": title[:32], "desp": content},  # Server酱标题上限 32 字
                timeout=20,
            )
            result = resp.json()
            if resp.ok and result.get("code") == 0:
                return {"status": "sent"}
            last = {"status": "failed", "detail": result}
        except Exception as e:
            last = {"status": "failed", "error": str(e)[:200]}
        time.sleep(2)
    return last


def compute_rsrs(index_df: pd.DataFrame, n: int = RSRS_N, m: int = RSRS_M) -> dict[str, Any]:
    df = index_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < n + 30:
        return {"ok": False, "reason": "index_data_too_short", "rsrs_z": None, "bullish": None}

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    scores = np.full(len(df), np.nan)

    for i in range(n - 1, len(df)):
        y = highs[i - n + 1 : i + 1]
        x = lows[i - n + 1 : i + 1]
        if np.std(x) < 1e-8:
            continue
        x_mean, y_mean = x.mean(), y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        slope = cov / (var + 1e-12)
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = max(0.0, min(1.0, 1.0 - ss_res / (ss_tot + 1e-12)))
        scores[i] = slope * r2

    z = np.full(len(df), np.nan)
    lookback = min(m, len(df) - n)
    for i in range(n - 1, len(df)):
        start = max(0, i - lookback + 1)
        window = scores[start : i + 1]
        window = window[np.isfinite(window)]
        if len(window) < 20:
            continue
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-12:
            continue
        z[i] = (scores[i] - mu) / sigma

    last_z = z[-1]
    if not np.isfinite(last_z):
        return {"ok": False, "reason": "rsrs_nan", "rsrs_z": None, "bullish": None}

    return {
        "ok": True,
        "rsrs_z": round(float(last_z), 3),
        "bullish": bool(last_z >= RSRS_BUY_THRESHOLD),
        "threshold": RSRS_BUY_THRESHOLD,
    }


def get_market_timing() -> dict[str, Any]:
    details = []
    crash_any = False
    rsrs_bull_cnt = 0
    rsrs_valid_cnt = 0
    ret_list = []

    for name, symbol in INDEX_LIST:
        item = {
            "name": name,
            "symbol": symbol,
            "close": None,
            "support_092": None,
            "support_095": None,
            "rsrs_z": None,
            "ret_1d": None,
            "bullish": None,
        }
        try:
            raw = ak.stock_zh_index_daily(symbol=symbol)
            if raw is None or raw.empty:
                details.append(item)
                continue
            df = raw.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(800).reset_index(drop=True)

            if len(df) >= 1:
                idx_close = float(df["close"].iloc[-1])
                item["close"] = round(idx_close, 2)
                item["support_092"] = round(idx_close * 0.92, 2)
                item["support_095"] = round(idx_close * 0.95, 2)

            if len(df) >= 2:
                ret = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
                item["ret_1d"] = round(ret, 4)
                ret_list.append(ret)
                if ret <= INDEX_CRASH_THRESHOLD:
                    crash_any = True

            rsrs = compute_rsrs(df)
            if rsrs.get("ok"):
                item["rsrs_z"] = rsrs["rsrs_z"]
                item["bullish"] = rsrs["bullish"]
                rsrs_valid_cnt += 1
                if rsrs["bullish"]:
                    rsrs_bull_cnt += 1
        except Exception as e:
            print(f"指数 {name} 获取/计算失败: {e}")
        details.append(item)

    rsrs_ok = (rsrs_valid_cnt > 0) and (rsrs_bull_cnt * 2 >= rsrs_valid_cnt)
    allow = (not crash_any) and (rsrs_ok if rsrs_valid_cnt > 0 else True)

    notes = []
    if crash_any:
        notes.append("存在指数昨日跌超3%")
    if rsrs_valid_cnt > 0 and not rsrs_ok:
        notes.append(f"RSRS偏多{rsrs_bull_cnt}/{rsrs_valid_cnt}")
    if rsrs_valid_cnt == 0:
        notes.append("RSRS数据缺失（默认放行，请人工判断大盘）")
    if not notes:
        notes.append("多指数择时偏多")

    avg_ret = round(float(np.mean(ret_list)), 4) if ret_list else None

    return {
        "ok": rsrs_valid_cnt > 0,
        "details": details,
        "rsrs_bull_cnt": rsrs_bull_cnt,
        "rsrs_valid_cnt": rsrs_valid_cnt,
        "index_ret_1d": avg_ret,
        "index_crash": crash_any,
        "allow_new_position": allow,
        "reason": "；".join(notes),
    }


def run_shard(
    shard_id: int,
    mtf_filter: str = MTF_FILTER_DEFAULT,
    universe: list[str] | None = None,
    breakout_only: bool = False,
    limitup_setup: bool = False,
    exclude_st: bool = False,
) -> None:
    print(f"\n{'='*60}")
    print(
        f"两路分打 - 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)} "
        f"| mtf_filter={mtf_filter} | breakout_only={breakout_only} "
        f"| limitup_setup={limitup_setup} | exclude_st={exclude_st}"
    )
    if universe is not None:
        print(f"universe 限定: {len(universe)} 只")
    print(f"{'='*60}")

    if not FULL_CHIP_AVAILABLE:
        print("错误：缺少 final_chip_research.py，红筹/获利盘硬过滤无法执行，终止。")
        sys.exit(1)

    out_file = CACHE_DIR / f"live_shard_{shard_id}.csv"
    if out_file.exists():
        out_file.unlink()  # 先清旧结果：本次无命中/中途失败时，merge 不会读到过期数据

    all_codes = list(universe) if universe is not None else get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表（或 universe 为空）")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")
    name_map = get_name_map() if exclude_st else {}

    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "data": 0, "red": 0, "profit": 0, "ma20": 0,
        "error": 0, "stale": 0, "st": 0,
    }

    for i, code in enumerate(codes):
        if is_beijing_stock(code):
            continue
        if exclude_st and is_st_name(name_map.get(code, "")):
            stats["st"] += 1
            continue

        try:
            df = get_stock_data(code)
            if df.empty or len(df) < MIN_BARS:
                continue
            if _is_stale(df):
                stats["stale"] += 1
                continue
            stats["data"] += 1
            row = apply_launch_filter(
                df,
                code,
                mtf_filter=mtf_filter,
                breakout_only=breakout_only,
                limitup_setup=limitup_setup,
                stats=stats,
            )
            if row:
                records.append(row)
        except Exception as e:  # 单只股票出错不应拖垮整个分片
            stats["error"] += 1
            stats["last_error"] = f"{code}: {type(e).__name__}: {e}"

        if (i + 1) % 30 == 0:
            print(
                f"  已处理 {i+1}/{len(codes)} | 有效{stats['data']} "
                f"红筹{stats['red']} 获利{stats['profit']} MA20内{stats['ma20']} 命中{len(records)}"
            )

    print(
        f"分片 {shard_id} 统计: 有效={stats['data']} 红筹={stats['red']} "
        f"获利={stats['profit']} MA20内={stats['ma20']} 最终命中={len(records)} "
        f"| 停牌/过期={stats['stale']} ST跳过={stats['st']} 异常={stats['error']}"
    )
    if stats.get("last_error"):
        print(f"  最近一次异常: {stats['last_error']}")

    if records:
        df_out = pd.DataFrame(records)
        df_out["code"] = df_out["code"].astype(str).str.zfill(6)
        for col in (
            "name",
            "date",
            "wide_state",
            "ma_signal",
            "mtf_label",
            "mtf_risk",
            "long_label",
            "breakout_label",
            "limitup_label",
            "ema_stack",
            "cluster_signal",
            "cluster_label",
            "ta_label",
            "ta_source",
        ):
            if col in df_out.columns:
                df_out[col] = df_out[col].map(
                    lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
                )
        df_out = fill_names(df_out)
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id} 完成，命中 {len(records)} 只 → {out_file}")
    else:
        print(f"分片 {shard_id} 无命中股票")


def _safe_z(s: pd.Series) -> pd.Series:
    """z-score；单行/常数列/NaN 时回落为 0，避免整列变 NaN 导致排序失效。"""
    s = pd.to_numeric(s, errors="coerce")
    return ((s - s.mean()) / (s.std() + 1e-8)).fillna(0.0)


def _to_bool(v: Any) -> bool:
    """CSV 读回的 bool 可能是 True/'True'/1/NaN，统一解析。"""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip().lower() in ("1", "true", "yes")


def _bool_series(s: pd.Series) -> pd.Series:
    return s.map(_to_bool).astype(bool)


def _penalize(score: pd.Series, mask: pd.Series, factor: float) -> pd.Series:
    """
    对 mask 内的得分做“惩罚”。z-score 得分有正有负，直接 *= 0.5 会让负分变得更靠前（反向）。
    这里按幅度惩罚：score - |score| * (1-factor)，正分缩小、负分更负。
    """
    return score - score.abs() * (1.0 - factor) * mask.astype(float)


def assign_strength_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    溢价 + HO + 换手 分层（研究观察，不硬剔除）。
    Turnover 口径为小数（18.65% → 0.1865）。
    """
    if df.empty:
        return df
    out = df.copy()
    prem = pd.to_numeric(out.get("ma20_premium", 0), errors="coerce").fillna(0.0)
    ho = pd.to_numeric(out.get("HighOpen", 0), errors="coerce").fillna(0.0)
    turn = pd.to_numeric(out.get("Turnover", 0), errors="coerce").fillna(0.0)

    # 若误存成百分数（如 18.65），折成小数
    turn = np.where(turn > 1.5, turn / 100.0, turn)
    out["Turnover"] = turn

    prem_ok = (prem >= -0.02) & (prem <= 0.08)
    ho_ok = (ho >= 0.03) & (ho <= 0.08)
    turn_ok = (turn >= 0.03) & (turn <= 0.12)

    prem_hot = prem > 0.12
    ho_hot = ho >= 0.10
    turn_hot = turn >= 0.15
    prem_weak = prem < -0.03
    ho_weak = ho < 0.02

    hits = prem_ok.astype(int) + ho_ok.astype(int) + turn_ok.astype(int)
    hot = prem_hot | ho_hot | turn_hot
    weak = prem_weak & ho_weak

    # 按含义分层：
    #   L1观察强势：三项全中；L2可观察：两项中
    #   L3偏热：任一项过热且命中<=1；L3一般：只中1项、不热不弱
    #   L4偏弱：明显偏弱，或三项都不沾边
    layer = np.select(
        [
            weak,
            hot & (hits <= 1),
            hits >= 3,
            hits == 2,
            hits == 1,
        ],
        ["L4偏弱", "L3偏热", "L1观察强势", "L2可观察", "L3一般"],
        default="L4偏弱",
    )
    out["layer"] = layer
    out["layer_hits"] = hits
    out["prem_ok"] = prem_ok
    out["ho_ok"] = ho_ok
    out["turn_ok"] = turn_ok
    return out


def run_tsfresh_research(codes: list[str], tag: str = "live") -> None:
    """
    【研究可选】对少量代码抽 tsfresh Minimal 特征，写入 CSV，不参与主排序。
    需: pip install tsfresh
    """
    if not TSFRESH_AVAILABLE:
        print("tsfresh 未安装，跳过 --tsfresh-research（pip install tsfresh）")
        return
    if not codes:
        print("tsfresh：无代码可研究")
        return
    rows = []
    for code in codes[:50]:  # 上限，避免过慢
        try:
            df = get_stock_data(code)
            if df is None or len(df) < 40:
                continue
            d = df.sort_values("date").tail(120).copy()
            d["date"] = pd.to_datetime(d["date"])
            d["code"] = str(code).zfill(6)
            d["time"] = range(len(d))
            rows.append(d[["code", "time", "close"]])
        except Exception:
            continue
    if not rows:
        print("tsfresh：无有效序列")
        return
    long_df = pd.concat(rows, ignore_index=True)
    print(f"tsfresh 研究：{long_df['code'].nunique()} 只 × 近120日 close（Minimal）")
    try:
        feat = extract_features(
            long_df,
            column_id="code",
            column_sort="time",
            column_value="close",
            default_fc_parameters=MinimalFCParameters(),
            n_jobs=0,
            disable_progressbar=True,
        )
        tsfresh_impute(feat)
        today = datetime.now().strftime("%Y%m%d")
        path = f"tsfresh_minimal_{tag}_{today}.csv"
        feat.to_csv(path, encoding="utf-8-sig")
        print(f"tsfresh 特征已写 → {path}（仅研究，未并入选股分数）")
    except Exception as e:
        print(f"tsfresh 提取失败: {e}")


def merge_and_select(
    top_n: int = 15,
    do_notify: bool = False,
    mtf_filter: str = MTF_FILTER_DEFAULT,
    breakout_only: bool = False,
    limitup_setup: bool = False,
    tsfresh_research: bool = False,
) -> None:
    files = sorted(CACHE_DIR.glob("live_shard_*.csv"))
    if not files:
        print("没有找到任何分片结果，请先运行分片")
        return
    _today_d = datetime.now().date()
    _old = [f.name for f in files if datetime.fromtimestamp(f.stat().st_mtime).date() != _today_d]
    if _old:
        print(f"提示：以下分片结果不是今天生成的（可能是旧数据）: {_old}")

    dfs = [pd.read_csv(f, dtype={"code": str}) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = fill_names(df)
    if "date" in df.columns:  # 同一只股票若出现在多个分片文件里，保留最新交易日那条
        df = df.sort_values("date").drop_duplicates("code", keep="last").reset_index(drop=True)
    else:
        df = df.drop_duplicates("code", keep="last").reset_index(drop=True)
    for _c in ("wide_score", "VwapClose", "HighOpen", "CloseLow", "Turnover", "profit_pct", "ma20_premium"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    # CSV 里空字符串会被读成 NaN，打印/推送时会出现 "nan"，这里统一还原为空串
    for _c in (
        "name", "wide_state", "ma_signal", "mtf_label", "mtf_risk", "long_label",
        "breakout_label", "limitup_label", "ema_stack", "cluster_signal",
        "cluster_label", "ta_label",
    ):
        if _c in df.columns:
            df[_c] = df[_c].fillna("").astype(str).replace({"nan": "", "None": ""})
    print(f"合并后共 {len(df)} 只命中股票 | mtf_filter={mtf_filter}")

    # 合并阶段可再按多周期过滤（分片若用 off 扫全量，这里可 strict 收紧）
    mode = (mtf_filter or "off").strip().lower()
    if mode not in ("", "off", "none", "0") and not df.empty:
        before = len(df)
        keep = []
        for _, row in df.iterrows():
            def _num(col: str) -> float | None:
                v = pd.to_numeric(row.get(col), errors="coerce") if col in df.columns else np.nan
                return float(v) if pd.notna(v) else None

            mtf = {
                "w_above_ma20": (
                    _to_bool(row.get("w_above_ma20"))
                    if "w_above_ma20" in df.columns and pd.notna(row.get("w_above_ma20"))
                    else None
                ),
                "m_pos_12": _num("m_pos_12"),
                "m_stretch": _num("m_stretch"),
            }
            prem = _num("ma20_premium")
            keep.append(pass_mtf_filter(mtf, prem if prem is not None else 0.0, mode))
        df = df.loc[keep].reset_index(drop=True)
        print(f"多周期过滤 {mode}: {before} → {len(df)}")

    if breakout_only and not df.empty and "near_breakout" in df.columns:
        before = len(df)
        flag = _bool_series(df["near_breakout"])
        df = df.loc[flag].reset_index(drop=True)
        print(f"突破过滤 breakout_only: {before} → {len(df)}")

    if limitup_setup and not df.empty and "near_limitup_setup" in df.columns:
        before = len(df)
        flag = _bool_series(df["near_limitup_setup"])
        df = df.loc[flag].reset_index(drop=True)
        print(f"冲板结构过滤 limitup_setup: {before} → {len(df)}")

    timing = get_market_timing()
    print(
        f"择时: allow={timing.get('allow_new_position')} | "
        f"RSRS偏多{timing.get('rsrs_bull_cnt')}/{timing.get('rsrs_valid_cnt')} | "
        f"均涨跌={timing.get('index_ret_1d')} | {timing.get('reason')}"
    )
    for d in timing.get("details") or []:
        print(
            f"  {d['name']}: 收盘={d.get('close')} "
            f"支撑0.92={d.get('support_092')} / 0.95={d.get('support_095')} "
            f"| RSRS={d.get('rsrs_z')} 昨日={d.get('ret_1d')} "
            f"{'偏多' if d.get('bullish') else '偏空' if d.get('bullish') is not None else 'N/A'}"
        )

    if df.empty:
        print("无股票通过过滤")
        if do_notify:
            title = f"选股 {datetime.now().strftime('%Y%m%d')}｜无标的"
            content = "今日无股票通过过滤。\n仅为研究输出，不构成投资建议。"
            print(notify_serverchan(title, content))
        return

    df["wide_z"] = _safe_z(df["wide_score"]) if "wide_score" in df.columns else 0.0
    df["VwapClose_z"] = _safe_z(df["VwapClose"]) if "VwapClose" in df.columns else 0.0
    df["HighOpen_z"] = _safe_z(df["HighOpen"]) if "HighOpen" in df.columns else 0.0
    df["CloseLow_z"] = _safe_z(df["CloseLow"]) if "CloseLow" in df.columns else 0.0
    df["Turnover_z"] = (
        _safe_z(df["Turnover"].clip(upper=TURN_OVER_CAP)) if "Turnover" in df.columns else 0.0
    )
    df["year_first_red"] = (
        pd.to_numeric(df["year_first_red"], errors="coerce").fillna(0.0)
        if "year_first_red" in df.columns
        else 0.0
    )
    df["profit_z"] = _safe_z(df["profit_pct"]) if "profit_pct" in df.columns else 0.0
    df["inv_wide_z"] = -df["wide_z"]

    df["score_wide"] = (
        WEIGHTS_WIDE["wide_score"] * df["wide_z"]
        + WEIGHTS_WIDE["year_first_red"] * df["year_first_red"]
        + WEIGHTS_WIDE["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_WIDE["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_WIDE["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_WIDE["Turnover"] * df["Turnover_z"]
    )
    wide_zone = _bool_series(df["is_wide_zone"]) if "is_wide_zone" in df.columns else None
    if wide_zone is not None:
        df["score_wide"] = _penalize(df["score_wide"], ~wide_zone, 0.6)

    df["score_peak"] = (
        WEIGHTS_PEAK["profit_pct"] * df["profit_z"]
        + WEIGHTS_PEAK["inv_wide"] * df["inv_wide_z"]
        + WEIGHTS_PEAK["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_PEAK["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_PEAK["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_PEAK["Turnover"] * df["Turnover_z"]
    )
    df["score_peak"] = _penalize(df["score_peak"], df["profit_pct"] < PEAK_PROFIT_MIN, 0.5)
    if wide_zone is not None:
        df["score_peak"] = _penalize(df["score_peak"], wide_zone, 0.7)

    # 多周期 + 年季线 小幅加减分（不改变硬过滤，只微调排序）
    if "mtf_bonus" in df.columns:
        df["mtf_bonus"] = pd.to_numeric(df["mtf_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["mtf_bonus"]
        df["score_peak"] = df["score_peak"] + df["mtf_bonus"]
    if "long_bonus" in df.columns:
        df["long_bonus"] = pd.to_numeric(df["long_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["long_bonus"]
        df["score_peak"] = df["score_peak"] + df["long_bonus"]
    if "breakout_bonus" in df.columns:
        df["breakout_bonus"] = pd.to_numeric(df["breakout_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["breakout_bonus"]
        df["score_peak"] = df["score_peak"] + df["breakout_bonus"]
    if "limitup_bonus" in df.columns:
        df["limitup_bonus"] = pd.to_numeric(df["limitup_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["limitup_bonus"]
        df["score_peak"] = df["score_peak"] + df["limitup_bonus"]
    if "cluster_bonus" in df.columns:
        df["cluster_bonus"] = pd.to_numeric(df["cluster_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["cluster_bonus"]
        df["score_peak"] = df["score_peak"] + df["cluster_bonus"]
    if "ta_bonus" in df.columns:
        df["ta_bonus"] = pd.to_numeric(df["ta_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["ta_bonus"]
        df["score_peak"] = df["score_peak"] + df["ta_bonus"]

    df["final_score"] = df[["score_wide", "score_peak"]].max(axis=1)
    df["main_track"] = np.where(df["score_peak"] > df["score_wide"], "尖峰", "宽幅启动")
    df = assign_strength_layer(df)

    # 同路内：先按层级 L1>L2>L3>L4，再按原分数
    layer_rank = {"L1观察强势": 1, "L2可观察": 2, "L3一般": 3, "L3偏热": 3.5, "L4偏弱": 4}
    df["layer_rank"] = df["layer"].map(layer_rank).fillna(9)

    top_wide = (
        df[df["main_track"] == "宽幅启动"]
        .sort_values(["layer_rank", "score_wide"], ascending=[True, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    top_peak = (
        df[df["main_track"] == "尖峰"]
        .sort_values(["layer_rank", "score_peak"], ascending=[True, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    today = datetime.now().strftime("%Y%m%d")
    top_wide.to_csv(f"live_select_wide_{today}.csv", index=False, encoding="utf-8-sig")
    top_peak.to_csv(f"live_select_peak_{today}.csv", index=False, encoding="utf-8-sig")
    layer_out = pd.concat([top_wide, top_peak], ignore_index=True)
    if not layer_out.empty:
        layer_out.to_csv(f"live_select_layer_{today}.csv", index=False, encoding="utf-8-sig")
    if "near_limitup_setup" in df.columns:
        lu_flag = _bool_series(df["near_limitup_setup"])
        top_lu = (
            df.loc[lu_flag]
            .sort_values(
                [c for c in ("limitup_score", "final_score") if c in df.columns],
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )
        if not top_lu.empty:
            top_lu.to_csv(f"live_select_limitup_{today}.csv", index=False, encoding="utf-8-sig")
            print(f"冲板结构 {len(top_lu)} 只 → live_select_limitup_{today}.csv")
        else:
            print("冲板结构：0 只（今日无命中）")
    print(f"宽幅启动 {len(top_wide)} 只 → live_select_wide_{today}.csv")
    print(f"红筹尖峰 {len(top_peak)} 只 → live_select_peak_{today}.csv")
    if "layer" in df.columns and not df.empty:
        print("分层统计(全量命中):")
        print(df["layer"].value_counts().to_string())
        print("分层统计(输出Top):")
        if not layer_out.empty:
            print(layer_out["layer"].value_counts().to_string())

    allow = timing.get("allow_new_position", True)
    timing_flag = "可关注" if allow else "建议观望/不新开仓"

    def _print_block(title: str, sub: pd.DataFrame, score_col: str) -> None:
        print("\n" + "=" * 90)
        print(f"{title} | 择时: {timing_flag}")
        print("=" * 90)
        if sub.empty:
            print("（本路无标的）")
            return
        for i, row in sub.iterrows():
            yf = "年线首红" if row.get("year_first_red", 0) > 0.5 else ""
            mtf = str(row.get("mtf_label") or "")
            long_l = str(row.get("long_label") or "")
            bo = str(row.get("breakout_label") or "")
            bo_s = f" | {bo}" if bo else ""
            lu = str(row.get("limitup_label") or "")
            lu_s = f" | {lu}" if lu else ""
            cl = str(row.get("cluster_signal") or "")
            cl_s = f" | EMA:{cl}" if cl and cl != "中性" else ""
            ta_l = str(row.get("ta_label") or "")
            ta_s = f" | TA:{ta_l}" if ta_l else ""
            print(
                f"{i+1:02d}. {row['code']} {row.get('name', '')} "
                f"| {row.get('layer', '')} "
                f"| 分{row[score_col]:.3f} | 收盘{row['close']} "
                f"| 支撑0.92={row.get('support_092', '-')} "
                f"| 获利{row.get('profit_pct', '-')}% "
                f"| MA20溢价{row.get('ma20_premium', 0):.1%} "
                f"| HO{row.get('HighOpen', 0):.3f} "
                f"| 换手{float(row.get('Turnover', 0)):.1%} "
                f"| {mtf} | 年季:{long_l}{bo_s}{lu_s}{cl_s}{ta_s} {yf}"
            )
        print("=" * 90)

    _print_block("A路 宽幅启动", top_wide, "score_wide")
    _print_block("B路 红筹尖峰", top_peak, "score_peak")

    if tsfresh_research:
        codes = []
        for sub in (top_wide, top_peak):
            if sub is not None and not sub.empty and "code" in sub.columns:
                codes.extend(sub["code"].astype(str).str.zfill(6).tolist())
        run_tsfresh_research(sorted(set(codes)), tag="live")

    if do_notify:
        def _timing_lines() -> list[str]:
            lines = [
                f"**择时**：{timing_flag}",
                f"- 综合：{timing.get('reason')}",
                f"- RSRS偏多 {timing.get('rsrs_bull_cnt')}/{timing.get('rsrs_valid_cnt')} | 均涨跌 {timing.get('index_ret_1d')}",
            ]
            for d in timing.get("details") or []:
                flag = (
                    "偏多"
                    if d.get("bullish")
                    else ("偏空" if d.get("bullish") is not None else "N/A")
                )
                lines.append(
                    f"- {d['name']}: 收盘={d.get('close')} "
                    f"支撑0.92={d.get('support_092')} / 0.95={d.get('support_095')} "
                    f"| RSRS={d.get('rsrs_z')} 昨日={d.get('ret_1d')} {flag}"
                )
            return lines

        def _build_desp(track_name: str, sub: pd.DataFrame, score_col: str) -> str:
            lines = [
                f"# {track_name} {today}",
                "",
            ]
            lines.extend(_timing_lines())
            lines.extend([
                "",
                f"共 {len(sub)} 只",
                "",
                "> 说明：获利盘/筹码为自研口径，可能与东财等软件不一致，请以实盘软件复核。",
                "",
            ])
            if not allow:
                lines.append("> 当前择时不建议新开仓，名单仅供观察。")
                lines.append("")
            if sub.empty:
                lines.append("本路无标的。")
            else:
                for i, row in sub.iterrows():
                    yf = "【年线首红】" if row.get("year_first_red", 0) > 0.5 else ""
                    nm = row.get("name") or ""
                    mtf = str(row.get("mtf_label") or "")
                    risk = str(row.get("mtf_risk") or "")
                    risk_s = f" 风险:{risk}" if risk and risk not in ("nan", "None") else ""
                    long_l = str(row.get("long_label") or "")
                    bo = str(row.get("breakout_label") or "")
                    bo_s = f" | {bo}" if bo else ""
                    lu = str(row.get("limitup_label") or "")
                    lu_s = f" | {lu}" if lu else ""
                    cl = str(row.get("cluster_signal") or "")
                    cl_detail = str(row.get("cluster_label") or "")
                    cl_s = f" | EMA:{cl}" + (f"({cl_detail})" if cl_detail else "") if cl and cl != "中性" else ""
                    ta_l = str(row.get("ta_label") or "")
                    ta_s = f" | TA:{ta_l}" if ta_l else ""
                    lines.append(
                        f"{i+1:02d}. {row['code']} {nm} "
                        f"| {row.get('layer', '')} "
                        f"| 分{row[score_col]:.3f} | 收盘{row['close']} "
                        f"| 支撑0.92={row.get('support_092', '-')} / 0.95={row.get('support_095', '-')} "
                        f"| 获利{row.get('profit_pct', '-')}% "
                        f"| MA20溢价{row.get('ma20_premium', 0):.1%} "
                        f"| HO{row.get('HighOpen', 0):.3f} "
                        f"| 换手{float(row.get('Turnover', 0)):.1%} "
                        f"| 周期:{mtf}{risk_s} | 年季:{long_l}{bo_s}{lu_s}{cl_s}{ta_s} {yf}"
                    )
            lines.append("\n仅为研究输出，不构成投资建议。")
            return "\n".join(lines)

        mtf_tag = {
            "resonate": "共振",
            "strict": "周线多",
            "soft": "软过滤",
            "off": "",
        }.get((mtf_filter or "off").lower(), mtf_filter or "")
        mtf_part = f"|{mtf_tag}" if mtf_tag else ""
        r1 = notify_serverchan(
            f"宽幅启动{today}|{'可做' if allow else '观望'}|{len(top_wide)}只{mtf_part}",
            _build_desp("宽幅启动", top_wide, "score_wide"),
        )
        print(f"Server酱 宽幅推送: {r1}")
        time.sleep(2)
        r2 = notify_serverchan(
            f"红筹尖峰{today}|{'可做' if allow else '观望'}|{len(top_peak)}只{mtf_part}",
            _build_desp("红筹尖峰", top_peak, "score_peak"),
        )
        print(f"Server酱 尖峰推送: {r2}")


def main():
    parser = argparse.ArgumentParser(description="两路分打+分开推送+多指数RSRS+名称补全")
    parser.add_argument("--shard", type=str, help="1-8 或 all")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument(
        "--mtf-filter",
        type=str,
        default=MTF_FILTER_DEFAULT,
        choices=["off", "soft", "strict", "resonate"],
        help="多周期过滤: off/soft/strict/resonate(共振:周线多+日线未深破+月线未极端高位破位)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help="只扫描该 CSV 内的股票（如 mtf_resonate_universe.csv）",
    )
    parser.add_argument(
        "--breakout-only",
        action="store_true",
        help="只保留「近端涨幅≥10%且接近20/60日高点」的股票（硬过滤）",
    )
    parser.add_argument(
        "--limitup-setup",
        action="store_true",
        help="冲板结构模块：只保留大阳收上沿/两日强势贴高/昨强今未崩等（研究用，非涨停预测）",
    )
    parser.add_argument(
        "--exclude-st",
        action="store_true",
        help="分片扫描时跳过 ST/*ST/退市整理股（按名称判断）",
    )
    parser.add_argument(
        "--tsfresh-research",
        action="store_true",
        help="研究：merge 后对 Top 结果抽取 tsfresh Minimal 特征（不参与打分，需 pip install tsfresh）",
    )
    args = parser.parse_args()

    if args.merge:
        merge_and_select(
            top_n=args.top,
            do_notify=args.notify,
            mtf_filter=args.mtf_filter,
            breakout_only=args.breakout_only,
            limitup_setup=args.limitup_setup,
            tsfresh_research=args.tsfresh_research,
        )
        return
    if not args.shard:
        parser.error("请指定 --shard 或 --merge")

    if args.shard != "all":
        try:
            shard_no = int(args.shard)
        except ValueError:
            parser.error("--shard 只能是 1-8 或 all")
        if shard_no not in SHARD_MAP:
            parser.error("--shard 只能是 1-8 或 all")

    universe_codes: list[str] | None = None
    if args.universe:
        universe_codes = load_universe(args.universe)
        print(f"已加载 universe: {args.universe} → {len(universe_codes)} 只")
        if not universe_codes:
            print("universe 为空，退出")
            return

    extra = ""
    if args.breakout_only:
        extra += " --breakout-only"
    if args.limitup_setup:
        extra += " --limitup-setup"

    if args.shard == "all":
        report_uncovered(universe_codes if universe_codes is not None else get_all_a_stocks())
        for i in range(1, 9):
            run_shard(
                i,
                mtf_filter=args.mtf_filter,
                universe=universe_codes,
                breakout_only=args.breakout_only,
                limitup_setup=args.limitup_setup,
                exclude_st=args.exclude_st,
            )
        print(
            "\n全部分片完成，请执行："
            f"python live_stock_selector.py --merge --top 15 --notify "
            f"--mtf-filter {args.mtf_filter}{extra}"
        )
    else:
        run_shard(
            int(args.shard),
            mtf_filter=args.mtf_filter,
            universe=universe_codes,
            breakout_only=args.breakout_only,
            limitup_setup=args.limitup_setup,
            exclude_st=args.exclude_st,
        )


if __name__ == "__main__":
    main()