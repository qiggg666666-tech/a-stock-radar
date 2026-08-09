#!/usr/bin/env python3
"""
多周期双坑底（双底 / W底）扫描器 v4.1 —— 矩阵全市场版
在 v4.0 生产加固版基础上做的矩阵化改造：
  1. 新增 --all 全市场模式（baostock 取全A列表，排除ST/退/北交所），SCAN_LIMIT 可截断
  2. get_bs_data 改为复用每个 worker 的 baostock 会话（不再每只票登录/登出），失败自动重连
  3. 结果输出到 OUTPUT_DIR（默认 output/），供 workflow artifact 上传
  4. 保留 v4 核心：baostock优先+akshare兜底 / 超时保护 / ZigZag / 动态突破评分 / RSI+MACD双背离
用法：
  python double_bottom_scanner_v4.py --all --workers 3 --push --top 30
  python double_bottom_scanner_v4.py --symbols 600519,000001 --periods W,M --min-score 60
"""

import argparse
import logging
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("double_bottom_v4")

try:
    import baostock as bs
    HAS_BS = True
except ImportError:
    HAS_BS = False

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ==================== 参数配置 ====================
ZIGZAG_DEVIATION_PCT = 0.05
MIN_TREND_DECLINE = 0.10
MAX_PRICE_DIFF_PCT = 0.03
MIN_REBOUND_PCT = 0.10
MIN_BARS_BETWEEN = 8
MAX_BARS_BETWEEN = 90
LOOKBACK_BARS = 150
BREAKOUT_BUFFER = 0.01
MIN_BREAKOUT_VOLUME_RATIO = 1.5
MAX_VOL2_RATIO = 1.2
MIN_HOLD_DAYS = 2
RSI_PERIOD = 14

WEIGHT_SYMMETRY = 25
WEIGHT_VOLUME = 15
WEIGHT_TREND = 15
WEIGHT_REBOUND = 15
WEIGHT_BREAKOUT = 15
WEIGHT_DIVERGENCE = 15

NUM_PROCESSES = 3
FETCH_TIMEOUT_SEC = 25
STAGGER_DELAY_RANGE = (0.5, 1.8)
MAX_RETRIES = 2

# ==================== 矩阵化: 全市场 + 每worker独立baostock会话 ====================
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT", "0"))
_BS_READY = False

def _bs_logout_quiet():
    try:
        if HAS_BS: bs.logout()
    except Exception: pass

def _worker_bs_init():
    """每个子进程启动时登录一次baostock并复用, 不再每只票登录/登出。"""
    global _BS_READY
    time.sleep(random.uniform(0, 2))
    if not HAS_BS:
        _BS_READY = False; return
    try: bs.logout()
    except Exception: pass
    try:
        lg = bs.login()
        _BS_READY = (lg.error_code == "0")
    except Exception:
        _BS_READY = False
    import atexit
    atexit.register(_bs_logout_quiet)

def get_all_a_share_symbols():
    """baostock全市场A股, 排除ST/退/北交所, 返回6位裸代码。"""
    if not HAS_BS: return []
    try:
        lg = bs.login()
        if lg.error_code != "0": return []
    except Exception:
        return []
    try:
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[(df["type"] == "1") & (df["status"] == "1")]
        df = df[~df["code_name"].astype(str).str.contains("ST|退", na=False)]
        out = []
        for c in df["code"].tolist():
            plain = str(c).split(".")[-1]
            if plain[:2] in ("60", "00", "30", "68"):
                out.append(plain)
        return sorted(set(out))
    except Exception as e:
        logger.warning(f"获取全市场列表失败: {e}")
        return []
    finally:
        _bs_logout_quiet()


class BreakoutQuality(Enum):
    STRONG = "强突破"
    WEAK = "弱突破"
    FORMING = "形成中"
    FAKE = "假突破"


@dataclass
class DoubleBottomResult:
    symbol: str
    period: str
    bottom1_date: datetime
    bottom1_price: float
    bottom1_idx: int
    bottom2_date: datetime
    bottom2_price: float
    bottom2_idx: int
    neckline: float
    neckline_zone: Tuple[float, float]
    bars_between: int
    price_diff_pct: float
    rebound_pct: float
    prior_decline_pct: float
    vol1: float
    vol2: float
    vol_ratio: float
    breakout_vol_ratio: float
    breakout: bool
    breakout_date: Optional[datetime]
    breakout_quality: BreakoutQuality
    breakout_candle_vol: float
    rsi_bottom1: float
    rsi_bottom2: float
    rsi_divergence: bool
    macd_divergence: bool
    symmetry_score: float
    volume_score: float
    trend_score: float
    rebound_score: float
    breakout_score: float
    divergence_score: float
    total_score: float
    current_close: float
    status: str
    target_price: float
    stop_loss_price: float
    distance_to_neckline_pct: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "bottom1_date": self.bottom1_date,
            "bottom1_price": round(self.bottom1_price, 3),
            "bottom2_date": self.bottom2_date,
            "bottom2_price": round(self.bottom2_price, 3),
            "neckline": round(self.neckline, 3),
            "neckline_zone_low": round(self.neckline_zone[0], 3),
            "neckline_zone_high": round(self.neckline_zone[1], 3),
            "bars_between": self.bars_between,
            "price_diff_pct": round(self.price_diff_pct, 2),
            "rebound_pct": round(self.rebound_pct, 2),
            "prior_decline_pct": round(self.prior_decline_pct, 2),
            "vol1": int(self.vol1),
            "vol2": int(self.vol2),
            "vol_ratio": round(self.vol_ratio, 2),
            "breakout_vol_ratio": round(self.breakout_vol_ratio, 2) if self.breakout else None,
            "breakout": self.breakout,
            "breakout_date": self.breakout_date,
            "breakout_quality": self.breakout_quality.value,
            "rsi_bottom1": round(self.rsi_bottom1, 2),
            "rsi_bottom2": round(self.rsi_bottom2, 2),
            "rsi_divergence": self.rsi_divergence,
            "macd_divergence": self.macd_divergence,
            "symmetry_score": round(self.symmetry_score, 1),
            "volume_score": round(self.volume_score, 1),
            "trend_score": round(self.trend_score, 1),
            "rebound_score": round(self.rebound_score, 1),
            "breakout_score": round(self.breakout_score, 1),
            "divergence_score": round(self.divergence_score, 1),
            "total_score": round(self.total_score, 1),
            "current_close": round(self.current_close, 3),
            "status": self.status,
            "target_price": round(self.target_price, 3),
            "stop_loss_price": round(self.stop_loss_price, 3),
            "distance_to_neckline_pct": round(self.distance_to_neckline_pct, 2),
        }


# ==================== 超时保护包装器 ====================

def fetch_with_timeout(fn, *args, timeout: int = FETCH_TIMEOUT_SEC, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(f"{fn.__name__} 超时({timeout}s)，参数={args}")
            return None
        except Exception as e:
            logger.warning(f"{fn.__name__} 异常: {e} | 参数={args}")
            return None


# ==================== 数据获取：baostock（复用worker会话） ====================

def to_bs_code(symbol: str) -> str:
    if symbol.startswith(("sh.", "sz.", "bj.")):
        return symbol
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj.{symbol}"
    return f"sh.{symbol}"


def get_bs_data(symbol: str, years: int = 12) -> Optional[pd.DataFrame]:
    """复用worker的baostock会话取前复权日线; 失败标记重连。"""
    global _BS_READY
    if not HAS_BS:
        return None
    bs_code = to_bs_code(symbol)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    if not _BS_READY:
        try:
            lg = bs.login()
            if lg.error_code != "0":
                return None
            _BS_READY = True
        except Exception:
            return None
    try:
        rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume",
            start_date=start, end_date=end, frequency="d", adjustflag="2")
        if rs.error_code != "0":
            _BS_READY = False
            return None
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.set_index("date").sort_index().dropna()[["open", "high", "low", "close", "volume"]]
    except Exception:
        _BS_READY = False
        return None


# ==================== 数据获取：akshare（兜底） ====================

def get_ak_data(symbol: str, period: str = "daily", years: int = 12) -> Optional[pd.DataFrame]:
    if not HAS_AK:
        return None
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ==================== 数据获取：yfinance（美股/港股） ====================

def get_yf_data(symbol: str, years: int = 12) -> Optional[pd.DataFrame]:
    if not HAS_YF:
        return None
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{years}y", auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ==================== 统一数据入口 ====================

def get_daily_data(symbol: str, is_a_share: bool, years: int = 12,
                    source: str = "auto") -> Optional[pd.DataFrame]:
    if not is_a_share:
        return fetch_with_timeout(get_yf_data, symbol, years)
    tried_sources = []
    if source in ("auto", "baostock") and HAS_BS:
        for attempt in range(MAX_RETRIES):
            df = fetch_with_timeout(get_bs_data, symbol, years)
            if df is not None and len(df) > 0:
                return df
            tried_sources.append("baostock")
            time.sleep(1.0)
        if source == "baostock":
            logger.warning(f"[{symbol}] baostock取数失败，已重试{MAX_RETRIES}次")
            return None
    if source in ("auto", "akshare") and HAS_AK:
        df = fetch_with_timeout(get_ak_data, symbol, "daily", years)
        if df is not None and len(df) > 0:
            return df
        tried_sources.append("akshare")
    logger.warning(f"[{symbol}] 所有数据源均取数失败（尝试过: {tried_sources or '无可用数据源'}）")
    return None


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(ohlc).dropna()


# ==================== 技术指标 ====================

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


# ==================== ZigZag 波段过滤 ====================

def zigzag_pivots(high: pd.Series, low: pd.Series, close: pd.Series,
                   deviation_pct: float = ZIGZAG_DEVIATION_PCT):
    n = len(close)
    if n < 5:
        return [], []
    pivots_idx = []
    pivots_type = []
    trend = 0
    last_extreme_idx = 0
    last_extreme_price = close.iloc[0]
    for i in range(1, n):
        current_price = close.iloc[i]
        if trend == 0:
            if current_price >= last_extreme_price * (1 + deviation_pct):
                trend = 1
                last_extreme_idx = i
                last_extreme_price = current_price
            elif current_price <= last_extreme_price * (1 - deviation_pct):
                trend = -1
                last_extreme_idx = i
                last_extreme_price = current_price
        elif trend == 1:
            if current_price > last_extreme_price:
                last_extreme_idx = i
                last_extreme_price = current_price
            elif current_price <= last_extreme_price * (1 - deviation_pct):
                pivots_idx.append(last_extreme_idx)
                pivots_type.append("H")
                trend = -1
                last_extreme_idx = i
                last_extreme_price = current_price
        elif trend == -1:
            if current_price < last_extreme_price:
                last_extreme_idx = i
                last_extreme_price = current_price
            elif current_price >= last_extreme_price * (1 + deviation_pct):
                pivots_idx.append(last_extreme_idx)
                pivots_type.append("L")
                trend = 1
                last_extreme_idx = i
                last_extreme_price = current_price
    if trend == -1 and (not pivots_idx or last_extreme_idx != pivots_idx[-1]):
        pivots_idx.append(last_extreme_idx)
        pivots_type.append("L")
    return pivots_idx, pivots_type


# ==================== 颈线计算 ====================

def find_neckline_zone(df: pd.DataFrame, idx1: int, idx2: int) -> Tuple[float, float, float]:
    middle_df = df.iloc[idx1:idx2 + 1]
    if len(middle_df) < 3:
        mx = middle_df["high"].max()
        return mx, mx, mx
    highs = middle_df["high"].values
    top_n = min(3, len(highs))
    significant_highs = np.partition(highs, -top_n)[-top_n:]
    neckline = np.mean(significant_highs)
    zone_low = np.min(significant_highs) * 0.998
    zone_high = np.max(significant_highs) * 1.002
    return neckline, zone_low, zone_high


# ==================== 突破质量分析 ====================

def analyze_breakout_quality(df: pd.DataFrame, neckline: float, breakout_idx: int, avg_volume: float):
    if breakout_idx is None or breakout_idx >= len(df):
        return BreakoutQuality.FORMING, 0.0
    breakout_vol = df.iloc[breakout_idx]["volume"]
    vol_ratio = breakout_vol / max(avg_volume, 1)
    hold_days = 0
    for i in range(breakout_idx + 1, min(breakout_idx + MIN_HOLD_DAYS + 1, len(df))):
        if df.iloc[i]["close"] >= neckline * (1 - BREAKOUT_BUFFER):
            hold_days += 1
        else:
            break
    if vol_ratio >= MIN_BREAKOUT_VOLUME_RATIO and hold_days >= MIN_HOLD_DAYS:
        return BreakoutQuality.STRONG, vol_ratio
    elif vol_ratio >= 1.0:
        return BreakoutQuality.WEAK, vol_ratio
    else:
        return BreakoutQuality.FAKE, vol_ratio


# ==================== 动态突破评分 ====================

def calculate_dynamic_breakout_score(breakout_quality: BreakoutQuality, current_close: float,
                                      neckline: float) -> float:
    if breakout_quality == BreakoutQuality.STRONG:
        return 100
    elif breakout_quality == BreakoutQuality.WEAK:
        return 60
    elif breakout_quality == BreakoutQuality.FAKE:
        return 20
    distance_pct = (neckline - current_close) / neckline * 100
    if distance_pct <= 1.0:
        return 85
    elif distance_pct <= 2.0:
        return 78
    elif distance_pct <= 3.0:
        return 70
    elif distance_pct <= 5.0:
        return 60
    elif distance_pct <= 10.0:
        return 45
    else:
        return 30


# ==================== 多维度评分 ====================

def calculate_scores(price_diff_pct, vol_ratio, prior_decline_pct, rebound_pct,
                      breakout_quality, breakout_vol_ratio, rsi_divergence, macd_divergence,
                      current_close, neckline):
    symmetry = max(0, 100 - (price_diff_pct / MAX_PRICE_DIFF_PCT) * 100)
    if vol_ratio <= 0.5:
        volume = 100
    elif vol_ratio >= MAX_VOL2_RATIO:
        volume = 0
    else:
        volume = 100 - ((vol_ratio - 0.5) / (MAX_VOL2_RATIO - 0.5)) * 100
    trend = min(100, 60 + (prior_decline_pct - MIN_TREND_DECLINE) / 0.20 * 40)
    trend = max(0, trend)
    rebound = min(100, 60 + (rebound_pct - MIN_REBOUND_PCT) / 0.15 * 40)
    rebound = max(0, rebound)
    breakout = calculate_dynamic_breakout_score(breakout_quality, current_close, neckline)
    divergence = 0
    if rsi_divergence:
        divergence += 50
    if macd_divergence:
        divergence += 50
    total = (symmetry * WEIGHT_SYMMETRY / 100 + volume * WEIGHT_VOLUME / 100 +
             trend * WEIGHT_TREND / 100 + rebound * WEIGHT_REBOUND / 100 +
             breakout * WEIGHT_BREAKOUT / 100 + divergence * WEIGHT_DIVERGENCE / 100)
    return symmetry, volume, trend, rebound, breakout, divergence, total


# ==================== 核心检测逻辑 v3 ====================

def detect_double_bottom_v3(df: pd.DataFrame, symbol: str = "", period: str = "",
                             price_tol: float = MAX_PRICE_DIFF_PCT, min_bars: int = MIN_BARS_BETWEEN,
                             max_bars: int = MAX_BARS_BETWEEN, min_decline: float = MIN_TREND_DECLINE,
                             min_rebound: float = MIN_REBOUND_PCT):
    if len(df) < 40:
        return []
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    lows = df["low"]
    highs = df["high"]
    closes = df["close"]
    volumes = df["volume"]
    rsi = df["rsi"]
    macd_hist = df["macd_hist"]
    pivot_indices, pivot_types = zigzag_pivots(highs, lows, closes, deviation_pct=ZIGZAG_DEVIATION_PCT)
    low_pivot_indices = [pivot_indices[i] for i in range(len(pivot_types)) if pivot_types[i] == "L"]
    if len(low_pivot_indices) < 2:
        return []
    results = []
    for i in range(len(low_pivot_indices) - 1):
        idx1 = low_pivot_indices[i]
        for j in range(i + 1, len(low_pivot_indices)):
            idx2 = low_pivot_indices[j]
            bars_between = idx2 - idx1
            if not (min_bars <= bars_between <= max_bars):
                continue
            price1 = lows.iloc[idx1]
            price2 = lows.iloc[idx2]
            price_diff_pct = abs(price1 - price2) / max(price1, 1e-9)
            if price_diff_pct > price_tol:
                continue
            pre_high = highs.iloc[max(0, idx1 - 30):idx1].max()
            prior_decline_pct = (pre_high - price1) / pre_high if pre_high > 0 else 0
            if prior_decline_pct < min_decline:
                continue
            middle_high = highs.iloc[idx1:idx2].max()
            rebound_pct = (middle_high - max(price1, price2)) / max(price1, price2)
            if rebound_pct < min_rebound:
                continue
            vol1 = volumes.iloc[idx1]
            vol2 = volumes.iloc[idx2]
            vol_ratio = vol2 / max(vol1, 1)
            if vol2 > vol1 * MAX_VOL2_RATIO:
                continue
            neckline, zone_low, zone_high = find_neckline_zone(df, idx1, idx2)
            after = closes.iloc[idx2 + 1:]
            breakout = False
            breakout_idx = None
            breakout_quality = BreakoutQuality.FORMING
            breakout_vol_ratio = 0.0
            avg_vol = volumes.iloc[max(0, idx2 - 20):idx2].mean()
            if len(after) > 0:
                for k, c in enumerate(after):
                    if c >= neckline * (1 + BREAKOUT_BUFFER):
                        breakout = True
                        breakout_idx = idx2 + 1 + k
                        breakout_quality, breakout_vol_ratio = analyze_breakout_quality(
                            df, neckline, breakout_idx, avg_vol)
                        break
            rsi1 = rsi.iloc[idx1] if not pd.isna(rsi.iloc[idx1]) else 50
            rsi2 = rsi.iloc[idx2] if not pd.isna(rsi.iloc[idx2]) else 50
            rsi_divergence = (price2 <= price1 * 1.01) and (rsi2 > rsi1 * 1.02)
            macd1 = macd_hist.iloc[idx1] if not pd.isna(macd_hist.iloc[idx1]) else 0
            macd2 = macd_hist.iloc[idx2] if not pd.isna(macd_hist.iloc[idx2]) else 0
            macd_divergence = (price2 <= price1 * 1.01) and (macd2 > macd1)
            current_close = closes.iloc[-1]
            distance_to_neckline = (neckline - current_close) / neckline * 100
            (symmetry_score, volume_score, trend_score, rebound_score,
             breakout_score, divergence_score, total_score) = calculate_scores(
                price_diff_pct, vol_ratio, prior_decline_pct, rebound_pct,
                breakout_quality, breakout_vol_ratio, rsi_divergence, macd_divergence,
                current_close, neckline
            )
            pattern_height = neckline - min(price1, price2)
            target_price = neckline + pattern_height if breakout else current_close * 1.05
            stop_loss_price = min(price1, price2) * 0.95
            if breakout:
                if breakout_quality == BreakoutQuality.STRONG:
                    status = "已突破(强)"
                elif breakout_quality == BreakoutQuality.WEAK:
                    status = "已突破(弱)"
                elif breakout_quality == BreakoutQuality.FAKE:
                    status = "假突破"
                else:
                    status = "已突破"
            else:
                status = "形成中"
            result = DoubleBottomResult(
                symbol=symbol, period=period,
                bottom1_date=df.index[idx1], bottom1_price=price1, bottom1_idx=idx1,
                bottom2_date=df.index[idx2], bottom2_price=price2, bottom2_idx=idx2,
                neckline=neckline, neckline_zone=(zone_low, zone_high),
                bars_between=bars_between, price_diff_pct=price_diff_pct * 100,
                rebound_pct=rebound_pct * 100, prior_decline_pct=prior_decline_pct * 100,
                vol1=vol1, vol2=vol2, vol_ratio=vol_ratio,
                breakout_vol_ratio=breakout_vol_ratio,
                breakout=breakout, breakout_date=df.index[breakout_idx] if breakout_idx is not None else None,
                breakout_quality=breakout_quality,
                breakout_candle_vol=df.iloc[breakout_idx]["volume"] if breakout_idx is not None else 0,
                rsi_bottom1=rsi1, rsi_bottom2=rsi2, rsi_divergence=rsi_divergence,
                macd_divergence=macd_divergence,
                symmetry_score=symmetry_score, volume_score=volume_score,
                trend_score=trend_score, rebound_score=rebound_score,
                breakout_score=breakout_score, divergence_score=divergence_score,
                total_score=total_score,
                current_close=current_close, status=status,
                target_price=target_price, stop_loss_price=stop_loss_price,
                distance_to_neckline_pct=distance_to_neckline
            )
            results.append(result)
    results = sorted(results, key=lambda x: x.total_score, reverse=True)[:5]
    return results


# ==================== 多周期扫描（单只票） ====================

def scan_symbol_v3(symbol: str, periods: List[str] = ("W", "M", "Q", "Y"),
                    is_a_share: bool = True, min_score: float = 50.0,
                    source: str = "auto"):
    daily = get_daily_data(symbol, is_a_share=is_a_share, years=12, source=source)
    if daily is None or len(daily) < 150:
        return {}
    period_map = {
        "W": ("W-FRI", "周线"),
        "M": ("ME", "月线"),
        "Q": ("QE", "季线"),
        "Y": ("YE", "年线"),
    }
    all_results = {}
    for code, (rule, name) in period_map.items():
        if code not in periods:
            continue
        try:
            df = resample_ohlcv(daily, rule)
        except Exception as e:
            logger.warning(f"[{symbol}] 重采样({rule})失败: {e}")
            continue
        if len(df) < 25:
            continue
        df = df.tail(LOOKBACK_BARS)
        try:
            patterns = detect_double_bottom_v3(df, symbol=symbol, period=name)
        except Exception as e:
            logger.warning(f"[{symbol}][{name}] 形态检测异常: {e}")
            continue
        patterns = [p for p in patterns if p.total_score >= min_score]
        if patterns:
            all_results[name] = patterns
    return all_results


# ==================== 多进程扫描（多只票） ====================

def _scan_worker(task: Tuple[str, List[str], bool, float, str]):
    symbol, periods, is_a, min_score, source = task
    time.sleep(random.uniform(*STAGGER_DELAY_RANGE))
    try:
        return symbol, scan_symbol_v3(symbol, periods=periods, is_a_share=is_a,
                                       min_score=min_score, source=source)
    except Exception as e:
        logger.warning(f"[{symbol}] 扫描worker异常: {e}")
        return symbol, {}


def scan_all(symbols: List[str], periods: List[str], min_score: float,
             source: str = "auto", workers: int = NUM_PROCESSES) -> Dict[str, Dict]:
    tasks = []
    for sym in symbols:
        is_a = sym.isdigit() and len(sym) == 6
        tasks.append((sym, periods, is_a, min_score, source))
    results_by_symbol: Dict[str, Dict] = {}
    if workers <= 1:
        iterator = tasks
        if HAS_TQDM:
            iterator = tqdm(tasks, desc="扫描中")
        for task in iterator:
            sym, res = _scan_worker(task)
            if res:
                results_by_symbol[sym] = res
        return results_by_symbol
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_bs_init) as executor:
        futures = {executor.submit(_scan_worker, task): task[0] for task in tasks}
        iterator = futures
        if HAS_TQDM:
            iterator = tqdm(list(futures), desc="扫描中")
            for fut in iterator:
                sym = futures[fut]
                try:
                    sym, res = fut.result()
                    if res:
                        results_by_symbol[sym] = res
                except Exception as e:
                    logger.warning(f"[{sym}] 进程执行异常: {e}")
        else:
            for fut in futures:
                sym = futures[fut]
                try:
                    sym, res = fut.result()
                    if res:
                        results_by_symbol[sym] = res
                except Exception as e:
                    logger.warning(f"[{sym}] 进程执行异常: {e}")
    return results_by_symbol


# ==================== Server酱 微信推送 ====================

def push_wechat(title: str, content: str) -> bool:
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        logger.info("未设置SENDKEY环境变量，跳过微信推送")
        return False
    if not HAS_REQUESTS:
        logger.warning("未安装requests库，无法推送微信")
        return False
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    try:
        resp = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        if resp.status_code == 200:
            logger.info("微信推送成功")
            return True
        logger.warning(f"微信推送失败: HTTP {resp.status_code}")
        return False
    except Exception as e:
        logger.warning(f"微信推送异常: {e}")
        return False


def build_push_content(all_rows: List[Dict], top: int = 10) -> str:
    if not all_rows:
        return "本次扫描未发现符合条件的双底形态。"
    df = pd.DataFrame(all_rows).sort_values("total_score", ascending=False).head(top)
    lines = [f"### 双坑底扫描 Top{min(top, len(df))}", ""]
    for _, r in df.iterrows():
        lines.append(
            f"**{r['symbol']}** [{r['period']}] 评分{r['total_score']:.1f} | {r['status']}  \n"
            f"颈线{r['neckline']:.2f} 现价{r['current_close']:.2f} "
            f"距颈线{r['distance_to_neckline_pct']:.2f}% 目标{r['target_price']:.2f}"
        )
        lines.append("")
    return "\n".join(lines)


# ==================== 输出格式化 ====================

def print_result(p: DoubleBottomResult):
    print(f"  ┌─【{p.period}】总分: {p.total_score:.1f}/100 | {p.status}")
    print(f"  │  底1: {p.bottom1_date.date()} @ {p.bottom1_price:.2f} (RSI:{p.rsi_bottom1:.1f})")
    print(f"  │  底2: {p.bottom2_date.date()} @ {p.bottom2_price:.2f} (RSI:{p.rsi_bottom2:.1f})")
    print(f"  │  价差: {p.price_diff_pct:.2f}% | 间隔: {p.bars_between}根K线")
    print(f"  │  颈线: {p.neckline:.2f} (区域: {p.neckline_zone[0]:.2f}~{p.neckline_zone[1]:.2f})")
    print(f"  │  前期跌幅: {p.prior_decline_pct:.1f}% | 中间反弹: {p.rebound_pct:.1f}%")
    print(f"  │  成交量: 底1={p.vol1:.0f} 底2={p.vol2:.0f} (比:{p.vol_ratio:.2f})")
    if p.breakout:
        print(f"  │  突破: {p.breakout_date.date() if p.breakout_date else 'N/A'} | "
              f"质量: {p.breakout_quality.value} | 放量: {p.breakout_vol_ratio:.2f}x")
    else:
        print(f"  │  突破: 尚未突破 | 距离颈线: {p.distance_to_neckline_pct:.2f}%")
    print(f"  │  背离: RSI={'✓' if p.rsi_divergence else '✗'} MACD={'✓' if p.macd_divergence else '✗'}")
    print(f"  │  评分: 对称{p.symmetry_score:.0f} 量{p.volume_score:.0f} 趋势{p.trend_score:.0f} "
          f"反弹{p.rebound_score:.0f} 突破{p.breakout_score:.0f} 背离{p.divergence_score:.0f}")
    print(f"  │  目标价: {p.target_price:.2f} | 止损价: {p.stop_loss_price:.2f} | 现价: {p.current_close:.2f}")
    print(f"  └─")


# ==================== 主程序 ====================

def load_symbols(args) -> List[str]:
    if getattr(args, "all_market", False):
        syms = get_all_a_share_symbols()
        logger.info(f"全市场模式: baostock取到 {len(syms)} 只")
        if SCAN_LIMIT and len(syms) > SCAN_LIMIT:
            syms = syms[:SCAN_LIMIT]
        return syms
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [s.strip() for s in args.symbols.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="多周期双坑底扫描器 v4.1（矩阵版·全市场）")
    parser.add_argument("--symbols", type=str, default="", help="股票代码,逗号分隔(与--all二选一)")
    parser.add_argument("--symbols-file", type=str, default=None, help="从文件读取代码列表")
    parser.add_argument("--all", dest="all_market", action="store_true", help="全市场扫描(baostock取全A)")
    parser.add_argument("--periods", type=str, default="W,M,Q,Y", help="周期 W/M/Q/Y")
    parser.add_argument("--min-score", type=float, default=50.0, help="最低形态评分(0-100)")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "baostock", "akshare"])
    parser.add_argument("--workers", type=int, default=NUM_PROCESSES, help="并行进程数")
    parser.add_argument("--push", action="store_true", help="Server酱推送(需SENDKEY)")
    parser.add_argument("--top", type=int, default=20, help="推送/展示Top N")
    args = parser.parse_args()

    symbols = load_symbols(args)
    periods = [p.strip().upper() for p in args.periods.split(",")]
    print("=" * 70)
    print("  多周期双坑底扫描器 v4.1 —— 矩阵版")
    print(f"  数据源: {args.source} | 并行度: {args.workers} | 股票数: {len(symbols)}")
    print("=" * 70)
    if not symbols:
        logger.error("无股票可扫描(--all取不到列表或未提供--symbols)")
        sys.exit(0)
    if not HAS_BS and not HAS_AK:
        logger.error("baostock 和 akshare 均未安装, 无法取数")

    results_by_symbol = scan_all(symbols, periods, args.min_score, source=args.source, workers=args.workers)

    all_rows = []
    for sym, period_results in results_by_symbol.items():
        for period_name, patterns in period_results.items():
            for p in patterns:
                all_rows.append(p.to_dict())
    print(f"\n命中 {len(all_rows)} 条形态, 涉及 {len(results_by_symbol)} 只股票")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(OUTPUT_DIR, f"double_bottom_{tag}.csv")
    if all_rows:
        df_out = pd.DataFrame(all_rows).sort_values("total_score", ascending=False)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 结果已保存到 {out_path}（共 {len(all_rows)} 条）")
        print(f"📊 最高评分: {df_out['total_score'].max():.1f} | 平均: {df_out['total_score'].mean():.1f}")
    else:
        pd.DataFrame(columns=["symbol", "period", "total_score", "status", "neckline",
                              "current_close", "target_price", "stop_loss_price"]).to_csv(out_path, index=False, encoding="utf-8-sig")
        print("全部扫描完成, 未发现符合条件的双底形态")

    if args.push:
        content = build_push_content(all_rows, top=args.top)
        push_wechat(f"双坑底扫描 命中{len(all_rows)}条 ({datetime.now():%Y-%m-%d %H:%M})", content)
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_double_bottom<<<
