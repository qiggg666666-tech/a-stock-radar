#!/usr/bin/env python3
"""
多周期双坑底（双底 / W底）扫描器 v5.3 —— 矩阵修复版
【本版修复】
  ① get_all_a_share_symbols 恢复 finally: bs.logout()
  ② 无股票/无数据源时 sys.exit(1) -> sys.exit(0), 防 Actions 瞬时失败红叉
  ③ 【矩阵接入】移除 random.shuffle，新增 SCAN_OFFSET 支持分段扫描，防止分段乱序。
其余沿用 v5.3: forming-only / near-neckline / list-symbols / 断点续扫 / 会话复用 / 查询超时。
"""
import argparse
import glob
import logging
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("double_bottom_v5_3")
try:
    import baostock as bs; HAS_BS = True
except ImportError: HAS_BS = False
try:
    import akshare as ak; HAS_AK = True
except ImportError: HAS_AK = False
try:
    import yfinance as yf; HAS_YF = True
except ImportError: HAS_YF = False
try:
    from tqdm import tqdm; HAS_TQDM = True
except ImportError: HAS_TQDM = False
try:
    import requests; HAS_REQUESTS = True
except ImportError: HAS_REQUESTS = False

# ==================== 终端颜色 ====================
class Colors:
    GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; BOLD = "\033[1m"; END = "\033[0m"
    @classmethod
    def green(cls, s): return f"{cls.GREEN}{s}{cls.END}"
    @classmethod
    def yellow(cls, s): return f"{cls.YELLOW}{s}{cls.END}"
    @classmethod
    def red(cls, s): return f"{cls.RED}{s}{cls.END}"
    @classmethod
    def cyan(cls, s): return f"{cls.CYAN}{s}{cls.END}"
    @classmethod
    def bold(cls, s): return f"{cls.BOLD}{s}{cls.END}"

# ==================== 全局参数 ====================
class Config:
    ZIGZAG_DEVIATION_PCT = 0.05; MIN_TREND_DECLINE = 0.10; MAX_PRICE_DIFF_PCT = 0.03; MIN_REBOUND_PCT = 0.10
    MIN_BARS_BETWEEN = 8; MAX_BARS_BETWEEN = 90; BREAKOUT_BUFFER = 0.01; MIN_BREAKOUT_VOLUME_RATIO = 1.5
    MAX_VOL2_RATIO = 1.2; MIN_HOLD_DAYS = 2; RSI_PERIOD = 14
    WEIGHT_SYMMETRY = 25; WEIGHT_VOLUME = 15; WEIGHT_TREND = 15; WEIGHT_REBOUND = 15; WEIGHT_BREAKOUT = 15; WEIGHT_DIVERGENCE = 15
    NUM_PROCESSES = int(os.getenv("NUM_WORKERS", "1"))
    # 关键：超时必须短于 GitHub job 的预算，且备用源只允许有限回退。
    AK_FETCH_TIMEOUT_SEC = int(os.getenv("AK_FETCH_TIMEOUT_SEC", "12"))
    BS_FETCH_TIMEOUT_SEC = int(os.getenv("BS_FETCH_TIMEOUT_SEC", "8"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
    BS_FAILURE_LIMIT = int(os.getenv("BS_FAILURE_LIMIT", "5"))
    STAGGER_DELAY_RANGE = (0.15, 0.45)
    # 提前正常退出并上传中间结果；0 表示不限制。默认约 5 小时 20 分。
    MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", "19200"))
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
    SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT", "0"))
    SCAN_OFFSET = int(os.environ.get("SCAN_OFFSET", "0"))  # 矩阵新增：分段扫描偏移量

@dataclass
class PeriodConfig:
    rule: str; name: str; min_years: int; lookback_bars: int; min_data_bars: int
    min_pivot_bars: int; max_pivot_bars: int; zigzag_deviation: float; min_decline: float; min_rebound: float

PERIOD_CONFIG: Dict[str, PeriodConfig] = {
    "W": PeriodConfig("W-FRI", "周线", 3, 150, 25, 8, 90, 0.05, 0.10, 0.10),
    "M": PeriodConfig("ME",    "月线", 5, 60,  20, 4, 36, 0.05, 0.10, 0.10),
    "Q": PeriodConfig("QE",    "季线", 10, 40, 15, 2, 16, 0.08, 0.10, 0.10),
    "Y": PeriodConfig("YE",    "年线", 20, 20, 8,  1, 6,  0.15, 0.10, 0.10),
}

class BreakoutQuality(Enum):
    STRONG = "强突破"; WEAK = "弱突破"; FORMING = "形成中"; FAKE = "假突破"

@dataclass
class DoubleBottomResult:
    symbol: str; period: str
    bottom1_date: datetime; bottom1_price: float; bottom1_idx: int
    bottom2_date: datetime; bottom2_price: float; bottom2_idx: int
    neckline: float; neckline_zone: Tuple[float, float]; bars_between: int
    price_diff_pct: float; rebound_pct: float; prior_decline_pct: float
    vol1: float; vol2: float; vol_ratio: float; breakout_vol_ratio: float
    breakout: bool; breakout_date: Optional[datetime]; breakout_quality: BreakoutQuality; breakout_candle_vol: float
    rsi_bottom1: float; rsi_bottom2: float; rsi_divergence: bool; macd_divergence: bool
    symmetry_score: float; volume_score: float; trend_score: float; rebound_score: float
    breakout_score: float; divergence_score: float; total_score: float
    current_close: float; status: str; target_price: float; stop_loss_price: float; distance_to_neckline_pct: float
    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "period": self.period,
            "bottom1_date": self.bottom1_date.strftime("%Y-%m-%d"), "bottom1_price": round(self.bottom1_price, 3),
            "bottom2_date": self.bottom2_date.strftime("%Y-%m-%d"), "bottom2_price": round(self.bottom2_price, 3),
            "neckline": round(self.neckline, 3), "neckline_zone_low": round(self.neckline_zone[0], 3),
            "neckline_zone_high": round(self.neckline_zone[1], 3), "bars_between": self.bars_between,
            "price_diff_pct": round(self.price_diff_pct, 2), "rebound_pct": round(self.rebound_pct, 2),
            "prior_decline_pct": round(self.prior_decline_pct, 2), "vol1": int(self.vol1), "vol2": int(self.vol2),
            "vol_ratio": round(self.vol_ratio, 2),
            "breakout_vol_ratio": round(self.breakout_vol_ratio, 2) if self.breakout else None,
            "breakout": self.breakout,
            "breakout_date": self.breakout_date.strftime("%Y-%m-%d") if self.breakout_date else None,
            "breakout_quality": self.breakout_quality.value,
            "rsi_bottom1": round(self.rsi_bottom1, 2), "rsi_bottom2": round(self.rsi_bottom2, 2),
            "rsi_divergence": self.rsi_divergence, "macd_divergence": self.macd_divergence,
            "symmetry_score": round(self.symmetry_score, 1), "volume_score": round(self.volume_score, 1),
            "trend_score": round(self.trend_score, 1), "rebound_score": round(self.rebound_score, 1),
            "breakout_score": round(self.breakout_score, 1), "divergence_score": round(self.divergence_score, 1),
            "total_score": round(self.total_score, 1), "current_close": round(self.current_close, 3),
            "status": self.status, "target_price": round(self.target_price, 3),
            "stop_loss_price": round(self.stop_loss_price, 3),
            "distance_to_neckline_pct": round(self.distance_to_neckline_pct, 2)}

# ==================== Baostock 会话管理 ====================
_BS_SESSION_READY = False
_BS_FAILURES = 0
_BS_CIRCUIT_OPEN = False
def _bs_logout_quiet():
    try:
        if HAS_BS: bs.logout()
    except Exception: pass
def _worker_bs_init():
    global _BS_SESSION_READY
    time.sleep(random.uniform(0, 2))
    if not HAS_BS:
        _BS_SESSION_READY = False; return
    _bs_logout_quiet()
    try:
        lg = bs.login(); _BS_SESSION_READY = (lg.error_code == "0")
        if _BS_SESSION_READY:
            import atexit; atexit.register(_bs_logout_quiet)
    except Exception as e:
        logger.warning(f"Worker baostock 登录失败: {e}"); _BS_SESSION_READY = False
def _ensure_bs_session() -> bool:
    global _BS_SESSION_READY
    if not HAS_BS: return False
    if _BS_SESSION_READY: return True
    try:
        lg = bs.login(); _BS_SESSION_READY = (lg.error_code == "0"); return _BS_SESSION_READY
    except Exception:
        _BS_SESSION_READY = False; return False

# ==================== 全市场列表 ====================
def get_all_a_share_symbols() -> List[str]:
    """AkShare 主路径获取股票池；BaoStock 仅在主路径失败时回退。"""
    if HAS_AK:
        try:
            spot = ak.stock_zh_a_spot_em()
            code_col = "代码" if "代码" in spot.columns else "code"
            name_col = "名称" if "名称" in spot.columns else "name"
            work = spot[[code_col, name_col]].copy()
            work[code_col] = work[code_col].astype(str).str.zfill(6)
            work = work[~work[name_col].astype(str).str.contains(r"ST|退", na=False, regex=True)]
            symbols = [code for code in work[code_col].tolist() if code[:2] in ("60", "00", "30", "68")]
            if symbols:
                return sorted(set(symbols))
        except Exception as exc:
            logger.warning(f"AkShare 股票池获取失败，尝试 BaoStock 回退: {exc}")
    if not HAS_BS:
        return []
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return []
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[(df["type"] == "1") & (df["status"] == "1")]
        df = df[~df["code_name"].astype(str).str.contains(r"ST|退", na=False, regex=True)]
        symbols = [str(code).split(".")[-1] for code in df["code"].tolist()]
        return sorted({code for code in symbols if code[:2] in ("60", "00", "30", "68")})
    except Exception as exc:
        logger.warning(f"BaoStock 股票池获取失败: {exc}")
        return []
    finally:
        try:
            bs.logout()
        except Exception:
            pass

def load_scanned_symbols(resume_pattern: Optional[str]) -> Set[str]:
    if not resume_pattern: return set()
    scanned = set()
    try:
        files = glob.glob(resume_pattern)
        for f in files:
            if os.path.exists(f):
                df = pd.read_csv(f, encoding="utf-8-sig")
                if "symbol" in df.columns: scanned.update(df["symbol"].astype(str).unique().tolist())
        logger.info(f"断点续扫: 从 {len(files)} 个历史文件读取到 {len(scanned)} 只已扫描股票")
    except Exception as e: logger.warning(f"读取历史文件失败: {e}")
    return scanned

def fetch_with_timeout(fn, *args, timeout: int, **kwargs):
    """不等待已超时线程结束；避免原 with 退出时 wait=True 抵消 timeout。"""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        logger.warning(f"{fn.__name__} 超时({timeout}s)，本标的快速跳过")
        return None
    except Exception as e:
        logger.warning(f"{fn.__name__} 异常: {e}")
        return None
    finally:
        # 不等待卡住的 SDK 请求；断路器会避免持续创建备用源请求。
        executor.shutdown(wait=False, cancel_futures=True)

def to_bs_code(symbol: str) -> str:
    if symbol.startswith(("sh.", "sz.", "bj.")): return symbol
    if symbol.startswith("6"): return f"sh.{symbol}"
    if symbol.startswith(("0", "3")): return f"sz.{symbol}"
    if symbol.startswith(("4", "8")): return f"bj.{symbol}"
    return f"sh.{symbol}"

def get_bs_data(symbol: str, years: int = 12) -> Optional[pd.DataFrame]:
    if not HAS_BS: return None
    if not _ensure_bs_session(): return None
    bs_code = to_bs_code(symbol)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    try:
        rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume",
            start_date=start, end_date=end, frequency="d", adjustflag="2")
        if rs.error_code != "0": return None
        rows = []
        while (rs.error_code == "0") and rs.next(): rows.append(rs.get_row_data())
        if not rows: return None
        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume"]: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.warning(f"[{symbol}] baostock 取数异常: {e}"); return None

def get_ak_data(symbol: str, years: int = 12) -> Optional[pd.DataFrame]:
    if not HAS_AK: return None
    try:
        end = datetime.now().strftime("%Y%m%d"); start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty: return None
        rm = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in rm.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date").sort_index()
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception as e:
        logger.warning(f"[{symbol}] akshare 取数失败: {e}"); return None

def get_yf_data(symbol: str, years: int = 12) -> Optional[pd.DataFrame]:
    if not HAS_YF: return None
    try:
        df = yf.Ticker(symbol).history(period=f"{years}y", auto_adjust=True)
        if df is None or df.empty: return None
        df = df.rename(columns=str.lower)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.warning(f"[{symbol}] yfinance 取数失败: {e}"); return None

def get_daily_data(symbol: str, is_a_share: bool = True, years: int = 12, source: str = "auto") -> Optional[pd.DataFrame]:
    """AkShare 主路径；BaoStock 仅作为短超时、有限次数的备用路径。"""
    global _BS_FAILURES, _BS_CIRCUIT_OPEN
    if not is_a_share:
        return fetch_with_timeout(get_yf_data, symbol, years, timeout=Config.AK_FETCH_TIMEOUT_SEC)
    tried = []
    if source in ("auto", "akshare") and HAS_AK:
        for _ in range(Config.MAX_RETRIES):
            df = fetch_with_timeout(get_ak_data, symbol, years, timeout=Config.AK_FETCH_TIMEOUT_SEC)
            if df is not None and len(df) > 50:
                return df
            tried.append("akshare")
        if source == "akshare":
            return None
    if source in ("auto", "baostock") and HAS_BS and not _BS_CIRCUIT_OPEN:
        for _ in range(Config.MAX_RETRIES):
            df = fetch_with_timeout(get_bs_data, symbol, years, timeout=Config.BS_FETCH_TIMEOUT_SEC)
            if df is not None and len(df) > 50:
                _BS_FAILURES = 0
                return df
            tried.append("baostock")
            _BS_FAILURES += 1
            if _BS_FAILURES >= Config.BS_FAILURE_LIMIT:
                _BS_CIRCUIT_OPEN = True
                logger.warning("BaoStock 连续失败达到阈值，已为本 worker 关闭备用源，剩余标的仅走 AkShare/快速跳过")
                break
        if source == "baostock":
            return None
    logger.warning(f"[{symbol}] 所有数据源均失败（尝试: {tried or '无可用源'}）")
    return None

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty or len(df) < 5: return pd.DataFrame()
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resampled = df.resample(rule).agg(ohlc)
    if rule in ("ME", "QE", "YE"):
        resampled[["open", "high", "low", "close"]] = resampled[["open", "high", "low", "close"]].ffill()
        resampled["volume"] = resampled["volume"].fillna(0)
    return resampled.dropna(subset=["open", "high", "low", "close"])

def compute_rsi(series: pd.Series, period: int = Config.RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0); loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean(); avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean(); ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def zigzag_pivots(high, low, close, deviation_pct):
    n = len(close)
    if n < 5: return [], []
    pivots_idx = []; pivots_type = []; trend = 0; last_extreme_idx = 0; last_extreme_price = close.iloc[0]
    for i in range(1, n):
        current_price = close.iloc[i]
        if trend == 0:
            if current_price >= last_extreme_price * (1 + deviation_pct):
                trend = 1; last_extreme_idx = i; last_extreme_price = current_price
            elif current_price <= last_extreme_price * (1 - deviation_pct):
                trend = -1; last_extreme_idx = i; last_extreme_price = current_price
        elif trend == 1:
            if current_price > last_extreme_price:
                last_extreme_idx = i; last_extreme_price = current_price
            elif current_price <= last_extreme_price * (1 - deviation_pct):
                pivots_idx.append(last_extreme_idx); pivots_type.append("H")
                trend = -1; last_extreme_idx = i; last_extreme_price = current_price
        elif trend == -1:
            if current_price < last_extreme_price:
                last_extreme_idx = i; last_extreme_price = current_price
            elif current_price >= last_extreme_price * (1 + deviation_pct):
                pivots_idx.append(last_extreme_idx); pivots_type.append("L")
                trend = 1; last_extreme_idx = i; last_extreme_price = current_price
    if trend == -1 and (not pivots_idx or last_extreme_idx != pivots_idx[-1]):
        pivots_idx.append(last_extreme_idx); pivots_type.append("L")
    return pivots_idx, pivots_type

def find_neckline_zone(df, idx1, idx2):
    middle_df = df.iloc[idx1: idx2 + 1]
    if len(middle_df) < 3:
        mx = float(middle_df["high"].max()); return mx, mx * 0.998, mx * 1.002
    highs = middle_df["high"].values; top_n = min(3, len(highs))
    significant_highs = np.partition(highs, -top_n)[-top_n:]
    neckline = float(np.mean(significant_highs))
    return neckline, float(np.min(significant_highs) * 0.998), float(np.max(significant_highs) * 1.002)

def analyze_breakout_quality(df, neckline, breakout_idx, avg_volume):
    if breakout_idx is None or breakout_idx >= len(df): return BreakoutQuality.FORMING, 0.0
    breakout_vol = df.iloc[breakout_idx]["volume"]; vol_ratio = breakout_vol / max(avg_volume, 1)
    hold_days = 0
    for i in range(breakout_idx + 1, min(breakout_idx + Config.MIN_HOLD_DAYS + 1, len(df))):
        if df.iloc[i]["close"] >= neckline * (1 - Config.BREAKOUT_BUFFER): hold_days += 1
        else: break
    if vol_ratio >= Config.MIN_BREAKOUT_VOLUME_RATIO and hold_days >= Config.MIN_HOLD_DAYS: return BreakoutQuality.STRONG, vol_ratio
    elif vol_ratio >= 1.0: return BreakoutQuality.WEAK, vol_ratio
    else: return BreakoutQuality.FAKE, vol_ratio

def calculate_dynamic_breakout_score(breakout_quality, current_close, neckline):
    if breakout_quality == BreakoutQuality.STRONG: return 100.0
    elif breakout_quality == BreakoutQuality.WEAK: return 60.0
    elif breakout_quality == BreakoutQuality.FAKE: return 20.0
    distance_pct = (neckline - current_close) / neckline * 100
    if distance_pct <= 1.0: return 85.0
    elif distance_pct <= 2.0: return 78.0
    elif distance_pct <= 3.0: return 70.0
    elif distance_pct <= 5.0: return 60.0
    elif distance_pct <= 10.0: return 45.0
    else: return 30.0

def calculate_scores(price_diff_pct, vol_ratio, prior_decline_pct, rebound_pct,
                     breakout_quality, breakout_vol_ratio, rsi_divergence, macd_divergence, current_close, neckline):
    symmetry = max(0.0, 100.0 - (price_diff_pct / Config.MAX_PRICE_DIFF_PCT) * 100.0)
    if vol_ratio <= 0.5: volume = 100.0
    elif vol_ratio >= Config.MAX_VOL2_RATIO: volume = 0.0
    else: volume = 100.0 - ((vol_ratio - 0.5) / (Config.MAX_VOL2_RATIO - 0.5)) * 100.0
    trend = min(100.0, 60.0 + (prior_decline_pct - Config.MIN_TREND_DECLINE) / 0.20 * 40.0); trend = max(0.0, trend)
    rebound = min(100.0, 60.0 + (rebound_pct - Config.MIN_REBOUND_PCT) / 0.15 * 40.0); rebound = max(0.0, rebound)
    breakout = calculate_dynamic_breakout_score(breakout_quality, current_close, neckline)
    divergence = 0.0
    if rsi_divergence: divergence += 50.0
    if macd_divergence: divergence += 50.0
    total = (symmetry * Config.WEIGHT_SYMMETRY / 100 + volume * Config.WEIGHT_VOLUME / 100 +
             trend * Config.WEIGHT_TREND / 100 + rebound * Config.WEIGHT_REBOUND / 100 +
             breakout * Config.WEIGHT_BREAKOUT / 100 + divergence * Config.WEIGHT_DIVERGENCE / 100)
    return symmetry, volume, trend, rebound, breakout, divergence, total

def detect_double_bottom(df, symbol="", period="", price_tol=Config.MAX_PRICE_DIFF_PCT,
                         min_decline=Config.MIN_TREND_DECLINE, min_rebound=Config.MIN_REBOUND_PCT,
                         min_data_bars=40, zigzag_deviation=Config.ZIGZAG_DEVIATION_PCT,
                         min_pivot_bars=Config.MIN_BARS_BETWEEN, max_pivot_bars=Config.MAX_BARS_BETWEEN):
    if len(df) < min_data_bars: return []
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    lows = df["low"]; highs = df["high"]; closes = df["close"]; volumes = df["volume"]
    rsi = df["rsi"]; macd_hist = df["macd_hist"]
    pivot_indices, pivot_types = zigzag_pivots(highs, lows, closes, deviation_pct=zigzag_deviation)
    low_pivot_indices = [pivot_indices[i] for i in range(len(pivot_types)) if pivot_types[i] == "L"]
    if len(low_pivot_indices) < 2: return []
    results = []
    for i in range(len(low_pivot_indices) - 1):
        idx1 = low_pivot_indices[i]
        for j in range(i + 1, len(low_pivot_indices)):
            idx2 = low_pivot_indices[j]; bars_between = idx2 - idx1
            if not (min_pivot_bars <= bars_between <= max_pivot_bars): continue
            price1 = float(lows.iloc[idx1]); price2 = float(lows.iloc[idx2])
            price_diff_pct = abs(price1 - price2) / max(price1, 1e-9)
            if price_diff_pct > price_tol: continue
            pre_high = float(highs.iloc[max(0, idx1 - 30): idx1].max())
            prior_decline_pct = (pre_high - price1) / pre_high if pre_high > 0 else 0
            if prior_decline_pct < min_decline: continue
            middle_high = float(highs.iloc[idx1: idx2].max())
            rebound_pct = (middle_high - max(price1, price2)) / max(price1, price2)
            if rebound_pct < min_rebound: continue
            vol1 = float(volumes.iloc[idx1]); vol2 = float(volumes.iloc[idx2]); vol_ratio = vol2 / max(vol1, 1)
            if vol2 > vol1 * Config.MAX_VOL2_RATIO: continue
            neckline, zone_low, zone_high = find_neckline_zone(df, idx1, idx2)
            after = closes.iloc[idx2 + 1:]
            breakout = False; breakout_idx = None; breakout_quality = BreakoutQuality.FORMING; breakout_vol_ratio = 0.0
            avg_vol = float(volumes.iloc[max(0, idx2 - 20): idx2].mean())
            if len(after) > 0:
                for k, c in enumerate(after):
                    if c >= neckline * (1 + Config.BREAKOUT_BUFFER):
                        breakout = True; breakout_idx = idx2 + 1 + k
                        breakout_quality, breakout_vol_ratio = analyze_breakout_quality(df, neckline, breakout_idx, avg_vol)
                        break
            rsi1 = float(rsi.iloc[idx1]) if not pd.isna(rsi.iloc[idx1]) else 50.0
            rsi2 = float(rsi.iloc[idx2]) if not pd.isna(rsi.iloc[idx2]) else 50.0
            rsi_divergence = (price2 <= price1 * 1.01) and (rsi2 > rsi1 * 1.02)
            macd1 = float(macd_hist.iloc[idx1]) if not pd.isna(macd_hist.iloc[idx1]) else 0.0
            macd2 = float(macd_hist.iloc[idx2]) if not pd.isna(macd_hist.iloc[idx2]) else 0.0
            macd_divergence = (price2 <= price1 * 1.01) and (macd2 > macd1)
            current_close = float(closes.iloc[-1]); distance_to_neckline = (neckline - current_close) / neckline * 100
            (symmetry_score, volume_score, trend_score, rebound_score, breakout_score, divergence_score, total_score) = calculate_scores(
                price_diff_pct, vol_ratio, prior_decline_pct, rebound_pct, breakout_quality, breakout_vol_ratio,
                rsi_divergence, macd_divergence, current_close, neckline)
            pattern_height = neckline - min(price1, price2)
            target_price = neckline + pattern_height if breakout else current_close * 1.05
            stop_loss_price = min(price1, price2) * 0.95
            if breakout:
                status = {BreakoutQuality.STRONG: "已突破(强)", BreakoutQuality.WEAK: "已突破(弱)",
                          BreakoutQuality.FAKE: "假突破", BreakoutQuality.FORMING: "已突破"}.get(breakout_quality, "已突破")
            else: status = "形成中"
            results.append(DoubleBottomResult(symbol=symbol, period=period,
                bottom1_date=df.index[idx1], bottom1_price=price1, bottom1_idx=idx1,
                bottom2_date=df.index[idx2], bottom2_price=price2, bottom2_idx=idx2,
                neckline=neckline, neckline_zone=(zone_low, zone_high), bars_between=bars_between,
                price_diff_pct=price_diff_pct * 100, rebound_pct=rebound_pct * 100, prior_decline_pct=prior_decline_pct * 100,
                vol1=vol1, vol2=vol2, vol_ratio=vol_ratio, breakout_vol_ratio=breakout_vol_ratio,
                breakout=breakout, breakout_date=df.index[breakout_idx] if breakout_idx is not None else None,
                breakout_quality=breakout_quality,
                breakout_candle_vol=df.iloc[breakout_idx]["volume"] if breakout_idx is not None else 0,
                rsi_bottom1=rsi1, rsi_bottom2=rsi2, rsi_divergence=rsi_divergence, macd_divergence=macd_divergence,
                symmetry_score=symmetry_score, volume_score=volume_score, trend_score=trend_score,
                rebound_score=rebound_score, breakout_score=breakout_score, divergence_score=divergence_score,
                total_score=total_score, current_close=current_close, status=status,
                target_price=target_price, stop_loss_price=stop_loss_price,
                distance_to_neckline_pct=distance_to_neckline))
    results = sorted(results, key=lambda x: x.total_score, reverse=True)[:5]
    return results

def scan_symbol(symbol, periods, is_a_share=True, min_score=50.0, source="auto"):
    max_years = max(PERIOD_CONFIG[p].min_years for p in periods if p in PERIOD_CONFIG)
    try:
        daily = get_daily_data(symbol, is_a_share=is_a_share, years=max_years, source=source)
    except Exception as e:
        logger.warning(f"[{symbol}] 取数异常: {e}"); return {}, "error"
    if daily is None: return {}, "no_data"
    if len(daily) < 100: return {}, "no_data"
    all_results = {}
    for code in periods:
        if code not in PERIOD_CONFIG: continue
        cfg = PERIOD_CONFIG[code]
        try: df = resample_ohlcv(daily, cfg.rule)
        except Exception: continue
        df = df.tail(cfg.lookback_bars)
        if len(df) < cfg.min_data_bars: continue
        try:
            patterns = detect_double_bottom(df, symbol=symbol, period=cfg.name,
                min_data_bars=cfg.min_data_bars, zigzag_deviation=cfg.zigzag_deviation,
                min_pivot_bars=cfg.min_pivot_bars, max_pivot_bars=cfg.max_pivot_bars,
                min_decline=cfg.min_decline, min_rebound=cfg.min_rebound)
        except Exception: continue
        patterns = [p for p in patterns if p.total_score >= min_score]
        if patterns: all_results[cfg.name] = patterns
    return all_results, "ok"

def _scan_worker(task):
    symbol, periods, is_a, min_score, source = task
    time.sleep(random.uniform(*Config.STAGGER_DELAY_RANGE))
    return symbol, scan_symbol(symbol, periods=periods, is_a_share=is_a, min_score=min_score, source=source)

def save_checkpoint(all_rows, tag, suffix="", processed_symbols=None, stats=None):
    """即便暂无命中也落盘进度，确保 GitHub 取消时 Upload results 仍有可恢复信息。"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    columns = ["symbol", "period", "total_score", "status", "distance_to_neckline_pct"]
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = frame.sort_values("total_score", ascending=False)
    result_path = os.path.join(Config.OUTPUT_DIR, f"double_bottom_{tag}{suffix}.csv")
    frame.to_csv(result_path, index=False, encoding="utf-8-sig")
    if processed_symbols is not None:
        progress_path = os.path.join(Config.OUTPUT_DIR, f"double_bottom_{tag}{suffix}_progress.csv")
        pd.DataFrame({"symbol": processed_symbols}).to_csv(progress_path, index=False, encoding="utf-8-sig")
    if stats is not None:
        status_path = os.path.join(Config.OUTPUT_DIR, f"double_bottom_{tag}{suffix}_status.json")
        import json
        Path(status_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"💾 检查点已保存: {result_path}；已处理 {len(processed_symbols or [])} 只")

def push_wechat(title, content):
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        logger.info("未设置 SENDKEY，跳过推送"); return False
    if not HAS_REQUESTS: return False
    try:
        resp = requests.post(f"https://sctapi.ftqq.com/{sendkey}.send", data={"title": title, "desp": content}, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"推送异常: {e}"); return False

def build_push_content(all_rows, top=10, forming_only=False):
    if not all_rows: return "本次扫描未发现符合条件的双底形态。"
    df = pd.DataFrame(all_rows).head(top)
    lines = [f"### {'正在形成中' if forming_only else '双坑底扫描'} Top {min(top, len(df))}", ""]
    for _, r in df.iterrows():
        emoji = "🚀" if r["breakout"] else "⏳"
        near = ""
        if not r["breakout"] and r.get("distance_to_neckline_pct", 999) < 3: near = " 🔥即将突破"
        lines.append(f"{emoji} **{r['symbol']}** [{r['period']}] 评分 **{r['total_score']:.1f}** | {r['status']}{near}  \n"
                     f"颈线 {r['neckline']:.2f} | 现价 {r['current_close']:.2f} | 距颈线 {r['distance_to_neckline_pct']:.1f}% | 目标 {r['target_price']:.2f}")
        lines.append("")
    return "\n".join(lines)

def colorize_distance(pct):
    if pct <= 3: return Colors.green(f"{pct:.2f}%")
    elif pct <= 10: return Colors.yellow(f"{pct:.2f}%")
    else: return Colors.red(f"{pct:.2f}%")

def load_symbols(args):
    if getattr(args, "all_market", False):
        syms = get_all_a_share_symbols()
        logger.info(f"全市场模式: 取到 {len(syms)} 只")
        
        # 【矩阵修复】移除 random.shuffle(syms)，确保分段扫描顺序固定
        
        # 矩阵新增：分段扫描逻辑
        if Config.SCAN_OFFSET > 0 and len(syms) > Config.SCAN_OFFSET:
            syms = syms[Config.SCAN_OFFSET:]
            logger.info(f"分段扫描: 跳过前 {Config.SCAN_OFFSET} 只, 本段剩余 {len(syms)} 只")
            
        if Config.SCAN_LIMIT > 0 and len(syms) > Config.SCAN_LIMIT:
            syms = syms[:Config.SCAN_LIMIT]
            logger.info(f"已限制扫描前 {Config.SCAN_LIMIT} 只")
        return syms
        
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]
    return []

def main():
    parser = argparse.ArgumentParser(description="多周期双坑底扫描器 v5.4 —— 快速容错分片版")
    parser.add_argument("--symbols", type=str, default="", help="股票代码,逗号分隔")
    parser.add_argument("--symbols-file", type=str, default=None, help="从文件读取代码列表")
    parser.add_argument("--all", dest="all_market", action="store_true", help="全市场扫描")
    parser.add_argument("--periods", type=str, default="W,M,Q,Y", help="周期 W/M/Q/Y")
    parser.add_argument("--min-score", type=float, default=50.0, help="最低形态评分")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "baostock", "akshare"])
    parser.add_argument("--workers", type=int, default=Config.NUM_PROCESSES, help="并行进程数(全市场建议1)")
    parser.add_argument("--push", action="store_true", help="Server酱推送")
    parser.add_argument("--top", type=int, default=20, help="推送/展示Top N")
    parser.add_argument("--batch-save", type=int, default=25, help="每N只保存一次中间结果")
    parser.add_argument("--resume", type=str, default=None, help="断点续扫：传入已有CSV路径通配符")
    parser.add_argument("--forming-only", action="store_true", help="只显示正在形成中(未突破)的形态")
    parser.add_argument("--near-neckline", type=float, default=100.0, help="形成中模式下只显示距颈线<=N%%的股票(默认100=不限)")
    parser.add_argument("--list-symbols", action="store_true", help="仅输出股票代码列表(供pipeline使用)")
    args = parser.parse_args()
    symbols = load_symbols(args)
    periods = [p.strip().upper() for p in args.periods.split(",")]
    scanned = load_scanned_symbols(args.resume)
    if scanned:
        before = len(symbols)
        symbols = [s for s in symbols if s not in scanned]
        logger.info(f"断点续扫: 跳过 {before - len(symbols)} 只已扫描股票，剩余 {len(symbols)} 只")
    print("=" * 70)
    print("  多周期双坑底扫描器 v5.4 —— 快速容错分片版")
    print(f"  数据源: {args.source} | 进程数: {args.workers} | 待扫股票: {len(symbols)}")
    print(f"  扫描周期: {', '.join(periods)} | 中间保存: 每{args.batch_save}只")
    if args.forming_only:
        print(f"  {Colors.bold(Colors.green('>>> 只显示正在形成中的形态（未突破颈线）<<<'))}")
        if args.near_neckline < 100:
            print(f"  {Colors.bold(Colors.yellow(f'>>> 距颈线 <= {args.near_neckline}% 过滤 <<<'))}")
    if scanned: print(f"  断点续扫: 已跳过 {len(scanned)} 只")
    print("=" * 70)
    if not symbols:
        logger.error("无股票可扫描"); sys.exit(0)
    if not HAS_BS and not HAS_AK:
        logger.error("baostock 和 akshare 均未安装"); sys.exit(0)
    for code in periods:
        if code in PERIOD_CONFIG:
            cfg = PERIOD_CONFIG[code]
            logger.info(f"[{cfg.name}] 配置: 需{cfg.min_years}年日线, 检测>={cfg.min_data_bars}根, "
                        f"底间距{cfg.min_pivot_bars}~{cfg.max_pivot_bars}, ZigZag={cfg.zigzag_deviation:.0%}")
    stats = {"total": len(symbols), "success": 0, "no_data": 0, "no_pattern": 0, "error": 0}
    all_rows: List[Dict[str, Any]] = []
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    start_time = time.time()
    # ==================== 限时扫描与检查点 ====================
    processed_symbols: List[str] = []
    deadline = start_time + Config.MAX_RUNTIME_SECONDS if Config.MAX_RUNTIME_SECONDS > 0 else None

    def out_of_budget() -> bool:
        return deadline is not None and time.time() >= deadline

    def record(sym: str, res, status: str):
        processed_symbols.append(sym)
        if status == "no_data":
            stats["no_data"] += 1
        elif status == "error":
            stats["error"] += 1
        elif res:
            stats["success"] += 1
            for _, patterns in res.items():
                for pattern in patterns:
                    all_rows.append(pattern.to_dict())
        else:
            stats["no_pattern"] += 1

    def checkpoint(count: int):
        save_checkpoint(all_rows, tag, suffix=f"_checkpoint_{count}", processed_symbols=processed_symbols, stats=stats)

    # ==================== 单线程模式 ====================
    if args.workers <= 1:
        iterator = tqdm(enumerate(symbols, start=1), total=len(symbols), desc="扫描中", ncols=80) if HAS_TQDM else enumerate(symbols, start=1)
        for count, sym in iterator:
            if out_of_budget():
                logger.warning("已达到安全运行预算，正常退出并保留检查点")
                checkpoint(count - 1)
                break
            try:
                res, status = scan_symbol(sym, periods=periods, is_a_share=sym.isdigit() and len(sym) == 6, min_score=args.min_score, source=args.source)
            except Exception as e:
                logger.warning(f"[{sym}] 扫描异常: {e}")
                res, status = {}, "error"
            record(sym, res, status)
            if count % args.batch_save == 0:
                checkpoint(count)
    # ==================== 多进程模式 ====================
    else:
        tasks = [(sym, periods, sym.isdigit() and len(sym) == 6, args.min_score, args.source) for sym in symbols]
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_bs_init)
        early_exit = False
        try:
            futures = {executor.submit(_scan_worker, task): task[0] for task in tasks}
            completed = as_completed(futures)
            iterator = tqdm(completed, desc="扫描中", ncols=80, total=len(futures)) if HAS_TQDM else completed
            for count, fut in enumerate(iterator, start=1):
                if out_of_budget():
                    early_exit = True
                    logger.warning("已达到安全运行预算，取消尚未开始的任务并保留检查点")
                    checkpoint(count - 1)
                    break
                sym = futures[fut]
                try:
                    _, (res, status) = fut.result()
                except Exception as e:
                    logger.warning(f"[{sym}] 进程异常: {e}")
                    res, status = {}, "error"
                record(sym, res, status)
                if count % args.batch_save == 0:
                    checkpoint(count)
        finally:
            if early_exit:
                for fut in futures:
                    fut.cancel()
            executor.shutdown(wait=not early_exit, cancel_futures=True)

    # 最终检查点确保匹配为空时也有可上传文件。
    checkpoint(len(processed_symbols))
    elapsed = time.time() - start_time

    # ==================== 过滤逻辑 ====================
    if args.forming_only:
        before_filter = len(all_rows)
        all_rows = [r for r in all_rows if r.get("status") == "形成中"]
        if args.near_neckline < 100:
            all_rows = [r for r in all_rows if r.get("distance_to_neckline_pct", 999) <= args.near_neckline]
        all_rows = sorted(all_rows, key=lambda x: x.get("distance_to_neckline_pct", 999))
        logger.info(f"形成中过滤: {before_filter} -> {len(all_rows)} 条")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(Config.OUTPUT_DIR, f"double_bottom_{tag}.csv")
    print("\n" + "=" * 70)
    print("  扫描统计")
    print("=" * 70)
    print(f"  总股票数 : {stats['total']}")
    print(f"  命中形态 : {stats['success']} 只")
    print(f"  无数据   : {stats['no_data']} 只")
    print(f"  无形态   : {stats['no_pattern']} 只")
    print(f"  异常失败 : {stats['error']} 只")
    print(f"  总耗时   : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if args.forming_only:
        print(f"  {Colors.green('>>> 仅显示形成中: ' + str(len(all_rows)) + ' 条 <<<')}")
    print("=" * 70)
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        if args.forming_only: df_out = df_out.sort_values("distance_to_neckline_pct", ascending=True)
        else: df_out = df_out.sort_values("total_score", ascending=False)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 结果已保存: {out_path}")
        if args.forming_only:
            near_breakout = (df_out["distance_to_neckline_pct"] <= 3).sum()
            print(f"📊 距颈线<3%(即将突破): {near_breakout} 条 | 总计: {len(df_out)} 条")
        else:
            print(f"📊 最高评分: {df_out['total_score'].max():.1f} | 平均: {df_out['total_score'].mean():.1f}")
        period_counts = df_out["period"].value_counts().to_dict()
        for pname, cnt in period_counts.items():
            print(f"   • {pname}: {cnt} 条")
        print("\n" + "=" * 70)
        print(f"  Top {min(20, len(df_out))} 详细结果")
        print("=" * 70)
        for _, row in df_out.head(20).iterrows():
            dist = row.get("distance_to_neckline_pct", 0)
            dist_str = colorize_distance(dist)
            header = f"┌─【{row['period']}】{row['symbol']} 评分:{row['total_score']:.1f} | {row['status']}"
            is_forming = (row['status'] == "形成中")
            print(f"  {Colors.cyan(header) if is_forming else Colors.bold(header)}{Colors.END}")
            print(f"  │  底1: {row['bottom1_date']} @ {row['bottom1_price']:.2f} (RSI:{row['rsi_bottom1']:.1f})")
            print(f"  │  底2: {row['bottom2_date']} @ {row['bottom2_price']:.2f} (RSI:{row['rsi_bottom2']:.1f})")
            print(f"  │  颈线: {row['neckline']:.2f} | 现价: {row['current_close']:.2f} | 距颈线: {dist_str}")
            print(f"  │  背离: RSI={'✓' if row['rsi_divergence'] else '✗'} MACD={'✓' if row['macd_divergence'] else '✗'}")
            print(f"  │  目标: {row['target_price']:.2f} | 止损: {row['stop_loss_price']:.2f}")
            print(f"  └─")
        if args.list_symbols:
            unique_symbols = df_out["symbol"].unique().tolist()
            print("\n" + "=" * 70)
            print("  股票代码列表（可直接复制到 --symbols 参数）")
            print("=" * 70)
            print(",".join(unique_symbols))
    else:
        pd.DataFrame(columns=["symbol", "period", "total_score", "status", "distance_to_neckline_pct"]).to_csv(
            out_path, index=False, encoding="utf-8-sig")
        print("全部扫描完成, 未发现符合条件的双底形态")
    if args.push:
        push_wechat(f"{'[形成中]' if args.forming_only else ''}双坑底扫描 命中{len(all_rows)}条 ({datetime.now():%Y-%m-%d %H:%M})",
                    build_push_content(all_rows, top=args.top, forming_only=args.forming_only))
    sys.exit(0)

if __name__ == "__main__":
    main()
