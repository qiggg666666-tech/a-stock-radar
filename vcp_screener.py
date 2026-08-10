# -*- coding: utf-8 -*-
"""
vcp_screener_github.py —— VCP 选股 · GitHub Actions 轻量版（矩阵修复版）
【本版修复】
  ① Notifier.send / _build_push_content 的 "\\n" 改为真实换行 "\n"（否则推送无换行、分块失效）
  ② baostock 单连接非线程安全：DataManager 加 threading.Lock, _bs_query 串行化（akshare 兜底仍并行）
  ③ compute_vcp_score / compute_stop_loss 的 groupby(axis=1) 改为 np.maximum（新版 pandas 弃用 axis=1）
设计: 零可视化依赖 / 向量化核心 / VCP收缩计数 / 大盘过滤 / ATR止损 / 回测 / ServerChan推送
依赖: pandas numpy scipy akshare baostock tqdm pyarrow scikit-learn
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import random
import logging
import threading
import traceback
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import akshare as ak
    AKSHARE_OK = True
except Exception:
    AKSHARE_OK = False

try:
    import baostock as bs
    BAOSTOCK_OK = True
except Exception:
    BAOSTOCK_OK = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import pyarrow
    PYARROW_OK = True
except Exception:
    PYARROW_OK = False

if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ====================== 配置中心 ======================
@dataclass
class ScreenerConfig:
    LOOKBACK_DAYS: int = 500
    MIN_REQUIRED: int = 252
    VCP_LOOKBACK: int = 120
    NUM_THREADS: int = 8
    SLEEP: float = 0.15
    AK_TIMEOUT: int = 25
    STAGE2_MIN_ABOVE_52W_LOW: float = 0.30
    STAGE2_MAX_DIST_52W_HIGH: float = 0.25
    VCP_SCORE_MIN: int = 5
    VCP_MIN_CONTRACTIONS: int = 2
    VCP_MAX_CONTRACTIONS: int = 4
    VCP_MAX_CONTRACTION_DEPTH: float = 0.35
    RS_RATING_MIN: float = 70.0
    TREND_SCORE_MIN: int = 7
    MARKET_REGIME_REQUIRED: bool = True
    RVOL_THRESHOLD: float = 1.5
    BREAKOUT_TOLERANCE: float = 0.995
    KC_TIGHTEN_PCT: float = 0.85
    ATR_TIGHTEN_PCT: float = 0.80
    VOL_DRY_RATIO: float = 0.65
    PRICE_RANGE_TIGHTEN: float = 0.60
    ANOMALY_CONTAMINATION: float = 0.08
    KEEP_PREFIX: Tuple[str, ...] = ("0", "3", "6")
    EXCLUDE_NAME: Tuple[str, ...] = ("ST", "退", "*ST", "DR", "N")
    MIN_PRICE: float = 3.0
    PRE_AMOUNT_MIN: float = 5.0e7
    PRE_TURNOVER_MIN: float = 0.3
    MAX_POSITION_RISK_PCT: float = 0.02
    ATR_STOP_MULTIPLIER: float = 2.5
    SCAN_LIMIT: int = 0
    OUTPUT_DIR: str = "output"
    PUSH_TOP: int = 15
    CLUSTER_TOP: int = 8
    HOT_SECTOR_TOP: int = 10
    HOT_SECTOR_MIN_PCT: float = 1.0
    CACHE_STALE_DAYS: int = 3
    SERVERCHAN_KEY: str = ""
    BACKTEST_MONTHS: int = 6
    BACKTEST_HOLD_DAYS: int = 20

    def __post_init__(self):
        for k in self.__dataclass_fields__:
            v = os.environ.get(k)
            if v is not None:
                t = type(getattr(self, k))
                if t == bool:
                    setattr(self, k, v.lower() in ("1", "true", "yes", "on"))
                elif t == tuple:
                    setattr(self, k, tuple(v.split(",")))
                else:
                    setattr(self, k, t(v))
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR = self.OUTPUT_DIR / "price_cache"
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ====================== 日志系统 ======================
def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("VCPGitHub")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / f"vcp_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ====================== 数据管理层 ======================
class DataManager:
    def __init__(self, cfg: ScreenerConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._bs_logged = False
        self._bs_lock = threading.Lock()   # 修复②: baostock 非线程安全, 串行化
        self.name_map: Dict[str, str] = {}
        self.industry_map: Dict[str, str] = {}

    def _bs_login(self, retries: int = 5) -> bool:
        if self._bs_logged:
            return True
        if not BAOSTOCK_OK:
            return False
        for i in range(retries):
            try:
                lg = bs.login()
                if getattr(lg, "error_code", "1") == "0":
                    self._bs_logged = True
                    return True
                self.logger.warning(f"baostock login failed({getattr(lg, 'error_msg', '')}), retry {i+1}/{retries}")
            except Exception as e:
                self.logger.warning(f"baostock login exception: {e}, retry {i+1}/{retries}")
            time.sleep(2 * (i + 1))
        return False

    def _bs_logout(self):
        if not self._bs_logged:
            return
        try:
            bs.logout()
        except Exception as e:
            self.logger.warning(f"baostock logout exception: {e}")
        finally:
            self._bs_logged = False

    def _bs_query(self, code: str, fields: str, start_date: str, timeout: int = 25) -> Optional[pd.DataFrame]:
        if not self._bs_logged:
            return None
        pref = ("sh." if code[:1] in ("6", "9") else "sz.") + code
        def _do():
            rs = bs.query_history_k_data_plus(pref, fields, start_date=start_date, adjustflag="2")
            return rs.get_data()
        try:
            with self._bs_lock:   # 修复②: 串行化 baostock 查询
                with ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(_do).result(timeout=timeout)
        except Exception as e:
            self.logger.debug(f"baostock query {code} failed: {e}")
            return None

    def _cache_path(self, code: str) -> Path:
        safe = code.replace(".", "_").replace("/", "_")
        return self.cfg.CACHE_DIR / f"{safe}.parquet"

    def load_cache(self, code: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(code)
        if not path.exists() or not PYARROW_OK:
            return None
        try:
            df = pd.read_parquet(path)
            if len(df) < self.cfg.MIN_REQUIRED:
                return None
            last_date = pd.to_datetime(df["date"]).max()
            if (datetime.now() - last_date).days > self.cfg.CACHE_STALE_DAYS:
                return None
            return df
        except Exception:
            return None

    def save_cache(self, code: str, df: pd.DataFrame):
        if not PYARROW_OK:
            return
        try:
            keep = ["date", "open", "high", "low", "close", "volume"]
            cols = [c for c in keep if c in df.columns]
            df[cols].to_parquet(self._cache_path(code), index=False)
        except Exception as e:
            self.logger.debug(f"cache write failed {code}: {e}")

    def fetch_hist(self, code: str) -> Optional[pd.DataFrame]:
        cached = self.load_cache(code)
        if cached is not None:
            return cached
        sd = (datetime.now() - timedelta(days=self.cfg.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        sy = sd.replace("-", "")
        end_str = datetime.now().strftime("%Y%m%d")
        if self._bs_logged:
            try:
                d = self._bs_query(code, "date,open,high,low,close,volume", sd)
                if d is not None and not d.empty:
                    for c in ["open", "high", "low", "close", "volume"]:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                    d["date"] = pd.to_datetime(d["date"])
                    d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                    if len(d) >= self.cfg.MIN_REQUIRED:
                        self.save_cache(code, d)
                        return d
            except Exception:
                pass
        if not AKSHARE_OK:
            return None
        for attempt in range(2):
            try:
                d = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=sy, end_date=end_str, adjust="qfq")
                if d is not None and not d.empty:
                    d = d.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
                    for c in ["open", "high", "low", "close", "volume"]:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                    d["date"] = pd.to_datetime(d["date"])
                    d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                    if len(d) >= self.cfg.MIN_REQUIRED:
                        self.save_cache(code, d)
                        return d
            except Exception as e:
                self.logger.debug(f"[hist] {code} akshare attempt {attempt+1} failed: {e}")
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
        return None

    def load_universe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        if self._bs_login():
            try:
                ind = bs.query_stock_industry().get_data()
                if ind is not None and not ind.empty and "code" in ind.columns:
                    self.industry_map = dict(zip(ind["code"], ind["industry"].fillna("")))
                    self.logger.info(f"baostock industry table: {len(self.industry_map)} entries")
                df = bs.query_stock_basic().get_data()
            except Exception as e:
                self.logger.warning(f"baostock list exception: {e}")
            finally:
                self._bs_logout()
        if df is None or df.empty or "code" not in df.columns:
            self.logger.info("baostock list invalid, fallback to akshare...")
            for attempt in range(3):
                try:
                    d = ak.stock_info_a_code_name()
                    if d is not None and not d.empty and "code" in d.columns:
                        nc = "name" if "name" in d.columns else d.columns[1]
                        d = d[["code", nc]].copy()
                        d.columns = ["code", "code_name"]
                        d["code"] = d["code"].astype(str).str.zfill(6)
                        d["code"] = d["code"].apply(lambda c: ("sh." if c[:1] in ("6", "9") else "sz.") + c)
                        d["type"] = "1"
                        d["status"] = "1"
                        df = d
                        break
                except Exception as e:
                    self.logger.warning(f"akshare list attempt {attempt+1} failed: {e}")
                time.sleep(2 + attempt)
        if df is None or df.empty:
            raise RuntimeError("Cannot get stock universe")
        df = df[df["code"].str.startswith(("sh.", "sz.")) & (df["type"] == "1") & (df["status"] == "1")].copy()
        df = df[~df["code_name"].astype(str).str.contains("ST|退|\\*ST", na=False, regex=True)]
        if "code_name" not in df.columns:
            df = df.rename(columns={df.columns[1]: "code_name"})
        self.name_map = dict(zip(df["code"], df["code_name"]))
        return df[["code", "code_name"]].rename(columns={"code_name": "name"})

    def fetch_market_index(self, symbol: str = "000300") -> Optional[pd.DataFrame]:
        cached = self.load_cache(f"idx_{symbol}")
        if cached is not None:
            return cached
        if not AKSHARE_OK:
            return None
        try:
            df = ak.index_zh_a_hist(symbol=symbol, period="daily")
            if df is not None and not df.empty:
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                self.save_cache(f"idx_{symbol}", df)
                return df
        except Exception as e:
            self.logger.warning(f"market index fetch failed: {e}")
        return None

    def fetch_industry_heat(self) -> pd.DataFrame:
        if not AKSHARE_OK:
            return pd.DataFrame()
        for i in range(3):
            try:
                d = ak.stock_board_industry_name_em()
                if d is not None and not d.empty:
                    return d
            except Exception as e:
                self.logger.debug(f"industry heat attempt {i+1} failed: {e}")
            time.sleep(2 + i)
        return pd.DataFrame()

    def fetch_industry(self, symbol: str) -> str:
        if not AKSHARE_OK:
            return "—"
        for attempt in range(2):
            try:
                info = ak.stock_individual_info_em(symbol=symbol)
                if info is not None and not info.empty and "item" in info.columns:
                    row = info[info["item"].isin(["行业", "所属行业"])]
                    if not row.empty:
                        return str(row.iloc[0]["value"])
            except Exception:
                time.sleep(1 + attempt)
        return "—"


# ====================== 向量化计算引擎 ======================
class VectorizedEngine:
    def __init__(self, cfg: ScreenerConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def build_wide_frames(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close, high, low, volume = {}, {}, {}, {}
        for code, df in stock_data.items():
            if df is None or len(df) < self.cfg.MIN_REQUIRED:
                continue
            df = df.set_index("date").sort_index()
            close[code] = df["close"]
            high[code] = df["high"]
            low[code] = df["low"]
            volume[code] = df["volume"]
        if not close:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(close), pd.DataFrame(high), pd.DataFrame(low), pd.DataFrame(volume)

    def compute_rs_rating(self, close: pd.DataFrame) -> pd.Series:
        r12 = close.pct_change(252)
        r6 = close.pct_change(126)
        r3 = close.pct_change(63)
        r1 = close.pct_change(21)
        rs = (0.4 * r12) + (0.2 * r6) + (0.2 * r3) + (0.2 * r1)
        return rs.iloc[-1].rank(pct=True) * 100

    def compute_trend_template(self, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.Series:
        sma50 = close.rolling(50).mean()
        sma150 = close.rolling(150).mean()
        sma200 = close.rolling(200).mean()
        high_52w = high.rolling(252).max()
        low_52w = low.rolling(252).min()
        sma200_shift = sma200.shift(30)
        c = close.iloc[-1]
        score = (
            (c > sma50.iloc[-1]).astype(int) +
            (c > sma150.iloc[-1]).astype(int) +
            (c > sma200.iloc[-1]).astype(int) +
            (sma50.iloc[-1] > sma150.iloc[-1]).astype(int) +
            (sma50.iloc[-1] > sma200.iloc[-1]).astype(int) +
            (sma150.iloc[-1] > sma200.iloc[-1]).astype(int) +
            (c >= low_52w.iloc[-1] * (1 + self.cfg.STAGE2_MIN_ABOVE_52W_LOW)).astype(int) +
            (c >= high_52w.iloc[-1] * (1 - self.cfg.STAGE2_MAX_DIST_52W_HIGH)).astype(int) +
            (sma200.iloc[-1] > sma200_shift.iloc[-1]).astype(int)
        )
        return score

    @staticmethod
    def _tr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        """真实波幅 TR（修复③: 用 np.maximum 逐元素, 避免 groupby(axis=1) 弃用）"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        return pd.DataFrame(np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)),
                            index=tr1.index, columns=tr1.columns)

    def compute_vcp_score(self, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        tr = self._tr(high, low, close)
        atr14 = tr.rolling(14).mean()
        atr10 = tr.rolling(10).mean()
        ema20 = close.ewm(span=20).mean()
        kc_width = (4 * atr14) / ema20
        kc_w20 = kc_width.rolling(20).mean()
        kc_w40 = kc_width.rolling(40).mean()
        kc_contracting = kc_w20.iloc[-1] < kc_w40.iloc[-1] * self.cfg.KC_TIGHTEN_PCT
        atr_earlier = atr14.iloc[-70:-10].mean() if len(atr14) > 70 else atr14.mean()
        atr_contracting = atr10.iloc[-1] < atr_earlier * self.cfg.ATR_TIGHTEN_PCT
        vol_avg5 = volume.rolling(5).mean()
        vol_avg50 = volume.rolling(50).mean()
        vol_dry = vol_avg5.iloc[-1] < vol_avg50.iloc[-1] * self.cfg.VOL_DRY_RATIO
        range_20 = (high.rolling(20).max() - low.rolling(20).min()) / close
        range_60 = (high.rolling(60).max() - low.rolling(60).min()) / close
        price_tight = range_20.iloc[-1] < range_60.iloc[-1] * self.cfg.PRICE_RANGE_TIGHTEN
        low_20 = low.rolling(20).min()
        low_40 = low.rolling(40).min()
        higher_lows = low_20.iloc[-1] >= low_40.iloc[-21] * 0.98
        score = (
            kc_contracting.astype(int) * 3 +
            atr_contracting.astype(int) * 2 +
            vol_dry.astype(int) * 2 +
            price_tight.astype(int) * 2 +
            higher_lows.astype(int) * 1
        )
        return pd.DataFrame({
            "VCP_Score": score,
            "KC_Contracting": kc_contracting,
            "ATR_Contracting": atr_contracting,
            "Vol_Dry": vol_dry,
            "Price_Tight": price_tight,
            "Higher_Lows": higher_lows
        })

    def compute_contraction_count(self, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
        from scipy.signal import argrelextrema
        results = []
        for code in close.columns:
            c = close[code].dropna().iloc[-lookback:]
            if len(c) < 40:
                results.append({"Contraction_Count": 0, "Max_Contraction": 0, "Avg_Contraction": 0})
                continue
            highs_idx = argrelextrema(c.values, np.greater, order=10)[0]
            if len(highs_idx) < 2:
                results.append({"Contraction_Count": 0, "Max_Contraction": 0, "Avg_Contraction": 0})
                continue
            contractions = []
            for i in range(1, len(highs_idx)):
                peak1 = c.iloc[highs_idx[i-1]]
                trough = c.iloc[highs_idx[i-1]:highs_idx[i]].min()
                depth = (peak1 - trough) / peak1 if peak1 > 0 else 0
                if depth > 0.05:
                    contractions.append(depth)
            results.append({
                "Contraction_Count": len(contractions),
                "Max_Contraction": max(contractions) if contractions else 0,
                "Avg_Contraction": np.mean(contractions) if contractions else 0
            })
        return pd.DataFrame(results, index=close.columns)

    def compute_breakout(self, close: pd.DataFrame, high: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        base_high = high.rolling(60).max()
        pivot = base_high.iloc[-1]
        breakout = close.iloc[-1] > pivot * self.cfg.BREAKOUT_TOLERANCE
        down_days = close.diff() < 0
        down_vol = volume.where(down_days)
        max_down_vol = down_vol.rolling(10).max()
        pocket_pivot = volume.iloc[-1] > max_down_vol.iloc[-1]
        avg_vol50 = volume.rolling(50).mean()
        rvol = volume.iloc[-1] / avg_vol50.iloc[-1]
        valid_breakout = breakout & ((rvol > self.cfg.RVOL_THRESHOLD) | pocket_pivot)
        near_pivot = close.iloc[-1] >= pivot * 0.97
        return pd.DataFrame({
            "Pivot": pivot,
            "Breakout": breakout,
            "PocketPivot": pocket_pivot,
            "RVOL": rvol,
            "Valid_Breakout": valid_breakout,
            "Near_Pivot": near_pivot
        })

    def compute_anomaly_filter(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
        if not SKLEARN_OK:
            returns = close.pct_change().iloc[-60:]
            vol_norm = volume / volume.rolling(50).mean()
            max_gap = returns.abs().max()
            vol_spike = vol_norm.iloc[-1]
            return (max_gap < 0.15) & (vol_spike < 5.0)
        returns = close.pct_change().iloc[-60:]
        vol_norm = volume / volume.rolling(50).mean()
        features = pd.DataFrame({
            "volatility": returns.std(),
            "volume_spike": vol_norm.iloc[-1],
            "max_gap": returns.abs().max(),
            "skew": returns.skew()
        }).fillna(0)
        if len(features) < 10:
            return pd.Series(True, index=features.index)
        clf = IsolationForest(contamination=self.cfg.ANOMALY_CONTAMINATION, random_state=42, n_estimators=100)
        pred = clf.fit_predict(features)
        return pd.Series(pred == 1, index=features.index)

    def compute_market_regime(self, market_close: pd.Series) -> bool:
        if len(market_close) < 200:
            return False
        sma50 = market_close.rolling(50).mean().iloc[-1]
        sma150 = market_close.rolling(150).mean().iloc[-1]
        sma200 = market_close.rolling(200).mean().iloc[-1]
        sma200_prev = market_close.rolling(200).mean().shift(30).iloc[-1]
        return market_close.iloc[-1] > sma50 > sma150 > sma200 and sma200 > sma200_prev

    def compute_stop_loss(self, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
        tr = self._tr(high, low, close)
        atr14 = tr.rolling(14).mean().iloc[-1]
        latest_close = close.iloc[-1]
        stop_price = latest_close - atr14 * self.cfg.ATR_STOP_MULTIPLIER
        risk_per_share = latest_close - stop_price
        account_value = 1_000_000
        max_risk = account_value * self.cfg.MAX_POSITION_RISK_PCT
        shares = (max_risk / risk_per_share).where(risk_per_share > 0, 0).astype(int)
        position_value = shares * latest_close
        return pd.DataFrame({
            "Stop_Price": stop_price,
            "ATR14": atr14,
            "Risk_Per_Share": risk_per_share,
            "Suggest_Shares": shares,
            "Position_Value": position_value
        })


# ====================== 回测引擎（纯数值，无图表）======================
class BacktestEngine:
    def __init__(self, cfg: ScreenerConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def run_rolling_backtest(self, stock_data: Dict[str, pd.DataFrame], signal_dates: List[str]) -> pd.DataFrame:
        results = []
        engine = VectorizedEngine(self.cfg, self.logger)
        for sig_date in tqdm(signal_dates, desc="backtest dates", unit="day"):
            sig_dt = pd.to_datetime(sig_date)
            hist_data = {}
            for code, df in stock_data.items():
                mask = df["date"] <= sig_dt
                if mask.sum() < self.cfg.MIN_REQUIRED:
                    continue
                hist_data[code] = df[mask].copy()
            if len(hist_data) < 10:
                continue
            C, H, L, V = engine.build_wide_frames(hist_data)
            if C.empty:
                continue
            rs = engine.compute_rs_rating(C)
            trend = engine.compute_trend_template(C, H, L)
            vcp = engine.compute_vcp_score(C, H, L, V)
            bo = engine.compute_breakout(C, H, V)
            signals = (
                (rs >= self.cfg.RS_RATING_MIN) &
                (trend >= self.cfg.TREND_SCORE_MIN) &
                (vcp["VCP_Score"] >= self.cfg.VCP_SCORE_MIN) &
                bo["Valid_Breakout"]
            )
            for code in signals[signals].index:
                entry_price = C[code].iloc[-1]
                full_df = stock_data.get(code)
                if full_df is None:
                    continue
                future = full_df[full_df["date"] > sig_dt].head(self.cfg.BACKTEST_HOLD_DAYS)
                if len(future) < self.cfg.BACKTEST_HOLD_DAYS:
                    continue
                exit_price = future.iloc[-1]["close"]
                return_pct = (exit_price - entry_price) / entry_price * 100
                max_price = future["high"].max()
                max_return = (max_price - entry_price) / entry_price * 100
                min_price = future["low"].min()
                max_drawdown = (min_price - entry_price) / entry_price * 100
                results.append({
                    "date": sig_date, "code": code, "entry": entry_price,
                    "exit": exit_price, "return_pct": return_pct,
                    "max_return": max_return, "max_drawdown": max_drawdown,
                    "hold_days": self.cfg.BACKTEST_HOLD_DAYS
                })
        return pd.DataFrame(results)


# ====================== 推送系统 ======================
class Notifier:
    def __init__(self, cfg: ScreenerConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.key = cfg.SERVERCHAN_KEY or os.environ.get("SERVERCHAN_KEY") or os.environ.get("SENDKEY", "")

    def send(self, title: str, content: str) -> bool:
        if not self.key:
            return False
        LIMIT = 3800
        lines = content.split("\n")   # 修复①: 真实换行
        chunks, cur, cur_len = [], [], 0
        for ln in lines:
            lnlen = len(ln) + 1
            if cur_len + lnlen > LIMIT and cur:
                chunks.append("\n".join(cur))   # 修复①
                cur, cur_len = [], 0
            cur.append(ln)
            cur_len += lnlen
        if cur:
            chunks.append("\n".join(cur))   # 修复①
        ok = True
        for i, ch in enumerate(chunks):
            t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
            ok = self._send_one(t, ch) and ok
            if i < len(chunks) - 1:
                time.sleep(1)
        self.logger.info(f"push complete: {len(chunks)} messages")
        return ok

    def _send_one(self, title: str, content: str) -> bool:
        try:
            from serverchan_sdk import sc_send
            sc_send(self.key, title, content)
            return True
        except Exception:
            pass
        try:
            import requests
            r = requests.post(f"https://sctapi.ftqq.com/{self.key}.send", data={"title": title, "desp": content}, timeout=15)
            return r.json().get("code") == 0
        except Exception as e:
            self.logger.warning(f"push failed: {e}")
            return False


# ====================== 主控制器 ======================
class VCPScreenerGitHub:
    def __init__(self, cfg: Optional[ScreenerConfig] = None):
        self.cfg = cfg or ScreenerConfig()
        self.logger = setup_logging(self.cfg.OUTPUT_DIR)
        self.data_mgr = DataManager(self.cfg, self.logger)
        self.engine = VectorizedEngine(self.cfg, self.logger)
        self.backtest = BacktestEngine(self.cfg, self.logger)
        self.notifier = Notifier(self.cfg, self.logger)
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.market_data: Optional[pd.DataFrame] = None
        self.cluster: List[Tuple[str, int]] = []
        self.hot: List[Tuple[str, float]] = []
        self.bt_df: Optional[pd.DataFrame] = None

    def run(self):
        self.logger.info("=" * 70)
        self.logger.info(f"📐 VCP GitHub Actions Screener | {datetime.now():%Y-%m-%d %H:%M}")
        self.logger.info(f"Lookback={self.cfg.LOOKBACK_DAYS}d | Limit={'All' if not self.cfg.SCAN_LIMIT else self.cfg.SCAN_LIMIT} | Vectorized | Backtest | Push")
        self.logger.info("=" * 70)

        # 1. Market regime
        if self.cfg.MARKET_REGIME_REQUIRED:
            self.logger.info("\n[1/6] Checking market regime...")
            self.market_data = self.data_mgr.fetch_market_index("000300")
            if self.market_data is not None and not self.market_data.empty:
                market_stage2 = self.engine.compute_market_regime(self.market_data.set_index("date")["close"])
                self.logger.info(f"  CSI300 Stage2: {'✓ OK' if market_stage2 else '✗ Weak market, observation only'}")
            else:
                self.logger.warning("  Market data failed, skip regime filter")
                market_stage2 = True
        else:
            market_stage2 = True

        # 2. Universe
        self.logger.info("\n[2/6] Loading stock universe...")
        universe = self.data_mgr.load_universe()
        codes = universe["code"].tolist()
        self.logger.info(f"  Initial universe: {len(codes)} stocks")

        # 3. Snapshot prefilter
        codes = self._snapshot_prefilter(codes)
        if self.cfg.SCAN_LIMIT and len(codes) > self.cfg.SCAN_LIMIT:
            codes = codes[:self.cfg.SCAN_LIMIT]
        self.logger.info(f"  After prefilter: {len(codes)} stocks")

        # 4. Fetch historical data
        self.logger.info(f"\n[3/6] Fetching historical data ({len(codes)} stocks, {self.cfg.NUM_THREADS} threads)...")
        self.stock_data = self._fetch_all(codes)
        self.logger.info(f"  Success: {len(self.stock_data)} stocks")
        if not self.stock_data:
            self.logger.error("No valid data")
            return

        # 5. Vectorized scan
        self.logger.info("\n[4/6] Vectorized core calculation...")
        candidates = self._vectorized_scan(market_stage2)
        if candidates.empty:
            self.logger.info("\nNo VCP candidates found (strict criteria, 0 hits is normal)")
            self._save_empty_result()
            return

        # 6. Enrich
        self.logger.info(f"\n[5/6] Enriching industry & hot sectors ({len(candidates)} candidates)...")
        candidates = self._enrich(candidates)

        # 7. Backtest (numeric only)
        self.logger.info("\n[6/6] Running rolling backtest...")
        self._run_backtest()

        # Save & push
        self._save_and_push(candidates)
        self.logger.info("\n✅ All done")

    def _snapshot_prefilter(self, codes: List[str]) -> List[str]:
        if not AKSHARE_OK:
            return codes
        try:
            spot = ak.stock_zh_a_spot_em()
            if spot is None or spot.empty or "代码" not in spot.columns:
                return codes
            spot["代码"] = spot["代码"].astype(str).str.zfill(6)
            for c in ["最新价", "成交额", "换手率"]:
                if c in spot.columns:
                    spot[c] = pd.to_numeric(spot[c], errors="coerce")
            m = (
                spot["代码"].str.startswith(self.cfg.KEEP_PREFIX)
                & ~spot["名称"].astype(str).str.contains("|".join(self.cfg.EXCLUDE_NAME), na=False, regex=True)
                & (spot["最新价"] >= self.cfg.MIN_PRICE)
            )
            if "成交额" in spot.columns:
                m &= spot["成交额"] >= self.cfg.PRE_AMOUNT_MIN
            if "换手率" in spot.columns:
                m &= spot["换手率"] >= self.cfg.PRE_TURNOVER_MIN
            keep = set(spot.loc[m, "代码"])
            out = [c for c in codes if c.split(".")[-1].zfill(6) in keep]
            self.logger.info(f"  Snapshot prefilter: {len(codes)} -> {len(out)} stocks")
            return out if out else codes
        except Exception as e:
            self.logger.warning(f"Snapshot prefilter failed: {e}")
            return codes

    def _fetch_all(self, codes: List[str]) -> Dict[str, pd.DataFrame]:
        self.data_mgr._bs_login()
        tasks = [(c, self.data_mgr.name_map.get(c, "")) for c in codes]
        results = {}
        fail = 0
        with ThreadPoolExecutor(max_workers=self.cfg.NUM_THREADS) as ex:
            futures = {ex.submit(self._fetch_one, c): c for c, _ in tasks}
            for future in tqdm(futures, desc="fetching data", unit="stock"):
                code = futures[future]
                try:
                    df = future.result(timeout=self.cfg.AK_TIMEOUT)
                    if df is not None:
                        results[code] = df
                    else:
                        fail += 1
                except Exception:
                    fail += 1
        self.logger.info(f"  Success {len(results)}, Failed {fail}")
        return results

    def _fetch_one(self, code: str) -> Optional[pd.DataFrame]:
        try:
            return self.data_mgr.fetch_hist(code.split(".")[-1].zfill(6))
        except Exception:
            return None

    def _vectorized_scan(self, market_stage2: bool) -> pd.DataFrame:
        C, H, L, V = self.engine.build_wide_frames(self.stock_data)
        if C.empty:
            return pd.DataFrame()
        self.logger.info(f"  Wide table: {C.shape[0]} days x {C.shape[1]} stocks")
        normal_mask = self.engine.compute_anomaly_filter(C, V)
        rs_rating = self.engine.compute_rs_rating(C)
        trend_score = self.engine.compute_trend_template(C, H, L)
        vcp_df = self.engine.compute_vcp_score(C, H, L, V)
        contraction_df = self.engine.compute_contraction_count(C, H, L, lookback=self.cfg.VCP_LOOKBACK)
        bo_df = self.engine.compute_breakout(C, H, V)
        stop_df = self.engine.compute_stop_loss(C, H, L)
        result = pd.DataFrame({
            "代码": C.columns,
            "名称": [self.data_mgr.name_map.get(c, "") for c in C.columns],
            "最新价": C.iloc[-1].values,
            "RS_Rating": rs_rating.reindex(C.columns).fillna(0).values,
            "Trend_Score": trend_score.reindex(C.columns).fillna(0).values,
            "VCP_Score": vcp_df["VCP_Score"].reindex(C.columns).fillna(0).values,
            "Contraction_Count": contraction_df["Contraction_Count"].reindex(C.columns).fillna(0).values,
            "Max_Contraction": contraction_df["Max_Contraction"].reindex(C.columns).fillna(0).values,
            "Avg_Contraction": contraction_df["Avg_Contraction"].reindex(C.columns).fillna(0).values,
            "KC_Contracting": vcp_df["KC_Contracting"].reindex(C.columns).fillna(False).values,
            "ATR_Contracting": vcp_df["ATR_Contracting"].reindex(C.columns).fillna(False).values,
            "Vol_Dry": vcp_df["Vol_Dry"].reindex(C.columns).fillna(False).values,
            "Price_Tight": vcp_df["Price_Tight"].reindex(C.columns).fillna(False).values,
            "Higher_Lows": vcp_df["Higher_Lows"].reindex(C.columns).fillna(False).values,
            "Pivot": bo_df["Pivot"].reindex(C.columns).fillna(0).values,
            "Breakout": bo_df["Breakout"].reindex(C.columns).fillna(False).values,
            "PocketPivot": bo_df["PocketPivot"].reindex(C.columns).fillna(False).values,
            "RVOL": bo_df["RVOL"].reindex(C.columns).fillna(0).values,
            "Valid_Breakout": bo_df["Valid_Breakout"].reindex(C.columns).fillna(False).values,
            "Near_Pivot": bo_df["Near_Pivot"].reindex(C.columns).fillna(False).values,
            "Stop_Price": stop_df["Stop_Price"].reindex(C.columns).fillna(0).values,
            "ATR14": stop_df["ATR14"].reindex(C.columns).fillna(0).values,
            "Risk_Per_Share": stop_df["Risk_Per_Share"].reindex(C.columns).fillna(0).values,
            "Suggest_Shares": stop_df["Suggest_Shares"].reindex(C.columns).fillna(0).values,
            "Position_Value": stop_df["Position_Value"].reindex(C.columns).fillna(0).values,
            "Normal": normal_mask.reindex(C.columns).fillna(False).values,
            "Market_Stage2": market_stage2
        })
        candidates = result[
            result["Normal"] &
            (result["RS_Rating"] >= self.cfg.RS_RATING_MIN) &
            (result["Trend_Score"] >= self.cfg.TREND_SCORE_MIN) &
            (result["VCP_Score"] >= self.cfg.VCP_SCORE_MIN) &
            (result["Contraction_Count"] >= self.cfg.VCP_MIN_CONTRACTIONS) &
            (result["Contraction_Count"] <= self.cfg.VCP_MAX_CONTRACTIONS) &
            (result["Max_Contraction"] <= self.cfg.VCP_MAX_CONTRACTION_DEPTH)
        ].copy()
        candidates = candidates.sort_values(
            ["Valid_Breakout", "VCP_Score", "RS_Rating", "Contraction_Count"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        self.logger.info(f"  Candidates: {len(candidates)} stocks")
        return candidates

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        def _get_ind(row):
            sym = row["代码"].split(".")[-1].zfill(6)
            return self.data_mgr.fetch_industry(sym)
        with ThreadPoolExecutor(max_workers=self.cfg.NUM_THREADS) as ex:
            industries = list(ex.map(_get_ind, df.to_dict("records")))
        df["行业"] = industries
        labeled = df[df["行业"].notna() & (df["行业"] != "—")]
        self.cluster = []
        if not labeled.empty:
            self.cluster = [(n, int(c)) for n, c in labeled["行业"].value_counts().head(self.cfg.CLUSTER_TOP).items()]
        self.logger.info(f"  VCP sectors: {self.cluster or 'None'}")
        heat = self.data_mgr.fetch_industry_heat()
        self.hot = []
        if not heat.empty and "板块名称" in heat.columns and "涨跌幅" in heat.columns:
            h = heat.copy()
            h["_chg"] = pd.to_numeric(h["涨跌幅"], errors="coerce")
            h = h[h["_chg"] >= self.cfg.HOT_SECTOR_MIN_PCT].sort_values("_chg", ascending=False)
            self.hot = [(str(row["板块名称"]), round(float(row["涨跌幅"]), 2)) for _, row in h.head(self.cfg.HOT_SECTOR_TOP).iterrows()]
        self.logger.info(f"  Hot sectors: {', '.join(f'{n}({c}%)' for n, c in self.hot) or 'None'}")
        hot_names = [n for n, _ in self.hot]
        df["resonance"] = False
        df["resonance_sector"] = ""
        for idx, row in df.iterrows():
            sector = row.get("行业", "")
            if not sector or sector == "—":
                continue
            for h_name in hot_names:
                if h_name and (h_name == sector or h_name in sector or sector in h_name):
                    df.at[idx, "resonance"] = True
                    df.at[idx, "resonance_sector"] = h_name
                    break
        n_reso = df["resonance"].sum()
        self.logger.info(f"  VCP + hot sector: {n_reso} stocks")
        df = df.sort_values(["resonance", "Valid_Breakout", "VCP_Score", "RS_Rating"], ascending=[False, False, False, False]).reset_index(drop=True)
        return df

    def _run_backtest(self):
        if len(self.stock_data) < 50:
            return
        end = datetime.now()
        signal_dates = []
        for i in range(self.cfg.BACKTEST_MONTHS, 0, -1):
            d = end - timedelta(days=30*i)
            signal_dates.append(d.strftime("%Y-%m-%d"))
        self.bt_df = self.backtest.run_rolling_backtest(self.stock_data, signal_dates)
        if self.bt_df is not None and not self.bt_df.empty:
            win_rate = (self.bt_df["return_pct"] > 0).mean() * 100
            avg_return = self.bt_df["return_pct"].mean()
            avg_max_return = self.bt_df["max_return"].mean()
            avg_max_dd = self.bt_df["max_drawdown"].mean()
            self.logger.info(f"  Backtest: {len(self.bt_df)} trades | Win rate {win_rate:.1f}% | Avg return {avg_return:.2f}% | Avg max return {avg_max_return:.2f}% | Avg max DD {avg_max_dd:.2f}%")
        else:
            self.logger.info("  Backtest: insufficient data")

    def _save_and_push(self, df: pd.DataFrame):
        tag = datetime.now().strftime("%Y%m%d")
        csv_path = self.cfg.OUTPUT_DIR / f"vcp_github_{tag}.csv"
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"\n📁 CSV saved: {csv_path}")
        except Exception as e:
            self.logger.error(f"CSV save failed: {e}")
        json_path = self.cfg.OUTPUT_DIR / f"vcp_github_{tag}.json"
        try:
            bt_records = []
            bt_summary = {}
            if self.bt_df is not None and not self.bt_df.empty:
                bt_records = self.bt_df.to_dict("records")
                bt_summary = {
                    "trades": len(self.bt_df),
                    "win_rate": float((self.bt_df["return_pct"] > 0).mean() * 100),
                    "avg_return": float(self.bt_df["return_pct"].mean()),
                    "avg_max_return": float(self.bt_df["max_return"].mean()),
                    "avg_max_drawdown": float(self.bt_df["max_drawdown"].mean())
                }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "date": tag,
                    "params": asdict(self.cfg),
                    "cluster": self.cluster,
                    "hot": self.hot,
                    "n": int(len(df)),
                    "n_resonance": int(df["resonance"].sum()) if "resonance" in df.columns else 0,
                    "n_breakout": int(df["Valid_Breakout"].sum()) if "Valid_Breakout" in df.columns else 0,
                    "backtest_summary": bt_summary,
                    "backtest_trades": bt_records,
                    "hits": df.to_dict("records")
                }, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"📁 JSON saved: {json_path}")
        except Exception as e:
            self.logger.error(f"JSON save failed: {e}")
        try:
            disp = df.copy()
            disp.insert(2, "板块", [self._sec_tag(r) for r in disp.to_dict("records")])
            drop_cols = ["resonance", "resonance_sector", "Normal",
                         "KC_Contracting", "ATR_Contracting", "Price_Tight", "Higher_Lows", "Market_Stage2"]
            disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns], errors="ignore")
            print("\n" + disp.head(self.cfg.PUSH_TOP).to_string(index=False))
        except Exception as e:
            self.logger.warning(f"Display error: {e}")
        if self.cfg.SERVERCHAN_KEY:
            try:
                n_reso = int(df["resonance"].sum()) if "resonance" in df.columns else 0
                n_bo = int(df["Valid_Breakout"].sum()) if "Valid_Breakout" in df.columns else 0
                bt_info = ""
                if self.bt_df is not None and not self.bt_df.empty:
                    wr = (self.bt_df["return_pct"] > 0).mean() * 100
                    ar = self.bt_df["return_pct"].mean()
                    bt_info = f" | 回测胜率{wr:.0f}% 均收益{ar:.1f}%"
                title = f"📐 VCP GitHub {len(df)}hits 🎯Hot{n_reso} 💥BO{n_bo}{bt_info}"
                content = self._build_push_content(df)
                self.notifier.send(title, content)
            except Exception as e:
                self.logger.error(f"Push error: {e}")

    def _save_empty_result(self):
        tag = datetime.now().strftime("%Y%m%d")
        json_path = self.cfg.OUTPUT_DIR / f"vcp_github_{tag}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"date": tag, "n": 0, "hits": []}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _sec_tag(self, r: dict) -> str:
        return ("🎯" + r.get("resonance_sector", "")) if r.get("resonance") else (r.get("行业") or "—")

    def _build_push_content(self, df: pd.DataFrame) -> str:
        reso = df[df["resonance"] == True] if "resonance" in df.columns else pd.DataFrame()
        broke = df[df["Valid_Breakout"] == True] if "Valid_Breakout" in df.columns else pd.DataFrame()
        L = [
            f"**📐 VCP GitHub 选股结果** | {len(df)}只命中 🎯风口{len(reso)} 💥突破{len(broke)}",
            "*(加权RS + TrendTemplate + Keltner/ATR/Vol多维度VCP + 收缩计数 + 大盘过滤 + ATR止损 + PocketPivot/RVOL)*",
            ""
        ]
        if self.hot:
            L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in self.hot[:6]))
            L.append("")
        if self.cluster:
            L.append("📐 **VCP蓄势板块**: " + "、".join(f"{n}({c})" for n, c in self.cluster))
            L.append("")
        if self.bt_df is not None and not self.bt_df.empty:
            wr = (self.bt_df["return_pct"] > 0).mean() * 100
            ar = self.bt_df["return_pct"].mean()
            amr = self.bt_df["max_return"].mean()
            amd = self.bt_df["max_drawdown"].mean()
            L.append(f"📊 **回测摘要**(近{self.cfg.BACKTEST_MONTHS}月): 胜率{wr:.1f}% | 均收益{ar:.2f}% | 均最大收益{amr:.2f}% | 均最大回撤{amd:.2f}%")
            L.append("")
        if not broke.empty:
            L.append(f"### 💥 已突破(PocketPivot+放量) 共{len(broke)}只")
            for _, r in broke.head(self.cfg.PUSH_TOP).iterrows():
                L.append(f"- **{r['名称']}({r['代码']})** [{self._sec_tag(r.to_dict())}] "
                        f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 收缩{r['Contraction_Count']}次 "
                        f"RS{r['RS_Rating']:.0f} RVOL{r['RVOL']:.1f} 止损{r['Stop_Price']:.2f} 仓位{r['Position_Value']:,.0f}")
            L.append("")
        if not reso.empty:
            L.append(f"### 🎯 VCP遇风口 共{len(reso)}只")
            for _, r in reso.iterrows():
                L.append(f"- **{r['名称']}({r['代码']})** [🎯{r['resonance_sector']}] "
                        f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 收缩{r['Contraction_Count']}次 "
                        f"RS{r['RS_Rating']:.0f} 突破{'✓' if r['Valid_Breakout'] else '✗'}")
            L.append("")
        L.append(f"### 📐 全部候选 共{len(df)}只")
        for _, r in df.iterrows():
            L.append(f"- **{r['名称']}({r['代码']})** [{self._sec_tag(r.to_dict())}] "
                    f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 收缩{r['Contraction_Count']}次 Trend{r['Trend_Score']} "
                    f"RS{r['RS_Rating']:.0f} 突破{'✓' if r['Valid_Breakout'] else '✗'} "
                    f"PP{'✓' if r['PocketPivot'] else '✗'} RVOL{r['RVOL']:.1f} "
                    f"止损{r['Stop_Price']:.2f}")
        return "\n".join(L)   # 修复①: 真实换行


# ====================== 入口 ======================
def main():
    cfg = ScreenerConfig()
    screener = VCPScreenerGitHub(cfg)
    screener.run()


if __name__ == "__main__":
    main()
# >>>FILE_END_vcp_gh<<<
