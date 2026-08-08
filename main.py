# -*- coding: utf-8 -*-
"""
main.py —— 主策略: 四周期即将站上20日线共振 + 7启动因子确认 + 周线宽松兜底 (融合版)
====================================================================
【融合 main_fusion 替代原主策略】
  主信号: 日/周/月/季 四周期即将站上20日线共振 (CROSS_MODE=price逼近MA20 / ma=MA5金叉MA20)
  启动确认: 7大启动因子加权(放量/均线多头/平台突破/低波压缩/回踩修复/无大回撤/KDJ反弹)
  兜底: 周线宽松信号(严格四周期0命中时兜住候选, 低权重) —— 移植自原主策略, 保证覆盖面
【过滤全部 env 可关】ATR_FILTER / BB_WIDTH_FILTER / VOLUME_CONFIRM / MACD_FILTER / MA20_DIRECTION
【数据】baostock主源+东财兜底(多进程每进程独立登录); 行业 baostock 行业表本地join; 需OHLCV+amount。
⚠️ 四周期严格共振+启动确认条件苛刻, 命中少属正常; 可 MIN_PERIODS=3/关过滤 放宽。
"""
import os
import sys
import time
import random
import requests
import warnings
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
try:
    import baostock as bs
except ImportError:
    raise ImportError("请先安装: pip install baostock")
import akshare as ak

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# ========================= 配置层 =========================
def _env_bool(k, d="1"):
    return os.environ.get(k, d).strip() in ("1", "true", "True")

@dataclass
class Config:
    SCAN_LIMIT: int = int(os.environ.get("SCAN_LIMIT", "0"))
    MIN_PRICE: float = float(os.environ.get("MIN_PRICE", "5"))
    EXCLUDE_ST: bool = True
    CROSS_MODE: str = os.environ.get("CROSS_MODE", "price")
    MIN_PERIODS: int = int(os.environ.get("MIN_PERIODS", "3"))
    DAY_THRESHOLD: float = float(os.environ.get("DAY_THRESHOLD", "0.005"))
    WEEK_THRESHOLD: float = float(os.environ.get("WEEK_THRESHOLD", "0.008"))
    MONTH_THRESHOLD: float = float(os.environ.get("MONTH_THRESHOLD", "0.012"))
    QUARTER_THRESHOLD: float = float(os.environ.get("QUARTER_THRESHOLD", "0.02"))
    WEEKLY_ONLY_THRESHOLD: float = float(os.environ.get("WEEKLY_ONLY_THRESHOLD", "0.015"))
    MA20_DIRECTION: bool = _env_bool("MA20_DIRECTION")
    VOLUME_CONFIRM: bool = _env_bool("VOLUME_CONFIRM")
    VOL_RATIO_MIN: float = float(os.environ.get("VOL_RATIO_MIN", "1.2"))
    MACD_FILTER: bool = _env_bool("MACD_FILTER")
    ATR_FILTER: bool = _env_bool("ATR_FILTER")
    ATR_MIN_PCT: float = float(os.environ.get("ATR_MIN_PCT", "0.015"))
    BB_WIDTH_FILTER: bool = _env_bool("BB_WIDTH_FILTER")
    BB_WIDTH_MIN: float = float(os.environ.get("BB_WIDTH_MIN", "0.03"))
    ENABLE_VOLUME_SURGE: bool = _env_bool("ENABLE_VOLUME_SURGE")
    ENABLE_MA_BULLISH: bool = _env_bool("ENABLE_MA_BULLISH")
    ENABLE_PLATFORM_BREAK: bool = _env_bool("ENABLE_PLATFORM_BREAK")
    ENABLE_LOW_VOL_COMPRESSION: bool = _env_bool("ENABLE_LOW_VOL_COMPRESSION")
    ENABLE_PULLBACK_RECOVERY: bool = _env_bool("ENABLE_PULLBACK_RECOVERY")
    ENABLE_NO_BIG_DROP: bool = _env_bool("ENABLE_NO_BIG_DROP")
    ENABLE_KDJ_BOUNCE: bool = _env_bool("ENABLE_KDJ_BOUNCE")
    VOL_SURGE_RATIO: float = 2.0
    MIN_TURNOVER: float = 2e8
    PLATFORM_LOOKBACK: int = 60
    PLATFORM_DEVIATION: float = 0.05
    COMPRESSION_DAYS: int = 20
    COMPRESSION_ATR_RATIO: float = 0.5
    PULLBACK_MAX_DEPTH: float = 0.08
    BIG_DROP_THRESHOLD: float = -0.07
    KDJ_J_THRESHOLD: int = 30
    W_DAY_GAP: float = 18.0
    W_WEEK_GAP: float = 20.0
    W_MONTH_GAP: float = 20.0
    W_QUARTER_GAP: float = 14.0
    W_EXTRA_PERIOD: float = 3.0
    W_RSI: float = 8.0
    W_TREND: float = 5.0
    W_VOLUME: float = 5.0
    BONUS_VOLUME_SURGE: float = 8.0
    BONUS_MA_BULLISH: float = 10.0
    BONUS_PLATFORM_BREAK: float = 12.0
    BONUS_LOW_VOL_COMPRESSION: float = 10.0
    BONUS_PULLBACK_RECOVERY: float = 8.0
    BONUS_NO_BIG_DROP: float = 5.0
    BONUS_KDJ_BOUNCE: float = 6.0
    DATA_START: str = "20190101"
    NUM_PROCESSES: int = int(os.environ.get("NUM_PROCESSES", "3"))
    SLEEP: float = float(os.environ.get("SLEEP", "0.1"))
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")
    PUSH_TOP: int = int(os.environ.get("PUSH_TOP", "20"))
    LABEL_TOP: int = int(os.environ.get("LABEL_TOP", "200"))
    CLUSTER_TOP: int = int(os.environ.get("CLUSTER_TOP", "8"))
    SERVERCHAN_KEY: str = os.environ.get("SERVERCHAN_KEY", "") or os.environ.get("SENDKEY", "")
    HOT_SECTOR_TOP: int = int(os.environ.get("HOT_SECTOR_TOP", "10"))
    HOT_SECTOR_MIN_PCT: float = float(os.environ.get("HOT_SECTOR_MIN_PCT", "1.0"))

CFG = Config()
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False
_INDUSTRY_MAP: Dict[str, str] = {}
_STRATEGY = None

# ========================= baostock 登录 =========================
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    if _BS_LOGGED:
        return True
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True; return True
        except Exception as e:
            print(f"  baostock登录异常: {e}")
        time.sleep(2 * (i + 1))
    return False

def _bs_logout():
    global _BS_LOGGED
    try:
        if _BS_LOGGED:
            bs.logout()
    except Exception:
        pass
    finally:
        _BS_LOGGED = False

def _init_worker():
    global _BS_LOGGED, _STRATEGY
    time.sleep(random.uniform(0, 2))
    _BS_LOGGED = False
    _bs_login_ok()
    _STRATEGY = MainFusionStrategy()

def send_serverchan(title, content, sendkey=""):
    key = sendkey or CFG.SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur:
        chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        try:
            from serverchan_sdk import sc_send
            ret = sc_send(key, t, ch)
            r_ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        except Exception:
            try:
                r = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": t, "desp": ch}, timeout=15)
                r_ok = r.json().get("code") == 0
            except Exception as e:
                print(f"  requests推送失败: {e}"); r_ok = False
        ok = r_ok and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print("📲 推送完成" + (" ✅" if ok else " ⚠️存在失败"))
    return ok

# ========================= 数据层 (baostock主源+东财兜底) =========================
def get_stock_list():
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock取列表异常: {e}"); stock_df = pd.DataFrame()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                nc = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'; d['status'] = '1'; stock_df = d
        except Exception as e:
            print(f"  akshare取列表失败: {e}")
    if stock_df is None or stock_df.empty:
        return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    if CFG.EXCLUDE_ST:
        stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    return stock_df[['code', 'code_name', 'type', 'status']]

def load_industry():
    global _INDUSTRY_MAP
    if _INDUSTRY_MAP:
        return
    try:
        ind = bs.query_stock_industry().get_data()
        if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
            import re as _re
            for _, row in ind.iterrows():
                s = row['industry']
                s = _re.sub(r'^[A-Z]\d+\s*', '', s.strip()) if s and isinstance(s, str) else "—"
                _INDUSTRY_MAP[row['code']] = s or "—"
            print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock)")
    except Exception as e:
        print(f"  行业表异常: {e}")

def _fetch_baostock(symbol):
    if not _BS_LOGGED:
        return None
    try:
        sd = f"{CFG.DATA_START[:4]}-{CFG.DATA_START[4:6]}-{CFG.DATA_START[6:]}"
        ed = datetime.now().strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(symbol, "date,open,high,low,close,volume,amount",
            start_date=sd, end_date=ed, frequency="d", adjustflag="2")
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close', 'volume']); df = df[df['volume'] > 0].sort_values('date').reset_index(drop=True)
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if 'amount' in df.columns:
            cols.append('amount')
        return df[cols] if len(df) >= 60 else None
    except Exception:
        return None

def _fetch_akshare(sym):
    end = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=CFG.DATA_START, end_date=end, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                                  "收盘": "close", "成交量": "volume", "成交额": "amount"})
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors="coerce")
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            cols = ["date", "open", "high", "low", "close", "volume"]
            if "amount" in d.columns:
                cols.append("amount")
            return d[cols] if len(d) >= 60 else None
        except Exception:
            time.sleep(1 + attempt)
    return None

def get_kline(symbol):
    df = _fetch_baostock(symbol)
    if df is not None and len(df) >= 60:
        return df
    sym = symbol.replace("sh.", "").replace("sz.", "")
    return _fetch_akshare(sym)

def get_hot_sectors():
    for i in range(3):
        try:
            d = ak.stock_board_industry_name_em()
            if d is not None and not d.empty:
                h = d.copy(); h["_chg"] = pd.to_numeric(h["涨跌幅"], errors="coerce")
                h = h[h["_chg"] >= CFG.HOT_SECTOR_MIN_PCT].sort_values("_chg", ascending=False)
                return [(str(row["板块名称"]), round(float(row["_chg"]), 2))
                        for _, row in h.head(CFG.HOT_SECTOR_TOP).iterrows()]
        except Exception as e:
            print(f"  行业热度榜第{i+1}次失败: {e}"); time.sleep(2 + i)
    return []

# ========================= 技术指标 =========================
class Technicals:
    @staticmethod
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean(); avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else None
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = Technicals.ema(series, fast); ema_slow = Technicals.ema(series, slow)
        dif = ema_fast - ema_slow; dea = Technicals.ema(dif, signal)
        macd_hist = 2 * (dif - dea)
        return {"dif": dif.iloc[-1] if len(dif) > 0 else None,
                "dea": dea.iloc[-1] if len(dea) > 0 else None,
                "macd": macd_hist.iloc[-1] if len(macd_hist) > 0 else None}
    @staticmethod
    def kdj(high, low, close, n=9, m1=3, m2=3):
        rsv = (close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min()).replace(0, 1e-9) * 100
        k = rsv.ewm(com=m1-1, adjust=False).mean(); d = k.ewm(com=m2-1, adjust=False).mean()
        j = 3 * k - 2 * d
        return {"k": k.iloc[-1] if not k.empty and pd.notna(k.iloc[-1]) else None,
                "d": d.iloc[-1] if not d.empty and pd.notna(d.iloc[-1]) else None,
                "j": j.iloc[-1] if not j.empty and pd.notna(j.iloc[-1]) else None}
    @staticmethod
    def atr(df, period=14):
        if len(df) < period + 1:
            return None
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    @staticmethod
    def bollinger_width(series, period=20, std_dev=2):
        if len(series) < period:
            return None
        ma = series.rolling(period).mean(); std = series.rolling(period).std()
        upper = ma + std_dev * std; lower = ma - std_dev * std
        width = (upper - lower) / ma
        return width.iloc[-1] if pd.notna(width.iloc[-1]) else None
    @staticmethod
    def resample(df, rule):
        rules = [rule] if isinstance(rule, str) else rule
        for r in rules:
            try:
                s = df.resample(r, on="date")["close"].last().dropna()
                if len(s) > 0:
                    return s
            except Exception:
                continue
        return pd.Series(dtype=float)

# ========================= 启动确认因子 =========================
class LaunchConfirmFactors:
    @staticmethod
    def volume_surge(df):
        if len(df) < 6 or "volume" not in df.columns:
            return False, 0.0
        vol = df["volume"].astype(float); vol_ma5 = vol.rolling(5).mean()
        ratio = vol.iloc[-1] / vol_ma5.iloc[-1] if vol_ma5.iloc[-1] > 0 else 0
        amount_ok = True
        if "amount" in df.columns:
            amount_ok = df["amount"].astype(float).iloc[-1] >= CFG.MIN_TURNOVER
        passed = ratio >= CFG.VOL_SURGE_RATIO and amount_ok
        return passed, (min(ratio / CFG.VOL_SURGE_RATIO, 2.0) if passed else 0.0)
    @staticmethod
    def ma_bullish(df):
        if len(df) < 30:
            return False, 0.0
        close = df["close"].astype(float)
        ma5 = close.rolling(5).mean().iloc[-1]; ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]; ma30 = close.rolling(30).mean().iloc[-1]
        ma30_prev = close.rolling(30).mean().iloc[-2]
        passed = (ma5 > ma10 > ma20 > ma30) and (ma30 > ma30_prev)
        spread = (ma5 - ma30) / ma30 if ma30 > 0 else 0
        return passed, (min(1 + spread * 10, 2.0) if passed else 0.0)
    @staticmethod
    def platform_breakout(df):
        if len(df) < CFG.PLATFORM_LOOKBACK + 5:
            return False, 0.0
        close = df["close"].astype(float); ma60 = close.rolling(60).mean()
        lookback = close.iloc[-CFG.PLATFORM_LOOKBACK:]; ma60_lookback = ma60.iloc[-CFG.PLATFORM_LOOKBACK:]
        deviation = ((lookback - ma60_lookback) / ma60_lookback).abs()
        platform_ratio = (deviation <= CFG.PLATFORM_DEVIATION).sum() / len(lookback)
        breakout = close.iloc[-1] >= ma60.iloc[-1] * 1.02
        vol = df["volume"].astype(float)
        vol_surge = vol.iloc[-1] >= vol.rolling(5).mean().iloc[-1] * 1.5
        passed = platform_ratio >= 0.6 and breakout and vol_surge
        return passed, (min(platform_ratio * 2, 2.0) if passed else 0.0)
    @staticmethod
    def low_vol_compression(df):
        if len(df) < CFG.COMPRESSION_DAYS * 2 + 5:
            return False, 0.0
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = tr.rolling(14).mean()
        recent_atr = atr.iloc[-CFG.COMPRESSION_DAYS:].mean()
        prior_atr = atr.iloc[-CFG.COMPRESSION_DAYS*2:-CFG.COMPRESSION_DAYS].mean()
        compressed = prior_atr > 0 and recent_atr / prior_atr <= CFG.COMPRESSION_ATR_RATIO
        vol = df["volume"].astype(float)
        today_break = close.iloc[-1] >= close.rolling(20).mean().iloc[-1]
        vol_surge = vol.iloc[-1] >= vol.rolling(5).mean().iloc[-1] * 1.5
        passed = compressed and today_break and vol_surge
        return passed, (1.5 if passed else 0.0)
    @staticmethod
    def pullback_recovery(df):
        if len(df) < 30:
            return False, 0.0
        close = df["close"].astype(float); ma20 = close.rolling(20).mean()
        ma20_up = ma20.iloc[-1] > ma20.iloc[-5]
        recent_low = close.iloc[-10:].min(); recent_ma20 = ma20.iloc[-10:].min()
        touched_ma = recent_low <= recent_ma20 * 1.02
        recovery = close.iloc[-1] > ma20.iloc[-1] and close.iloc[-1] > close.iloc[-2]
        max_close_10 = close.iloc[-10:].max()
        pullback_depth = (max_close_10 - recent_low) / max_close_10 if max_close_10 > 0 else 0
        depth_ok = pullback_depth <= CFG.PULLBACK_MAX_DEPTH
        passed = ma20_up and touched_ma and recovery and depth_ok
        return passed, (min(1 + (1 - pullback_depth / CFG.PULLBACK_MAX_DEPTH), 2.0) if passed else 0.0)
    @staticmethod
    def no_big_drop(df):
        if len(df) < 20:
            return True, 1.0
        close = df["close"].astype(float)
        big_drop = (close.pct_change() <= CFG.BIG_DROP_THRESHOLD).any()
        two_day_return = (close - close.shift(2)) / close.shift(2)
        two_day_big_drop = (two_day_return <= -0.10).any()
        passed = not big_drop and not two_day_big_drop
        return passed, (1.0 if passed else 0.0)
    @staticmethod
    def kdj_oversold_bounce(df):
        if len(df) < 35:
            return False, 0.0
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        kdj = Technicals.kdj(high, low, close); j_val = kdj.get("j")
        if j_val is None or pd.isna(j_val):
            return False, 0.0
        oversold = j_val <= CFG.KDJ_J_THRESHOLD
        macd_data = Technicals.macd(close); macd_bull = (macd_data.get("dif", 0) or 0) > 0
        kdj_prev = Technicals.kdj(high.iloc[:-1], low.iloc[:-1], close.iloc[:-1]); j_prev = kdj_prev.get("j")
        j_turn_up = j_prev is not None and not pd.isna(j_prev) and j_val > j_prev
        passed = oversold and macd_bull and j_turn_up
        return passed, (min((CFG.KDJ_J_THRESHOLD - j_val) / 10 + 1, 2.0) if passed else 0.0)
    @classmethod
    def evaluate_all(cls, df):
        factors = {}; total_bonus = 0.0; pass_count = 0
        checks = [
            ("放量上涨", CFG.ENABLE_VOLUME_SURGE, cls.volume_surge, CFG.BONUS_VOLUME_SURGE),
            ("均线多头", CFG.ENABLE_MA_BULLISH, cls.ma_bullish, CFG.BONUS_MA_BULLISH),
            ("突破平台", CFG.ENABLE_PLATFORM_BREAK, cls.platform_breakout, CFG.BONUS_PLATFORM_BREAK),
            ("低波突破", CFG.ENABLE_LOW_VOL_COMPRESSION, cls.low_vol_compression, CFG.BONUS_LOW_VOL_COMPRESSION),
            ("回踩修复", CFG.ENABLE_PULLBACK_RECOVERY, cls.pullback_recovery, CFG.BONUS_PULLBACK_RECOVERY),
            ("无大回撤", CFG.ENABLE_NO_BIG_DROP, cls.no_big_drop, CFG.BONUS_NO_BIG_DROP),
            ("KDJ反弹", CFG.ENABLE_KDJ_BOUNCE, cls.kdj_oversold_bounce, CFG.BONUS_KDJ_BOUNCE),
        ]
        for name, enabled, check_fn, bonus in checks:
            if not enabled:
                factors[name] = {"通过": None, "得分": 0.0}; continue
            passed, mult = check_fn(df)
            score = bonus * mult if passed else 0.0
            factors[name] = {"通过": passed, "得分": round(score, 1), "倍数": round(mult, 2)}
            if passed:
                total_bonus += score; pass_count += 1
        enabled_cnt = len([c for c in checks if c[1]])
        confirm_ratio = pass_count / enabled_cnt if enabled_cnt else 1.0
        return {"因子详情": factors, "通过数": pass_count, "总加分": round(total_bonus, 1),
                "确认系数": round(0.7 + 0.3 * confirm_ratio, 2),
                "启动强度": "强" if pass_count >= 5 else "中" if pass_count >= 3 else "弱"}

# ========================= 策略层 =========================
@dataclass
class PeriodSignal:
    name: str; gap: float; threshold: float
    hit: bool = False; ma20_up: bool = False; volume_ok: bool = False; macd_ok: bool = False

@dataclass
class MainResult:
    code: str; name: str; close: float
    signals: List[PeriodSignal] = field(default_factory=list)
    hit_count: int = 0; hit_periods: str = ""
    rsi: Optional[float] = None; atr_pct: Optional[float] = None; bb_width: Optional[float] = None
    trend_score: float = 0.0; volume_score: float = 0.0; main_score: float = 0.0
    launch_bonus: float = 0.0; launch_coeff: float = 1.0; total_score: float = 0.0
    launch_details: Dict = field(default_factory=dict)
    weekly_score: Optional[float] = None; signal: str = ""
    industry: str = "—"; hot_meet: bool = False; hot_sector: str = ""; error: Optional[str] = None

def strategy_weekly_only(df):
    """周线宽松兜底: MA5略低于MA20且抬头(移植自原主策略)。"""
    try:
        if len(df) < 150:
            return None
        d = df.copy(); d["close"] = d["close"].astype(float); d["date"] = pd.to_datetime(d["date"])
        df_week = d.resample("W-FRI", on="date")["close"].last().dropna()
        w_ma5 = df_week.rolling(5).mean().dropna(); w_ma20 = df_week.rolling(20).mean().dropna()
        if len(w_ma5) < 2 or len(w_ma20) < 2:
            return None
        latest_w5, prev_w5 = w_ma5.iloc[-1], w_ma5.iloc[-2]; latest_w20 = w_ma20.iloc[-1]
        gap = (latest_w20 - latest_w5) / latest_w20
        if latest_w5 < latest_w20 and 0 <= gap < CFG.WEEKLY_ONLY_THRESHOLD and latest_w5 > prev_w5:
            return {"gap": gap, "close": d["close"].iloc[-1]}
        return None
    except Exception:
        return None

class MainFusionStrategy:
    def _quad_signals(self, df):
        close = df["close"].astype(float)
        if len(df) < 260:
            return None
        volume = df.get("volume", pd.Series([0] * len(df))).astype(float)
        d_close = close
        w_close = Technicals.resample(df, ["W-FRI", "W"])
        m_close = Technicals.resample(df, ["ME", "M"])
        q_close = Technicals.resample(df, ["QE-DEC", "Q-DEC", "QE", "Q"])
        d_vol = volume
        w_vol = Technicals.resample(df[["date", "volume"]].rename(columns={"volume": "close"}), ["W-FRI", "W"])
        m_vol = Technicals.resample(df[["date", "volume"]].rename(columns={"volume": "close"}), ["ME", "M"])
        periods = [
            ("日", d_close, d_vol, CFG.DAY_THRESHOLD, 20),
            ("周", w_close, w_vol, CFG.WEEK_THRESHOLD, 20),
            ("月", m_close, m_vol, CFG.MONTH_THRESHOLD, 20),
            ("季", q_close, pd.Series(dtype=float), CFG.QUARTER_THRESHOLD, max(5, min(20, len(q_close) - 1))),
        ]
        signals = []; hit_periods = []
        for pname, pclose, pvol, threshold, ma_window in periods:
            if len(pclose) < ma_window + 2:
                continue
            ma20 = pclose.rolling(ma_window).mean(); ma5 = pclose.rolling(5).mean()
            ma20_val = ma20.iloc[-1]
            ma20_prev = ma20.iloc[-2] if len(ma20) > 1 else ma20_val
            ma5_val = ma5.iloc[-1]; p_close_val = pclose.iloc[-1]
            if CFG.CROSS_MODE == 'ma':
                gap = (ma20_val - ma5_val) / ma20_val if ma20_val != 0 else 999
            else:
                gap = (ma20_val - p_close_val) / ma20_val if ma20_val != 0 else 999
            hit = (gap > 0) and (gap < threshold)
            ma20_up = ma20_val > ma20_prev if CFG.MA20_DIRECTION else True
            vol_ma20 = pvol.rolling(20).mean().iloc[-1] if len(pvol) > 20 else 0
            vol_ok = True
            if CFG.VOLUME_CONFIRM and len(pvol) > 20 and vol_ma20 > 0:
                vol_ok = pvol.iloc[-1] >= vol_ma20 * CFG.VOL_RATIO_MIN
            macd_ok = True
            if CFG.MACD_FILTER and pname in ("日", "周") and len(pclose) > 35:
                md = Technicals.macd(pclose)
                macd_ok = (md.get("dif", 0) or 0) >= (md.get("dea", 0) or 0)
            effective = hit and ma20_up and vol_ok and macd_ok
            signals.append(PeriodSignal(name=pname, gap=gap, threshold=threshold, hit=effective,
                                        ma20_up=ma20_up, volume_ok=vol_ok, macd_ok=macd_ok))
            if effective:
                hit_periods.append(pname)
        return {"signals": signals, "hit_count": len(hit_periods), "hit_periods": "+".join(hit_periods)}

    def analyze(self, code, name, df):
        if df is None or len(df) < 60:
            return None
        df = df.copy(); df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
        close = df["close"].astype(float)
        if close.iloc[-1] < CFG.MIN_PRICE:
            return None
        quad = self._quad_signals(df)
        weekly = strategy_weekly_only(df)
        quad_hit = quad is not None and quad["hit_count"] >= CFG.MIN_PERIODS
        result = MainResult(code=code, name=name, close=round(close.iloc[-1], 2))
        tags = []
        if quad_hit:
            result.signals = quad["signals"]; result.hit_count = quad["hit_count"]; result.hit_periods = quad["hit_periods"]
            result.rsi = Technicals.rsi(close)
            atr_val = Technicals.atr(df); result.atr_pct = atr_val / close.iloc[-1] if atr_val else None
            result.bb_width = Technicals.bollinger_width(close)
            rejected = False
            if CFG.ATR_FILTER and result.atr_pct is not None and result.atr_pct < CFG.ATR_MIN_PCT:
                rejected = True
            if CFG.BB_WIDTH_FILTER and result.bb_width is not None and result.bb_width < CFG.BB_WIDTH_MIN:
                rejected = True
            if not rejected:
                result.main_score = self._calc_main_score(result)
                result.trend_score = self._calc_trend_score(result.signals)
                result.volume_score = self._calc_volume_score(result.signals)
                launch = LaunchConfirmFactors.evaluate_all(df)
                result.launch_bonus = launch["总加分"]; result.launch_coeff = launch["确认系数"]; result.launch_details = launch
                result.total_score = round(result.main_score * result.launch_coeff + result.launch_bonus, 1)
                tags.append("四周期共振")
        if weekly is not None:
            wk = round(max(0, (1 - weekly["gap"] / CFG.WEEKLY_ONLY_THRESHOLD)) * 100, 1)
            result.weekly_score = wk
            tags.append("周线宽松")
            if "四周期共振" not in tags:
                result.total_score = round(wk * 0.5, 1)
        if not tags:
            return None
        result.signal = "+".join(tags)
        return result

    def _calc_main_score(self, r):
        score = 0.0
        for s in r.signals:
            if not s.hit:
                continue
            weight = {"日": CFG.W_DAY_GAP, "周": CFG.W_WEEK_GAP, "月": CFG.W_MONTH_GAP, "季": CFG.W_QUARTER_GAP}.get(s.name, 10)
            score += max(0, (1 - s.gap / s.threshold)) * weight
        score += max(0, r.hit_count - CFG.MIN_PERIODS) * CFG.W_EXTRA_PERIOD
        if r.rsi is not None:
            if 35 <= r.rsi <= 55:
                score += CFG.W_RSI
            elif r.rsi > 75:
                score -= 10
            elif r.rsi < 25:
                score += 5
        score += min(5, r.trend_score); score += min(5, r.volume_score)
        return round(max(0, min(100, score)), 1)
    def _calc_trend_score(self, signals):
        return round(sum(max(0, 1 - s.gap / s.threshold) + (0.2 if s.ma20_up else 0) for s in signals if s.hit), 2)
    def _calc_volume_score(self, signals):
        return round(sum(1 for s in signals if s.hit and s.volume_ok) * 1.5, 2)

# ========================= 扫描 (多进程) =========================
def _process_worker(args):
    code, name = args
    try:
        df = get_kline(code)
        if df is None:
            return MainResult(code=code, name=name, close=0, error="无数据")
        time.sleep(CFG.SLEEP)
        return _STRATEGY.analyze(code, name, df)
    except Exception as e:
        return MainResult(code=code, name=name, close=0, error=str(e))

def run_scan():
    global _INDUSTRY_MAP
    if not _bs_login_ok():
        print("主进程 baostock 登录失败, 列表/行业走 akshare 兜底")
    try:
        stock_df = get_stock_list()
        load_industry()
    finally:
        _bs_logout()
    if stock_df.empty:
        print("⚠️ 无法获取股票列表"); return []
    codes = stock_df["code"].tolist()
    if CFG.SCAN_LIMIT and len(codes) > CFG.SCAN_LIMIT:
        codes = codes[:CFG.SCAN_LIMIT]
    code_to_name = dict(zip(stock_df["code"], stock_df["code_name"]))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]
    results = []; errors = 0
    print(f"开始扫描 {len(tasks)} 只 (进程={CFG.NUM_PROCESSES}, 模式={CFG.CROSS_MODE}, MIN_PERIODS={CFG.MIN_PERIODS})...")
    with mp.Pool(processes=CFG.NUM_PROCESSES, initializer=_init_worker) as pool:
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="主策略", unit="只")
        for res in pool.imap_unordered(_process_worker, tasks):
            if res is not None:
                if res.error:
                    errors += 1
                else:
                    results.append(res)
                    pbar.write(f"  ✅ {res.code} {res.name} | {res.signal} | 综合:{res.total_score}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=errors)
    print(f"\n扫描完成: 命中 {len(results)} 只, 失败 {errors} 只")
    results.sort(key=lambda x: x.total_score, reverse=True)
    return results

# ========================= 输出与推送 =========================
def enrich(results):
    if not results:
        return pd.DataFrame(), [], []
    for r in results:
        r.industry = _INDUSTRY_MAP.get(r.code, "—")
    labeled = [r for r in results if r.industry not in ("—", "未知", "")]
    cluster = []
    if labeled:
        vc = pd.Series([r.industry for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CFG.CLUSTER_TOP).items()]
    print(f"🌀 共振板块: {cluster or '无'}")
    hot = get_hot_sectors()
    hot_names = [n for n, _ in hot]
    meet_cnt = 0
    for r in results:
        m = _match_sector(r.industry, hot_names)
        if m:
            r.hot_meet = True; r.hot_sector = m; meet_cnt += 1
    print(f"🎯 共振遇风口: {meet_cnt} 只")
    results.sort(key=lambda r: (1 if r.hot_meet else 0, r.total_score), reverse=True)
    rows = []
    for r in results:
        ld = r.launch_details
        row = {"代码": r.code, "名称": r.name, "行业": r.industry, "最新价": r.close, "信号": r.signal,
               "满足周期": r.hit_periods, "满足数": r.hit_count, "主评分": r.main_score,
               "启动加分": r.launch_bonus, "确认系数": r.launch_coeff, "综合评分": r.total_score,
               "启动强度": ld.get("启动强度", "—") if ld else "—", "周线宽松评分": r.weekly_score,
               "趋势分": r.trend_score, "量价分": r.volume_score, "RSI": r.rsi,
               "ATR%": round(r.atr_pct * 100, 2) if r.atr_pct else None,
               "BB宽度": round(r.bb_width * 100, 2) if r.bb_width else None,
               "hot_meet": r.hot_meet, "hot_sector": r.hot_sector}
        for s in r.signals:
            row[f"{s.name}_gap"] = round(s.gap * 100, 3); row[f"{s.name}_hit"] = s.hit
        rows.append(row)
    return pd.DataFrame(rows), cluster, hot

def _match_sector(sector, hot_names):
    if not sector or sector in ("—", "未知", "") or not hot_names:
        return ""
    s = sector.strip()
    for h in hot_names:
        if h and (h == s or h in s or s in h):
            return h
    return ""

def build_push_content(df, cluster, hot):
    P = CFG.PUSH_TOP
    lines = []
    if hot:
        lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); lines.append("")
    if cluster:
        lines.append("🌀 **共振板块**: " + "、".join(f"{n}({c}只)" for n, c in cluster)); lines.append("")
    meet = df[df["hot_meet"] == True] if "hot_meet" in df.columns else pd.DataFrame()
    if not meet.empty:
        lines.append(f"### 🎯 共振遇风口 Top{min(len(meet), P)}")
        for _, row in meet.head(P).iterrows():
            tag = f"[🎯{row['hot_sector']}]" if row.get("hot_sector") else ""
            lines.append(f"- {row['名称']}({row['代码']}){tag} 价{row['最新价']} | {row['信号']} | "
                         f"周期:{row['满足周期'] or '—'} | 综合:{row['综合评分']} | 强度:{row['启动强度']}")
        lines.append("")
    lines.append(f"### 📋 全部共振 Top{min(len(df), P)}")
    for _, row in df.head(P).iterrows():
        tag = f"[🎯{row['hot_sector']}]" if row.get("hot_sector") and row["hot_meet"] else f"[{row['行业']}]"
        lines.append(f"- {row['名称']}({row['代码']}){tag} 价{row['最新价']} | {row['信号']} | "
                     f"周期:{row['满足周期'] or '—'} | 主分:{row['主评分']} | 综合:{row['综合评分']} | 强度:{row['启动强度']}")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output*")
    return "\n".join(lines)

def save_and_push(df, cluster, hot):
    tag = datetime.now().strftime("%Y%m%d")
    csv_path = f"{CFG.OUTPUT_DIR}/main_screener_{tag}.csv"
    json_path = f"{CFG.OUTPUT_DIR}/main_screener_{tag}.json"
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"\n💾 已保存: {csv_path} (共 {len(df)} 只)")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        disp = df.head(CFG.PUSH_TOP).copy()
        disp["板块标签"] = disp.apply(lambda r: f"🎯{r['hot_sector']}" if r.get("hot_meet") else r.get("行业", "—"), axis=1)
        disp = disp.drop(columns=["hot_meet", "hot_sector"], errors="ignore")
        print("\n" + disp.to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if CFG.SERVERCHAN_KEY and not df.empty:
        try:
            meet_n = int(df["hot_meet"].sum()) if "hot_meet" in df.columns else 0
            title = f"主策略 命中{len(df)}只 🌀板块{len(cluster)} 🎯风口{meet_n}"
            content = (f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                       f"参数: 模式={CFG.CROSS_MODE} MIN_PERIODS={CFG.MIN_PERIODS} (四周期共振+启动确认+周线宽松兜底)\n\n"
                       ) + build_push_content(df, cluster, hot)
            send_serverchan(title, content)
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")

def main():
    print("=" * 70)
    print(f"主策略 四周期共振+启动确认+周线宽松兜底 | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"模式={CFG.CROSS_MODE} | MIN_PERIODS={CFG.MIN_PERIODS} | 过滤: ATR={CFG.ATR_FILTER} BB={CFG.BB_WIDTH_FILTER} "
          f"量能={CFG.VOLUME_CONFIRM} MACD={CFG.MACD_FILTER} MA20方向={CFG.MA20_DIRECTION}")
    print("=" * 70)
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule':
        try:
            d = ak.tool_trade_date_hist_sina()
            dates = set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
            if datetime.now().strftime('%Y-%m-%d') not in dates:
                print("非交易日且为定时触发, 跳过"); sys.exit(0)
        except Exception as e:
            print(f"  交易日历获取失败, 默认继续: {e}")
    results = run_scan()
    if results:
        df, cluster, hot = enrich(results)
        save_and_push(df, cluster, hot)
    else:
        print("\n本次未找到符合条件的股票(可降 MIN_PERIODS 或关过滤)。")
        if CFG.SERVERCHAN_KEY:
            send_serverchan("主策略 | 0命中", "**主策略(四周期共振+启动确认)** | 本次无满足条件的票(门槛严, 0命中属正常; 可降MIN_PERIODS/关过滤)。")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_main<<<
