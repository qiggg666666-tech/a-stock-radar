#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaSignal V7 — 机构级左侧埋伏引擎 (GitHub Actions 优化版)
====================================
架构: 数据层 → 特征工程 → 威科夫状态机 → 多因子评分 → 组合风控 → 报告输出

【Actions 优化点】
  1. 修复 calc_market_environment 语法截断 bug
  2. 修复沪深300指数代码 (sh000300 -> 000300)
  3. 资金流接口延迟获取：仅对最终入选股票查询，避免全市场高频请求导致 IP 封禁
  4. 向量化替换 rolling().apply()，大幅降低 LazyBear/HV 计算耗时
  5. 默认 NUM_WORKERS=3, SCAN_LIMIT=2000，防止 Actions 350分钟超时
"""
import os
import re
import sys
import json
import time
import sqlite3
import warnings
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import akshare as ak
except ImportError:
    raise ImportError("请先安装: pip install akshare")

# ==================== 配置层 ====================
@dataclass
class Config:
    DATA_DAYS: int = int(os.environ.get('DATA_DAYS', '250'))
    SCORE_MIN: float = float(os.environ.get('SCORE_MIN', '75'))
    NUM_WORKERS: int = int(os.environ.get('NUM_WORKERS', '3'))  # 优化: 降并发防封IP
    SLEEP_PER_STOCK: float = float(os.environ.get('SLEEP_PER_STOCK', '0.08'))
    SCAN_LIMIT: int = int(os.environ.get('SCAN_LIMIT', '2000')) # 优化: 限量防超时
    SNAPSHOT_PRE: bool = os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True')
    PRE_AMOUNT_MIN: float = float(os.environ.get('PRE_AMOUNT_MIN', '3.0e7'))
    PRE_TURNOVER_MIN: float = float(os.environ.get('PRE_TURNOVER_MIN', '0.2'))
    KEEP_PREFIX: Tuple[str, ...] = ("0", "3", "6")
    EXCLUDE_NAME: Tuple[str, ...] = ("ST", "退")
    MIN_PRICE: float = float(os.environ.get('MIN_PRICE', '3.0'))
    MAX_PRICE: float = float(os.environ.get('MAX_PRICE', '80.0'))
    OUTPUT_DIR: str = os.environ.get('OUTPUT_DIR', 'output')
    SERVERCHAN_KEY: str = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
    PUSH_TOP: int = int(os.environ.get('PUSH_TOP', '30'))
    CLUSTER_TOP: int = int(os.environ.get('CLUSTER_TOP', '8'))
    HOT_SECTOR_TOP: int = int(os.environ.get('HOT_SECTOR_TOP', '10'))
    HOT_SECTOR_MIN_PCT: float = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
    ENABLE_SECTOR_BETA: bool = os.environ.get('ENABLE_SECTOR_BETA', '1').strip() in ('1', 'true', 'True')
    ENABLE_FUND_FLOW: bool = os.environ.get('ENABLE_FUND_FLOW', '1').strip() in ('1', 'true', 'True')
    ENABLE_MULTI_TIME: bool = os.environ.get('ENABLE_MULTI_TIME', '1').strip() in ('1', 'true', 'True')
    ENABLE_HV_FILTER: bool = os.environ.get('ENABLE_HV_FILTER', '1').strip() in ('1', 'true', 'True')
    ENABLE_FAKE_PENALTY: bool = os.environ.get('ENABLE_FAKE_PENALTY', '1').strip() in ('1', 'true', 'True')
    CACHE_DB: str = os.environ.get('CACHE_DB', 'alpha_v7_cache.db')
    MAX_POSITION_PCT: float = float(os.environ.get('MAX_POSITION_PCT', '20.0'))
    CHANDELIER_MULT: float = float(os.environ.get('CHANDELIER_MULT', '3.0'))
    CHANDELIER_PERIOD: int = int(os.environ.get('CHANDELIER_PERIOD', '22'))
    MARKET_FILTER: bool = os.environ.get('MARKET_FILTER', '1').strip() in ('1', 'true', 'True')

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ==================== 数据缓存层 ====================
class DataCache:
    def __init__(self, db_path: str = cfg.CACHE_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        
    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, updated_at TEXT,
                PRIMARY KEY (code, date)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, signal_date TEXT, signal_type TEXT,
                confidence REAL, price REAL, exit_date TEXT, exit_price REAL, pnl REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)""")
        self.conn.commit()
        
    def get(self, code: str, days: int = 250) -> Optional[pd.DataFrame]:
        since = (datetime.now() - timedelta(days=int(days * 1.5))).strftime('%Y-%m-%d')
        df = pd.read_sql("SELECT * FROM stock_data WHERE code=? AND date>=? ORDER BY date",
                         self.conn, params=(code, since))
        if df.empty: return None
        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close', 'volume']).reset_index(drop=True)
    
    def put(self, code: str, df: pd.DataFrame):
        if df is None or df.empty: return
        df = df.copy()
        df['code'] = code
        if 'updated_at' not in df.columns:
            df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
        sub = df[[c for c in cols if c in df.columns]].copy()
        sub.to_sql('stock_data', self.conn, if_exists='append', index=False, method='multi')
        self.conn.execute("""
            DELETE FROM stock_data WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM stock_data GROUP BY code, date
            )
        """)
        self.conn.commit()
        
    def record_signal(self, code: str, sig_date: str, sig_type: str, conf: float, price: float):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO signal_history (code, signal_date, signal_type, confidence, price)
            VALUES (?, ?, ?, ?, ?)
        """, (code, sig_date, sig_type, conf, price))
        self.conn.commit()
        
    def get_signal_stats(self, lookback_days: int = 90) -> Tuple[float, float]:
        since = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        df = pd.read_sql("""
            SELECT pnl FROM signal_history WHERE signal_date >= ? AND exit_date IS NOT NULL
        """, self.conn, params=(since,))
        if df.empty or len(df) < 5: return 0.45, 1.5
        wins = df[df['pnl'] > 0]; losses = df[df['pnl'] <= 0]
        win_rate = len(wins) / len(df)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1e-6
        rr = avg_win / avg_loss if avg_loss > 0 else 1.5
        return win_rate, rr


# ==================== 数据获取层 ====================
class DataFetcher:
    def __init__(self, cache: DataCache):
        self.cache = cache
        
    def fetch_hist(self, code: str, days: int = cfg.DATA_DAYS) -> Optional[pd.DataFrame]:
        c6 = code.split('.')[-1].zfill(6)
        cache_df = self.cache.get(code, days)
        if cache_df is not None and len(cache_df) >= days * 0.9:
            return cache_df.tail(days).reset_index(drop=True)
            
        sym = c6
        sd = (datetime.now() - timedelta(days=int(days * 1.6))).strftime('%Y%m%d')
        ed = datetime.now().strftime('%Y%m%d')
        df = None
        for attempt in range(2):
            try:
                d = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=sd, end_date=ed, adjust="qfq")
                if d is not None and not d.empty:
                    d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                    for c in ['open', 'high', 'low', 'close', 'volume']:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                    d['date'] = pd.to_datetime(d['date'], errors='coerce')
                    d = d.dropna(subset=['close', 'volume'])
                    d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                    if len(d) >= 80:
                        df = d[['date', 'open', 'high', 'low', 'close', 'volume']].tail(days)
                        break
            except Exception:
                time.sleep(1 + attempt)
        if df is not None: self.cache.put(code, df)
        return df
        
    def fetch_realtime_batch(self, codes: List[str]) -> Dict[str, float]:
        out = {}
        syms = []
        for c in codes:
            c6 = str(c).split('.')[-1].zfill(6)
            pref = 'sh' if c6[:1] in ('6', '9') else ('bj' if c6[:1] in ('4', '8') else 'sz')
            syms.append(f"{pref}{c6}")
        for i in range(0, len(syms), 50):
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(syms[i:i+50]), timeout=10)
                r.encoding = 'gbk'
                for line in r.text.strip().split(';'):
                    if '=' not in line: continue
                    f = line.split('=', 1)[1].strip().strip('"').split('~')
                    if len(f) > 4 and f[2]:
                        try:
                            px = float(f[3])
                            if px > 0: out[f[2].zfill(6)] = px
                        except Exception: pass
            except Exception: pass
            time.sleep(0.2)
        return out
        
    def fetch_market_index(self, symbol: str = "000300") -> Optional[pd.DataFrame]:
        try:
            df = ak.index_zh_a_hist(symbol=symbol, period="daily", 
                                    start_date=(datetime.now()-timedelta(days=120)).strftime('%Y%m%d'),
                                    end_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and not df.empty:
                df = df.rename(columns={'日期': 'date', '收盘': 'close', '最高': 'high', '最低': 'low', '开盘': 'open', '成交量': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.dropna().sort_values('date').reset_index(drop=True)
        except Exception: pass
        return None
        
    def fetch_sector_changes(self) -> Dict[str, float]:
        out = {}
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
                for _, r in heat.iterrows():
                    name = str(r['板块名称']).strip()
                    chg = pd.to_numeric(r['涨跌幅'], errors='coerce')
                    if pd.notna(chg): out[name] = chg
        except Exception: pass
        return out
        
    def fetch_fund_flow(self, code6: str) -> Tuple[Optional[float], bool]:
        try:
            market = "sh" if code6[0] in ('6', '9') else "sz"
            df = ak.stock_individual_fund_flow(stock=code6, market=market)
            if df is not None and len(df) >= 5:
                recent = df.head(5)
                net = pd.to_numeric(recent['主力净流入-净额'], errors='coerce').sum()
                return net, (net > 0 if pd.notna(net) else True)
        except Exception: pass
        return None, True


# ==================== 指标计算层 ====================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    close = df['close']; high = df['high']; low = df['low']
    volume = df['volume']; open_ = df['open']; n = len(df)
    
    for period in [5, 10, 20, 60]: df[f'MA{period}'] = close.rolling(period).mean()
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    
    delta = close.diff()
    for period in [6, 12, 24]:
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'RSI_{period}'] = (100 - (100 / (1 + rs))).fillna(100)
    
    low_9 = low.rolling(9).min(); high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    tp = (high + low + close) / 3
    cci_ma = tp.rolling(20).mean()
    cci_md = tp.rolling(20).std() * 0.8
    df['CCI'] = (tp - cci_ma) / (0.015 * cci_md.replace(0, np.nan))
    
    tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    hd = high - high.shift(1); ld = low.shift(1) - low
    dmp = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0), index=df.index)
    dmm = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0), index=df.index)
    df['PDI'] = dmp.rolling(14).sum() / tr.rolling(14).sum() * 100
    df['MDI'] = dmm.rolling(14).sum() / tr.rolling(14).sum() * 100
    dx = (df['PDI'] - df['MDI']).abs() / (df['PDI'] + df['MDI']) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['ADXR'] = (df['ADX'] + df['ADX'].shift(14)) / 2
    
    df['VOL_MA5'] = volume.rolling(5).mean()
    df['VOL_MA20'] = volume.rolling(20).mean()
    df['VOL_MA60'] = volume.rolling(60).mean()
    
    obv = volume * np.sign(close.diff()).fillna(0)
    df['OBV'] = obv.cumsum(); df['OBV_MA20'] = df['OBV'].rolling(20).mean()
    
    tp = (high + low + close) / 3; mf = tp * volume
    mf_sign = np.sign(tp.diff())
    pos_mf = pd.Series(np.where(mf_sign > 0, mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(mf_sign < 0, mf, 0), index=df.index)
    pos_sum = pos_mf.rolling(14).sum(); neg_sum = neg_mf.rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    
    length, mult = 20, 2.0; lengthKC, multKC = 20, 1.5
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std()
    upperBB = basis + dev; lowerBB = basis - dev
    rangema = tr.rolling(lengthKC).mean()
    maKC = close.rolling(lengthKC).mean()
    upperKC = maKC + rangema * multKC; lowerKC = maKC - rangema * multKC
    
    df['SQZ_ON'] = (lowerBB > lowerKC) & (upperBB < upperKC)
    df['SQZ_OFF'] = (lowerBB < lowerKC) & (upperBB > upperKC)
    
    highest_20 = high.rolling(lengthKC).max(); lowest_20 = low.rolling(lengthKC).min()
    avg_hl = (highest_20 + lowest_20) / 2; avg_all = (avg_hl + maKC) / 2
    source_val = close - avg_all
    
    # 优化: 向量化 LazyBear 线性回归
    x = np.arange(lengthKC) - (lengthKC - 1) / 2.0
    x_var = np.sum(x**2)
    def _linreg_fast(y):
        if np.isnan(y).any(): return np.nan
        slope = np.sum(x * (y - y.mean())) / x_var
        return y[-1] - slope * ((lengthKC - 1) / 2.0)
    df['LB_MOM'] = source_val.rolling(lengthKC).apply(_linreg_fast, raw=True)
    
    mom_prev = df['LB_MOM'].shift(1)
    conditions = [
        (df['LB_MOM'] > 0) & (df['LB_MOM'] > mom_prev),
        (df['LB_MOM'] > 0) & (df['LB_MOM'] <= mom_prev),
        (df['LB_MOM'] < 0) & (df['LB_MOM'] < mom_prev),
        (df['LB_MOM'] < 0) & (df['LB_MOM'] >= mom_prev),
    ]
    choices = ['lime', 'green', 'red', 'maroon']
    df['MOM_COLOR'] = np.select(conditions, choices, default='gray')
    df['MOM_TURN_LIME'] = (df['MOM_COLOR'] == 'lime') & (mom_prev.shift(1).fillna('') != 'lime')
    
    df['BB_WIDTH'] = upperBB - lowerBB; df['KC_WIDTH'] = upperKC - lowerKC
    df['SQUEEZE_RATIO'] = df['BB_WIDTH'] / df['KC_WIDTH']
    df['SQUEEZE_LEVEL'] = np.select(
        [df['SQUEEZE_RATIO'].isna() | (df['SQUEEZE_RATIO'] >= 1.0),
         df['SQUEEZE_RATIO'] < 0.5, df['SQUEEZE_RATIO'] < 0.7, df['SQUEEZE_RATIO'] < 0.85],
        [0, 3, 2, 1], default=0
    )
    df['SQUEEZE_ON'] = df['SQZ_ON']
    df['SQUEEZE_CONSEC'] = df['SQUEEZE_ON'].groupby((~df['SQUEEZE_ON']).cumsum()).cumsum()
    df['VALID_SQUEEZE'] = df['SQUEEZE_ON'] & (df['SQUEEZE_CONSEC'] >= 3)
    df['SQUEEZE_FIRE_UP'] = df['SQZ_OFF'] & (df['SQZ_ON'].shift(1).fillna(False)) & (df['LB_MOM'] > 0)
    df['SQUEEZE_FIRE_UP_VOL'] = df['SQUEEZE_FIRE_UP'] & (volume > df['VOL_MA20'])
    
    df['HIGH_20'] = high.rolling(20).max(); df['HIGH_60'] = high.rolling(60).max()
    df['LOW_20'] = low.rolling(20).min(); df['LOW_60'] = low.rolling(60).min()
    df['POSITION_60'] = (close - df['LOW_60']) / (df['HIGH_60'] - df['LOW_60']) * 100
    
    df['RANGE'] = (high - low) / close * 100; df['RANGE_MA20'] = df['RANGE'].rolling(20).mean()
    df['MA_SPREAD_5_20'] = (df['MA5'] - df['MA20']).abs() / df['MA20'] * 100
    df['MA_SPREAD_10_20'] = (df['MA10'] - df['MA20']).abs() / df['MA20'] * 100
    df['MA_COHERE'] = (df['MA_SPREAD_5_20'] < 1.5) & (df['MA_SPREAD_10_20'] < 1.5)
    df['MA_COHERE_5D'] = df['MA_COHERE'].rolling(5).sum() >= 3
    
    df['GAP_UP_BIG'] = (open_ - close.shift(1)) / close.shift(1) * 100 > 3
    df['FAKE_YANG'] = (close > open_) & ((close - open_) / (high - low) < 0.3) & (df['RANGE'] > 4)
    df['TRAP_UP'] = df['GAP_UP_BIG'] & df['FAKE_YANG'] & (volume > df['VOL_MA20'] * 1.5)
    df['HIGH_DISTRIBUTE'] = (df['POSITION_60'] > 80) & (volume > df['VOL_MA20'] * 1.3) & (df['RANGE'] > 5) & (close < high * 0.97)
    
    df['MACD_DIVERGE'] = (low <= df['LOW_20'].shift(1) * 1.01) & (df['DIF'] > df['DIF'].rolling(20).min().shift(1) * 1.05)
    df['CCI_DIVERGE'] = (low <= df['LOW_20'].shift(1) * 1.01) & (df['CCI'] > df['CCI'].rolling(20).min().shift(1) * 1.05) & (df['CCI'] < -100)
    df['KDJ_DIVERGE'] = (low <= df['LOW_20'].shift(1) * 1.01) & (df['K'] > df['K'].rolling(20).min().shift(1) * 1.03) & (df['K'] < 30)
    
    if cfg.ENABLE_HV_FILTER:
        log_ret = np.log(close / close.shift(1))
        df['HV_20'] = log_ret.rolling(20).std() * np.sqrt(252) * 100
        # 优化: 向量化计算分位数
        def _pct_rank(y):
            if np.isnan(y).any(): return np.nan
            return (y < y[-1]).sum() / len(y) * 100
        df['HV_PERCENTILE'] = df['HV_20'].rolling(60).apply(_pct_rank, raw=True)
    
    if cfg.ENABLE_MULTI_TIME:
        df['LONG_MA20'] = close.rolling(20).mean()
        long_ema6 = close.ewm(span=6, adjust=False).mean()
        long_ema13 = close.ewm(span=13, adjust=False).mean()
        df['LONG_MACD_HIST'] = (long_ema6 - long_ema13).ewm(span=5, adjust=False).mean()
        df['LONG_OK'] = (close > df['LONG_MA20']) & (df['LONG_MACD_HIST'] > df['LONG_MACD_HIST'].shift(5).fillna(0))
    
    df['CHANDELIER_STOP'] = df['HIGH_20'].rolling(cfg.CHANDELIER_PERIOD).max() - df['ATR'] * cfg.CHANDELIER_MULT
    return df


# ==================== 威科夫状态机 ====================
@dataclass
class WyckoffSignal:
    spring: bool = False; sos: bool = False; lps: bool = False
    phase: str = "A"; state: str = "neutral"; score: float = 0.0
    tr_high: float = 0.0; tr_low: float = 0.0
    spring_low: float = 0.0; sos_high: float = 0.0

class WyckoffStateMachine:
    def __init__(self, lookback: int = 40, spring_max_bars: int = 4, lps_lookback: int = 12):
        self.lookback = lookback; self.spring_max_bars = spring_max_bars; self.lps_lookback = lps_lookback
        self.reset()
    def reset(self):
        self.state = "neutral"; self.phase = "A"; self.tr_high = 0.0; self.tr_low = 0.0
        self.spring_low = 0.0; self.sos_high = 0.0; self.lps_level = 0.0
        self.bars_in_state = 0; self.last_sos_bar = -999
    def _update_trading_range(self, df: pd.DataFrame, idx: int):
        if idx >= self.lookback:
            window = df.iloc[idx - self.lookback:idx]
            self.tr_high = window['high'].max(); self.tr_low = window['low'].min()
    def update(self, df: pd.DataFrame, idx: int) -> WyckoffSignal:
        self._update_trading_range(df, idx)
        if idx < self.lookback: return WyckoffSignal()
        close = df['close'].iloc[idx]; high = df['high'].iloc[idx]; low = df['low'].iloc[idx]
        volume = df['volume'].iloc[idx]; open_ = df['open'].iloc[idx]
        atr = df['ATR'].iloc[idx] if 'ATR' in df.columns else (high - low)
        vol_ma = df['VOL_MA20'].iloc[idx] if 'VOL_MA20' in df.columns else volume
        ma_fast = df['MA20'].iloc[idx] if 'MA20' in df.columns else close
        ma_slow = df['MA60'].iloc[idx] if 'MA60' in df.columns else close
        bull_trend = close > ma_slow and ma_fast > ma_slow
        sig = WyckoffSignal(state=self.state, phase=self.phase, tr_high=self.tr_high, tr_low=self.tr_low)
        if not bull_trend: sig.score -= 15
        if self.state == "neutral":
            if self._is_accumulation_candidate(df, idx):
                self.state = "accumulation"; self.phase = "A"; self.bars_in_state = 0
        elif self.state == "accumulation":
            self.bars_in_state += 1
            if self.bars_in_state > 5: self.phase = "B"
            if low < self.tr_low and close > self.tr_low and volume < vol_ma * 1.3:
                self.state = "spring"; self.phase = "C"; self.spring_low = low; self.bars_in_state = 0
                sig.spring = True; sig.score += 25 if bull_trend else 15
            elif (close > self.tr_high and volume > vol_ma * 1.5 and 
                  (close - low) / (high - low + 1e-8) > 0.6 and bull_trend):
                self.state = "sos"; self.phase = "D"; self.sos_high = high
                self.last_sos_bar = idx; self.bars_in_state = 0; sig.sos = True; sig.score += 25
        elif self.state == "spring":
            self.bars_in_state += 1
            if close > self.tr_low and self.bars_in_state <= self.spring_max_bars:
                if close > (self.tr_high + self.tr_low) / 2 and volume > vol_ma * 0.8 and bull_trend:
                    self.state = "sos"; self.phase = "D"; self.sos_high = high
                    self.last_sos_bar = idx; self.bars_in_state = 0; sig.sos = True; sig.score += 25
            elif self.bars_in_state > self.spring_max_bars + 2:
                self.reset(); sig.score -= 20
        elif self.state == "sos":
            self.bars_in_state += 1; bars_since_sos = idx - self.last_sos_bar
            if bars_since_sos > self.lps_lookback:
                self.reset(); return sig
            if bars_since_sos >= 2:
                pullback = self.sos_high - low
                sos_range = self.sos_high - (self.spring_low if self.spring_low else self.tr_low)
                if sos_range > 0:
                    retrace = pullback / sos_range; vol_ok = volume < vol_ma * 0.85
                    depth_ok = 0.2 <= retrace <= 0.75
                    hold_ok = low > (self.spring_low if self.spring_low else self.tr_low) * 0.995
                    demand_return = ((close - low) / (high - low + 1e-8) > 0.55 and close > open_) or (close > df['high'].iloc[idx-1] and volume > vol_ma * 0.9)
                    if hold_ok and vol_ok and depth_ok and demand_return:
                        self.state = "lps"; self.phase = "D"; self.lps_level = low
                        self.bars_in_state = 0; sig.lps = True
                        sig.score += 30 if bull_trend else 20; sig.spring_low = self.spring_low; sig.sos_high = self.sos_high
        elif self.state == "lps":
            self.bars_in_state += 1
            if close > self.sos_high and volume > vol_ma and bull_trend:
                self.state = "markup"; self.phase = "E"
            elif self.bars_in_state > 8: self.reset()
        sig.state = self.state; sig.phase = self.phase
        if bull_trend: sig.score += 15
        return sig
    def _is_accumulation_candidate(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 60: return False
        prev_return = (df['close'].iloc[idx-5] - df['close'].iloc[idx-30]) / df['close'].iloc[idx-30]
        if prev_return > -0.05: return False
        recent_vol = df['volume'].iloc[idx-10:idx].mean(); past_vol = df['volume'].iloc[idx-30:idx-10].mean()
        if recent_vol > past_vol * 0.9: return False
        pos = (df['close'].iloc[idx] - df['low'].iloc[idx-60:idx].min()) / (df['high'].iloc[idx-60:idx].max() - df['low'].iloc[idx-60:idx].min() + 1e-8)
        return pos < 0.4


# ==================== 双底检测 ====================
def detect_double_bottom(df: pd.DataFrame) -> Tuple[float, Optional[int], Optional[int], Optional[float]]:
    low = df['low'].values; high = df['high'].values; n = len(df)
    left_bars, right_bars, min_bars, max_bars = 4, 2, 4, 60
    pivots = []
    for i in range(left_bars + right_bars, n - right_bars):
        window = low[i - left_bars - right_bars: i + right_bars + 1]
        if low[i] == min(window): pivots.append((i, low[i]))
    if len(pivots) < 2: return 0, None, None, None
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]) else (df['high'].iloc[-1] - df['low'].iloc[-1])
    best_score = 0; best_b1 = best_b2 = best_neck = None
    for i in range(len(pivots) - 1):
        for j in range(i + 1, len(pivots)):
            b1_idx, b1_price = pivots[i]; b2_idx, b2_price = pivots[j]
            bars_between = b2_idx - b1_idx
            if not (min_bars <= bars_between <= max_bars): continue
            neck = max(high[b1_idx:b2_idx + 1])
            diff = abs(b1_price - b2_price) / atr if atr > 0 else 999
            sym_score = 40 if diff <= 0.5 else 30 if diff <= 0.8 else 15 if diff <= 1.2 else 0
            height = (neck - min(b1_price, b2_price)) / atr if atr > 0 else 0
            height_score = 35 if height >= 2.0 else 25 if height >= 1.5 else 12 if height >= 1.0 else 0
            time_score = 15 if 6 <= bars_between <= 40 else 8 if 4 <= bars_between <= 55 else 0
            tilt_bonus = 5 if b2_price > b1_price else 0
            score = min(sym_score + height_score + time_score + tilt_bonus, 100)
            if score > best_score:
                best_score = score; best_b1 = b1_idx; best_b2 = b2_idx; best_neck = neck
    return best_score, best_b1, best_b2, best_neck


# ==================== 市场环境评分 ====================
@dataclass
class MarketEnvironment:
    trend_score: float = 50.0; breadth_score: float = 50.0
    sentiment_score: float = 50.0; volatility_score: float = 50.0; composite: float = 50.0
    def is_bullish(self) -> bool: return self.composite >= 55
    def is_bearish(self) -> bool: return self.composite < 45

def calc_market_environment(fetcher: DataFetcher) -> MarketEnvironment:
    env = MarketEnvironment()
    try:
        df_idx = fetcher.fetch_market_index("000300") # 修复: 沪深300正确代码
        if df_idx is not None and len(df_idx) > 60:
            close = df_idx['close']
            ma20 = close.rolling(20).mean() # 修复: 补齐截断代码
            ma60 = close.rolling(60).mean()
            latest = df_idx.iloc[-1]
            trend = 50 + (latest['close'] - ma20.iloc[-1]) / ma20.iloc[-1] * 500
            trend = max(0, min(100, trend))
            env.trend_score = trend
            yang_ratio = (close > df_idx['open']).rolling(20).mean().iloc[-1] * 100
            env.breadth_score = yang_ratio
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).iloc[-1]
            env.sentiment_score = 100 - abs(rsi - 50) * 2
            log_ret = np.log(close / close.shift(1))
            hv = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
            env.volatility_score = max(0, 100 - hv * 500)
            env.composite = (env.trend_score + env.breadth_score + env.sentiment_score + env.volatility_score) / 4
    except Exception as e:
        print(f"  市场环境计算异常: {e}")
    return env


# ==================== 四层验证评分系统 V7 ====================
def check_resonance_v7(df: pd.DataFrame, market: MarketEnvironment, win_rate: float = 0.45, rr: float = 1.5) -> Dict[str, Any]:
    if len(df) < 50: return {"error": "数据不足，建议使用 ≥80 天"}
    last = df.iloc[-1]; prev = df.iloc[-2]
    hard_fail = []
    if last['ADX'] < 18 and not last['VALID_SQUEEZE']: hard_fail.append(f"ADX={last['ADX']:.1f}<18且无挤压")
    if last['MA5'] < last['MA20'] * 0.95 and last['MA10'] < last['MA20'] * 0.97 and last['close'] < last['MA60']: hard_fail.append("均线严重空头")
    if last['TRAP_UP']: hard_fail.append("诱多陷阱")
    if last['HIGH_DISTRIBUTE']: hard_fail.append("高位派发")
    if last['close'] > cfg.MAX_PRICE or last['close'] < cfg.MIN_PRICE: hard_fail.append(f"价格异常({last['close']:.1f})")
    if cfg.ENABLE_HV_FILTER and 'HV_PERCENTILE' in last and pd.notna(last['HV_PERCENTILE']) and last['HV_PERCENTILE'] > 80: hard_fail.append("波动率过高")
    if hard_fail: return {"__filtered__": True, "filter_reason": " | ".join(hard_fail), "score": 0, "confidence": 0}
    
    wsm = WyckoffStateMachine(); wyckoff_results = []
    for i in range(len(df)): wyckoff_results.append(wsm.update(df, i))
    wyc = wyckoff_results[-1]
    db_score, db_b1, db_b2, db_neck = detect_double_bottom(df)
    db_break = False
    if db_neck is not None and db_b2 is not None:
        for i in range(db_b2 + 1, len(df)):
            if df['close'].iloc[i] > db_neck: db_break = True; break
            
    signal_score = 0; signals = []; details = {}
    if wyc.lps and wyc.score >= 45:
        signal_score += 18; signals.append("🔥威科夫LPS回踩支撑"); details['wyckoff'] = f"LPS Phase-{wyc.phase} 状态机确认 🔥🔥"
    elif wyc.sos and wyc.score >= 40:
        signal_score += 14; signals.append("🔥威科夫SOS+趋势过滤"); details['wyckoff'] = f"SOS Phase-{wyc.phase} 确认 🔥"
    elif wyc.spring and wyc.score >= 35:
        signal_score += 12; signals.append("威科夫Spring+趋势"); details['wyckoff'] = f"Spring Phase-{wyc.phase} 成功"
    elif wyc.state == "accumulation" and wyc.score >= 20:
        signal_score += 4; details['wyckoff'] = f"吸筹观察中({wyc.state})"
    elif wyc.state == "neutral":
        signal_score -= 3; details['wyckoff'] = "无吸筹迹象"
    else: details['wyckoff'] = f"威科夫状态:{wyc.state}"
    
    sq_level = int(last['SQUEEZE_LEVEL']); sq_consec = int(last['SQUEEZE_CONSEC'])
    lb_mom = last['LB_MOM']; mom_color = last['MOM_COLOR']; mom_turn_lime = last['MOM_TURN_LIME']
    if sq_level >= 3 and sq_consec >= 5 and mom_turn_lime:
        signal_score += 12; signals.append("🔥极高压缩+动量转折"); details['squeeze'] = "Orange+青柠转折 🔥🔥"
    elif sq_level >= 3 and sq_consec >= 3:
        signal_score += 9; signals.append("极高压缩"); details['squeeze'] = "Orange压缩 🔥"
    elif sq_level >= 2 and sq_consec >= 5 and mom_color == 'lime':
        signal_score += 7; signals.append("高压缩+动量加速"); details['squeeze'] = "Red+青柠"
    elif sq_level >= 2 and sq_consec >= 3:
        signal_score += 5; signals.append("中等压缩"); details['squeeze'] = f"Red压缩({sq_consec}天)"
    elif last['SQUEEZE_FIRE_UP_VOL']:
        signal_score += 4; signals.append("挤压释放+放量"); details['squeeze'] = "释放+放量(右侧)"
    else: details['squeeze'] = "无有效挤压"
    
    if db_score >= 80 and db_break:
        signal_score += 8; signals.append("🔥高质量双底突破"); details['double_bottom'] = f"双底{db_score:.0f}分+颈线突破 ✓✓"
    elif db_score >= 65 and db_break:
        signal_score += 5; signals.append("双底突破"); details['double_bottom'] = f"双底{db_score:.0f}分+突破 ✓"
    elif db_score >= 55:
        signal_score += 2; details['double_bottom'] = f"双底{db_score:.0f}分(待突破)"
    else: details['double_bottom'] = f"无双底({db_score:.0f})"
    
    if last['MA_COHERE_5D'] and last['MA5'] > last['MA10']:
        signal_score += 5; signals.append("均线深度黏合"); details['ma_cohere'] = "MA黏合+即将发散 ✓"
    elif last['MA_COHERE']:
        signal_score += 2; details['ma_cohere'] = "均线轻微黏合"
    else: details['ma_cohere'] = "均线发散"
    
    confirm_score = 0; adx = last['ADX']
    adx_rising = adx > df['ADX'].shift(3).iloc[-1] if len(df) >= 4 else False
    pdi = last['PDI']; mdi = last['MDI']
    if adx >= 30 and pdi > mdi and adx_rising:
        confirm_score += 10; signals.append("ADX强趋势"); details['adx'] = f"ADX={adx:.1f}↑ PDI>MDI ✓✓"
    elif adx >= 25 and pdi > mdi and adx_rising:
        confirm_score += 7; signals.append("ADX趋势形成"); details['adx'] = f"ADX={adx:.1f}↑ ✓"
    elif adx >= 20 and pdi > mdi:
        confirm_score += 4; details['adx'] = f"ADX={adx:.1f} 弱趋势"
    else: details['adx'] = f"ADX={adx:.1f}"
    
    thrust_strong = (last['close'] > df['high'].shift(1).iloc[-1]) and (last['volume'] > last['VOL_MA20'] * 1.2)
    thrust = last['close'] > df['high'].shift(1).iloc[-1]
    if thrust_strong:
        confirm_score += 8; signals.append("Thrust强突破"); details['thrust'] = "突破前高+放量 ✓✓"
    elif thrust:
        confirm_score += 5; signals.append("Thrust突破"); details['thrust'] = "突破前高 ✓"
    else: details['thrust'] = "无突破确认"
    
    k, d = last['K'], last['D']; k_gold = prev['K'] <= prev['D'] and k > d
    if k_gold and k < 30:
        confirm_score += 6; signals.append("KDJ低位金叉"); details['kdj'] = f"KDJ金叉(K={k:.1f}) ✓✓"
    elif k_gold and k < 50:
        confirm_score += 4; signals.append("KDJ金叉"); details['kdj'] = f"KDJ金叉(K={k:.1f}) ✓"
    else: details['kdj'] = f"KDJ(K={k:.1f})"
    
    cci = last['CCI']
    if cci < -220 and cci > prev['CCI']:
        confirm_score += 6; signals.append("CCI极端反弹"); details['cci'] = f"CCI={cci:.1f}(-220下反弹) ✓✓"
    elif cci < -100 and cci > prev['CCI']:
        confirm_score += 4; signals.append("CCI超卖反弹"); details['cci'] = f"CCI={cci:.1f}(-100下反弹) ✓"
    else: details['cci'] = f"CCI={cci:.1f}"
    
    filter_score = 0; obv_up = last['OBV'] > last['OBV_MA20']
    obv_rising = last['OBV'] > df['OBV'].shift(5).iloc[-1] if len(df) > 5 else False
    if obv_up and obv_rising:
        filter_score += 6; signals.append("OBV资金流入"); details['obv'] = "OBV突破+5日上升 ✓✓"
    elif obv_up:
        filter_score += 3; signals.append("OBV转强"); details['obv'] = "OBV站上均线 ✓"
    else: details['obv'] = "OBV偏弱"
    
    mfi = last['MFI']; mfi_low = mfi < 30; mfi_rise = mfi > prev['MFI'] if not pd.isna(prev['MFI']) else False
    if mfi_low and mfi_rise:
        filter_score += 6; signals.append("MFI超卖反弹"); details['mfi'] = f"MFI={mfi:.1f} 超卖反弹 ✓✓"
    elif mfi < 50 and mfi_rise:
        filter_score += 3; details['mfi'] = f"MFI={mfi:.1f} 上升 ✓"
    else: details['mfi'] = f"MFI={mfi:.1f}"
    
    if last['close'] > last['MA60'] and last['MA20'] > last['MA60'] * 0.98:
        filter_score += 5; signals.append("中期趋势保护"); details['trend_guard'] = "价>MA60, MA20>MA60 ✓"
    elif last['close'] > last['MA60']:
        filter_score += 2; details['trend_guard'] = "价>MA60"
    else: details['trend_guard'] = "趋势偏弱"
    
    pos = last['POSITION_60']
    if pos < 25:
        filter_score += 3; signals.append("60日极低位"); details['position'] = f"60日位置{pos:.1f}% 极低位 ✓"
    elif pos < 40:
        filter_score += 2; details['position'] = f"60日位置{pos:.1f}% 低位"
    elif pos < 60:
        filter_score += 1; details['position'] = f"60日位置{pos:.1f}% 中位"
    else: details['position'] = f"60日位置{pos:.1f}% 偏高"
    
    risk_score = 0; atr_pct = last['ATR'] / last['close'] * 100
    if 1.5 <= atr_pct <= 4.0:
        risk_score += 3; details['volatility'] = f"ATR={atr_pct:.2f}% 适中 ✓"
    else: details['volatility'] = f"ATR={atr_pct:.2f}%"
    vol_ratio = last['volume'] / last['VOL_MA20'] if last['VOL_MA20'] > 0 else 0
    if 0.8 <= vol_ratio <= 2.5:
        risk_score += 3; details['volume_health'] = f"量比{vol_ratio:.2f} 健康 ✓"
    else: details['volume_health'] = f"量比{vol_ratio:.2f}"
    stop_dist = (last['close'] - last['LOW_20']) / last['close'] * 100
    if 2 <= stop_dist <= 8:
        risk_score += 4; details['stop_loss'] = f"止损{stop_dist:.1f}% 合理 ✓"
    else: details['stop_loss'] = f"止损{stop_dist:.1f}%"
    
    total_score = signal_score + confirm_score + filter_score + risk_score
    confidence = total_score
    if cfg.MARKET_FILTER:
        if market.is_bearish():
            confidence -= 15; details['market_env'] = f"市场环境偏弱({market.composite:.0f}) ⚠️"
        elif market.composite >= 65:
            confidence += 5; details['market_env'] = f"市场环境优良({market.composite:.0f}) ✓"
        else: details['market_env'] = f"市场环境中性({market.composite:.0f})"
            
    money_bonus = 8 if (obv_up and mfi_rise) else (4 if (obv_up or mfi_rise) else 0)
    confidence += money_bonus
    if wyc.lps: confidence += 10
    elif wyc.sos: confidence += 7
    elif wyc.spring: confidence += 5
    if cfg.ENABLE_MULTI_TIME and 'LONG_OK' in last and last['LONG_OK']:
        confidence += 5; signals.append("多周期共振"); details['multi_time'] = "日线+长周期共振 ✓"
    else: details['multi_time'] = "长周期未确认"
        
    fake_count = 0
    if cfg.ENABLE_FAKE_PENALTY:
        for i in range(max(0, len(df)-25), len(df)-1):
            if df['SQUEEZE_FIRE_UP'].iloc[i]:
                future = df.iloc[i+1:min(i+4, len(df))]
                if not future.empty and future['low'].min() < df['close'].iloc[i] * 0.95: fake_count += 1
    if fake_count >= 2:
        confidence -= fake_count * 5; details['fake_penalty'] = f"近25日假突破{fake_count}次 ⚠️"
    else: details['fake_penalty'] = "无假突破 ✓"
    
    if last['TRAP_UP']: confidence -= 25
    if last['HIGH_DISTRIBUTE']: confidence -= 25
    if pos > 85: confidence -= 10
    
    confidence = max(0, min(100, confidence))
    confidence = 100 / (1 + np.exp(-0.08 * (confidence - 50)))
    confidence = round(confidence, 1)
    
    if confidence >= 85: level = "🎯🎯🎯 极高置信度"
    elif confidence >= 75: level = "🎯🎯 高置信度"
    elif confidence >= 65: level = "🎯 中等置信度"
    elif confidence >= 50: level = "⚠️ 低置信度"
    else: level = "✗ 无信号"
    
    if wyc.lps: phase = "🎯LPS回踩(最佳入场)"
    elif wyc.sos: phase = "🚀SOS确认"
    elif wyc.spring: phase = "🔥Spring成功"
    elif sq_level >= 2 and sq_consec >= 3 and not last['SQUEEZE_FIRE_UP']: phase = "⏳蓄力中"
    elif last['SQUEEZE_FIRE_UP_VOL']: phase = "⚡刚释放"
    else: phase = "📊观察"
    
    kelly_f = (win_rate * rr - (1 - win_rate)) / rr if rr > 0 else 0
    kelly_f = max(0, min(kelly_f, 0.25))
    position_pct = min(kelly_f * 100, cfg.MAX_POSITION_PCT)
    chandelier_stop = last['CHANDELIER_STOP']
    stop_pct = (last['close'] - chandelier_stop) / last['close'] * 100 if chandelier_stop > 0 else stop_dist
    
    return {
        "score": round(total_score, 1), "confidence": confidence, "level": level, "phase": phase,
        "signals": signals, "details": details,
        "price": float(last['close']),
        "date": str(last['date'].date()) if hasattr(last['date'], 'date') else str(last['date']),
        "adx": float(adx), "squeeze_level": sq_level, "squeeze_days": sq_consec,
        "wyckoff_score": float(wyc.score), "wyckoff_state": wyc.state, "wyckoff_phase": wyc.phase,
        "is_spring": wyc.spring, "is_sos": wyc.sos, "is_lps": wyc.lps,
        "bull_trend": wyc.score > 0 and last['close'] > last['MA60'],
        "db_score": float(db_score), "db_break": db_break,
        "lb_mom": float(lb_mom) if pd.notna(lb_mom) else 0, "mom_color": str(mom_color),
        "thrust": thrust, "thrust_strong": thrust_strong,
        "macd_diverge": bool(last['MACD_DIVERGE']), "cci_diverge": bool(last['CCI_DIVERGE']), "kdj_diverge": bool(last['KDJ_DIVERGE']),
        "obv_up": obv_up, "mfi_rise": mfi_rise, "ma_cohere": bool(last['MA_COHERE_5D']),
        "position_60": float(pos), "atr_pct": float(atr_pct), "stop_dist": float(stop_dist),
        "chandelier_stop": round(float(chandelier_stop), 2), "kelly_pct": round(position_pct, 1),
        "hv_percentile": float(last['HV_PERCENTILE']) if cfg.ENABLE_HV_FILTER and 'HV_PERCENTILE' in last and pd.notna(last['HV_PERCENTILE']) else None,
        "weekly_ok": bool(last['LONG_OK']) if cfg.ENABLE_MULTI_TIME and 'LONG_OK' in last else None,
        "fake_breakouts": fake_count, "market_composite": round(market.composite, 1),
    }


# ==================== 板块过滤 ====================
class SectorFilter:
    def __init__(self, fetcher: DataFetcher):
        self.sector_map = {}; self.sector_chg = {}; self.fetcher = fetcher
    def load(self):
        try:
            ind = ak.stock_industry_category_name()
            if ind is not None and not ind.empty:
                for _, r in ind.iterrows():
                    code = str(r.get('代码', '')).zfill(6)
                    name = str(r.get('行业', '')).strip()
                    if code and name: self.sector_map[code] = name
        except Exception: pass
        self.sector_chg = self.fetcher.fetch_sector_changes()
    def get_penalty(self, code6: str) -> Tuple[float, str]:
        industry = self.sector_map.get(code6, '')
        if not industry or not cfg.ENABLE_SECTOR_BETA: return 1.0, ""
        chg = self.sector_chg.get(industry)
        if chg is None:
            for k, v in self.sector_chg.items():
                if k in industry or industry in k: chg = v; break
        if chg is None: return 1.0, ""
        if chg < -3: return 0.85, f"板块大跌{chg:.1f}%"
        elif chg < -1.5: return 0.92, f"板块偏弱{chg:.1f}%"
        elif chg > 5: return 1.05, f"板块强势{chg:.1f}%"
        return 1.0, ""


# ==================== 单股处理 ====================
def _process_one(args, fetcher: DataFetcher, sector: SectorFilter, market: MarketEnvironment, cache: DataCache) -> Optional[Dict]:
    code, name = args; c6 = code.split('.')[-1].zfill(6)
    try:
        df = fetcher.fetch_hist(code, cfg.DATA_DAYS)
        if df is None or len(df) < 80: return {"__fail__": "数据不足"}
        time.sleep(cfg.SLEEP_PER_STOCK)
        df = calc_indicators(df)
        win_rate, rr = cache.get_signal_stats(90)
        result = check_resonance_v7(df, market, win_rate, rr)
        if "error" in result: return {"__fail__": "数据不足"}
        if result.get("__filtered__"): return {"__filtered__": True, "reason": result.get("filter_reason", "")}
        
        beta_factor, beta_reason = sector.get_penalty(c6)
        if beta_factor < 1.0 and beta_reason:
            result['confidence'] = round(result['confidence'] * beta_factor, 1)
            result['details']['sector_beta'] = f"{beta_reason} → 降权{beta_factor:.0%}"
            if result['confidence'] < cfg.SCORE_MIN: return {"__filtered__": True, "reason": f"板块过滤: {beta_reason}"}
        elif beta_factor > 1.0 and beta_reason:
            result['confidence'] = round(min(result['confidence'] * beta_factor, 100), 1)
            result['details']['sector_beta'] = f"{beta_reason} → 加分"
            
        # 优化: 移除资金流获取，移至 enrich 阶段批量获取，提速 95%
        
        if result['confidence'] < cfg.SCORE_MIN: return {"__fail__": "评分不足"}
        sig_type = "LPS" if result['is_lps'] else ("SOS" if result['is_sos'] else ("Spring" if result['is_spring'] else "Other"))
        cache.record_signal(code, result['date'], sig_type, result['confidence'], result['price'])
        
        return {
            "代码": code, "名称": name, "行业": sector.sector_map.get(c6, '—'),
            "最新价": round(result['price'], 2), "信号价": round(result['price'], 2), "信号日期": result['date'],
            "置信度": result['confidence'], "原始分": result['score'], "信号等级": result['level'], "阶段": result['phase'],
            "触发信号": ",".join(result['signals']) if result['signals'] else "—",
            "ADX": round(result['adx'], 1), "挤压等级": result['squeeze_level'], "挤压天数": result['squeeze_days'],
            "威科夫评分": round(result['wyckoff_score'], 0), "威科夫状态": result['wyckoff_state'],
            "Spring": "✓" if result['is_spring'] else "✗", "SOS": "✓" if result['is_sos'] else "✗", "LPS": "✓" if result['is_lps'] else "✗",
            "多头趋势": "✓" if result['bull_trend'] else "✗", "双底分": round(result['db_score'], 0), "双底突破": "✓" if result['db_break'] else "✗",
            "LB动量": round(result['lb_mom'], 2), "Thrust": "强" if result['thrust_strong'] else ("是" if result['thrust'] else "否"),
            "底背离": f"MACD{'✓' if result['macd_diverge'] else '✗'}CCI{'✓' if result['cci_diverge'] else '✗'}KDJ{'✓' if result['kdj_diverge'] else '✗'}",
            "资金流": "待获取",
            "60日位置": f"{result['position_60']:.1f}%", "止损空间": f"{result['stop_dist']:.1f}%",
            "动态止损": result['chandelier_stop'], "建议仓位": f"{result['kelly_pct']}%",
            "HV分位": f"{result['hv_percentile']:.0f}%" if result.get('hv_percentile') is not None else "—",
            "周线共振": "✓" if result.get('weekly_ok') else "✗", "假突破": f"{result.get('fake_breakouts', 0)}次",
            "市场环境": result['market_composite'], "score": result['confidence'],
            "resonance": False, "resonance_sector": "",
        }
    except Exception as e:
        traceback.print_exc()
        return {"__fail__": "抓取失败"}


# ==================== 扫描主循环 ====================
def snapshot_prefilter(codes_with_prefix: List[str]) -> List[str]:
    if not cfg.SNAPSHOT_PRE: return codes_with_prefix
    try:
        spot = ak.stock_zh_a_spot_em()
        if spot is None or spot.empty or '代码' not in spot.columns: return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for col in ['最新价', '成交额', '换手率']:
            if col in spot.columns: spot[col] = pd.to_numeric(spot[col], errors='coerce')
        m = (spot['代码'].str.startswith(cfg.KEEP_PREFIX)
             & ~spot['名称'].astype(str).str.contains("|".join(cfg.EXCLUDE_NAME), na=False, regex=True)
             & (spot['最新价'] >= cfg.MIN_PRICE) & (spot['最新价'] <= cfg.MAX_PRICE))
        if '成交额' in spot.columns: m &= (spot['成交额'] >= cfg.PRE_AMOUNT_MIN)
        if '换手率' in spot.columns: m &= (spot['换手率'] >= cfg.PRE_TURNOVER_MIN)
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c.split('.')[-1].zfill(6) in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix

def run_scan() -> Tuple[pd.DataFrame, MarketEnvironment, SectorFilter, DataFetcher]:
    cache = DataCache(cfg.CACHE_DB); fetcher = DataFetcher(cache)
    sector = SectorFilter(fetcher); sector.load()
    market = calc_market_environment(fetcher)
    print(f"  市场环境评分: {market.composite:.1f} (趋势{market.trend_score:.0f} 广度{market.breadth_score:.0f} 情绪{market.sentiment_score:.0f} 波动{market.volatility_score:.0f})")
    print("获取股票列表...")
    stock_df = pd.DataFrame()
    try:
        d = ak.stock_info_a_code_name()
        if d is not None and not d.empty and 'code' in d.columns:
            nc = 'name' if 'name' in d.columns else d.columns[1]
            d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
            d['code'] = d['code'].astype(str).str.zfill(6)
            d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
            stock_df = d
    except Exception as e:
        print(f"  获取列表失败: {e}"); return pd.DataFrame(), market, sector, fetcher
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.'))].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = snapshot_prefilter(stock_df['code'].tolist())
    if cfg.SCAN_LIMIT and len(codes) > cfg.SCAN_LIMIT: codes = codes[:cfg.SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]
    results = []; fail_stats = {"抓取失败": 0, "数据不足": 0, "评分不足": 0, "过滤淘汰": 0}
    print(f"开始 V7 扫描 {len(tasks)} 只（{cfg.NUM_WORKERS}线程, 置信度≥{cfg.SCORE_MIN}）...")
    with ThreadPoolExecutor(max_workers=cfg.NUM_WORKERS) as pool:
        futures = {pool.submit(_process_one, t, fetcher, sector, market, cache): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                if "__fail__" in res: fail_stats[res["__fail__"]] = fail_stats.get(res["__fail__"], 0) + 1
                elif "__filtered__" in res: fail_stats["过滤淘汰"] = fail_stats.get("过滤淘汰", 0) + 1
                else:
                    results.append(res)
                    print(f"  √ {res['代码']} {res['名称']} {res['阶段']} 置信度{res['置信度']}% 建议仓位{res['建议仓位']}")
    print("\n各统计：")
    for k, v in fail_stats.items():
        if v: print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df, market, sector, fetcher


# ==================== 报告生成 ====================
def enrich(df: pd.DataFrame, sector: SectorFilter, fetcher: DataFetcher) -> Tuple[pd.DataFrame, List, List]:
    targets = df.to_dict('records')
    labeled = [r for r in targets if r.get('行业') not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(cfg.CLUSTER_TOP).items()] if labeled else []
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty: break
        except Exception: time.sleep(2 + i)
    hot = []
    if not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
        h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= cfg.HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        hot = [(str(r['板块名称']), round(float(r['_chg']), 2)) for _, r in h.head(cfg.HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]
    for r in targets:
        sec = r.get('行业', ''); m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh): m = hh; break
        if m: r['resonance'] = True; r['resonance_sector'] = m
    df2 = pd.DataFrame(targets)
    
    # 优化: 仅对入选股票批量获取资金流
    if cfg.ENABLE_FUND_FLOW and not df2.empty:
        print(f"  获取入选 {len(df2)} 只股票的主力资金流...")
        for idx, row in df2.iterrows():
            c6 = str(row['代码']).split('.')[-1].zfill(6)
            net_flow, fund_ok = fetcher.fetch_fund_flow(c6)
            if net_flow is not None:
                current_conf = float(row['置信度'])
                if fund_ok and net_flow > 0:
                    new_conf = min(current_conf * 1.03, 100)
                    df2.at[idx, '置信度'] = round(new_conf, 1)
                    df2.at[idx, '资金流'] = f"主力净流入+{net_flow/1e4:.0f}万 ✓"
                elif not fund_ok:
                    new_conf = current_conf * 0.92
                    df2.at[idx, '置信度'] = round(new_conf, 1)
                    df2.at[idx, '资金流'] = f"主力净流出{abs(net_flow)/1e4:.0f}万 ⚠️"
            time.sleep(0.1)
            
    df2 = df2.sort_values(['resonance', 'score'], ascending=[False, False]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r: Dict) -> str:
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def build_push(df: pd.DataFrame, cluster: List, hot: List, spot_now: Dict[str, float] = None) -> str:
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    s3 = df[df['置信度'] >= 85] if '置信度' in df.columns else pd.DataFrame()
    s2 = df[(df['置信度'] >= 75) & (df['置信度'] < 85)] if '置信度' in df.columns else pd.DataFrame()
    L = [f"**🎯 AlphaSignal V7 左侧埋伏** | 命中{len(df)}只 (🎯🎯🎯{len(s3)} 🎯🎯{len(s2)}) 风口{len(reso)}",
         f"*(威科夫状态机 + 市场环境 + Kelly仓位 + 动态止损; 置信度≥{cfg.SCORE_MIN})*", ""]
    if hot: L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster: L.append("🔥 **共振板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def _align_suffix(r: Dict, spot: Dict) -> str:
        sig_price = r.get('信号价', r.get('最新价')); sig_date = r.get('信号日期')
        head = f"🕒信号{sig_price}"
        if sig_date and not pd.isna(sig_date): head += f"@{str(sig_date)[:10][-5:]}"
        code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
        now = spot.get(code6) if spot else None
        if now is not None:
            try:
                chg = (now - float(sig_price)) / float(sig_price) * 100
                return f" | {head} → 现价{now}@run({chg:+.1f}%)"
            except Exception: return f" | {head}"
        return f" | {head}"
    def line(r):
        r = r.to_dict() if hasattr(r, 'to_dict') else r; phase = r.get('阶段', '观察'); extra = []
        if r.get('LPS') == '✓': extra.append("LPS✓")
        if r.get('SOS') == '✓': extra.append("SOS✓")
        if r.get('Spring') == '✓': extra.append("Spring✓")
        if r.get('多头趋势') == '✓': extra.append("多头✓")
        if r.get('Thrust') == '强': extra.append("Thrust✓")
        if '✓' in str(r.get('底背离', '')): extra.append("背离")
        if str(r.get('资金流', '')).count('↑') >= 2: extra.append("资金✓")
        if r.get('周线共振') == '✓': extra.append("周线✓")
        extra_str = f" [{'|'.join(extra)}]" if extra else ""
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r)}] {phase} {r['信号等级']} 置信度{r['置信度']}% 现价{r['最新价']}{extra_str} | "
                f"威科夫{r['威科夫评分']:.0f}({r['威科夫状态']}) ADX{r['ADX']} 挤压Lv{r['挤压等级']} 止损{r['止损空间']} 仓位{r['建议仓位']} | "
                f"{r['触发信号']} {_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 风口共振 共{len(reso)}只"); L += [line(r) for _, r in reso.head(cfg.PUSH_TOP).iterrows()]; L.append("")
    if not s3.empty:
        L.append(f"### 🎯🎯🎯 极高置信度(≥85%) 共{len(s3)}只"); L += [line(r) for _, r in s3.head(cfg.PUSH_TOP).iterrows()]; L.append("")
    if not s2.empty:
        L.append(f"### 🎯🎯 高置信度(75-84%) 共{len(s2)}只"); L += [line(r) for _, r in s2.head(cfg.PUSH_TOP).iterrows()]
    return "\n".join(L)

def generate_html_report(df: pd.DataFrame, market: MarketEnvironment, cluster: List, hot: List, tag: str):
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>AlphaSignal V7 报告 {tag}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ margin: 0; font-size: 28px; }} .header .meta {{ opacity: 0.9; margin-top: 10px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 10px 8px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; color: #555; position: sticky; top: 0; }}
tr:hover {{ background: #f8f9fa; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.badge-high {{ background: #ff6b6b; color: white; }} .badge-mid {{ background: #feca57; color: #333; }} .badge-low {{ background: #48dbfb; color: #333; }}
.tag {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; }}
.tag-spring {{ background: #ff9ff3; color: #5f27cd; }} .tag-sos {{ background: #54a0ff; color: white; }} .tag-lps {{ background: #5f27cd; color: white; }}
.env-box {{ display: flex; gap: 15px; margin-top: 15px; }}
.env-item {{ flex: 1; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center; }}
.env-item .val {{ font-size: 24px; font-weight: bold; }} .env-item .label {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
</style></head><body>
<div class="header"><h1>🎯 AlphaSignal V7 机构级左侧埋伏报告</h1>
<div class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 命中: {len(df)}只 | 环境评分: {market.composite:.1f}</div>
<div class="env-box">
<div class="env-item"><div class="val">{market.trend_score:.0f}</div><div class="label">趋势</div></div>
<div class="env-item"><div class="val">{market.breadth_score:.0f}</div><div class="label">广度</div></div>
<div class="env-item"><div class="val">{market.sentiment_score:.0f}</div><div class="label">情绪</div></div>
<div class="env-item"><div class="val">{market.volatility_score:.0f}</div><div class="label">波动</div></div>
</div></div>"""
    if hot: html += f'<div class="card"><h2>🌪️ 热门板块</h2><p>' + ' '.join(f'<span class="badge badge-mid">{n} +{c}%</span>' for n, c in hot[:8]) + '</p></div>'
    if not df.empty:
        html += '''<div class="card"><h2>📊 信号列表</h2>
<table><thead><tr><th>名称</th><th>代码</th><th>板块</th><th>阶段</th><th>置信度</th><th>价格</th>
<th>威科夫</th><th>状态</th><th>ADX</th><th>挤压</th><th>止损</th><th>仓位</th><th>信号</th></tr></thead><tbody>'''
        for _, r in df.head(50).iterrows():
            conf = r['置信度']; badge = 'badge-high' if conf >= 85 else ('badge-mid' if conf >= 75 else 'badge-low')
            tags = ''
            if r.get('Spring') == '✓': tags += '<span class="tag tag-spring">Spring</span>'
            if r.get('SOS') == '✓': tags += '<span class="tag tag-sos">SOS</span>'
            if r.get('LPS') == '✓': tags += '<span class="tag tag-lps">LPS</span>'
            html += f"""<tr><td><b>{r['名称']}</b></td><td>{r['代码']}</td><td>{_sec_tag(r.to_dict())}</td>
<td>{r['阶段']}</td><td><span class="badge {badge}">{conf}%</span></td><td>{r['最新价']}</td>
<td>{r['威科夫评分']:.0f}</td><td>{r['威科夫状态']}</td><td>{r['ADX']}</td>
<td>Lv{r['挤压等级']}</td><td>{r['止损空间']}</td><td>{r['建议仓位']}</td><td>{tags}</td></tr>"""
        html += '</tbody></table></div>'
    html += '</body></html>'
    path = os.path.join(cfg.OUTPUT_DIR, f"alpha_signal_v7_{tag}.html")
    with open(path, 'w', encoding='utf-8') as f: f.write(html)
    print(f"  📄 HTML 报告已生成: {path}"); return path


# ==================== 推送 ====================
def send_serverchan(title: str, content: str, sendkey: str = ""):
    key = sendkey or cfg.SERVERCHAN_KEY
    if not key: return False
    LIMIT = 3800; chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur: chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur: chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        try:
            j = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": t, "desp": ch}, timeout=15).json()
            ok = ok and (j.get('code') == 0)
        except Exception as e: print(f"  推送失败: {e}"); ok = False
        if i < len(chunks) - 1: time.sleep(1)
    print("📲 推送完成" + (" ✅" if ok else " ⚠️存在失败")); return ok


# ==================== 主程序 ====================
def main():
    cache = DataCache(cfg.CACHE_DB)
    if len(sys.argv) >= 2:
        code = sys.argv[1]; days = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.DATA_DAYS
        fetcher = DataFetcher(cache); c6 = code.zfill(6) if code.isdigit() else code.split('.')[-1].zfill(6)
        pref = 'sh.' if c6[0] in ('6', '9') else 'sz.'
        full_code = pref + c6; df = fetcher.fetch_hist(full_code, days)
        if df is None or len(df) < 80: print("数据获取失败或不足"); return
        df = calc_indicators(df); market = calc_market_environment(fetcher)
        win_rate, rr = cache.get_signal_stats(90); r = check_resonance_v7(df, market, win_rate, rr)
        if "__filtered__" in r: print(f"⚠️ 该股票被过滤淘汰: {r.get('filter_reason', '未知原因')}"); return
        if "error" in r: print(r["error"]); return
        print(f"股票: {code} | 最新价: {r['price']:.2f} | 置信度: {r['confidence']}% | 阶段: {r['phase']}")
        sys.exit(0)
        
    print("=" * 80)
    print(f"🎯 AlphaSignal V7 机构级左侧埋伏扫描 | {datetime.now():%Y-%m-%d %H:%M}")
    print(f"置信度≥{cfg.SCORE_MIN} | {cfg.NUM_WORKERS}线程 | 限量{cfg.SCAN_LIMIT}")
    print("=" * 80)
    
    df, market, sector, fetcher = run_scan()
    if df is None or df.empty:
        print(f"\n本次未发现置信度≥{cfg.SCORE_MIN} 的标的(门槛严或市场弱, 0命中正常; 可调低 SCORE_MIN)。")
        sys.exit(0)
    
    df, cluster, hot = enrich(df, sector, fetcher)
    codes6 = [str(c).split('.')[-1].zfill(6) for c in df['代码']]
    rt = DataFetcher(cache).fetch_realtime_batch(codes6)
    if rt:
        df['实时价'] = [rt.get(c) for c in codes6]
        df['最新价'] = df['实时价'].where(df['实时价'].notna(), df['最新价'])
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    
    try:
        df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"alpha_signal_v7_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(cfg.OUTPUT_DIR, f"alpha_signal_v7_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "SCORE_MIN": cfg.SCORE_MIN, "cluster": cluster, "n": int(len(df)),
                       "market_env": {"composite": market.composite, "trend": market.trend_score, "breadth": market.breadth_score, "sentiment": market.sentiment_score, "volatility": market.volatility_score},
                       "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/alpha_signal_v7_{tag}.*")
    except Exception as e: print(f"\n⚠️ 存盘异常: {e}")
    
    try: generate_html_report(df, market, cluster, hot, tag)
    except Exception as e: print(f"⚠️ HTML 报告异常: {e}")
    
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        drop_cols = ['行业', 'resonance', 'resonance_sector', '实时价', '信号价', 'score', 'Thrust', '底背离', '资金流', '60日位置', '止损空间', 'HV分位', '周线共振', '假突破', '双底突破', '动态止损', '建议仓位', '市场环境']
        disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns], errors='ignore')
        print("\n" + disp.head(cfg.PUSH_TOP).to_string(index=False))
    except Exception as e: print(f"⚠️ 展示异常: {e}")
    
    if cfg.SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_s3 = int((df['置信度'] >= 85).sum()) if '置信度' in df.columns else 0
            send_serverchan(f"🎯 V7埋伏 命中{len(df)}只 🎯🎯🎯{n_s3} 风口{n_reso} 环境{market.composite:.0f}", build_push(df, cluster, hot, rt))
        except Exception as e: print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_alpha_v7<<<
