# -*- coding: utf-8 -*-
"""
vagas_obv_screener.py —— VAGAS融合版(结构突破×OBV×Phase×RS×Minervini×VCP) · 矩阵规格
六部分整合+补齐: 断线重连/无花括号对齐/统一_FAIL_STATS/补缺失辅助函数与PARAMS/删除悬空.exit(0)。
信号四级: 🟢BUY / 🟡WATCH / APPROACH / ⚠️SELL; 含市场氛围+三段式推送+实时复核。
⚠️ 多维过滤极严, 0命中属正常; APPROACH为左侧预警风险更高; 必止损; 非预测。
"""
import os, re, sys, json, time, random, warnings, traceback, requests
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import akshare as ak
import baostock as bs
from tqdm import tqdm

warnings.filterwarnings("ignore")
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# ==================== 基础 PARAMS ====================
PARAMS = dict(
    lookbackSwing=5, useFVG=True, fastLen=12, slowLen=26, signalLen=9, confirmWindow=3,
    atrLen=14, atrMultSL=2.0, atrMultTP=4.0, INCLUDE_WARNINGS=False,
    PRE_SIGNAL=True, APPROACH_PCT=0.02, lookback_days=400, min_data_len=120,
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
# ==================== 融合版 PARAMS(补齐) ====================
PARAMS.update(dict(
    PHASE_MA_SHORT=50, PHASE_MA_MID=150, PHASE_MA_LONG=200, PHASE_SLOPE_DAYS=20,
    RS_BENCHMARK="sh.000300", RS_PERIOD=60,
    MINERVINI_ENABLE=True, MINERVINI_MIN_PASS=6,
    MINERVINI_ABOVE_LOW_PCT=0.30, MINERVINI_NEAR_HIGH_PCT=0.25,
    VCP_ENABLE=True, VCP_LOOKBACK=120, VCP_VOL_CONTRACTION=True,
    MARKET_REGIME_ENABLE=True, BACKTEST_ENABLE=False, RISK_PER_TRADE=0.01,
    SCORE_BUY=70, SCORE_WATCH=55, SCORE_APPROACH=40, RS_MIN_SCORE=6, MIN_RR_RATIO=2.0,
    SCORE_WEIGHTS=dict(structure=0.25, volume=0.20, phase=0.20, rs=0.15, pattern=0.10, fundamental=0.10),
))
PARAMS['PRE_SIGNAL'] = os.environ.get('PRE_SIGNAL', str(PARAMS['PRE_SIGNAL'])).strip() in ('1', 'true', 'True')
PARAMS['APPROACH_PCT'] = float(os.environ.get('APPROACH_PCT', str(PARAMS['APPROACH_PCT'])))
PARAMS['useFVG'] = os.environ.get('USE_FVG', str(PARAMS['useFVG'])).strip() in ('1', 'true', 'True')
PARAMS['MARKET_REGIME_ENABLE'] = os.environ.get('MARKET_REGIME', str(PARAMS['MARKET_REGIME_ENABLE'])).strip() in ('1', 'true', 'True')
PARAMS['BACKTEST_ENABLE'] = os.environ.get('BACKTEST', str(PARAMS['BACKTEST_ENABLE'])).strip() in ('1', 'true', 'True')
REALTIME_RECHECK = os.environ.get('REALTIME_RECHECK', '1').strip() in ('1', 'true', 'True')
CHASE_MAX = float(os.environ.get('CHASE_MAX', '0.15'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = Path(OUTPUT_DIR) / "vagas_cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
_BENCHMARK_DF = None
_FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0}

# ------------------ 推送 ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok: return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15).json().get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False

def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key: return False
    LIMIT = 3800; chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur: chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1: time.sleep(1)
    return ok

# ------------------ baostock(含断线重连) ------------------
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    if _BS_LOGGED: return True
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True; return True
        except Exception as e:
            print(f"  baostock 登录异常: {e}")
        time.sleep(2 * (i + 1))
    return False

def _bs_logout():
    global _BS_LOGGED
    try:
        if _BS_LOGGED: bs.logout()
    except Exception:
        pass
    finally:
        _BS_LOGGED = False

def _init_worker():
    global _BS_LOGGED
    time.sleep(random.uniform(0, 2)); _BS_LOGGED = False
    _bs_login_ok()

def _bs_q(code, fields, sd, ed, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed, frequency="d", adjustflag="2").get_data()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)

def _bs_fetch(code, fields, sd, ed, timeout):
    """【修复】断线(Broken pipe)自动登出+重登+重试一次"""
    global _BS_LOGGED
    for attempt in range(2):
        if not _BS_LOGGED:
            if not _bs_login_ok(): return None
        try:
            d = _bs_q(code, fields, sd, ed, timeout=timeout)
            return d if (d is not None and not d.empty) else None
        except Exception:
            try: bs.logout()
            except Exception: pass
            _BS_LOGGED = False
            time.sleep(1.0 + attempt)
    return None

def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)

def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

def _clean_industry(s):
    if not s or not isinstance(s, str): return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

# ------------------ 缓存 ------------------
def _cache_load(key):
    p = CACHE_DIR / f"{key}.json"
    if not p.exists(): return None
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def _cache_save(key, data):
    try:
        with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception: pass

# ------------------ 数学/指标辅助(补齐) ------------------
def linear_regression_slope(series, days=20):
    s = series.tail(days).dropna()
    if len(s) < 3: return 0.0
    y = s.values; x = np.arange(len(y))
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0: return 0.0
    slope = ((x - x.mean()) * (y - y.mean())).sum() / denom
    mean = y.mean()
    return float(slope / mean) if mean else 0.0

def calc_rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))

def calc_kdj(df, n=9):
    low_n = df['low'].rolling(n).min(); high_n = df['high'].rolling(n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n + 1e-12) * 100
    k = rsv.ewm(com=2, adjust=False).mean(); d = k.ewm(com=2, adjust=False).mean()
    return k, d, 3*k - 2*d

def calc_bollinger(df, period=20, std=2):
    mid = df['close'].rolling(period).mean(); sd = df['close'].rolling(period).std()
    return mid + std*sd, mid, mid - std*sd

def calc_macd(close, fast=12, slow=26, sig=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea, (dif - dea) * 2

def calc_atr(df, period=14):
    high = df['high'].astype(float); low = df['low'].astype(float); close = df['close'].astype(float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_obv(df):
    close = df['close'].astype(float); vol = df['volume'].astype(float)
    return (np.sign(close.diff()).fillna(0) * vol).cumsum()

def last_swing_high(a, left=5, right=5):
    n = len(a)
    for i in range(n - 1 - right, left - 1, -1):
        if a[i] >= np.max(a[i - left:i + right + 1]): return float(a[i])
    return None

def last_swing_low(a, left=5, right=5):
    n = len(a)
    for i in range(n - 1 - right, left - 1, -1):
        if a[i] <= np.min(a[i - left:i + right + 1]): return float(a[i])
    return None
  # ------------------ 实时价+复核 ------------------
def _fetch_realtime_tencent(codes):
    out = {}
    try:
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
            time.sleep(0.3)
    except Exception as e:
        print(f"  腾讯实时价异常: {e}")
    return out

def _refresh_realtime_price(df):
    if df is None or df.empty: return df, {}
    df = df.copy()
    if '信号价' not in df.columns: df['信号价'] = df['最新价']
    codes6 = [str(c).split('.')[-1].zfill(6) for c in df['代码']]
    rt = _fetch_realtime_tencent(codes6)
    if rt:
        df['实时价'] = [rt.get(c) for c in codes6]
        df['最新价'] = df['实时价'].where(df['实时价'].notna(), df['最新价'])
    return df, rt

def _realtime_recheck(df):
    if not REALTIME_RECHECK or df is None or df.empty: return df
    keep = []; ap = PARAMS['APPROACH_PCT']
    for r in df.to_dict('records'):
        px = r.get('最新价'); sh = r.get('摆动高'); sl = r.get('摆动低'); sig = r.get('信号价', r.get('最新价'))
        if px is None or pd.isna(px): keep.append(r); continue
        st = str(r.get('信号', ''))
        if 'BUY' in st:
            if sh is not None and not pd.isna(sh) and px < float(sh): continue
            if sig and not pd.isna(sig) and sig > 0 and (px / sig - 1) > CHASE_MAX: continue
        elif ('WATCH' in st) or ('APPROACH' in st):
            if sh is not None and not pd.isna(sh):
                if px < float(sh) * (1 - ap) or px > float(sh) * (1 + CHASE_MAX): continue
        elif 'SELL' in st:
            if sl is not None and not pd.isna(sl) and px > float(sl): continue
        keep.append(r)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame()

# ------------------ Phase 分类 ------------------
def classify_phase(df, current_price=None):
    if len(df) < PARAMS['PHASE_MA_LONG'] + 10:
        return {'phase': 0, 'phase_name': '未知', 'confidence': 0, 'reasons': ['数据不足']}
    close = df['close'].astype(float)
    if current_price is None: current_price = float(close.iloc[-1])
    sma_50 = close.rolling(PARAMS['PHASE_MA_SHORT']).mean().iloc[-1]
    sma_150 = close.rolling(PARAMS['PHASE_MA_MID']).mean().iloc[-1]
    sma_200 = close.rolling(PARAMS['PHASE_MA_LONG']).mean().iloc[-1]
    s50s = close.rolling(PARAMS['PHASE_MA_SHORT']).mean()
    s200s = close.rolling(PARAMS['PHASE_MA_LONG']).mean()
    slope_50 = linear_regression_slope(s50s.dropna(), PARAMS['PHASE_SLOPE_DAYS'])
    slope_200 = linear_regression_slope(s200s.dropna(), PARAMS['PHASE_SLOPE_DAYS'])
    reasons = []; p2 = 0
    if current_price > sma_50: p2 += 1; reasons.append('价>50MA')
    if current_price > sma_150: p2 += 1; reasons.append('价>150MA')
    if current_price > sma_200: p2 += 1; reasons.append('价>200MA')
    if sma_50 > sma_150: p2 += 1; reasons.append('50>150MA')
    if sma_150 > sma_200: p2 += 1; reasons.append('150>200MA')
    if slope_50 > 0: p2 += 1; reasons.append('50MA向上')
    if slope_200 > 0: p2 += 1; reasons.append('200MA向上')
    base = dict(sma_50=sma_50, sma_150=sma_150, sma_200=sma_200, slope_50=slope_50, slope_200=slope_200)
    if p2 >= 6:
        return {'phase': 2, 'phase_name': '上升📈', 'confidence': min(100, p2 * 12.5), 'reasons': reasons, **base}
    p4 = 0; r4 = []
    if current_price < sma_50: p4 += 1; r4.append('价<50MA')
    if current_price < sma_200: p4 += 1; r4.append('价<200MA')
    if sma_50 < sma_200: p4 += 1; r4.append('50<200MA')
    if slope_50 < 0: p4 += 1; r4.append('50MA向下')
    if slope_200 < 0: p4 += 1; r4.append('200MA向下')
    if p4 >= 4:
        return {'phase': 4, 'phase_name': '下跌📉', 'confidence': min(100, p4 * 15), 'reasons': r4, **base}
    p3 = 0; r3 = []
    if slope_50 < 0 and slope_200 > 0: p3 += 1; r3.append('50MA转下/200MA仍上')
    if current_price < sma_50 and current_price > sma_200: p3 += 1; r3.append('价破50MA未破200MA')
    if abs(slope_50) < 0.05: p3 += 1; r3.append('50MA走平')
    if p3 >= 2:
        return {'phase': 3, 'phase_name': '派发⛔', 'confidence': min(100, 50 + p3 * 15), 'reasons': r3, **base}
    return {'phase': 1, 'phase_name': '筑底⏳', 'confidence': 50, 'reasons': ['均线整理中'], **base}

# ------------------ RS 相对强度 ------------------
def _load_benchmark():
    global _BENCHMARK_DF
    if _BENCHMARK_DF is not None and len(_BENCHMARK_DF) > 100: return _BENCHMARK_DF
    cache_key = f"benchmark_{PARAMS['RS_BENCHMARK']}"
    cached = _cache_load(cache_key)
    if cached:
        _BENCHMARK_DF = pd.DataFrame(cached); return _BENCHMARK_DF
    sd = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    try:
        if _bs_login_ok():
            d = _bs_q(PARAMS['RS_BENCHMARK'], "date,close", sd, ed, timeout=15)
            if d is not None and not d.empty and len(d) > 100:
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna().sort_values('date').reset_index(drop=True)
                _BENCHMARK_DF = d; _cache_save(cache_key, d.to_dict('records')); return d
    except Exception: pass
    try:
        d = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=sd.replace('-', ''), end_date=ed.replace('-', ''))
        if d is not None and not d.empty:
            d = d.rename(columns={'日期': 'date', '收盘': 'close'})
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'])
            d = d.dropna().sort_values('date').reset_index(drop=True)
            _BENCHMARK_DF = d; _cache_save(cache_key, d.to_dict('records')); return d
    except Exception: pass
    return pd.DataFrame()

def calc_relative_strength(df, benchmark_df=None):
    if benchmark_df is None or benchmark_df.empty: benchmark_df = _load_benchmark()
    if benchmark_df is None or benchmark_df.empty or len(df) < PARAMS['RS_PERIOD'] + 5:
        return {'rs_score': 5.0, 'rs_slope': 0.0, 'rs_ratio': 1.0}
    try:
        sc = df[['date', 'close']].copy(); sc['date'] = pd.to_datetime(sc['date'])
        bc = benchmark_df[['date', 'close']].copy(); bc['date'] = pd.to_datetime(bc['date'])
        m = pd.merge(sc, bc, on='date', suffixes=('_stock', '_bench'))
        if len(m) < PARAMS['RS_PERIOD']: return {'rs_score': 5.0, 'rs_slope': 0.0, 'rs_ratio': 1.0}
        m['sr'] = m['close_stock'].pct_change(); m['br'] = m['close_bench'].pct_change()
        m['ratio'] = (1 + m['sr']) / (1 + m['br']); m['cum'] = m['ratio'].cumprod()
        slope = linear_regression_slope(m['cum'].dropna(), PARAMS['RS_PERIOD'])
        score = np.clip((slope + 0.3) / 0.6 * 10, 0, 10)
        return {'rs_score': round(float(score), 2), 'rs_slope': round(float(slope), 4), 'rs_ratio': round(float(m['cum'].iloc[-1]), 4)}
    except Exception:
        return {'rs_score': 5.0, 'rs_slope': 0.0, 'rs_ratio': 1.0}

# ------------------ Minervini ------------------
def check_minervini(df, phase_info, rs_info):
    if len(df) < 250: return 0, ['数据不足(<250日)'], []
    close = df['close'].astype(float); c = float(close.iloc[-1])
    passed = []; failed = []
    s150 = close.rolling(150).mean().iloc[-1]; s200 = close.rolling(200).mean().iloc[-1]
    if c > s150 and c > s200: passed.append('价>150/200MA')
    else: failed.append('价未站150/200MA')
    if s150 > s200: passed.append('150>200MA')
    else: failed.append('150MA未上穿200MA')
    if linear_regression_slope(close.rolling(200).mean().dropna(), 20) > 0: passed.append('200MA向上')
    else: failed.append('200MA未向上')
    s50 = close.rolling(50).mean().iloc[-1]
    if s50 > s150 > s200: passed.append('50>150>200MA')
    else: failed.append('均线未级联多头')
    if c > s50: passed.append('价>50MA')
    else: failed.append('价未站50MA')
    low52 = close.rolling(252).min().iloc[-1]
    if c >= low52 * (1 + PARAMS['MINERVINI_ABOVE_LOW_PCT']): passed.append('距52周低足够')
    else: failed.append('距52周低不足')
    high52 = close.rolling(252).max().iloc[-1]
    if c >= high52 * (1 - PARAMS['MINERVINI_NEAR_HIGH_PCT']): passed.append('距52周高足够近')
    else: failed.append('距52周高太远')
    if rs_info.get('rs_score', 0) >= 7.0: passed.append('RS强度≥70')
    else: failed.append('RS强度不足')
    return len(passed), failed, passed

# ------------------ VCP ------------------
def detect_vcp(df):
    if len(df) < PARAMS['VCP_LOOKBACK']:
        return {'is_vcp': False, 'contractions': 0, 'tightness': 0, 'vol_contracting': False}
    lb = df.iloc[-PARAMS['VCP_LOOKBACK']:].copy()
    seg = len(lb) // 3
    if seg < 5: return {'is_vcp': False, 'contractions': 0, 'tightness': 0, 'vol_contracting': False}
    ranges = []; vols = []
    for i in range(3):
        s = lb.iloc[i*seg:(i+1)*seg]
        sh = s['high'].astype(float).max(); sl = s['low'].astype(float).min()
        ranges.append((sh - sl) / sl * 100); vols.append(s['volume'].astype(float).mean())
    contr = (1 if ranges[0] > ranges[1] else 0) + (1 if ranges[1] > ranges[2] else 0)
    tight = ranges[2]
    volc = vols[0] > vols[1] > vols[2]
    is_vcp = contr >= 2 and tight < 8 and (not PARAMS['VCP_VOL_CONTRACTION'] or volc)
    return {'is_vcp': is_vcp, 'contractions': contr, 'tightness': round(tight, 2), 'vol_contracting': volc}

# ------------------ K线形态 ------------------
def detect_candle_patterns(df):
    if len(df) < 5: return []
    p = []; o = df['open'].astype(float); h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    body = abs(c - o); rng = h - l
    if c.iloc[-1] > o.iloc[-1] and body.iloc[-1] > rng.iloc[-1] * 0.6 and body.iloc[-1] > body.iloc[-2:-1].mean() * 1.5: p.append('大阳线')
    if (c.iloc[-3] < o.iloc[-3] and body.iloc[-2] < body.iloc[-3] * 0.3 and c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2): p.append('早晨之星')
    ls = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]; us = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
    if ls > body.iloc[-1] * 2 and us < body.iloc[-1] * 0.5: p.append('锤子线')
    if c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2]: p.append('阳线吞没')
    v = df['volume'].astype(float)
    if len(v) >= 5 and v.iloc[-1] < v.iloc[-5:-1].mean() * 0.7: p.append('缩量整理')
    return p
  # ------------------ 市场氛围 ------------------
def analyze_market_regime():
    if not PARAMS['MARKET_REGIME_ENABLE']:
        return {'regime': '中性', 'breadth': 50.0, 'hs300_phase': {}, 'advice': '市场氛围判断已关闭'}
    try:
        bench = _load_benchmark()
        if bench.empty or len(bench) < 200:
            return {'regime': '未知', 'breadth': 50.0, 'hs300_phase': {}, 'advice': '基准数据不足'}
        ph = classify_phase(bench)
        try:
            spot = ak.stock_zh_a_spot_em()
            if spot is not None and not spot.empty and '涨跌幅' in spot.columns:
                spot['涨跌幅'] = pd.to_numeric(spot['涨跌幅'], errors='coerce')
                breadth = (spot['涨跌幅'] > 0).sum() / spot['涨跌幅'].notna().sum() * 100
            else: breadth = 50
        except Exception: breadth = 50
        phase = ph.get('phase', 0)
        if phase == 2 and breadth > 55: regime, advice = '偏多🟢', '大盘Phase2上升+涨多跌少，可积极布局结构突破股'
        elif phase == 2: regime, advice = '震荡偏多🟡', '大盘上升但个股分化，精选RS强势的结构突破股'
        elif phase == 1: regime, advice = '筑底⚪', '大盘筑底中，控制仓位，关注接近突破的预警股'
        elif phase == 3: regime, advice = '派发🟠', '大盘派发阶段，降低仓位，只保留最强趋势股'
        elif phase == 4: regime, advice = '偏空🔴', '大盘下跌趋势，空仓或极轻仓'
        else: regime, advice = '中性⚪', '市场方向不明，观望为主'
        return {'regime': regime, 'breadth': round(breadth, 1), 'hs300_phase': ph, 'advice': advice}
    except Exception as e:
        return {'regime': '未知', 'breadth': 50.0, 'hs300_phase': {}, 'advice': f'判断异常: {e}'}

# ------------------ 回测 ------------------
def backtest_signal(df, hold_days=20, tp=0.10, sl=0.08):
    if len(df) < hold_days * 3: return {}
    close = df['close'].astype(float)
    wins = 0; pnl = 0; n = 0; max_dd = 0
    sw = PARAMS['lookbackSwing']
    for i in range(sw * 2 + 1, len(df) - hold_days):
        wh = df['high'].astype(float).iloc[i - sw*2:i].max()
        if close.iloc[i] > wh and close.iloc[i-1] <= wh:
            n += 1; entry = close.iloc[i]; peak = entry; done = False
            for d in range(1, hold_days + 1):
                if i + d >= len(close): break
                px = close.iloc[i + d]; peak = max(peak, px)
                max_dd = max(max_dd, (peak - px) / peak)
                if px >= entry * (1 + tp): wins += 1; pnl += tp; done = True; break
                if px <= entry * (1 - sl): pnl -= sl; done = True; break
            if not done:
                ret = close.iloc[min(i + hold_days, len(close) - 1)] / entry - 1
                pnl += ret
                if ret > 0: wins += 1
    if n == 0: return {}
    return {'win_rate': round(wins / n * 100, 1), 'avg_return': round(pnl / n * 100, 2), 'sample_size': n, 'max_dd': round(max_dd * 100, 1)}

# ------------------ 核心信号(融合) ------------------
def check_one_stock(df):
    if df is None or len(df) < PARAMS['min_data_len']: return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns: return None, "数据不足"
    close = df['close'].astype(float); high = df['high'].astype(float); low = df['low'].astype(float); vol = df['volume'].astype(float)
    c_now = float(close.iloc[-1]); c_prev = float(close.iloc[-2])
    last_sh = last_swing_high(high.to_numpy(), PARAMS['lookbackSwing'], PARAMS['lookbackSwing'])
    last_sl = last_swing_low(low.to_numpy(), PARAMS['lookbackSwing'], PARAMS['lookbackSwing'])
    bull_break = last_sh is not None and c_now > last_sh
    bear_break = last_sl is not None and c_now < last_sl
    bull_break_new = bull_break and not (last_sh is not None and c_prev > last_sh)
    bear_break_new = bear_break and not (last_sl is not None and c_prev < last_sl)
    bull_fvg = bool(low.iloc[-1] > high.iloc[-3]); bear_fvg = bool(high.iloc[-1] < low.iloc[-3])
    vagas_bull = bull_break_new and (not PARAMS['useFVG'] or bull_fvg)
    vagas_bear = bear_break_new and (not PARAMS['useFVG'] or bear_fvg)
    obv = calc_obv(df)
    obv_macd = obv.ewm(span=PARAMS['fastLen'], adjust=False).mean() - obv.ewm(span=PARAMS['slowLen'], adjust=False).mean()
    obv_signal = obv_macd.ewm(span=PARAMS['signalLen'], adjust=False).mean()
    obv_bull_cross = (obv_macd > obv_signal) & (obv_macd.shift(1) <= obv_signal.shift(1))
    obv_bear_cross = (obv_macd < obv_signal) & (obv_macd.shift(1) >= obv_signal.shift(1))
    obv_bull = bool(obv_macd.iloc[-1] > obv_signal.iloc[-1]); obv_bear = bool(obv_macd.iloc[-1] < obv_signal.iloc[-1])
    cw = PARAMS['confirmWindow']
    recent_bull = bool(obv_bull_cross.iloc[-cw:].any()); recent_bear = bool(obv_bear_cross.iloc[-cw:].any())
    phase_info = classify_phase(df, c_now); phase = phase_info['phase']
    rs_info = calc_relative_strength(df); rs_raw = rs_info.get('rs_score', 5)
    minervini_pass = 0
    if PARAMS['MINERVINI_ENABLE']:
        minervini_pass, _, _ = check_minervini(df, phase_info, rs_info)
    vcp_info = detect_vcp(df)
    candles = detect_candle_patterns(df)
    rsi = calc_rsi(close).iloc[-1]
    k, d, j = calc_kdj(df)
    bu, bm, bl = calc_bollinger(df)
    atr = calc_atr(df, PARAMS['atrLen']); at = float(atr.iloc[-1])
    if pd.isna(at) or at <= 0: return None, "数据不足"
    swing_stop = last_sl * 0.98 if last_sl else c_now * 0.92
    atr_stop = c_now - at * PARAMS['atrMultSL']
    stop = max(atr_stop, swing_stop); tp_price = c_now + at * PARAMS['atrMultTP']
    risk = c_now - stop; reward = tp_price - c_now
    rr = reward / risk if risk > 0 else 0
    w = PARAMS['SCORE_WEIGHTS']; sd = {}
    st = 0
    if vagas_bull: st = 100 + (10 if bull_fvg else 0)
    elif bull_break: st = 60
    elif last_sh and c_now >= last_sh * (1 - PARAMS['APPROACH_PCT']): st = 40
    sd['structure'] = min(100, st)
    sd['volume'] = 100 if (obv_bull and recent_bull) else (70 if obv_bull else (50 if recent_bull else 0))
    sd['phase'] = phase_info.get('confidence', 60) if phase == 2 else (phase_info.get('confidence', 40) * 0.6 if phase == 1 else (20 if phase == 3 else 5))
    sd['rs'] = rs_raw * 10
    pt = 0
    if vcp_info['is_vcp']: pt += 40
    if '大阳线' in candles: pt += 30
    if '阳线吞没' in candles: pt += 25
    if '早晨之星' in candles: pt += 35
    if '锤子线' in candles: pt += 20
    if '缩量整理' in candles: pt += 15
    if 40 <= rsi <= 70: pt += 10
    sd['pattern'] = min(100, pt)
    fd = 50
    if vagas_bull and vol.iloc[-1] > vol.iloc[-20:].mean() * 1.2: fd += 30
    if c_now > bm.iloc[-1]: fd += 10
    if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]: fd += 10
    sd['fundamental'] = min(100, fd)
    total = round(sum(sd[key] * w[key] for key in w), 1)
    is_buy = (vagas_bull and obv_bull and phase == 2 and total >= PARAMS['SCORE_BUY']
              and rs_raw >= PARAMS['RS_MIN_SCORE'] and rr >= PARAMS['MIN_RR_RATIO'])
    if PARAMS['MINERVINI_ENABLE'] and is_buy and minervini_pass < PARAMS['MINERVINI_MIN_PASS']:
        is_buy = False
    is_watch = (not is_buy and last_sh and c_now >= last_sh * (1 - PARAMS['APPROACH_PCT']) and not bull_break
                and obv_bull and phase in (1, 2) and PARAMS['SCORE_WATCH'] <= total < PARAMS['SCORE_BUY'])
    is_app = (not is_buy and not is_watch and PARAMS['PRE_SIGNAL'] and last_sh and not bull_break
              and c_now >= last_sh * (1 - PARAMS['APPROACH_PCT']) and obv_bull
              and PARAMS['SCORE_APPROACH'] <= total < PARAMS['SCORE_WATCH'])
    is_sell = vagas_bear and obv_bear and phase in (3, 4) and PARAMS['INCLUDE_WARNINGS']
    if not (is_buy or is_watch or is_app or is_sell): return None, "无信号"
    bt = backtest_signal(df) if PARAMS['BACKTEST_ENABLE'] else {}
    sig = "🟢BUY" if is_buy else ("🟡WATCH" if is_watch else ("🔔APPROACH" if is_app else "⚠️SELL"))
    tier = 3 if is_buy else (2 if is_watch else (1 if is_app else 0))
    trig = []
    if is_buy:
        trig.append("VAGAS突破摆动高")
        if PARAMS['useFVG'] and bull_fvg: trig.append("FVG缺口")
        if bool(obv_bull_cross.iloc[-1]): trig.append("OBV金叉")
        elif recent_bull: trig.append("OBV近期金叉")
        trig.append(f"Phase{phase_info['phase_name']}"); trig.append(f"RS{rs_raw:.1f}")
        if vcp_info['is_vcp']: trig.append("VCP收缩")
        if minervini_pass >= 6: trig.append(f"Minervini{minervini_pass}/8")
    elif is_watch or is_app:
        trig.append(f"接近摆动高(差{(last_sh - c_now) / last_sh * 100:.1f}%)"); trig.append("OBV蓄势")
        if vcp_info['is_vcp']: trig.append("VCP收缩")
    else:
        trig.append("跌破摆动低")
        if bool(obv_bear_cross.iloc[-1]): trig.append("OBV死叉")
    trig.extend(candles[:2])
    sig_date = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    pos_pct = min(PARAMS['RISK_PER_TRADE'] / (risk / c_now) * 100, 20) if risk > 0 else 5
    return {"代码": None, "名称": None, "行业": "", "最新价": round(c_now, 2), "信号价": round(c_now, 2), "信号日期": sig_date,
            "信号": sig, "触发": "+".join(trig) if trig else "—", "是否做多": bool(is_buy or is_watch or is_app), "tier": tier,
            "综合评分": total, "评分详情": sd,
            "摆动高": round(last_sh, 2) if last_sh else None, "摆动低": round(last_sl, 2) if last_sl else None,
            "FVG": "↑缺口" if bull_fvg else ("↓缺口" if bear_fvg else "无"), "OBV状态": "多" if obv_bull else ("空" if obv_bear else "—"),
            "Phase": phase_info['phase_name'], "RS评分": rs_raw, "Minervini通过": minervini_pass,
            "VCP": "✓" if vcp_info['is_vcp'] else "—", "RSI": round(float(rsi), 1) if not pd.isna(rsi) else None,
            "ATR": round(at, 3), "建议止损": round(stop, 2), "建议止盈": round(tp_price, 2), "风险回报比": round(rr, 2),
            "建议仓位%": round(pos_pct, 1), "回测胜率%": bt.get('win_rate'), "回测样本": bt.get('sample_size', 0),
            "score": round(at / c_now * 100, 2), "resonance": False, "resonance_sector": ""}, None
  # ------------------ 数据获取 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception: time.sleep(1 + attempt)
    return None

def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    d = _bs_fetch(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
    if d is not None and not d.empty:
        for c in ['open', 'high', 'low', 'close', 'volume']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        if len(d) >= PARAMS['min_data_len']:
            return d
    return _fetch_hist_em(code, sy, ey)

def snapshot_prefilter(codes):
    if not PARAMS['SNAPSHOT_PRE']: return codes
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns: return codes
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns: spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns: m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns: m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes if c[3:] in keep]
        print(f"  快照预筛: {len(codes)} → {len(out)} 只")
        return out if out else codes
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes

def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None: return {"__fail__": "抓取失败"}
        if len(df) < PARAMS['min_data_len']: return {"__fail__": "数据不足"}
        time.sleep(PARAMS['SLEEP'])
        info, reason = check_one_stock(df)
        if info is None: return {"__fail__": reason}
        info["代码"] = code; info["名称"] = name
        return info
    except Exception:
        return {"__fail__": "抓取失败"}

def run_scan():
    global _INDUSTRY_MAP, _FAIL_STATS
    _FAIL_STATS = {k: 0 for k in _FAIL_STATS}
    print("连接 Baostock（行业表+列表+子进程登录）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
                for _, row in ind.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条")
        except Exception as e: print(f"  取行业表异常: {e}")
        try: stock_df = bs.query_stock_basic().get_data()
        except Exception as e: print(f"  baostock 取列表异常: {e}"); stock_df = pd.DataFrame()
        _bs_logout()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        for attempt in range(3):
            try:
                d = ak.stock_info_a_code_name()
                if d is not None and not d.empty and 'code' in d.columns:
                    nc = 'name' if 'name' in d.columns else d.columns[1]
                    d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                    d['code'] = d['code'].astype(str).str.zfill(6)
                    d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                    d['type'] = '1'; d['status'] = '1'; stock_df = d; break
            except Exception as e: print(f"  akshare列表第{attempt+1}次失败: {e}")
            time.sleep(2 + attempt)
    if stock_df is None or stock_df.empty: print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = snapshot_prefilter(stock_df['code'].tolist())
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT: codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]
    results = []; fail = 0
    print(f"开始VAGAS融合扫描 {len(tasks)} 只...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="vagas融合", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res: fail += 1; _FAIL_STATS[res["__fail__"]] = _FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['信号']} 评分={res['综合评分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail)
    print("\n各失败原因统计：")
    for k, v in _FAIL_STATS.items():
        if v: print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values(['tier', '综合评分'], ascending=[False, False]).reset_index(drop=True)
    return df

# ------------------ 行业+风口 ------------------
def enrich(df):
    targets = df.to_dict('records')
    for r in targets: r['行业'] = _INDUSTRY_MAP.get(r['代码'], '—')
    labeled = [r for r in targets if r.get('行业') not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty: break
        except Exception: time.sleep(2 + i)
    hot = []
    if not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
        h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        hot = [(str(r['板块名称']), round(float(r['_chg']), 2)) for _, r in h.head(HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]; cnt = 0
    for r in targets:
        sec = r.get('行业', ''); m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh): m = hh; break
        if m: r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', 'tier', '综合评分'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def _align_suffix(r, spot_now):
    """【修复】纯字符串拼接(无花括号), 防 unmatched '}'"""
    sig_price = r.get('信号价', r.get('最新价')); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price): return ""
    head = "🕒信号" + str(sig_price)
  
    if sig_date and not pd.isna(sig_date): head += "@" + str(sig_date)[:10][-5:]
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return " | " + head + " → 现价" + str(now) + "@run(" + format(chg, "+.1f") + "%)"
        except Exception:
            return " | " + head
    return " | " + head
  # ------------------ 三段式推送 ------------------
def build_push(df, cluster, hot, market_regime, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    buys = df[df['信号'] == '🟢BUY'] if '信号' in df.columns else pd.DataFrame()
    watches = df[df['信号'] == '🟡WATCH'] if '信号' in df.columns else pd.DataFrame()
    approaches = df[df['信号'] == '🔔APPROACH'] if '信号' in df.columns else pd.DataFrame()
    warns = df[df['信号'].astype(str).str.contains('SELL|警示')] if '信号' in df.columns else pd.DataFrame()
    n_buy, n_watch, n_app, n_warn, n_reso = len(buys), len(watches), len(approaches), len(warns), len(reso)
    L = [f"**🚀 VAGAS融合版 | 结构突破×OBV×Phase×RS** | {datetime.now():%Y-%m-%d %H:%M}",
         "*(VAGAS突破+OBV确认+Phase趋势+RS强度+Minervini+VCP; 非预测)*", ""]
    if market_regime:
        L.append(f"🌡️ **市场氛围**: {market_regime.get('regime', '未知')} | 上涨家数占比 {market_regime.get('breadth', 50)}%")
        L.append(f"💡 {market_regime.get('advice', '')}"); L.append("")
    L.append(f"📊 **信号统计**: BUY {n_buy} | WATCH {n_watch} | APPROACH {n_app} | 警示 {n_warn} | 🎯风口 {n_reso}"); L.append("")
    if hot: L.append("🌪️ **风口板块**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster: L.append("🚀 **结构突破板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        st = str(r['信号'])
        if '🟢BUY' in st: flag, pos = "🟢", f"止损{r['建议止损']}/止盈{r['建议止盈']} | 仓位{r.get('建议仓位%', '—')}%"
        elif '🟡WATCH' in st: flag, pos = "🟡", f"观察, 止损{r['建议止损']}"
        elif 'APPROACH' in st: flag, pos = "🔔", f"预警观察, 破位止损{r['建议止损']}"
        else: flag, pos = "⚠️", "回避"
        extra = []
        if r.get('Phase'): extra.append(f"Phase:{r['Phase']}")
        if r.get('RS评分') is not None: extra.append(f"RS:{r['RS评分']}")
        if r.get('VCP') == '✓': extra.append("VCP✓")
        if r.get('RSI') is not None: extra.append(f"RSI:{r['RSI']}")
        if r.get('回测胜率%') is not None: extra.append(f"胜率{r['回测胜率%']}%({r.get('回测样本', 0)}次)")
        extra_str = (" | " + " | ".join(extra)) if extra else ""
        return (f"- {flag} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 评分{r.get('综合评分', '—')} "
                f"触发:{r['触发']} | 现价{r['最新价']} OBV={r['OBV状态']} FVG={r['FVG']}{extra_str} | {pos}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 遇风口共振 共{n_reso}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    if not buys.empty:
        L.append(f"### 🟢 BUY(已突破确认·最高优先级) 共{n_buy}只")
        L += [line(r) for _, r in buys.iterrows()]; L.append("")
    if not watches.empty:
        L.append(f"### 🟡 WATCH(接近突破·观察等待) 共{n_watch}只")
        L += [line(r) for _, r in watches.iterrows()]; L.append("")
    if not approaches.empty:
        L.append(f"### 🔔 APPROACH(左侧预警·风险更高) 共{n_app}只")
        L += [line(r) for _, r in approaches.iterrows()]; L.append("")
    if not warns.empty:
        L.append(f"### ⚠️ 空头警示(回避) 共{n_warn}只")
        L += [line(r) for _, r in warns.iterrows()]
    L.append(""); L.append("---"); L.append("### 🧠 今日操作建议"); L.append("")
    if n_buy > 0:
        tb = buys.iloc[0]
        L.append(f"🔑 **最值得关注**: {tb['名称']}({tb['代码']}) — 评分{tb['综合评分']}，{tb['Phase']}，RS{tb.get('RS评分', '—')}")
        L.append(f"  若进场: 进 {tb['最新价']} → 损 {tb['建议止损']} / 标 {tb['建议止盈']} | R:R={tb.get('风险回报比', '—')}:1")
    if market_regime:
        rg = market_regime.get('regime', '')
        if '偏多' in rg or '上升' in rg: L.append("📌 **操作方向**: 市场偏多，可挑选BUY信号分批进场，优先RS>7+遇风口标的")
        elif '派发' in rg or '偏空' in rg: L.append("📌 **操作方向**: 市场偏弱，降低仓位，只保留最强BUY信号，或空仓观望")
        elif '筑底' in rg: L.append("📌 **操作方向**: 市场筑底中，控制仓位，关注WATCH列表等待突破确认")
        else: L.append("📌 **操作方向**: 市场方向不明，观望为主，只关注最高评分BUY信号")
    L.append(""); L.append("⚠️ **风险提示**: APPROACH为左侧预警，风险远高于BUY；务必按建议位止损；历史胜率仅供参考。")
    return "\n".join(L)

def build_console_report(df, spot_now=None):
    if df is None or df.empty:
        return "无信号"
    disp = df.copy()
    show = ['代码', '名称', '信号', '综合评分', 'Phase', 'RS评分', '最新价', '建议止损', '建议止盈', '风险回报比', '触发']
    show = [c for c in show if c in disp.columns]
    return disp[show].head(PUSH_TOP).to_string(index=False)

# ------------------ 主函数 ------------------
def main():
    print("=" * 80)
    print(f"🚀 VAGAS融合版 | {datetime.now():%Y-%m-%d %H:%M}")
    print(f"   预警={'开' if PARAMS['PRE_SIGNAL'] else '关'} | 复核={'开' if REALTIME_RECHECK else '关'} | FVG={'开' if PARAMS['useFVG'] else '关'}")
    print(f"   Minervini={'开' if PARAMS['MINERVINI_ENABLE'] else '关'} | VCP={'开' if PARAMS['VCP_ENABLE'] else '关'} | 回测={'开' if PARAMS['BACKTEST_ENABLE'] else '关'} | 氛围={'开' if PARAMS['MARKET_REGIME_ENABLE'] else '关'}")
    print("=" * 80)
    market_regime = {}
    if PARAMS['MARKET_REGIME_ENABLE']:
        print("\n🌡️ 分析市场氛围...")
        market_regime = analyze_market_regime()
        print(f"  市场氛围: {market_regime.get('regime', '未知')} | 上涨占比 {market_regime.get('breadth', 50)}%")
    df = run_scan()
    if df is None or df.empty:
        print("\n本次无有效信号 (多维度过滤极严, 0命中属正常)。")
        if SERVERCHAN_KEY:
            send_serverchan("🚀 VAGAS融合版 | 当日无有效信号",
                f"**VAGAS融合版** | {datetime.now():%Y-%m-%d}\n\n市场氛围: {market_regime.get('regime', '未知')}\n扫描结果: 无通过多维度过滤的信号。\n\n*(Phase+RS+Minervini+VCP多维过滤严格，无信号属正常，非故障)*")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    df, rt = _refresh_realtime_price(df)
    if not df.empty:
        _b = len(df); df = _realtime_recheck(df)
        print(f"  实时复核: {_b} → {len(df)}")
    if df is None or df.empty:
        print("\n实时复核后无有效信号。")
        if SERVERCHAN_KEY:
            send_serverchan("🚀 VAGAS融合版 | 复核后无有效信号", "**VAGAS融合版** | 当日信号经实时复核全部失效(破位/追高/离开接近区)。\n\n*(防失效信号, 非故障)*")
        sys.exit(0)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"vagas_fusion_{tag}.csv"), index=False, encoding="utf-8-sig")
        export = {"date": tag, "market_regime": market_regime, "cluster": cluster, "hot_sectors": hot,
                  "n_total": int(len(df)),
                  "n_buy": int((df['信号'] == '🟢BUY').sum()),
                  "n_watch": int((df['信号'] == '🟡WATCH').sum()),
                  "n_approach": int((df['信号'] == '🔔APPROACH').sum()),
                  "n_warn": int(df['信号'].astype(str).str.contains('SELL|警示').sum()),
                  "fail_stats": _FAIL_STATS, "hits": df.to_dict('records')}
        with open(os.path.join(OUTPUT_DIR, f"vagas_fusion_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump(export, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/vagas_fusion_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        print("\n" + build_console_report(df, rt))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_buy = int((df['信号'] == '🟢BUY').sum()) if '信号' in df.columns else 0
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"🚀 VAGAS融合版 BUY{n_buy}只 🎯风口{n_reso} | {market_regime.get('regime', '')}",
                            build_push(df, cluster, hot, market_regime, rt))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_vagas_obv<<<
  
  
