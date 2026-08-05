# -*- coding: utf-8 -*-
"""
trend_judge_screener.py —— 综合走势判断+选股(5维评分·精确版) × 三脚本形态共振整合版 · 矩阵规格
5维走势评分(趋势/动量/量价/MACD/位置)+6道质量门槛, 全市场选技术面最强票。
【整合 macd_ma_crossover + boll_wbottom + vagas_obv 三脚本】
  在质量门槛①-⑥之上叠加【形态共振】: 逐票用同一份日线跑三个形态检测器——
   🧩 MACD金叉趋势启动: 近N根金叉+站上年线/季线+零轴上   (来自 macd_ma_crossover)
   🧩 布林带W底: 最新根突破上轨放量+中轨颈线+两低点抬高缩量 (来自 boll_wbottom)
   🧩 VAGAS结构突破+OBV: 刚突破摆动高+OBV-MACD多头, 含🔔接近突破预警 (来自 vagas_obv)
  每触发一个 +PATTERN_BONUS 分(封顶100); PATTERN_ONLY=1 时只推"至少触发一个形态"的票。
【实时价】腾讯实时价刷新"最新价"(午盘=午盘实时/盘后=当日收盘), 信号日收盘存"信号价", 对齐列显示
  🕒信号价@日期 → 现价@run(涨跌幅); 腾讯失败回退日线收盘。
【诚实定位】概率性技术面强弱评分+形态共振, 非预测; 提质减量, 弱市可能0命中; 必结合仓位/止损。
【工程规格】双源baostock+东财+硬超时; 多进程; 快照预筛; 行业本地join+聚类+风口🎯; 推送分页; 收尾防护。
"""
import os, re, sys, json, time, random, warnings, traceback, requests
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta
import pandas as pd

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==================== 参数 ====================
PARAMS = dict(
    W_TREND=0.30, W_MOMENTUM=0.20, W_VOLUME=0.15, W_MACD=0.20, W_POSITION=0.15,
    LABEL_STRONG_BULL=70, LABEL_BULL=55, LABEL_NEUTRAL=40, LABEL_BEAR=25,
    SCORE_MIN=65.0,
    lookback_days=400, min_data_len=120,
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SCORE_MIN_ENV = float(os.environ.get('SCORE_MIN', str(PARAMS['SCORE_MIN'])))
MIN_TREND_SCORE = float(os.environ.get('MIN_TREND_SCORE', '0.5'))
MIN_STRONG_DIMS = int(os.environ.get('MIN_STRONG_DIMS', '3'))
STRONG_DIM_THRESH = float(os.environ.get('STRONG_DIM_THRESH', '0.55'))
MIN_VOLUME_SCORE = float(os.environ.get('MIN_VOLUME_SCORE', '0.4'))
MAX_RET20 = float(os.environ.get('MAX_RET20', '0.5'))
MAX_RSI = float(os.environ.get('MAX_RSI', '80'))
MIN_AMOUNT = float(os.environ.get('MIN_AMOUNT', '1.0e8'))
# ---- 三脚本形态共振 ----
PATTERN_BONUS = float(os.environ.get('PATTERN_BONUS', '8'))
PATTERN_ONLY = os.environ.get('PATTERN_ONLY', '0').strip() in ('1', 'true', 'True')
PRE_SIGNAL = os.environ.get('PRE_SIGNAL', '1').strip() in ('1', 'true', 'True')
APPROACH_PCT = float(os.environ.get('APPROACH_PCT', '0.02'))
USE_FVG = os.environ.get('USE_FVG', '1').strip() in ('1', 'true', 'True')

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "趋势不足": 0, "共振不足": 0, "量价不足": 0,
              "涨幅过大": 0, "RSI过热": 0, "流动性不足": 0, "评分不足": 0, "无形态": 0}

# ------------------ 推送 ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok:
            return True
        print(f"  sdk返回非成功({ret}), 回退requests")
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        j = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15).json()
        return j.get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False

def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
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
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print("📲 推送完成" + (" ✅" if ok else " ⚠️存在失败"))
    return ok

# ------------------ baostock ------------------
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
            print(f"  baostock 登录异常: {e}")
        time.sleep(2 * (i + 1))
    return False

def _bs_logout():
    global _BS_LOGGED
    try:
        if _BS_LOGGED:
            bs.logout()
    except Exception as e:
        print(f"  baostock 登出异常: {e}")
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

def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)

def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

# ==================== 基础指标 ====================
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n):
    return s.rolling(n).mean()

def _rsi(s, n):
    d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean(); al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-9))

def _obv(close, volume):
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()

# ==================== 实时价(腾讯源) ====================
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
                    if '=' not in line:
                        continue
                    f = line.split('=', 1)[1].strip().strip('"').split('~')
                    if len(f) > 4 and f[2]:
                        try:
                            px = float(f[3])
                            if px > 0:
                                out[f[2].zfill(6)] = px
                        except Exception:
                            pass
            except Exception as e:
                print(f"   [实时价] 批次失败: {e}")
            time.sleep(0.3)
    except Exception as e:
        print(f"  腾讯实时价异常: {e}")
    return out

def _refresh_realtime_price(df):
    if df is None or df.empty:
        return df, {}
    df = df.copy()
    if '信号价' not in df.columns:
        df['信号价'] = df['最新价']
    codes6 = [str(c).split('.')[-1].zfill(6) for c in df['代码']]
    rt = _fetch_realtime_tencent(codes6)
    if rt:
        df['实时价'] = [rt.get(c) for c in codes6]
        df['最新价'] = df['实时价'].where(df['实时价'].notna(), df['最新价'])
        print(f"  实时价刷新: 腾讯取到 {int(df['实时价'].notna().sum())}/{len(df)} 只")
    else:
        print("  实时价刷新: 腾讯未取到, 沿用日线收盘")
    return df, rt

def _align_suffix(r, spot_now):
    sig_price = r.get('信号价', r.get('最新价')); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price):
        return ""
    head = f"🕒信号{sig_price}"
    if sig_date and not pd.isna(sig_date):
        head += f"@{str(sig_date)[:10][-5:]}"
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return f" | {head} → 现价{now}@run({chg:+.1f}%)"
        except Exception:
            return f" | {head}"
    return f" | {head}"

# ==================== 三脚本形态检测器 ====================
def detect_macd_ma_cross(df, cross_lookback=3):
    try:
        close = df['close'].astype(float); L = len(close) - 1
        if L < 60:
            return False, None
        cL = float(close.iloc[L])
        ma250 = close.rolling(250, min_periods=250).mean()
        ma60 = close.rolling(60, min_periods=60).mean()
        ma_type = None
        if pd.notna(ma250.iloc[L]) and cL > ma250.iloc[L]:
            ma_type = "年线"
        elif pd.notna(ma60.iloc[L]) and cL > ma60.iloc[L]:
            ma_type = "季线"
        if not ma_type:
            return False, None
        dif = _ema(close, 12) - _ema(close, 26); dea = _ema(dif, 9)
        if not (dif.iloc[L] > 0 and dea.iloc[L] > 0):
            return False, None
        cross = (dif.shift(1) <= dea.shift(1)) & (dif > dea)
        if not bool(cross.iloc[-cross_lookback:].any()):
            return False, None
        return True, f"趋势启动·{ma_type}"
    except Exception:
        return False, None

def detect_boll_wbottom(df, bb_period=20, bb_std=2.0, lookback=75, alpha=0.006,
                        min_gap=8, max_gap=45, shrink=0.85, expand=1.5):
    try:
        if len(df) < 100:
            return False
        close = df['close'].astype(float); volume = df['volume'].astype(float)
        std = close.rolling(bb_period, min_periods=bb_period).std()
        mid = close.rolling(bb_period, min_periods=bb_period).mean()
        upper = mid + bb_std * std; lower = mid - bb_std * std
        vol_ma = volume.rolling(10, min_periods=1).mean()
        c = close.to_numpy(); v = volume.to_numpy(); u = upper.to_numpy()
        m = mid.to_numpy(); lo = lower.to_numpy(); vm = vol_ma.to_numpy()
        n = len(c); i = n - 1
        if np.isnan(u[i]) or np.isnan(vm[i]) or vm[i] <= 0:
            return False
        if not (c[i] > u[i] and v[i] >= vm[i] * expand):
            return False
        lo_i = max(i - lookback, 0)
        for j in range(i - 1, lo_i, -1):
            if np.isnan(m[j]) or c[j] <= 0:
                continue
            if abs(c[j] - m[j]) < alpha * c[j]:
                for k in range(j - 1, lo_i, -1):
                    if np.isnan(lo[k]) or c[k] <= 0:
                        continue
                    if abs(c[k] - lo[k]) < alpha * c[k]:
                        threshold = c[k]
                        for mm in range(i - 1, j, -1):
                            if np.isnan(lo[mm]):
                                continue
                            if (abs(c[mm] - lo[mm]) < alpha * c[mm] and c[mm] > lo[mm] and c[mm] > threshold * 0.995):
                                gap = abs(mm - k)
                                if not (min_gap <= gap <= max_gap):
                                    continue
                                if v[k] <= 0 or v[mm] >= v[k] * shrink:
                                    continue
                                return True
        return False
    except Exception:
        return False

def _swing_high(high_arr, left=5, right=5):
    n = len(high_arr)
    for i in range(n - 1 - right, left - 1, -1):
        if high_arr[i] >= np.max(high_arr[i - left:i + right + 1]):
            return float(high_arr[i])
    return None

def detect_vagas_obv(df, use_fvg=True, confirm_window=3, pre_signal=True, approach_pct=0.02):
    try:
        if len(df) < 20:
            return False, None
        close = df['close'].astype(float); high = df['high'].astype(float)
        low = df['low'].astype(float); volume = df['volume'].astype(float)
        obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
        obv_fast = obv.ewm(span=12, adjust=False).mean(); obv_slow = obv.ewm(span=26, adjust=False).mean()
        obv_macd = obv_fast - obv_slow; obv_signal = obv_macd.ewm(span=9, adjust=False).mean()
        obv_bull_cross = (obv_macd > obv_signal) & (obv_macd.shift(1) <= obv_signal.shift(1))
        last_sh = _swing_high(high.to_numpy())
        c_now = float(close.iloc[-1]); c_prev = float(close.iloc[-2])
        bull_break = (last_sh is not None) and (c_now > last_sh)
        bull_break_prev = (last_sh is not None) and (c_prev > last_sh)
        bull_break_new = bull_break and not bull_break_prev
        bull_fvg = bool(low.iloc[-1] > high.iloc[-3])
        vagas_bull = bull_break_new and (not use_fvg or bull_fvg)
        obv_bull = bool(obv_macd.iloc[-1] > obv_signal.iloc[-1])
        recent_bull = bool(obv_bull_cross.iloc[-confirm_window:].any())
        if vagas_bull and obv_bull and (bool(obv_bull_cross.iloc[-1]) or recent_bull):
            return True, "VAGAS突破"
        if pre_signal and last_sh is not None and (not bull_break) and c_now >= last_sh * (1 - approach_pct) and obv_bull:
            return True, "🔔VAGAS接近"
        return False, None
    except Exception:
        return False, None

# ==================== 综合走势评分 + 形态共振 ====================
def _label(score):
    if score >= PARAMS['LABEL_STRONG_BULL']:
        return "🟢强多"
    if score >= PARAMS['LABEL_BULL']:
        return "🟢偏多"
    if score >= PARAMS['LABEL_NEUTRAL']:
        return "⚪中性"
    if score >= PARAMS['LABEL_BEAR']:
        return "🔴偏空"
    return "🔴强空"

def check_one_stock(df):
    if df is None or len(df) < PARAMS['min_data_len']:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"
    close = df['close'].astype(float); high = df['high'].astype(float)
    low = df['low'].astype(float); volume = df['volume'].astype(float)
    L = len(df) - 1
    ma5 = _sma(close, 5); ma10 = _sma(close, 10); ma20 = _sma(close, 20); ma60 = _sma(close, 60)
    if pd.isna(ma60.iloc[L]):
        return None, "数据不足"
    cL = float(close.iloc[L])

    m5, m10, m20, m60 = float(ma5.iloc[L]), float(ma10.iloc[L]), float(ma20.iloc[L]), float(ma60.iloc[L])
    trend = 0.0
    if m5 > m10 > m20: trend += 0.3
    if m20 > m60: trend += 0.1
    if cL > m20: trend += 0.2
    if m20 > float(ma20.iloc[L - 6]): trend += 0.2
    if m60 > float(ma60.iloc[L - 6]): trend += 0.2

    ret20 = (cL / float(close.iloc[L - 20]) - 1) if close.iloc[L - 20] else 0.0
    ret_score = _clip((ret20 + 0.10) / 0.30)
    rsi = float(_rsi(close, 14).iloc[L])
    if 40 <= rsi <= 70: rsi_score = 1.0
    elif 30 <= rsi < 40 or 70 < rsi <= 80: rsi_score = 0.5
    else: rsi_score = 0.2
    momentum = 0.6 * ret_score + 0.4 * rsi_score

    vol_ma20 = float(_sma(volume, 20).iloc[L])
    vol_ratio = (float(volume.iloc[L]) / vol_ma20) if vol_ma20 > 0 else 0.0
    vol_score = _clip((vol_ratio - 0.5) / 1.5)
    obv = _obv(close, volume); obv_ma20 = _sma(obv, 20)
    obv_score = 1.0 if (pd.notna(obv_ma20.iloc[L]) and obv.iloc[L] > obv_ma20.iloc[L]) else 0.4
    volume_dim = 0.5 * vol_score + 0.5 * obv_score

    dif = _ema(close, 12) - _ema(close, 26); dea = _ema(dif, 9)
    dL, eL = float(dif.iloc[L]), float(dea.iloc[L])
    macd = 0.0
    if dL > eL: macd += 0.4
    if dL > 0: macd += 0.3
    cross = (dif.shift(1) <= dea.shift(1)) & (dif > dea)
    if bool(cross.iloc[-5:].any()): macd += 0.3

    h20 = float(high.iloc[-20:].max()); l20 = float(low.iloc[-20:].min())
    pos = _clip((cL - l20) / (h20 - l20)) if h20 > l20 else 0.5
    near_high_pct = round((h20 - cL) / h20 * 100, 1) if h20 else None

    total = (PARAMS['W_TREND'] * trend + PARAMS['W_MOMENTUM'] * momentum +
             PARAMS['W_VOLUME'] * volume_dim + PARAMS['W_MACD'] * macd +
             PARAMS['W_POSITION'] * pos)
    score = round(total * 100, 1)

    # 质量门槛 ①-⑥
    if trend < MIN_TREND_SCORE:
        return None, "趋势不足"
    dims = [trend, momentum, volume_dim, macd, pos]
    strong_dims = sum(1 for d in dims if d >= STRONG_DIM_THRESH)
    if strong_dims < MIN_STRONG_DIMS:
        return None, "共振不足"
    if volume_dim < MIN_VOLUME_SCORE:
        return None, "量价不足"
    if ret20 > MAX_RET20:
        return None, "涨幅过大"
    if rsi > MAX_RSI:
        return None, "RSI过热"
    avg_amount_yi = None
    if 'amount' in df.columns:
        amt = pd.to_numeric(df['amount'], errors='coerce').iloc[-20:].mean()
        if not pd.isna(amt):
            avg_amount_yi = round(float(amt) / 1e8, 2)
            if float(amt) < MIN_AMOUNT:
                return None, "流动性不足"

    # 形态共振: 三脚本检测器
    patterns = []
    ok_mc, lab_mc = detect_macd_ma_cross(df)
    if ok_mc: patterns.append(lab_mc)
    if detect_boll_wbottom(df): patterns.append("布林W底")
    ok_vg, lab_vg = detect_vagas_obv(df, use_fvg=USE_FVG, pre_signal=PRE_SIGNAL, approach_pct=APPROACH_PCT)
    if ok_vg: patterns.append(lab_vg)
    n_pattern = len(patterns)
    score = round(min(100.0, score + n_pattern * PATTERN_BONUS), 1)

    if score < SCORE_MIN_ENV:
        return None, "评分不足"
    if PATTERN_ONLY and n_pattern == 0:
        return None, "无形态"

    sig_date = pd.to_datetime(df['date'].iloc[L]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(cL, 2), "信号价": round(cL, 2), "信号日期": sig_date,
            "走势评分": score, "多空标签": _label(score), "共振维度": strong_dims,
            "形态共振": n_pattern, "形态标签": "+".join(patterns) if patterns else "—",
            "趋势分": round(trend, 2), "动量分": round(momentum, 2), "量价分": round(volume_dim, 2),
            "MACD分": round(macd, 2), "位置分": round(pos, 2),
            "RSI": round(rsi, 1), "量比": round(vol_ratio, 2), "日均成交额亿": avg_amount_yi,
            "近20日涨幅%": round(ret20 * 100, 1), "距20日高%": near_high_pct,
            "resonance": False, "resonance_sector": ""}, None

# ------------------ 历史双源 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'})
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'] if c in d.columns]
                    return d[cols]
        except Exception:
            time.sleep(1 + attempt)
    return None

def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume,amount", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d
        except Exception:
            pass
    return _fetch_hist_em(code, sy, ey)

def snapshot_prefilter(codes_with_prefix):
    if not PARAMS['SNAPSHOT_PRE']:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for col in ['最新价', '成交额', '换手率']:
            if col in spot.columns:
                spot[col] = pd.to_numeric(spot[col], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes_with_prefix

def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < PARAMS['min_data_len']:
            return {"__fail__": "数据不足"}
        time.sleep(PARAMS['SLEEP'])
        info, reason = check_one_stock(df)
        if info is None:
            return {"__fail__": reason}
        info["代码"] = code; info["名称"] = name
        return info
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}

def run_scan():
    global _INDUSTRY_MAP, FAIL_STATS
    FAIL_STATS = {k: 0 for k in FAIL_STATS}
    print("连接 Baostock（行业表+列表+子进程登录）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
                for _, row in ind.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}"); stock_df = pd.DataFrame()
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
            except Exception as e:
                print(f"  akshare列表第{attempt+1}次失败: {e}")
            time.sleep(2 + attempt)
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = snapshot_prefilter(stock_df['code'].tolist())
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]
    results = []; fail_count = 0
    print(f"开始综合走势判断+形态共振扫描 {len(tasks)} 只（PATTERN_ONLY={'开' if PATTERN_ONLY else '关'}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="走势判断", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  {res['多空标签']} {res['代码']} {res['名称']} 评分{res['走势评分']} 共振{res['共振维度']}/5 形态{res['形态共振']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各淘汰原因统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['形态共振', '共振维度', '走势评分'], ascending=[False, False, False]).reset_index(drop=True)
    return df

# ------------------ 行业+风口 ------------------
def enrich(df):
    targets = df.to_dict('records')
    for r in targets:
        r['行业'] = _INDUSTRY_MAP.get(r['代码'], '—')
    labeled = [r for r in targets if r.get('行业') not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty:
                break
        except Exception:
            time.sleep(2 + i)
    hot = []
    if not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
        h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        hot = [(str(r['板块名称']), round(float(r['_chg']), 2)) for _, r in h.head(HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]
    cnt = 0
    for r in targets:
        sec = r.get('行业', ''); m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh):
                    m = hh; break
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '形态共振', '共振维度', '走势评分'], ascending=[False, False, False, False]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    patterned = df[df['形态共振'] >= 1] if '形态共振' in df.columns else pd.DataFrame()
    jingxuan = df[(df['走势评分'] >= 70) & (df['共振维度'] >= 4)] if ('共振维度' in df.columns and '走势评分' in df.columns) else pd.DataFrame()
    strong = df[df['走势评分'] >= PARAMS['LABEL_STRONG_BULL']] if '走势评分' in df.columns else pd.DataFrame()
    bull = df[(df['走势评分'] >= PARAMS['LABEL_BULL']) & (df['走势评分'] < PARAMS['LABEL_STRONG_BULL'])] if '走势评分' in df.columns else pd.DataFrame()
    L = [f"**📊 综合走势判断+形态共振** | ★精选{len(jingxuan)} 强多{len(strong)} 偏多{len(bull)} 🧩形态{len(patterned)} 🎯风口{len(reso)} (现价=实时价)",
         "*(5维评分+6道质量门槛+三脚本形态共振[MACD金叉趋势启动/布林W底/VAGAS+OBV]; 概率性, 非预测; 结合仓位/止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📊 **走势强势板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        amt = f" 额{r['日均成交额亿']}亿" if r.get('日均成交额亿') is not None else ""
        pat = f" 🧩{r['形态标签']}" if r.get('形态共振', 0) >= 1 else ""
        return (f"- {r['多空标签']} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 评分{r['走势评分']} 共振{r['共振维度']}/5{pat} | "
                f"现价{r['最新价']} 涨{r['近20日涨幅%']}% RSI{r['RSI']} 量比{r['量比']}{amt}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 走势强势遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not patterned.empty:
        L.append(f"### 🧩 形态共振(MACD金叉/W底/VAGAS) 共{len(patterned)}只")
        L += [line(r) for _, r in patterned.head(PUSH_TOP).iterrows()]; L.append("")
    if not jingxuan.empty:
        L.append(f"### ★ 精选 共{len(jingxuan)}只 (评分≥70且共振≥4维)")
        L += [line(r) for _, r in jingxuan.head(PUSH_TOP).iterrows()]; L.append("")
    if not strong.empty:
        L.append(f"### 🟢 强多 共{len(strong)}只")
        L += [line(r) for _, r in strong.head(PUSH_TOP).iterrows()]; L.append("")
    if not bull.empty:
        L.append(f"### 🟢 偏多 共{len(bull)}只")
        L += [line(r) for _, r in bull.head(PUSH_TOP).iterrows()]
    return "\n".join(L)

def main():
    print("=" * 70)
    print(f"📊 综合走势判断+形态共振 | {datetime.now():%Y-%m-%d %H:%M} | PATTERN_ONLY={'开' if PATTERN_ONLY else '关'} 形态加分{PATTERN_BONUS}")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print(f"\n本次未发现通过质量门槛的票(市场偏弱或门槛严; 可调低 SCORE_MIN/MIN_STRONG_DIMS, 或 PATTERN_ONLY=0)。")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    df, rt = _refresh_realtime_price(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"trend_judge_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"trend_judge_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"SCORE_MIN": SCORE_MIN_ENV, "PATTERN_BONUS": PATTERN_BONUS,
                       "PATTERN_ONLY": PATTERN_ONLY, "PRE_SIGNAL": PRE_SIGNAL, "APPROACH_PCT": APPROACH_PCT, "USE_FVG": USE_FVG},
                       "cluster": cluster, "n": int(len(df)),
                       "n_pattern": int((df['形态共振'] >= 1).sum()) if '形态共振' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/trend_judge_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', '距20日高%', '实时价'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_pat = int((df['形态共振'] >= 1).sum()) if '形态共振' in df.columns else 0
            n_strong = int((df['走势评分'] >= PARAMS['LABEL_STRONG_BULL']).sum()) if '走势评分' in df.columns else 0
            send_serverchan(f"📊 走势判断+形态共振 强多{n_strong} 🧩形态{n_pat} 共{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, rt))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_trend_judge<<<
