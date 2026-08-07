#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resonance_signal.py —— 大涨前多指标共振检测 · 矩阵双模式版
指标: 成交量/均线/MACD/RSI/连阳/突破前高/跳空缺口 + 布林带 + 肯特纳通道 + 量化挤压强度。
【双模式】
  ① 全市场扫描(矩阵/workflow, 不带参数): baostock+东财股票池→快照预筛→逐只共振评分→
     得分≥SCORE_MIN→行业本地join+风口🎯→腾讯实时价对齐→Server酱分页推送→存output/。
  ② 单股分析(CLI, 带代码): python resonance_signal.py 600519 [天数]  (保留原详细报告)
⚠️ 共振=技术面概率信号, 非买入保证; 挤压/突破假信号常见, 必止损。仅供学习, 不构成投资建议。
"""
import os
import re
import sys
import json
import time
import random
import warnings
import traceback
import requests
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import baostock as bs
except ImportError:
    raise ImportError("请先安装: pip install baostock")
import akshare as ak
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# ==================== 参数 (env 可调) ====================
DATA_DAYS = int(os.environ.get('DATA_DAYS', '250'))        # 拉取历史天数(指标预热+挤压分位)
SCORE_MIN = float(os.environ.get('SCORE_MIN', '9'))        # 扫描模式起推分(9=较强共振★★)
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = float(os.environ.get('SLEEP_PER_STOCK', '0.1'))
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '15'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SNAPSHOT_PRE = os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True')
PRE_AMOUNT_MIN = float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7'))
PRE_TURNOVER_MIN = float(os.environ.get('PRE_TURNOVER_MIN', '0.3'))
KEEP_PREFIX = ("0", "3", "6"); EXCLUDE_NAME = ("ST", "退"); MIN_PRICE = float(os.environ.get('MIN_PRICE', '3.0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '30'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "评分不足": 0}

# ------------------ 推送 ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok:
            return True
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
    except Exception:
        pass
    finally:
        _BS_LOGGED = False

def _init_worker():
    global _BS_LOGGED
    time.sleep(random.uniform(0, 2)); _BS_LOGGED = False
    _bs_login_ok()

def _bs_q(code, fields, sd, ed, timeout=QUERY_TIMEOUT_SEC):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed, frequency="d", adjustflag="2").get_data()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)

def _call_with_timeout(fn, *a, timeout=20, **kw):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)

def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

# ------------------ 实时价(腾讯) ------------------
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
            except Exception:
                pass
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

# ------------------ 历史双源 (扫描模式) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust="qfq", timeout=25)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= 80:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']].tail(DATA_DAYS).reset_index(drop=True)
        except Exception:
            time.sleep(1 + attempt)
    return None

def _fetch_hist_screener(code):
    sd = (datetime.now() - timedelta(days=int(DATA_DAYS * 1.6))).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=QUERY_TIMEOUT_SEC)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= 80:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']].tail(DATA_DAYS).reset_index(drop=True)
        except Exception:
            pass
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    return _fetch_hist_em(sym, sd.replace('-', ''), ed.replace('-', ''))

# ==================== 指标计算 (原逻辑保留) ====================
def calc_indicators(df, boll_period=20, boll_std=2.0, kc_period=20, kc_atr_period=10, kc_mult=1.5,
                    squeeze_ratio_th=0.85, min_squeeze_days=5, momentum_period=12, strength_lookback=60):
    df = df.copy()
    close = df['close']; high = df['high']; low = df['low']; volume = df['volume']
    df['MA5'] = close.rolling(5).mean(); df['MA10'] = close.rolling(10).mean(); df['MA20'] = close.rolling(20).mean()
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(100)
    df['VOL_MA5'] = volume.rolling(5).mean(); df['VOL_MA20'] = volume.rolling(20).mean()
    df['HIGH_20'] = high.rolling(20).max()
    df['prev_high'] = high.shift(1); df['prev_low'] = low.shift(1)
    df['gap_up'] = df['open'] > df['prev_high']; df['gap_down'] = df['open'] < df['prev_low']
    df['gap_up_pct'] = (df['open'] - df['prev_high']) / df['prev_high'] * 100
    df['gap_down_pct'] = (df['prev_low'] - df['open']) / df['prev_low'] * 100
    df['BOLL_MID'] = close.rolling(boll_period).mean(); df['BOLL_STD'] = close.rolling(boll_period).std()
    df['BOLL_UPPER'] = df['BOLL_MID'] + boll_std * df['BOLL_STD']; df['BOLL_LOWER'] = df['BOLL_MID'] - boll_std * df['BOLL_STD']
    df['BOLL_WIDTH'] = (df['BOLL_UPPER'] - df['BOLL_LOWER']) / df['BOLL_MID'] * 100
    df['BOLL_PB'] = (close - df['BOLL_LOWER']) / (df['BOLL_UPPER'] - df['BOLL_LOWER'])
    df['BOLL_WIDTH_MA'] = df['BOLL_WIDTH'].rolling(10).mean(); df['BOLL_WIDTH_MIN'] = df['BOLL_WIDTH'].rolling(20).min()
    df['KC_MID'] = close.ewm(span=kc_period, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1 / kc_atr_period, adjust=False).mean()
    df['KC_UPPER'] = df['KC_MID'] + kc_mult * df['ATR']; df['KC_LOWER'] = df['KC_MID'] - kc_mult * df['ATR']
    df['BB_WIDTH'] = df['BOLL_UPPER'] - df['BOLL_LOWER']; df['KC_WIDTH'] = df['KC_UPPER'] - df['KC_LOWER']
    df['SQUEEZE_RATIO'] = df['BB_WIDTH'] / df['KC_WIDTH']
    df['SQUEEZE_PERCENTILE'] = df['SQUEEZE_RATIO'].rolling(strength_lookback).apply(
        lambda x: x.rank(pct=True).iloc[-1] * 100 if len(x.dropna()) > 10 else np.nan, raw=False)
    ratio = df['SQUEEZE_RATIO']
    df['SQUEEZE_SCORE'] = np.select([ratio.isna() | (ratio >= 1.0), ratio <= 0.4], [0.0, 100.0], default=(1.0 - ratio) / 0.6 * 100)
    df['SQUEEZE_ON'] = df['SQUEEZE_RATIO'] < squeeze_ratio_th
    df['SQUEEZE_STRONG'] = df['SQUEEZE_RATIO'] < 0.70
    df['SQUEEZE_DAYS'] = df['SQUEEZE_ON'].groupby((~df['SQUEEZE_ON']).cumsum()).cumsum()
    df['TIME_WEIGHT'] = np.minimum(1 + df['SQUEEZE_DAYS'] / 10, 2.0)
    df['SQUEEZE_STRENGTH'] = df['SQUEEZE_SCORE'] * df['TIME_WEIGHT']
    df['VALID_SQUEEZE'] = df['SQUEEZE_ON'] & (df['SQUEEZE_DAYS'] >= min_squeeze_days)
    df['HIGH_STRENGTH_SQUEEZE'] = (df['SQUEEZE_STRENGTH'] >= 100) & (df['SQUEEZE_DAYS'] >= min_squeeze_days)
    df['MOMENTUM'] = close - close.shift(momentum_period)
    squeeze_just_off = (~df['SQUEEZE_ON']) & (df['SQUEEZE_ON'].shift(1).fillna(False))
    prev_days_ok = df['SQUEEZE_DAYS'].shift(1) >= min_squeeze_days
    df['SQUEEZE_FIRE_UP'] = squeeze_just_off & prev_days_ok & (df['MOMENTUM'] > 0) & (close > df['KC_MID'])
    df['SQUEEZE_FIRE_DOWN'] = squeeze_just_off & prev_days_ok & (df['MOMENTUM'] < 0) & (close < df['KC_MID'])
    df['SQUEEZE_FIRE_UP_VOL'] = df['SQUEEZE_FIRE_UP'] & (volume > df['VOL_MA20'] * 1.3)
    return df

def check_resonance(df):
    if len(df) < 50:
        return {"error": "数据不足，建议使用 ≥80 天"}
    last = df.iloc[-1]; prev = df.iloc[-2]; recent5 = df.tail(5)
    signals = []; score = 0; details = {}
    vol_ratio = last['volume'] / last['VOL_MA5'] if last['VOL_MA5'] > 0 else 0
    if vol_ratio >= 1.5:
        signals.append("放量突破"); score += 2; details['volume'] = f"量比 {vol_ratio:.2f}倍 ✓"
    else:
        details['volume'] = f"量比 {vol_ratio:.2f}倍 ✗"
    if last['MA5'] > last['MA10'] > last['MA20'] and last['close'] > last['MA5']:
        signals.append("均线多头"); score += 2; details['ma'] = "MA5>MA10>MA20 多头 ✓"
    elif last['close'] > last['MA5'] > last['MA10']:
        signals.append("均线偏多"); score += 1; details['ma'] = "短期均线向上 ✓"
    else:
        details['ma'] = "均线未多头 ✗"
    macd_cross = prev['DIF'] <= prev['DEA'] and last['DIF'] > last['DEA']
    hist_up = last['MACD_HIST'] > 0 and last['MACD_HIST'] > prev['MACD_HIST']
    if macd_cross or (last['DIF'] > 0 and hist_up):
        signals.append("MACD转强"); score += 2; details['macd'] = "金叉/红柱放大 ✓"
    else:
        details['macd'] = "MACD未转强 ✗"
    if last['RSI'] >= 50:
        signals.append("RSI强势"); score += 1; details['rsi'] = f"RSI={last['RSI']:.1f} 强势 ✓"
    elif last['RSI'] < 30:
        signals.append("RSI超卖反弹"); score += 1; details['rsi'] = f"RSI={last['RSI']:.1f} 超卖 ✓"
    else:
        details['rsi'] = f"RSI={last['RSI']:.1f} 中性"
    yang_count = (recent5['close'] > recent5['open']).sum()
    if yang_count >= 3:
        signals.append("连阳蓄力"); score += 1; details['kline'] = f"近5日{yang_count}根阳线 ✓"
    else:
        details['kline'] = f"近5日{yang_count}根阳线"
    if last['close'] >= last['HIGH_20'] * 0.998:
        signals.append("突破前高"); score += 2; details['breakout'] = "突破/接近20日高点 ✓"
    else:
        details['breakout'] = "未突破前高 ✗"
    if last['gap_up']:
        gap_pct = last['gap_up_pct']
        if gap_pct >= 1.0:
            signals.append("向上跳空缺口"); score += 2; details['gap'] = f"向上跳空 {gap_pct:.2f}% ✓（强）"
        elif gap_pct >= 0.3:
            signals.append("小幅向上跳空"); score += 1; details['gap'] = f"向上跳空 {gap_pct:.2f}% ✓"
        else:
            details['gap'] = f"向上跳空过小 {gap_pct:.2f}%"
    elif last['gap_down']:
        details['gap'] = f"向下跳空 {last['gap_down_pct']:.2f}%（偏空）"
    else:
        details['gap'] = "无跳空缺口"
    boll_score = 0; boll_desc = []
    if last['close'] > last['BOLL_MID']:
        boll_score += 1; boll_desc.append("站上中轨")
    if last['close'] > last['BOLL_UPPER']:
        boll_score += 1; boll_desc.append("突破上轨"); signals.append("布林上轨突破")
    if last['BOLL_WIDTH'] > last['BOLL_WIDTH_MA'] * 1.08:
        boll_score += 1; boll_desc.append("带宽扩张"); signals.append("布林带宽扩张")
    was_tight = last['BOLL_WIDTH_MIN'] < last['BOLL_WIDTH_MA'] * 0.75
    now_expand = last['BOLL_WIDTH'] > last['BOLL_WIDTH_MIN'] * 1.25
    if was_tight and now_expand and last['close'] > last['BOLL_MID']:
        boll_score += 1; boll_desc.append("收口后开口"); signals.append("布林收口后开口")
    if last['BOLL_PB'] <= 0.15 and last['close'] > prev['close']:
        boll_score += 1; boll_desc.append("下轨反弹"); signals.append("布林下轨反弹")
    if boll_score >= 2:
        score += 2; details['boll'] = " + ".join(boll_desc) + " ✓"
    elif boll_score == 1:
        score += 1; details['boll'] = " + ".join(boll_desc) + " ✓"
    else:
        details['boll'] = " + ".join(boll_desc) if boll_desc else "中性 ✗"
    strength = last['SQUEEZE_STRENGTH']; days = int(last['SQUEEZE_DAYS'])
    if last['SQUEEZE_FIRE_UP_VOL']:
        signals.append("挤压释放向上+放量"); score += 3
        details['squeeze'] = f"释放向上（强度{strength:.0f}，持续{days}天）+ 放量 ✓✓✓"
    elif last['SQUEEZE_FIRE_UP']:
        signals.append("挤压释放向上"); score += 2
        details['squeeze'] = f"释放向上（强度{strength:.0f}，持续{days}天）✓✓"
    elif last['HIGH_STRENGTH_SQUEEZE']:
        details['squeeze'] = f"高强度挤压中（综合强度{strength:.0f}，已持续{days}天）【重点关注】"
    elif last['VALID_SQUEEZE']:
        details['squeeze'] = f"有效挤压中（强度{strength:.0f}，持续{days}天）"
    elif last['SQUEEZE_ON']:
        details['squeeze'] = f"轻微挤压（强度{strength:.0f}，{days}天）"
    elif last['close'] > last['KC_UPPER']:
        signals.append("肯特纳上轨突破"); score += 1; details['squeeze'] = "突破肯特纳上轨 ✓"
    else:
        details['squeeze'] = "无有效挤压"
    if score >= 12:
        level = "★★★ 强烈共振"
    elif score >= 9:
        level = "★★ 较强共振"
    elif score >= 6:
        level = "★ 弱共振"
    else:
        level = "无明确共振"
    return {"score": score, "level": level, "signals": signals, "details": details,
            "price": float(last['close']),
            "date": str(last['date'].date()) if hasattr(last['date'], 'date') else str(last['date']),
            "squeeze_strength": float(last['SQUEEZE_STRENGTH']), "squeeze_days": days,
            "fire_up": bool(last['SQUEEZE_FIRE_UP']), "fire_up_vol": bool(last['SQUEEZE_FIRE_UP_VOL']),
            "high_strength": bool(last['HIGH_STRENGTH_SQUEEZE']), "squeeze_on": bool(last['SQUEEZE_ON'])}

# ------------------ 扫描模式: 单只处理 ------------------
def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist_screener(code)
        if df is None or len(df) < 80:
            return {"__fail__": "数据不足"}
        time.sleep(SLEEP_PER_STOCK)
        df = calc_indicators(df)
        result = check_resonance(df)
        if "error" in result:
            return {"__fail__": "数据不足"}
        if result['score'] < SCORE_MIN:
            return {"__fail__": "评分不足"}
        return {"代码": code, "名称": name, "行业": "",
                "最新价": round(result['price'], 2), "信号价": round(result['price'], 2), "信号日期": result['date'],
                "共振得分": result['score'], "共振等级": result['level'],
                "触发信号": ",".join(result['signals']) if result['signals'] else "—",
                "挤压强度": round(result['squeeze_strength'], 1), "挤压天数": result['squeeze_days'],
                "score": result['score'], "resonance": False, "resonance_sector": ""}
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}

def snapshot_prefilter(codes_with_prefix):
    if not SNAPSHOT_PRE:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for col in ['最新价', '成交额', '换手率']:
            if col in spot.columns:
                spot[col] = pd.to_numeric(spot[col], errors='coerce')
        m = (spot['代码'].str.startswith(KEEP_PREFIX)
             & ~spot['名称'].astype(str).str.contains("|".join(EXCLUDE_NAME), na=False, regex=True)
             & (spot['最新价'] >= MIN_PRICE))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PRE_AMOUNT_MIN)
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PRE_TURNOVER_MIN)
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes_with_prefix

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
    print(f"开始多指标共振扫描 {len(tasks)} 只（{NUM_PROCESSES}进程, 起推分≥{SCORE_MIN}）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="共振扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['共振等级']} 分{res['共振得分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
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
    df2 = df2.sort_values(['resonance', 'score'], ascending=[False, False]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    s3 = df[df['共振得分'] >= 12] if '共振得分' in df.columns else pd.DataFrame()
    s2 = df[(df['共振得分'] >= 9) & (df['共振得分'] < 12)] if '共振得分' in df.columns else pd.DataFrame()
    L = [f"**🔥 大涨前多指标共振** | 命中{len(df)}只 (★★★{len(s3)} ★★{len(s2)}) 🎯风口{len(reso)} (现价=实时价)",
         f"*(量+均线+MACD+RSI+连阳+突破+跳空+布林/肯特纳挤压; 评分≥{SCORE_MIN:.0f}才推; 概率性非预测, 必止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🔥 **共振板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] {r['共振等级']} 分{r['共振得分']} 现价{r['最新价']} | "
                f"{r['触发信号']} | 挤压{r['挤压强度']}({r['挤压天数']}天){_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 共振遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not s3.empty:
        L.append(f"### ★★★ 强烈共振(≥12分) 共{len(s3)}只")
        L += [line(r) for _, r in s3.head(PUSH_TOP).iterrows()]; L.append("")
    if not s2.empty:
        L.append(f"### ★★ 较强共振(9-11分) 共{len(s2)}只")
        L += [line(r) for _, r in s2.head(PUSH_TOP).iterrows()]
    return "\n".join(L)

# ------------------ 单股分析模式 (原 CLI 报告) ------------------
def single_stock_report(code, days):
    print(f"\n正在分析 {code} （最近 {days} 天）...\n")
    df = _fetch_hist_screener(_pref(code) if code.isdigit() else code) if code.isdigit() else _fetch_hist_em(code, (datetime.now() - timedelta(days=int(days * 1.6))).strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d'))
    if df is None or len(df) < 80:
        print("数据获取失败或不足"); return
    df = calc_indicators(df)
    r = check_resonance(df)
    if "error" in r:
        print(r["error"]); return
    print("=" * 66)
    print(f"股票: {code}  |  最新价: {r['price']:.2f}  |  日期: {r['date']}")
    print("=" * 66)
    print(f"共振得分: {r['score']} / 18")
    print(f"共振等级: {r['level']}")
    print(f"触发信号: {', '.join(r['signals']) if r['signals'] else '无'}")
    print("-" * 66)
    print("详细指标:")
    for k, v in r['details'].items():
        print(f"  {k:12}: {v}")
    print("=" * 66)
    print("\n提示: 仅供技术分析学习，不构成任何投资建议。")

# ------------------ 主程序 ------------------
def main():
    # 模式②: 带参数 = 单股分析
    if len(sys.argv) >= 2:
        code = sys.argv[1]
        days = int(sys.argv[2]) if len(sys.argv) > 2 else DATA_DAYS
        single_stock_report(code, days)
        sys.exit(0)

    # 模式①: 无参数 = 全市场扫描 (workflow)
    print("=" * 70)
    print(f"🔥 大涨前多指标共振扫描 | {datetime.now():%Y-%m-%d %H:%M} | 起推分≥{SCORE_MIN} | 进程{NUM_PROCESSES}")
    print("⚠️ 共振=技术面概率信号, 非买入保证; 必止损; 不构成投资建议")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print(f"\n本次未发现评分≥{SCORE_MIN} 的共振票(门槛严或市场弱, 0命中属正常; 可调低 SCORE_MIN)。")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    df, rt = _refresh_realtime_price(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"resonance_signal_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"resonance_signal_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "SCORE_MIN": SCORE_MIN, "cluster": cluster, "n": int(len(df)),
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/resonance_signal_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', '实时价', '信号价', 'score'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = rt
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_s3 = int((df['共振得分'] >= 12).sum()) if '共振得分' in df.columns else 0
            send_serverchan(f"🔥 多指标共振 命中{len(df)}只 ★★★{n_s3} 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_resonance_signal<<<
