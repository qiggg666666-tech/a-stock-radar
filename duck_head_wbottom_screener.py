# -*- coding: utf-8 -*-
"""
duck_head_wbottom_screener.py —— 周线老鸭头 + 日线W底 + MACD/RSI双重背离 综合选股(日线扫描版) · 矩阵规格
策略内核(保留原脚本, 一字未动):
  ① 周线老鸭头(detect_weekly_duck_head): 60周线向上 + 5/10上穿60 + 回调幅度合适 + 短期均线多头
     + 量能放量-缩量 + 周线MACD金叉 + MACD/RSI双重底背离加分(顶背离扣分)。
  ② 日线W底(detect_daily_w_bottom): 找两低点(接近) + 突破颈线 + 突破放量 + 日线MACD金叉。
  ③ MACD+RSI双重背离(detect_combined_divergence): 显著性极值匹配(ATR/std/fixed prominence)。
  综合(composite_signal): 周线valid+日线valid=强信号; 周线≥3+日线≥2 或 周线valid+日线≥2=中等信号。

【本版完善·相对原本地脚本】
 1 修两个致命bug: 原 main 里 `df[\~df['name']]` 的 `\~` 转义错误(SyntaxError) -> `~`;
   `ak.stock_info_a_code_name()` 返回列是 code/name, 原 `df['symbol']` KeyError -> `df['code']`。
 2 单源akshare串行 -> 双源baostock+东财+硬超时 + 多进程并发 + 快照预筛砍量。
 3 行业映射原"逐板块调stock_board_industry_cons_em(~90次接口, 极慢易限流)" -> baostock query_stock_industry
   本地join(零接口); 板块"强制强势过滤" -> 全市场扫描+东财风口🎯标记(不强制过滤, 风口优先排序)。
 4 加 Server酱推送(全发分页, 严格检查返回) + 信号vs实时对齐列; 不拦交易日; 收尾防护sys.exit(0); append补丁。
 5 策略内核(老鸭头/W底/背离/极值检测/MACD/RSI)一字未动; 极值检测保留 scipy.signal(argrelextrema/find_peaks),
   故 requirements.txt 必须含 scipy。
⚠️ 概率性形态+背离+板块综合判断, 非预测; 老鸭头/W底均可能失败, 务必结合仓位/止损, 不构成投资建议。
⚠️ 周线老鸭头需~80周线, 故拉~1800天日线 resample 周线, min_data_len=450; 信号极严, 0命中属正常。
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

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

try:
    import akshare as ak
    import baostock as bs
    from scipy.signal import argrelextrema, find_peaks
except ImportError as e:
    raise ImportError(f"缺少依赖(需 akshare/baostock/scipy): {e}; 请 pip install scipy 并确认 requirements.txt 含 scipy")
from tqdm import tqdm


# ==================== 参数 (env 可调) ====================
MIN_DATA_LEN = int(os.environ.get('MIN_DATA_LEN', '450'))     # 日线最小根数(约90周, 供60周线+预热)
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '1800'))  # 拉取日历天数(约7年)
PUSH_MEDIUM = os.environ.get('PUSH_MEDIUM', '1').strip() in ('1', 'true', 'True')  # 是否推中等信号
MIN_PRICE = float(os.environ.get('MIN_PRICE', '3.0'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = float(os.environ.get('SLEEP', '0.1'))
FETCH_TIMEOUT = int(os.environ.get('FETCH_TIMEOUT', '15'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SNAPSHOT_PRE = os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True')
PRE_AMOUNT_MIN = float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7'))
PRE_TURNOVER_MIN = float(os.environ.get('PRE_TURNOVER_MIN', '0.3'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0}


# ------------------ 推送 (全发分页) ------------------
def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    lines = content.split("\n")
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
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
        print(f"  推送第{i+1}/{len(chunks)}条 ({len(ch)}字符)")
        try:
            from serverchan_sdk import sc_send
            sc_send(key, t, ch); r_ok = True
        except Exception as e:
            print(f"  sdk失败回退requests: {e}")
            try:
                j = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": t, "desp": ch}, timeout=15).json()
                r_ok = j.get('code') == 0
            except Exception as e2:
                print(f"  requests推送失败: {e2}"); r_ok = False
        ok = r_ok and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print("📲 推送完成" + (" ✅" if ok else " ⚠️存在失败"))
    return ok


def is_trading_day():
    try:
        d = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        print(f"  交易日历获取失败, 默认继续: {e}"); return True


# ------------------ baostock 登录 / 超时 ------------------
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    if _BS_LOGGED:
        return True
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True; return True
            print(f"  baostock 登录失败({getattr(lg,'error_msg','')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _init_worker():
    global _BS_LOGGED
    time.sleep(random.uniform(0, 2))
    _BS_LOGGED = False
    _bs_login_ok()


def _query_with_timeout(code, fields, start_date, timeout=FETCH_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2").get_data()
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
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


# ==================== 历史双源 (日线; 周线由日线 resample) ====================
def _fetch_hist_em(sym, start_y):
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    sy = sd.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _query_with_timeout(code, "date,open,high,low,close,volume", sd)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            pass
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    return _fetch_hist_em(sym, sy)


def _resample_weekly(daily):
    w = daily.set_index('date').resample('W').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
    return w


def _fetch_list_akshare():
    for attempt in range(3):
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                nc = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'; d['status'] = '1'; return d
        except Exception as e:
            print(f"  akshare列表第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])


def snapshot_prefilter(tasks):
    if not SNAPSHOT_PRE:
        return tasks
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫"); return tasks
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (spot['代码'].str.startswith(("0", "3", "6"))
             & ~spot['名称'].astype(str).str.contains("ST|退", na=False, regex=True)
             & (spot['最新价'] >= MIN_PRICE))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PRE_AMOUNT_MIN)
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PRE_TURNOVER_MIN)
        keep = set(spot.loc[m, '代码'])
        out = [(c, n) for c, n in tasks if c[3:] in keep]
        print(f"  快照预筛: {len(tasks)} → {len(out)} 只 (宽松, 失败退化全扫)")
        return out if out else tasks
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return tasks


# ==================== 策略工具 (保留原脚本) ====================
def calc_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    dif = exp1 - exp2; dea = dif.ewm(span=signal, adjust=False).mean()
    df = df.copy(); df['dif'] = dif; df['dea'] = dea; df['macd_hist'] = (dif - dea) * 2
    return df


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def find_significant_extrema(series, high=None, low=None, order=3, method="pct",
                             min_prominence_pct=0.015, atr_mult=0.6, std_mult=0.8,
                             fixed_prominence=None, min_distance=None):
    values = series.values.astype(float)
    n = len(values)
    if n < order * 2 + 3:
        return np.array([]), np.array([])
    if method == "atr" and high is not None and low is not None:
        prev_close = np.roll(values, 1)
        tr = np.maximum(high.values - low.values,
                        np.maximum(np.abs(high.values - prev_close), np.abs(low.values - prev_close)))
        atr = pd.Series(tr).rolling(14, min_periods=5).mean().iloc[-1]
        if np.isnan(atr) or atr <= 0:
            atr = np.nanstd(values) * 1.2
        min_prom = max(atr * atr_mult, 1e-8)
    elif method == "std":
        vol = np.nanstd(values)
        if np.isnan(vol) or vol <= 0:
            vol = np.nanmean(np.abs(values)) * 0.02 + 1e-8
        min_prom = vol * std_mult
    elif method == "fixed":
        min_prom = fixed_prominence if fixed_prominence is not None else 1.0
    else:
        mean_abs = abs(np.nanmean(values)); vol = np.nanstd(values)
        if np.isnan(vol):
            vol = mean_abs * 0.02
        min_prom = max(vol * 0.7, mean_abs * min_prominence_pct, 1e-8)
    low_idx = argrelextrema(values, np.less_equal, order=order)[0]
    high_idx = argrelextrema(values, np.greater_equal, order=order)[0]

    def filter_prom(idxs, is_low=True):
        filtered = []
        for i in idxs:
            left = max(0, i - order * 2); right = min(n, i + order * 2 + 1)
            window = values[left:right]
            prom = (np.nanmax(window) - values[i]) if is_low else (values[i] - np.nanmin(window))
            if prom >= min_prom:
                filtered.append(i)
        return filtered

    lows = filter_prom(low_idx, True); highs = filter_prom(high_idx, False)

    def apply_min_dist(idxs, min_dist):
        if not idxs or min_dist is None or min_dist <= 1:
            return np.array(idxs)
        idxs = sorted(idxs); keep = [idxs[0]]
        for x in idxs[1:]:
            if x - keep[-1] >= min_dist:
                keep.append(x)
        return np.array(keep)

    if min_distance is not None:
        lows = apply_min_dist(lows, min_distance); highs = apply_min_dist(highs, min_distance)
    else:
        lows = np.array(lows); highs = np.array(highs)
    return lows, highs


def match_extrema(price_idxs, indicator_idxs, max_distance=5):
    if len(price_idxs) == 0 or len(indicator_idxs) == 0:
        return []
    matches = []; used = set()
    for p_idx in sorted(price_idxs, reverse=True):
        best, best_dist = None, max_distance + 1
        for i_idx in indicator_idxs:
            if i_idx in used:
                continue
            dist = abs(p_idx - i_idx)
            if dist <= max_distance and dist < best_dist:
                best_dist = dist; best = i_idx
        if best is not None:
            matches.append((p_idx, best, best_dist)); used.add(best)
    matches.sort(key=lambda x: x[0])
    return matches


def detect_volume_pattern(df, lookback=30):
    if len(df) < lookback + 10:
        return {"valid": False, "breakthrough_volume": False}
    vol_ma = df['volume'].rolling(20).mean()
    recent = df.iloc[-lookback:]
    up_mask = recent['close'] > recent['close'].shift(1)
    down_mask = recent['close'] < recent['close'].shift(1)
    up_vol = recent.loc[up_mask, 'volume'].mean() if up_mask.any() else 0
    down_vol = recent.loc[down_mask, 'volume'].mean() if down_mask.any() else 0
    recent_vol = df['volume'].iloc[-3:].mean()
    vol_ma_last = vol_ma.iloc[-1] if not pd.isna(vol_ma.iloc[-1]) else 1
    return {"valid": (up_vol > vol_ma_last * 1.3) and (down_vol < vol_ma_last * 0.75),
            "breakthrough_volume": recent_vol > vol_ma_last * 1.5}


def detect_pullback_ratio(df, lookback=40):
    if len(df) < lookback:
        return {"valid": False, "ratio": 0}
    window = df.iloc[-lookback:]
    peak, trough = window['high'].max(), window['low'].min()
    if peak <= 0:
        return {"valid": False, "ratio": 0}
    ratio = (peak - trough) / peak
    return {"valid": 0.08 <= ratio <= 0.35, "ratio": round(ratio * 100, 2)}


def detect_macd_golden_cross(df, timeframe="daily", lookback=6):
    if len(df) < 40:
        return {"valid": False, "score": 0, "reasons": ["数据不足"]}
    df = calc_macd(df)
    dif, dea, hist = df['dif'], df['dea'], df['macd_hist']
    cross = (dif.shift(1) < dea.shift(1)) & (dif > dea)
    recent_cross = cross.iloc[-lookback:].any()
    current_dif, current_dea = dif.iloc[-1], dea.iloc[-1]
    if timeframe == "weekly":
        above_zero = current_dif > 0 and current_dea > 0
        near_zero = current_dif > -abs(current_dif) * 0.5 if current_dif != 0 else True
        zero_score = 2 if above_zero else (1 if near_zero else 0)
    else:
        above_zero = current_dif > 0
        near_zero = current_dif > -abs(df['close'].iloc[-1]) * 0.01
        zero_score = 1 if (above_zero or near_zero) else 0
    hist_turning = (hist.iloc[-1] > hist.iloc[-2] > 0) or (hist.iloc[-2] <= 0 < hist.iloc[-1])
    dif_slope_up = dif.iloc[-1] > dif.iloc[-2]
    spread_expanding = (dif.iloc[-1] - dea.iloc[-1]) > (dif.iloc[-2] - dea.iloc[-2])
    score, reasons = 0, []
    if recent_cross:
        score += 2; reasons.append(f"近{lookback}{'周' if timeframe=='weekly' else '日'}内金叉")
    if zero_score >= 2:
        score += 2; reasons.append("DIF/DEA均在零轴上方")
    elif zero_score == 1:
        score += 1; reasons.append("接近零轴")
    if hist_turning:
        score += 1; reasons.append("柱状图转强")
    if dif_slope_up and spread_expanding:
        score += 1; reasons.append("DIF向上且开口扩大")
    valid = (score >= 4 and recent_cross) if timeframe == "weekly" else (score >= 3 and recent_cross)
    return {"valid": valid, "score": score, "reasons": reasons,
            "dif": round(float(current_dif), 4), "dea": round(float(current_dea), 4)}


def detect_combined_divergence(df, lookback=35, order=3):
    if len(df) < lookback + 20:
        return {"bullish_div": False, "bearish_div": False, "double_bullish": False,
                "double_bearish": False, "score": 0, "reasons": ["数据不足"], "rsi_value": None}
    df = calc_macd(df.copy()); df['rsi'] = calc_rsi(df['close'], 14)
    recent = df.iloc[-lookback:].reset_index(drop=True)
    price_lows, price_highs = find_significant_extrema(recent['low'], high=recent['high'], low=recent['low'],
                                                       order=order, method="atr", atr_mult=0.55)
    dif_lows, dif_highs = find_significant_extrema(recent['dif'], order=order, method="std", std_mult=0.7)
    rsi_lows, rsi_highs = find_significant_extrema(recent['rsi'], order=order, method="fixed",
                                                   fixed_prominence=3.5, min_distance=4)
    score, reasons = 0, []
    bullish_count = bearish_count = 0
    current_rsi = float(recent['rsi'].iloc[-1])
    low_matches = match_extrema(price_lows, dif_lows, max_distance=5)
    if len(low_matches) >= 2:
        m1, m2 = low_matches[-2], low_matches[-1]
        p1, d1 = recent['low'].iloc[m1[0]], recent['dif'].iloc[m1[1]]
        p2, d2 = recent['low'].iloc[m2[0]], recent['dif'].iloc[m2[1]]
        between_high = recent['high'].iloc[m1[0]:m2[0] + 1].max()
        if p2 < p1 and d2 > d1 and (between_high - min(p1, p2)) / min(p1, p2) > 0.03:
            bullish_count += 1; score += 2
            reasons.append(f"MACD底背离 价{p1:.2f}→{p2:.2f} DIF{d1:.4f}→{d2:.4f}")
    high_matches = match_extrema(price_highs, dif_highs, max_distance=5)
    if len(high_matches) >= 2:
        m1, m2 = high_matches[-2], high_matches[-1]
        p1, d1 = recent['high'].iloc[m1[0]], recent['dif'].iloc[m1[1]]
        p2, d2 = recent['high'].iloc[m2[0]], recent['dif'].iloc[m2[1]]
        between_low = recent['low'].iloc[m1[0]:m2[0] + 1].min()
        if p2 > p1 and d2 < d1 and (max(p1, p2) - between_low) / max(p1, p2) > 0.03:
            bearish_count += 1; score -= 2
            reasons.append(f"MACD顶背离 价{p1:.2f}→{p2:.2f} DIF{d1:.4f}→{d2:.4f}")
    low_matches = match_extrema(price_lows, rsi_lows, max_distance=5)
    if len(low_matches) >= 2:
        m1, m2 = low_matches[-2], low_matches[-1]
        p1, r1 = recent['low'].iloc[m1[0]], recent['rsi'].iloc[m1[1]]
        p2, r2 = recent['low'].iloc[m2[0]], recent['rsi'].iloc[m2[1]]
        between_high = recent['high'].iloc[m1[0]:m2[0] + 1].max()
        if p2 < p1 and r2 > r1 and (between_high - min(p1, p2)) / min(p1, p2) > 0.03:
            bullish_count += 1; score += 2
            extra = "（超卖区）" if r2 < 40 else ""
            reasons.append(f"RSI底背离{extra} 价{p1:.2f}→{p2:.2f} RSI{r1:.1f}→{r2:.1f}")
            if r2 < 40:
                score += 1
    high_matches = match_extrema(price_highs, rsi_highs, max_distance=5)
    if len(high_matches) >= 2:
        m1, m2 = high_matches[-2], high_matches[-1]
        p1, r1 = recent['high'].iloc[m1[0]], recent['rsi'].iloc[m1[1]]
        p2, r2 = recent['high'].iloc[m2[0]], recent['rsi'].iloc[m2[1]]
        between_low = recent['low'].iloc[m1[0]:m2[0] + 1].min()
        if p2 > p1 and r2 < r1 and (max(p1, p2) - between_low) / max(p1, p2) > 0.03:
            bearish_count += 1; score -= 2
            extra = "（超买区）" if r2 > 60 else ""
            reasons.append(f"RSI顶背离{extra} 价{p1:.2f}→{p2:.2f} RSI{r1:.1f}→{r2:.1f}")
    if bullish_count >= 2:
        score += 2; reasons.append("★ MACD+RSI双重底背离")
    if bearish_count >= 2:
        score -= 2; reasons.append("☆ MACD+RSI双重顶背离")
    if not reasons:
        reasons.append("无明显背离")
    return {"bullish_div": bullish_count > 0, "bearish_div": bearish_count > 0,
            "double_bullish": bullish_count >= 2, "double_bearish": bearish_count >= 2,
            "score": score, "reasons": reasons, "rsi_value": round(current_rsi, 1)}


def detect_weekly_duck_head(df_weekly):
    if len(df_weekly) < 80:
        return {"valid": False, "score": 0, "reasons": ["数据不足"]}
    df = df_weekly.copy()
    df['ma5'] = df['close'].rolling(5).mean(); df['ma10'] = df['close'].rolling(10).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    score, reasons = 0, []
    if (df['ma60'].iloc[-1] - df['ma60'].iloc[-20]) / 20 > 0:
        score += 1; reasons.append("60周线向上")
    cross5 = (df['ma5'].shift(1) < df['ma60'].shift(1)) & (df['ma5'] > df['ma60'])
    cross10 = (df['ma10'].shift(1) < df['ma60'].shift(1)) & (df['ma10'] > df['ma60'])
    if (cross5 | cross10).iloc[-30:].any():
        score += 1; reasons.append("出现过5/10上穿60")
    pb = detect_pullback_ratio(df, 25)
    if pb["valid"]:
        score += 1; reasons.append(f"回调幅度合适({pb['ratio']}%)")
    if df['ma5'].iloc[-1] > df['ma10'].iloc[-1] and df['ma10'].iloc[-1] > df['ma60'].iloc[-1] * 0.98:
        score += 1; reasons.append("短期均线多头")
    vol = detect_volume_pattern(df, 20)
    if vol["valid"]:
        score += 1; reasons.append("量能符合放量-缩量")
    macd = detect_macd_golden_cross(df, timeframe="weekly", lookback=6)
    if macd["valid"]:
        score += 2; reasons.extend(macd["reasons"])
    elif macd["score"] >= 2:
        score += 1; reasons.append("周线MACD有转强迹象")
    div = detect_combined_divergence(df, lookback=35, order=3)
    if div["double_bullish"]:
        score += 3; reasons.append("MACD+RSI双重底背离（强）")
    elif div["bullish_div"]:
        score += 2; reasons.extend([r for r in div["reasons"] if "底背离" in r])
    if div["double_bearish"]:
        score -= 3; reasons.append("MACD+RSI双重顶背离（强警告）")
    elif div["bearish_div"]:
        score -= 2; reasons.extend([r for r in div["reasons"] if "顶背离" in r])
    if div["double_bearish"]:
        valid = score >= 7
    elif div["bearish_div"]:
        valid = score >= 5
    else:
        valid = score >= 4
    return {"valid": valid, "score": max(score, 0), "reasons": reasons, "divergence": div}


def detect_daily_w_bottom(df_daily):
    if len(df_daily) < 60:
        return {"valid": False, "score": 0, "reasons": ["数据不足"]}
    df = df_daily.copy()
    recent = df.iloc[-40:]
    try:
        lows = recent['low'].values
        inv = -lows
        peaks, props = find_peaks(inv, distance=8, prominence=max(np.std(lows) * 0.3, 0.01))
        if len(peaks) < 2:
            return {"valid": False, "score": 0, "reasons": ["找不到两个明显低点"]}
        p1, p2 = peaks[-2], peaks[-1]
        low1, low2 = lows[p1], lows[p2]
        bottom_diff = abs(low1 - low2) / min(low1, low2)
        bottoms_close = bottom_diff <= 0.05
        neck = recent['high'].iloc[p1:p2 + 1].max()
        breakthrough = df['close'].iloc[-1] > neck * 0.995
        vol = detect_volume_pattern(df, 25)
        macd = detect_macd_golden_cross(df, timeframe="daily", lookback=6)
        score, reasons = 0, []
        if bottoms_close:
            score += 1; reasons.append(f"两底接近(差{bottom_diff*100:.1f}%)")
        if breakthrough:
            score += 1; reasons.append("已突破颈线")
        if vol.get("breakthrough_volume"):
            score += 1; reasons.append("突破放量")
        if macd["valid"]:
            score += 1; reasons.append("日线MACD金叉")
        return {"valid": score >= 3, "score": score, "reasons": reasons, "neck": round(float(neck), 2)}
    except Exception:
        return {"valid": False, "score": 0, "reasons": ["检测异常"]}


def composite_signal(df_daily, df_weekly):
    weekly = detect_weekly_duck_head(df_weekly)
    daily = detect_daily_w_bottom(df_daily)
    strong = weekly["valid"] and daily["valid"]
    medium = (weekly["score"] >= 3 and daily["score"] >= 2) or (weekly["valid"] and daily["score"] >= 2)
    return {"signal": "强信号" if strong else ("中等信号" if medium else "无信号"),
            "strong": strong, "weekly": weekly, "daily": daily,
            "total_score": weekly.get("score", 0) + daily.get("score", 0)}


# ==================== 单只处理 ====================
def _process_one(args):
    code, name = args
    try:
        daily = _fetch_hist(code)
        if daily is None:
            return {"__fail__": "抓取失败"}
        if len(daily) < MIN_DATA_LEN:
            return {"__fail__": "数据不足"}
        time.sleep(SLEEP_PER_STOCK)
        weekly = _resample_weekly(daily)
        signal = composite_signal(daily, weekly)
        if signal["signal"] == "无信号":
            return {"__fail__": "无信号"}
        cL = float(daily['close'].iloc[-1])
        sig_date = pd.to_datetime(daily['date'].iloc[-1]).strftime('%Y-%m-%d') if 'date' in daily.columns else ""
        return {"代码": code, "名称": name, "行业": "",
                "最新价": round(cL, 2), "信号日期": sig_date,
                "信号强度": signal["signal"], "总分": signal["total_score"],
                "周线得分": signal["weekly"].get("score", 0), "日线得分": signal["daily"].get("score", 0),
                "周线理由": " | ".join(signal["weekly"].get("reasons", [])[:4]),
                "日线理由": " | ".join(signal["daily"].get("reasons", [])[:4]),
                "score": signal["total_score"], "resonance": False, "resonance_sector": ""}
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}


# ------------------ 主扫描 ------------------
def run_scan():
    global _INDUSTRY_MAP, FAIL_STATS
    FAIL_STATS = {k: 0 for k in FAIL_STATS}
    print("连接 Baostock（行业表 + 列表 + 子进程登录）...")
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
        try:
            bs.logout()
        except Exception:
            pass
        global _BS_LOGGED
        _BS_LOGGED = False
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        stock_df = _fetch_list_akshare()
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = stock_df['code'].tolist()
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]
    tasks = snapshot_prefilter(tasks)

    results = []; fail_count = 0
    print(f"开始老鸭头+W底扫描 {len(tasks)} 只（{NUM_PROCESSES}进程, 双源, 周线{LOOKBACK_DAYS}天）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="老鸭头W底", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  ★ {res['代码']} {res['名称']} {res['信号强度']} 总分{res['总分']} 周{res['周线得分']}日{res['日线得分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('总分', ascending=False).reset_index(drop=True)
    return df


# ------------------ 行业 + 聚类 + 风口 ------------------
def fetch_industry(symbol):
    for attempt in range(2):
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            if info is not None and not info.empty and 'item' in info.columns:
                row = info[info['item'].isin(['行业', '所属行业'])]
                if not row.empty:
                    return row.iloc[0]['value']
        except Exception:
            time.sleep(1 + attempt)
    return "—"


def get_industry_heat():
    for i in range(3):
        try:
            d = ak.stock_board_industry_name_em()
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"  行业热度榜第{i+1}次失败: {e}")
        time.sleep(2 + i)
    return pd.DataFrame()


def get_hot_sectors(heat):
    if heat.empty or '板块名称' not in heat.columns or '涨跌幅' not in heat.columns:
        return []
    h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
    h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
    return [(str(row['板块名称']), round(float(row['_chg']), 2)) for _, row in h.head(HOT_SECTOR_TOP).iterrows()]


def match_sector(sector, hot_names):
    if not sector or sector in ('—', '未知', '') or not hot_names:
        return ""
    s = sector.strip()
    for h in hot_names:
        if h and h == s:
            return h
    for h in hot_names:
        if h and (h in s or s in h):
            return h
    return ""


def sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def enrich(results):
    if not results:
        return pd.DataFrame(), [], []
    targets = results[:200]
    print(f"为 {len(targets)} 只命中标的补行业 ...")
    def _q(r):
        ind = _INDUSTRY_MAP.get(r['代码'], '')
        if not ind or ind in ('—', '未知', ''):
            sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
            ind = fetch_industry(sym)
        r['行业'] = ind
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🦆 老鸭头W底板块: {cluster or '无'}")
    heat = get_industry_heat(); hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in results:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 老鸭头W底遇风口 {cnt} 只")
    results.sort(key=lambda r: (1 if r.get('resonance') else 0, 1 if r['信号强度'] == '强信号' else 0, r['总分']), reverse=True)
    return pd.DataFrame(results), cluster, hot


# ------------------ 实时对齐 ------------------
def _fetch_spot_now():
    try:
        d = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if d is None or d.empty or '代码' not in d.columns:
            print("  实时对齐: 快照空(限流), 对齐列降级"); return {}
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        if '最新价' in d.columns:
            d['最新价'] = pd.to_numeric(d['最新价'], errors='coerce')
        out = {r['代码']: float(r['最新价']) for _, r in d.iterrows() if pd.notna(r.get('最新价'))}
        print(f"  实时对齐: 取到 {len(out)} 只现价"); return out
    except Exception as e:
        print(f"  实时对齐: 快照失败({e}), 对齐列降级"); return {}


def _align_suffix(r, spot_now):
    sig_price = r.get('最新价'); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price):
        return ""
    head = f"🕒信号{sig_price}"
    if sig_date and not pd.isna(sig_date):
        sd = str(sig_date)[:10]; head += f"@{sd[-5:]}"
        try:
            days = (datetime.now().date() - pd.to_datetime(sd).date()).days
            if days >= 0:
                head += f"(距今{days}天)"
        except Exception:
            pass
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return f" | {head} → 现价{now}@run({chg:+.1f}%)"
        except Exception:
            return f" | {head}"
    return f" | {head}"


def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    strong = df[df['信号强度'] == '强信号'] if '信号强度' in df.columns else pd.DataFrame()
    medium = df[df['信号强度'] == '中等信号'] if '信号强度' in df.columns else pd.DataFrame()
    L = [f"**🦆 周线老鸭头+日线W底** | 强信号{len(strong)} 中等{len(medium)} 🎯风口{len(reso)} (全发)",
         "*(周线老鸭头×日线W底×MACD/RSI双重背离×板块; 概率性形态判断, 非预测; 结合仓位/止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🦆 **老鸭头W底板块**: " + "、".join(f"{n}({c}只)" for n, c in cluster)); L.append("")
    def line(r):
        flag = "🟢" if r['信号强度'] == '强信号' else "🟡"
        return (f"- {flag} **{r['名称']}({r['代码']})** [{sec_tag(r.to_dict())}] {r['信号强度']} 总分{r['总分']} "
                f"(周{r['周线得分']}/日{r['日线得分']}) 现价{r['最新价']}{_align_suffix(r, spot_now)}<br>"
                f" 周:{r['周线理由']}<br> 日:{r['日线理由']}")
    if not reso.empty:
        L.append(f"### 🎯 遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not strong.empty:
        L.append(f"### 🟢 强信号 共{len(strong)}只 (周线老鸭头+日线W底双确认)")
        L += [line(r) for _, r in strong.head(PUSH_TOP).iterrows()]; L.append("")
    if PUSH_MEDIUM and not medium.empty:
        L.append(f"### 🟡 中等信号 共{len(medium)}只")
        L += [line(r) for _, r in medium.head(PUSH_TOP).iterrows()]
    return "\n".join(L)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"🦆 周线老鸭头+日线W底 (日线扫描) | {datetime.now():%Y-%m-%d %H:%M} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 多进程{NUM_PROCESSES}; 预筛={'开' if SNAPSHOT_PRE else '关'}; "
          f"周线老鸭头×日线W底×双重背离; 不拦交易日; 推送全列+分页")
    print("⚠️ 概率性形态+背离+板块综合判断, 非预测; 结合仓位/止损, 不构成投资建议")
    print("=" * 70)
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule' and not is_trading_day():
        print("非交易日且为定时触发(schedule), 跳过; 手动/本地不受此限"); sys.exit(0)

    df = run_scan()
    if df is None or df.empty:
        print("\n本次未发现满足 老鸭头+W底 的票 (周线+日线双确认极严, 0命中属正常)。")
        sys.exit(0)
    df, cluster, hot = enrich(df.to_dict('records'))
    tag = datetime.now().strftime('%Y%m%d')
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"duck_head_wbottom_{tag}.csv"), index=False, encoding="utf-8-sig")
        df.to_json(os.path.join(OUTPUT_DIR, f"duck_head_wbottom_{tag}.json"), orient='records', force_ascii=False, indent=2)
        print(f"\n📁 已存 output/duck_head_wbottom_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}"); traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = _fetch_spot_now()
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_strong = int((df['信号强度'] == '强信号').sum()) if '信号强度' in df.columns else 0
            send_serverchan(f"🦆 老鸭头W底 强信号{n_strong}只 共{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_duck_wbottom<<<
