# -*- coding: utf-8 -*-
"""
first_red_wbottom_screener.py —— 120日首红W底 + MACD底背离 + 均线粘合 全市场选股(日线扫描版) · 矩阵规格
源自 TradingView Pine strategy "120日首红W底 + 优化MACD背离 + 均线粘合 (ATR自适应·预设)",
移植其【入场信号 finalSignal】为日线批量截面扫描(不含持仓态的止损止盈触发/均线退出, 只给初始建议位)。

入场信号 = 三共振同时满足:
  ① W底突破首红(wBottomSignal): 长期低位(close<MA120×1.08)下, pivotlow找两底(间隔15~300根、容差ATR自适应),
     颈线=max(两底)×1.02, 收盘突破颈线 + 首红(收阳实体>0.4% + 放量×volMult + 距上次站上MA120>70根)。
  ② MACD底背离有效(bullDiv): pivotlow价格创新低 但 MACD柱抬高(零轴下+柱上升), 背离确认后divValidWindow(20)根内有效。
  ③ 均线粘合(isStrongGlue or isGlueNow): MA5/10/20粘合(ATR自适应阈值), 连续粘合≥minGlueBars。
  finalSignal = ① and ② and ③。截面扫描看最近 SIGNAL_FRESH_BARS(5) 根内是否触发。

【ATR自适应预设】低波/中波(默认)/高波/极高波, 按ATR%自动调W底容差/粘合阈值/止损止盈倍数/放量倍数(PRESET env可选)。
【截面扫描适配】Pine是逐bar策略含持仓管理; 本脚本只移植入场信号+初始建议位(止损=价-ATR×stopMult, 止盈=价+ATR×limitMult),
  止损止盈触发/均线跌破退出是持仓后动态管理, 扫描不执行, 拿到建议位后手动管理或回测。
【数据长度】W底两底间隔最多300根+MA120预热120根+pivot右5根+首红barssince70, 故拉~1100天、min_data_len=450。
【工程规格】双源baostock+东财+硬超时; 多进程+快照预筛砍量; baostock行业本地join+聚类+东财风口🎯;
  推送全发分页+信号vs实时对齐列; 不拦交易日; 收尾防护sys.exit(0); append补丁。阈值多env可调。
⚠️ 三共振信号极严, 全市场0命中属正常, 非bug。⚠️ 信号非买入保证, W底/背离均可能失败, 务必按建议位止损。
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

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


# ==================== ATR自适应预设 (对应Pine getPresetParams) ====================
# [bottomAtrMult, glueAtrMult, minBottomTol%, maxBottomTol%, minGlueThresh%, maxGlueThresh%, minGlueBars, stopAtrMult, limitAtrMult, volMult]
PRESETS = {
    "低波":   dict(bottomAtrMult=0.70, glueAtrMult=0.35, minBottomTol=0.8, maxBottomTol=2.0, minGlueThresh=0.4, maxGlueThresh=1.0, minGlueBars=2, stopAtrMult=1.5, limitAtrMult=2.8, volMult=1.5),
    "中波":   dict(bottomAtrMult=0.90, glueAtrMult=0.45, minBottomTol=1.0, maxBottomTol=3.5, minGlueThresh=0.5, maxGlueThresh=1.5, minGlueBars=2, stopAtrMult=1.8, limitAtrMult=3.0, volMult=1.6),
    "高波":   dict(bottomAtrMult=1.10, glueAtrMult=0.55, minBottomTol=1.5, maxBottomTol=4.5, minGlueThresh=0.7, maxGlueThresh=2.0, minGlueBars=2, stopAtrMult=2.0, limitAtrMult=3.5, volMult=1.8),
    "极高波": dict(bottomAtrMult=1.25, glueAtrMult=0.65, minBottomTol=2.0, maxBottomTol=5.5, minGlueThresh=0.9, maxGlueThresh=2.5, minGlueBars=1, stopAtrMult=2.2, limitAtrMult=4.0, volMult=2.0),
}

# ==================== 参数 (固定项=Pine默认; env可调项) ====================
PRESET = os.environ.get('PRESET', '中波')
if PRESET not in PRESETS:
    PRESET = '中波'
P = PRESETS[PRESET]

# 固定参数 (Pine 默认)
LOOKBACK120 = 120
MIN_BARS_BETWEEN = 15
MAX_BARS_BETWEEN = 300
STALE_BARS = 250
ATR_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
PIVOT_LEFT, PIVOT_RIGHT = 5, 5
MIN_DIV_BARS, MAX_DIV_BARS = 10, 60
REQUIRE_BELOW_ZERO = True
REQUIRE_HIST_RISING = True
DIV_VALID_WINDOW = 20
GLUE_LEN1, GLUE_LEN2, GLUE_LEN3 = 5, 10, 20

# env 可调
SIGNAL_FRESH_BARS = int(os.environ.get('SIGNAL_FRESH_BARS', '5'))
MIN_DATA_LEN = int(os.environ.get('MIN_DATA_LEN', '450'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '1100'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '3.0'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = float(os.environ.get('SLEEP', '0.1'))
FETCH_TIMEOUT = int(os.environ.get('FETCH_TIMEOUT', '15'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SNAPSHOT_PRE = os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True')
PRE_AMOUNT_MIN = float(os.environ.get('PRE_AMOUNT_MIN', '3.0e7'))
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
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c


def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


# ------------------ 历史双源 (OHLCV) ------------------
def _fetch_hist_em(sym, start_y):
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume', '换手率': 'turn'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    return d
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
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    return d
        except Exception:
            pass
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    return _fetch_hist_em(sym, sy)


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


# ==================== 指标工具 (numpy) ====================
def _sma(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).mean().to_numpy()


def _atr_arr(high, low, close, n):
    prev = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def _macd_arr(close, fast, slow, sig):
    c = pd.Series(close)
    ml = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml.to_numpy(), sl.to_numpy(), (ml - sl).to_numpy()


def _pivotlow(low, left, right):
    """Pine ta.pivotlow: 在 bar (i+right) 处返回 bar i 的局部最低(窗口[i-left, i+right])。"""
    n = len(low)
    pl = np.full(n, np.nan)
    for i in range(left, n - right):
        w = low[i - left:i + right + 1]
        if low[i] <= np.nanmin(w):
            pl[i + right] = low[i]
    return pl


def _barssince(cond):
    n = len(cond)
    out = np.full(n, 10 ** 9)
    last = -10 ** 9
    for i in range(n):
        if cond[i]:
            last = i
        out[i] = i - last
    return out


def _consecutive_count(cond):
    n = len(cond)
    out = np.zeros(n, dtype=int)
    c = 0
    for i in range(n):
        c = c + 1 if cond[i] else 0
        out[i] = c
    return out


# ==================== W底状态机 (复刻Pine var逻辑) ====================
def _detect_wbottom_state(close, ma120, pLow, bottomTol, firstRed, n):
    wSig = np.zeros(n, dtype=bool)
    b1 = np.nan; b1Bar = -10 ** 9; neck = np.nan; wPending = False
    for i in range(n):
        pl = pLow[i]
        if not np.isnan(pl):
            pivotBar = i - PIVOT_RIGHT
            if close[i] < ma120[i] * 1.08:
                if np.isnan(b1) or (i - b1Bar > STALE_BARS):
                    b1 = pl; b1Bar = pivotBar; neck = np.nan; wPending = False
                elif (pivotBar - b1Bar > MIN_BARS_BETWEEN) and (pivotBar - b1Bar < MAX_BARS_BETWEEN):
                    if b1 > 0 and abs(pl - b1) / b1 <= bottomTol[i]:
                        neck = max(b1, pl) * 1.02
                        wPending = True
                    else:
                        b1 = pl; b1Bar = pivotBar; neck = np.nan; wPending = False
        if wPending and not np.isnan(neck) and close[i] > neck and firstRed[i]:
            wSig[i] = True
            wPending = False; b1 = np.nan; neck = np.nan
    return wSig


# ==================== MACD底背离状态机 (复刻Pine var逻辑) ====================
def _detect_macd_div_state(macdLine, hist, pricePivot, n):
    bullDivRaw = np.zeros(n, dtype=bool)
    lastPriceLow = np.nan; lastMacdLow = np.nan; lastPivotBar = -10 ** 9
    for i in range(n):
        pp = pricePivot[i]
        if not np.isnan(pp):
            pivotBar = i - PIVOT_RIGHT
            if pivotBar >= 1 and not np.isnan(lastPriceLow) and not np.isnan(lastMacdLow):
                barsSince = pivotBar - lastPivotBar
                if MIN_DIV_BARS <= barsSince <= MAX_DIV_BARS:
                    priceLowerLow = pp < lastPriceLow
                    macdHigherLow = hist[pivotBar] > lastMacdLow
                    belowZero = (macdLine[pivotBar] < 0) if REQUIRE_BELOW_ZERO else True
                    histRising = (hist[pivotBar] > hist[pivotBar - 1]) if REQUIRE_HIST_RISING else True
                    if priceLowerLow and macdHigherLow and belowZero and histRising:
                        bullDivRaw[i] = True
            lastPriceLow = pp
            lastMacdLow = hist[pivotBar] if pivotBar >= 0 else np.nan
            lastPivotBar = pivotBar
    return _barssince(bullDivRaw) <= DIV_VALID_WINDOW


# ==================== 策略内核: 三共振 finalSignal ====================
def detect_signal(df):
    if df is None or len(df) < MIN_DATA_LEN:
        return None, "数据不足"
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
    if len(df) < MIN_DATA_LEN:
        return None, "数据不足"

    close = df['close'].to_numpy(float); open_ = df['open'].to_numpy(float)
    high = df['high'].to_numpy(float); low = df['low'].to_numpy(float)
    volume = df['volume'].to_numpy(float)
    n = len(close)

    ma120 = _sma(close, LOOKBACK120)
    ma1 = _sma(close, GLUE_LEN1); ma2 = _sma(close, GLUE_LEN2); ma3 = _sma(close, GLUE_LEN3)
    atr = _atr_arr(high, low, close, ATR_LEN)
    with np.errstate(invalid='ignore', divide='ignore'):
        atrPct = atr / close * 100
    bottomTol = np.clip(atrPct * P['bottomAtrMult'], P['minBottomTol'], P['maxBottomTol']) / 100
    glueThresh = np.clip(atrPct * P['glueAtrMult'], P['minGlueThresh'], P['maxGlueThresh']) / 100

    # 均线粘合
    with np.errstate(invalid='ignore', divide='ignore'):
        avgMA = (ma1 + ma2 + ma3) / 3
        isGlueNow = (np.abs(ma1 - ma2) / avgMA <= glueThresh) & (np.abs(ma2 - ma3) / avgMA <= glueThresh)
    isGlueNow = np.nan_to_num(isGlueNow, nan=False).astype(bool)
    isStrongGlue = _consecutive_count(isGlueNow) >= P['minGlueBars']

    # 120日首红
    volMa20 = _sma(volume, 20)
    volCond = volume > volMa20 * P['volMult']
    with np.errstate(invalid='ignore', divide='ignore'):
        redBody = (close > open_) & ((close - open_) / open_ > 0.004)
    firstRed = np.nan_to_num(redBody & volCond & (_barssince(close > ma120) > 70), nan=False).astype(bool)

    # ① W底突破首红
    pLow = _pivotlow(low, PIVOT_LEFT, PIVOT_RIGHT)
    wSig = _detect_wbottom_state(close, ma120, pLow, bottomTol, firstRed, n)

    # ② MACD底背离有效
    macdLine, sigLine, hist = _macd_arr(close, MACD_FAST, MACD_SLOW, MACD_SIG)
    bullDiv = _detect_macd_div_state(macdLine, hist, pLow, n)

    # ③ finalSignal = ① and ② and (强粘合 or 粘合)
    finalSignal = wSig & bullDiv & (isStrongGlue | isGlueNow)

    fresh = SIGNAL_FRESH_BARS
    tail = finalSignal[-fresh:]
    if not tail.any():
        return None, "无信号"
    sig_idx = n - fresh + int(np.where(tail)[0][-1])
    days_ago = (n - 1) - sig_idx

    sig_date = pd.to_datetime(df['date'].iloc[sig_idx]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    cL = float(close[-1]); atrL = float(atr[-1])
    if not (atrL > 0):
        return None, "数据不足"
    info = {
        "代码": None, "名称": None, "行业": "",
        "最新价": round(cL, 2), "信号日期": sig_date, "距今天数": int(days_ago),
        "ATR%": round(float(atrPct[-1]), 2),
        "W底容差%": round(float(bottomTol[-1] * 100), 2),
        "粘合阈值%": round(float(glueThresh[-1] * 100), 2),
        "均线粘合": "是" if bool(isStrongGlue[-1] or isGlueNow[-1]) else "否",
        "MACD背离": "是" if bool(bullDiv[-1]) else "否",
        "建议止损": round(cL - atrL * P['stopAtrMult'], 2),
        "建议止盈": round(cL + atrL * P['limitAtrMult'], 2),
        "预设": PRESET,
        "score": round(100 - days_ago * 5, 1),
        "resonance": False, "resonance_sector": "",
    }
    return info, None


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < MIN_DATA_LEN:
            return {"__fail__": "数据不足"}
        time.sleep(SLEEP_PER_STOCK)
        info, reason = detect_signal(df)
        if info is None:
            return {"__fail__": reason}
        info["代码"] = code; info["名称"] = name
        return info
    except FutureTimeoutError:
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
    print(f"开始120日首红W底三共振扫描 {len(tasks)} 只（{NUM_PROCESSES}进程, 双源, 预设={PRESET}, 信号窗口={SIGNAL_FRESH_BARS}根）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="W底首红", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 信号{res['信号日期']} 距今{res['距今天数']}天 价={res['最新价']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
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
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🔴 W底首红板块: {cluster or '无'}")
    heat = get_industry_heat(); hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in results:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 W底首红遇风口 {cnt} 只")
    results.sort(key=lambda r: (1 if r.get('resonance') else 0, r['score']), reverse=True)
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
    L = [f"**🔴 120日首红W底·三共振** | 命中{len(df)}只 🎯风口{len(reso)} (全发, 预设{PRESET})",
         "*(W底突破首红×MACD底背离×均线粘合; 信号极严0命中正常; 建议位=入场初始位, 止损止盈触发需持仓后手动; 非买入保证, 必止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🔴 **W底首红板块**: " + "、".join(f"{n}({c}只)" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{sec_tag(r.to_dict())}] 信号{r['信号日期']}(距今{r['距今天数']}天) "
                f"现价{r['最新价']} ATR{r['ATR%']}% 粘合{r['均线粘合']} 背离{r['MACD背离']} "
                f"止损{r['建议止损']}/止盈{r['建议止盈']}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 W底首红遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    L.append(f"### 🔴 全部W底首红三共振 共{len(df)}只 (按新鲜度)")
    L += [line(r) for _, r in df.head(PUSH_TOP).iterrows()]
    if len(df) > PUSH_TOP:
        L.append(f"\n*…另有 {len(df)-PUSH_TOP} 只, 详见 output 报告*")
    return "\n".join(L)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"🔴 120日首红W底·三共振 (日线扫描) | {datetime.now():%Y-%m-%d %H:%M} | 预设={PRESET} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 多进程{NUM_PROCESSES}; 预筛={'开' if SNAPSHOT_PRE else '关'}; "
          f"W底首红×MACD背离×均线粘合; 信号窗口={SIGNAL_FRESH_BARS}根; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule' and not is_trading_day():
        print("非交易日且为定时触发(schedule), 跳过; 手动/本地不受此限"); sys.exit(0)

    df = run_scan()
    if df is None or df.empty:
        print("\n本次未发现满足 120日首红W底三共振 的信号 (三条件叠加极严, 0命中属正常)。")
        print("可调: PRESET改高波/极高波(放宽容差) / SIGNAL_FRESH_BARS调大(放宽信号新鲜度)")
        sys.exit(0)
    df, cluster, hot = enrich(df.to_dict('records'))
    tag = datetime.now().strftime('%Y%m%d')
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"first_red_wbottom_{tag}.csv"), index=False, encoding="utf-8-sig")
        df.to_json(os.path.join(OUTPUT_DIR, f"first_red_wbottom_{tag}.json"), orient='records', force_ascii=False, indent=2)
        print(f"\n📁 已存 output/first_red_wbottom_{tag}.*")
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
            send_serverchan(f"🔴 120日首红W底 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_first_red_wbottom<<<
