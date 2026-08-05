# -*- coding: utf-8 -*-
"""
duck_wbottom_kdj_screener.py —— 回踩筑底四共振 全市场选股(日线扫描版) · 矩阵规格
四共振(AND): ① 周线老鸭头 ② 日线W底 ③ KDJ超卖(J≤J_MAX) ④ MACD零轴上(DIF>0)。
【本版·实时价】新增腾讯实时行情(qt.gtimg.cn, 海外IP可访问): 推送前把"最新价"刷成实时价
  (午盘收盘后跑=午盘实时价, 下午收盘后跑=收盘实时价), 原日线收盘价存为"信号价"供对齐列
  (🕒信号价@日期 → 现价@run 涨跌幅); 东财快照被墙也有实时价。加场次标签区分每天两次推送。
【KDJ/MACD 原样搬自 kdj_macd_screener.py 真码】口径一致。
【周线老鸭头/日线W底 本脚本自带 numpy 实现, 不依赖 scipy】。
【工程规格】双源baostock+东财+硬超时; 多进程+快照预筛; 行业本地join+聚类+风口🎯;
  推送全发分页+信号vs实时对齐列; 收尾防护sys.exit(0); append补丁。阈值全env可调。
【数据】老鸭头需~80周线, LOOKBACK_DAYS=1800, MIN_DATA_LEN=450; 建议 SCAN_LIMIT≤1500。
⚠️ 四共振极严, 0命中属正常; 非买入保证, 等J拐头/放量确认+止损。
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
from datetime import datetime, timedelta, timezone

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
KDJ_N = int(os.environ.get('KDJ_N', '9'))
J_MAX = float(os.environ.get('J_MAX', '0.0'))
MACD_MODE = os.environ.get('MACD_MODE', 'zero')
REQUIRE_KDJ_OS = os.environ.get('REQUIRE_KDJ_OS', '1').strip() in ('1', 'true', 'True')
REQUIRE_DIF_POS = os.environ.get('REQUIRE_DIF_POS', '1').strip() in ('1', 'true', 'True')
DUCK_SCORE_MIN = int(os.environ.get('DUCK_SCORE_MIN', '4'))
WBOTTOM_TOL = float(os.environ.get('WBOTTOM_TOL', '0.05'))
NEAR_NECK = float(os.environ.get('NEAR_NECK', '1.03'))
WBOTTOM_LOOKBACK = int(os.environ.get('WBOTTOM_LOOKBACK', '60'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '1800'))
MIN_DATA_LEN = int(os.environ.get('MIN_DATA_LEN', '450'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '3.0'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP = float(os.environ.get('SLEEP', '0.1'))
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "周线老鸭头未成": 0, "日线W底未成": 0,
              "价格飞出底区": 0, "KDJ未超卖": 0, "MACD未上0轴": 0}

_BJ = timezone(timedelta(hours=8))


def _bj_now():
    return datetime.now(_BJ)


def _session_tag():
    """区分午盘盘后 / 盘中 / 收盘后, 让每天两次推送可辨识。"""
    h = _bj_now().hour
    if h < 13:
        return "午盘盘后"
    elif h < 15:
        return "盘中"
    return "收盘后"


# ------------------ 推送 (全发分页) ------------------
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
        if j.get('code') != 0:
            print(f"  requests返回非0: {j}")
        return j.get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


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
        ok = _send_one(t, ch, key) and ok
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


def _bs_q(code, fields, sd, timeout=FETCH_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
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


# ------------------ 【本版】腾讯实时价 (海外可访问) ------------------
def _fetch_realtime_tencent(codes):
    """腾讯实时行情, 分批查询, 返回 {6位代码: 现价}。失败返回 {}。"""
    out = {}
    try:
        syms = []
        for c in codes:
            c6 = str(c).split('.')[-1].zfill(6)
            if c6[:1] in ('6', '9'):
                pref = 'sh'
            elif c6[:1] in ('4', '8'):
                pref = 'bj'
            else:
                pref = 'sz'
            syms.append(f"{pref}{c6}")
        for i in range(0, len(syms), 50):
            batch = syms[i:i + 50]
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(batch), timeout=10)
                r.encoding = 'gbk'
                for line in r.text.strip().split(';'):
                    line = line.strip()
                    if '=' not in line:
                        continue
                    body = line.split('=', 1)[1].strip().strip('"')
                    f = body.split('~')
                    if len(f) > 4 and f[2]:
                        try:
                            price = float(f[3])
                            if price > 0:
                                out[f[2].zfill(6)] = price
                        except Exception:
                            pass
            except Exception as e:
                print(f"   [实时价] 批次{i // 50 + 1}失败: {e}")
            time.sleep(0.3)
    except Exception as e:
        print(f"  腾讯实时价异常: {e}")
    return out


def _refresh_realtime_price(df):
    """把"最新价"刷成腾讯实时价, 原日线收盘价保留为"信号价"供对齐列。返回 (df, rt)。"""
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
        n_rt = int(df['实时价'].notna().sum())
        print(f"  实时价刷新: 腾讯取到 {n_rt}/{len(df)} 只实时价")
    else:
        print("  实时价刷新: 腾讯未取到, 沿用日线收盘价")
    return df, rt


# ------------------ 历史双源 ------------------
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
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
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


# ==================== KDJ / MACD ====================
def calc_kdj(df, n=9, k_period=3, d_period=3):
    df = df.copy()
    low_n = df["low"].rolling(n).min(); high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    df["K"] = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / d_period, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def _macd_ok(dif, dea):
    if MACD_MODE == "zero_cross":
        return bool(dif > 0 and dif > dea)
    return bool(dif > 0)


def _macd_series(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean(); es = close.ewm(span=slow, adjust=False).mean()
    dif = ef - es; dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


# ==================== 周线老鸭头 ====================
def _detect_duck(df_daily):
    try:
        w = df_daily.set_index('date').resample('W').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        if len(w) < 80:
            return False, 0
        w['ma5'] = w['close'].rolling(5).mean(); w['ma10'] = w['close'].rolling(10).mean()
        w['ma60'] = w['close'].rolling(60).mean()
        if pd.isna(w['ma60'].iloc[-1]) or pd.isna(w['ma60'].iloc[-20]):
            return False, 0
        score = 0
        if (w['ma60'].iloc[-1] - w['ma60'].iloc[-20]) / 20 > 0:
            score += 1
        cross5 = (w['ma5'].shift(1) < w['ma60'].shift(1)) & (w['ma5'] > w['ma60'])
        cross10 = (w['ma10'].shift(1) < w['ma60'].shift(1)) & (w['ma10'] > w['ma60'])
        if (cross5 | cross10).iloc[-30:].any():
            score += 1
        win = w.iloc[-25:]; peak, trough = win['high'].max(), win['low'].min()
        if peak > 0 and 0.08 <= (peak - trough) / peak <= 0.35:
            score += 1
        if w['ma5'].iloc[-1] > w['ma10'].iloc[-1] and w['ma10'].iloc[-1] > w['ma60'].iloc[-1] * 0.98:
            score += 1
        dif, dea, hist = _macd_series(w['close'])
        cross = (dif.shift(1) < dea.shift(1)) & (dif > dea)
        if cross.iloc[-6:].any():
            score += 2
        elif dif.iloc[-1] > dea.iloc[-1]:
            score += 1
        return score >= DUCK_SCORE_MIN, score
    except Exception:
        return False, 0


# ==================== 日线W底 ====================
def _detect_wbottom(df_daily):
    out = {"bottoms_close": False, "neck": None, "low_min": None, "in_zone": False}
    try:
        if len(df_daily) < WBOTTOM_LOOKBACK:
            return out
        recent = df_daily.iloc[-WBOTTOM_LOOKBACK:].reset_index(drop=True)
        lows = recent['low'].to_numpy(float)
        peaks = []
        for i in range(1, len(lows) - 1):
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                peaks.append(i)
        filt = []
        for p in peaks:
            if not filt or p - filt[-1] >= 8:
                filt.append(p)
        if len(filt) < 2:
            return out
        p1, p2 = filt[-2], filt[-1]
        low1, low2 = lows[p1], lows[p2]
        if min(low1, low2) <= 0:
            return out
        out["bottoms_close"] = bool(abs(low1 - low2) / min(low1, low2) <= WBOTTOM_TOL)
        neck = float(recent['high'].iloc[p1:p2 + 1].max())
        low_min = float(min(low1, low2))
        out["neck"] = round(neck, 2); out["low_min"] = round(low_min, 2)
        close_now = float(df_daily['close'].iloc[-1])
        out["in_zone"] = bool(close_now <= neck * NEAR_NECK and close_now >= low_min * 0.97)
        return out
    except Exception:
        return out


# ==================== 策略内核: 四共振 AND ====================
def check_one_stock(df):
    if df is None or len(df) < MIN_DATA_LEN:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"
    df = df.copy(); df['date'] = pd.to_datetime(df['date'])
    df = calc_kdj(df, n=KDJ_N); df = calc_macd(df)
    L = df.iloc[-1]
    J = float(L['J']); K = float(L['K']); D = float(L['D'])
    DIF = float(L['DIF']); DEA = float(L['DEA']); close = float(L['close'])

    if REQUIRE_KDJ_OS and J > J_MAX:
        return None, "KDJ未超卖"
    if REQUIRE_DIF_POS and not _macd_ok(DIF, DEA):
        return None, "MACD未上0轴"
    duck_ok, duck_score = _detect_duck(df)
    if not duck_ok:
        return None, "周线老鸭头未成"
    wb = _detect_wbottom(df)
    if not wb["bottoms_close"]:
        return None, "日线W底未成"
    if not wb["in_zone"]:
        return None, "价格飞出底区"

    j_score = min(40.0, max(0.0, -J))
    dif_pct = DIF / close * 100 if close else 0
    dif_score = min(10.0, max(0.0, dif_pct))
    neck = wb["neck"] or close
    near_neck_pct = (close - neck) / neck * 100 if neck else 0
    zone_score = max(0.0, 10.0 - abs(near_neck_pct))
    score = round(j_score + duck_score * 2 + zone_score + dif_score, 1)

    sig_date = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(close, 2), "信号价": round(close, 2), "信号日期": sig_date,
            "J": round(J, 2), "K": round(K, 2), "D": round(D, 2),
            "DIF": round(DIF, 4), "DEA": round(DEA, 4),
            "鸭头分": duck_score, "颈线": wb["neck"], "底": wb["low_min"],
            "距颈线%": round(near_neck_pct, 1),
            "score": score, "resonance": False, "resonance_sector": ""}, None


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < MIN_DATA_LEN:
            return {"__fail__": "数据不足"}
        time.sleep(SLEEP)
        info, reason = check_one_stock(df)
        if info is None:
            return {"__fail__": reason}
        info["代码"] = code; info["名称"] = name
        return info
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
    codes = snapshot_prefilter(codes)
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail_count = 0
    print(f"开始回踩筑底四共振扫描 {len(tasks)} 只（{NUM_PROCESSES}进程, 双源, 周线{LOOKBACK_DAYS}天）...")
    print(f"四共振: 老鸭头≥{DUCK_SCORE_MIN} + W底(容差{WBOTTOM_TOL}/底区≤颈线×{NEAR_NECK}) + J≤{J_MAX}(卡={REQUIRE_KDJ_OS}) + DIF>0(卡={REQUIRE_DIF_POS}, {MACD_MODE})")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="回踩筑底", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} J={res['J']} DIF={res['DIF']} 鸭头{res['鸭头分']} 距颈线{res['距颈线%']}% 分{res['score']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各条件淘汰统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 淘汰{fail_count}")
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
        ind = _INDUSTRY_MAP.get(r['代码'], '')
        if not ind or ind in ('—', '未知', ''):
            sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
            ind = fetch_industry(sym)
        r['行业'] = ind
    with cf.ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🦆 回踩筑底板块: {cluster or '无'}")
    heat = get_industry_heat(); hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in results:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 回踩筑底遇风口 {cnt} 只")
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
    sig_price = r.get('信号价', r.get('最新价'))
    sig_date = r.get('信号日期')
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
    dif_tag = ('零轴上+金叉' if MACD_MODE == 'zero_cross' else '站上0轴')
    L = [f"**🦆📉 回踩筑底四共振·{_session_tag()}** | 命中{len(df)}只 🎯风口{len(reso)} (全发) | 现价=实时价",
         f"*(周线老鸭头×日线W底×KDJ超卖J≤{J_MAX}×MACD{dif_tag}; 趋势框架内强势回踩双底超卖=跌下来的蓄势/筑底; 追高被J超卖+底区双排除; 非买入保证, 等J拐头/放量+止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🦆 **回踩筑底板块**: " + "、".join(f"{n}({c}只)" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{sec_tag(r.to_dict())}] 现价{r['最新价']} "
                f"J={r['J']} DIF={r['DIF']} 鸭头{r['鸭头分']} 颈线{r['颈线']}(距{r['距颈线%']}%) 分{r['score']}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    L.append(f"### 🦆 全部四共振 共{len(df)}只 (按超卖+共振强度)")
    L += [line(r) for _, r in df.head(PUSH_TOP).iterrows()]
    if len(df) > PUSH_TOP:
        L.append(f"\n*…另有 {len(df)-PUSH_TOP} 只, 详见 output 报告*")
    return "\n".join(L)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    dif_tag = ('零轴上+金叉' if MACD_MODE == 'zero_cross' else '站上0轴')
    print(f"🦆📉 回踩筑底四共振 (日线扫描) | 北京 {_bj_now():%Y-%m-%d %H:%M} | 场次: {_session_tag()} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 多进程{NUM_PROCESSES}; 预筛={'开' if SNAPSHOT_PRE else '关'}; "
          f"老鸭头×W底×J≤{J_MAX}(卡{REQUIRE_KDJ_OS})×{dif_tag}(卡{REQUIRE_DIF_POS}); 不拦交易日; 推送全列+分页")
    print("⚠️ 趋势框架内强势回踩型'跌下来'(非跌崩抄底, 要后者设REQUIRE_DIF_POS=0); 四共振极严0命中正常; 非买入保证")
    print("=" * 70)
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule' and not is_trading_day():
        print("非交易日且为定时触发(schedule), 跳过; 手动/本地不受此限"); sys.exit(0)

    df = run_scan()
    if df is None or df.empty:
        print("\n本次未发现满足 回踩筑底四共振 的票 (四条件叠加极严, 0命中属正常)。")
        print("看上面淘汰统计定位瓶颈; 放宽: J_MAX调大 / REQUIRE_DIF_POS=0(允许跌深) / NEAR_NECK调大(放宽底区) / DUCK_SCORE_MIN调小")
        sys.exit(0)
    df, cluster, hot = enrich(df.to_dict('records'))
    # 【本版】实时价刷新: 腾讯行情, 午盘/收盘两次推送的现价都是实时价
    df, rt = _refresh_realtime_price(df)
    tag = datetime.now().strftime('%Y%m%d')
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"duck_wbottom_kdj_{tag}.csv"), index=False, encoding="utf-8-sig")
        df.to_json(os.path.join(OUTPUT_DIR, f"duck_wbottom_kdj_{tag}.json"), orient='records', force_ascii=False, indent=2)
        print(f"\n📁 已存 output/duck_wbottom_kdj_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}"); traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'K', 'D', 'DEA', '底', '信号价', '实时价'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = rt if rt else _fetch_spot_now()
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"🦆 回踩筑底·{_session_tag()} 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_duck_wbottom_kdj<<<
