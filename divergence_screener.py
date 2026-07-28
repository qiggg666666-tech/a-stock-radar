# -*- coding: utf-8 -*-
"""
divergence_screener.py —— 多重底背离加权评分 全市场选股(日线扫描版) · 矩阵规格
源自"MA+VWMA+BB+斐波那契+RSI/MACD/OBV/VWMA多重底背离(加权评分)+ATR止损止盈"策略,
移植其【买入信号】为日线批量扫描。买入=趋势回调 × 多重底背离 × 斐波那契支撑 三重共振:
  ① 趋势OK(trend_ok): 收盘>MA50 且 收盘>VWMA20 且 收盘<=布林中轨 (上升趋势中的回调位)
  ② 多重底背离(加权评分): RSI/MACD/OBV/VWMA 四项底背离, 默认满足≥3项即通过
     (STRICT_MODE=True 可切回四项全满足); 底背离=价格创新低而指标抬高
  ③ 斐波那契支撑: 收盘靠近 0.5/0.618/0.786 回撤位(距离≤1倍ATR)
对命中票, 用当前ATR给【初始建议位】: 止损=价-2*ATR, 止盈=价+4*ATR (盈亏比1:2)。

【重要-移植修正】原回测脚本的枢轴低点用 right=5 的【未来数据】确认却标在枢轴当天买入
  (look-ahead 乐观偏差)。本扫描版改为【已确认枢轴(右边数据齐全, 无未来数据)+新鲜度窗口
  DIV_FRESH_BARS】: 最近窗口内确认的底背离才算当前有效, 比原脚本更诚实。
【纯多头】原策略仅 Buy_Signal 开多、无开空逻辑, 故本扫描只输出做多机会, 无做空转警示。
【指标手写】SMA/VWMA/BB/RSI(Wilder)/MACD/OBV/ATR 全部手写, 不依赖 pandas_ta; 不画图, 无 matplotlib。
【本版修复】_fetch_hist 加 baostock 长连接 Broken pipe 自愈: 查询失败(连接被服务端断开)时
  强制重登一次再试, 不再直接放弃, 把跑久后漏扫的票补回(提升覆盖率)。
【本版规格】双源baostock+东财+硬超时; 全市场多进程(每子进程独立登录baostock, 命门已修);
  宽松快照预筛; baostock行业本地join+聚类+东财风口🎯; 推送全发分页(严格检查返回);
  存output/+收尾防护+sys.exit(0); 不拦交易日(周末用上一交易日数据)。
⚠️ 三重共振+四项背离评分, 信号稀少, 全市场0命中属正常, 非bug。
⚠️ 底背离=左侧反转提示, 非买入保证; 务必按建议位止损。
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


# ==================== 参数配置 (对应原策略) ====================
PARAMS = dict(
    # 均线/通道
    MA_FAST=20, MA_MID=50, MA_SLOW=200, VWMA_LENGTH=20, BB_LENGTH=20, BB_STD=2.0,
    # 动量/量能
    RSI_LENGTH=14, MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, OBV_MA=20, ATR_LENGTH=14,
    # 背离
    PIVOT_LEFT=5, PIVOT_RIGHT=5, MIN_PIVOT_DIST=8, RSI_DIV_MAX=45,
    STRICT_MODE=False,                 # True=四项全满足; False=加权评分(推荐)
    DIVERGENCE_SCORE_THRESHOLD=3,      # 非严格模式: 四项至少满足几项
    DIV_FRESH_BARS=10,                 # 已确认底背离的新鲜度窗口(修正未来函数)
    # 斐波那契
    FIB_LOOKBACK=80, FIB_PROXIMITY_ATR_MULT=1.0,
    # 风险(初始建议位)
    STOP_LOSS_ATR_MULT=2.0, TAKE_PROFIT_ATR_MULT=4.0,
    # 数据
    lookback_days=600, min_data_len=220,   # MA200需≥200根+预热
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0}


# ------------------ 推送 (全发分页 + 严格检查返回) ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        if isinstance(ret, dict):
            ok = ret.get('code', ret.get('errno', -1)) == 0
        elif isinstance(ret, bool):
            ok = ret
        else:
            ok = ret not in (None, False)
        if ok:
            return True
        print(f"  sdk返回非成功({ret}), 回退requests")
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=15)
        j = r.json()
        if j.get('code') != 0:
            print(f"  requests返回非0: {j} (多为额度/限流/key问题)")
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
    if not chunks:
        chunks = [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        print(f"  推送第{i+1}/{len(chunks)}条 ({len(ch)}字符)")
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    if len(chunks) > 1:
        print(f"📲 共发送{len(chunks)}条(全发分页) {'✅全部成功' if ok else '⚠️存在失败(查额度/限流)'}")
    else:
        print("📲 推送成功 ✅" if ok else "⚠️ 推送返回失败(查Server酱额度/限流/微信端)")
    return ok


# ------------------ baostock 登录(命门已修) / 超时 ------------------
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
    time.sleep(random.uniform(0, 2))
    _BS_LOGGED = False
    _bs_login_ok()


def _bs_q(code, fields, sd, ed, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed,
                                            frequency="d", adjustflag="2").get_data()
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


# ==================== 指标 (全部手写, 无 pandas_ta) ====================
def _sma(s, n):
    return s.rolling(n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n):
    d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean(); al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-9))


def _atr(df, n):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _vwma(c, v, n):
    cv = (c * v).rolling(n).sum(); vs = v.rolling(n).sum()
    return cv / vs.replace(0, 1e-9)


def _obv(c, v):
    direction = np.sign(c.diff()).fillna(0)
    return (direction * v).cumsum()


# ==================== 枢轴低点 + 底背离 (numpy, 无未来数据) ====================
def _is_pivot_low(arr, left, right):
    """已确认枢轴低点: arr[i] 是 [i-left, i+right] 窗口最小值 (i+right<=n-1, 无未来数据)"""
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        w = arr[i - left:i + right + 1]
        if np.isnan(w).any():
            continue
        if arr[i] <= w.min():
            out[i] = True
    return out


def _last_div_idx(price_arr, ind_arr, left, right, min_dist, max_indicator):
    """返回最后一个触发底背离的枢轴位置(无未来数据); 无则 -1。
    底背离=价格枢轴创新低 且 指标枢轴抬高 且 指标在zone内。"""
    pp = _is_pivot_low(price_arr, left, right)
    ip = _is_pivot_low(ind_arr, left, right)
    both_idx = np.where(pp & ip)[0]
    last_price = last_ind = last_pos = None
    last_div = -1
    for i in both_idx:
        cp = float(price_arr[i]); ci = float(ind_arr[i])
        if last_pos is not None:
            if (i - last_pos) >= min_dist:
                zone_ok = (max_indicator is None) or (ci < max_indicator)
                if (cp < last_price) and (ci > last_ind) and zone_ok:
                    last_div = int(i)
        last_price = cp; last_ind = ci; last_pos = int(i)
    return last_div


# ==================== 入场信号检测 (日线, 只看最新一根) ====================
def check_one_stock(df: pd.DataFrame):
    if df is None or len(df) < PARAMS['min_data_len']:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"

    close = df['close'].astype(float); high = df['high'].astype(float)
    low = df['low'].astype(float); volume = df['volume'].astype(float)

    ma50 = _sma(close, PARAMS['MA_MID'])
    vwma = _vwma(close, volume, PARAMS['VWMA_LENGTH'])
    bb_mid = _sma(close, PARAMS['BB_LENGTH'])
    bb_std = close.rolling(PARAMS['BB_LENGTH']).std()
    rsi = _rsi(close, PARAMS['RSI_LENGTH'])
    dif_line = _ema(close, PARAMS['MACD_FAST']) - _ema(close, PARAMS['MACD_SLOW'])
    obv = _obv(close, volume)
    atr = _atr(df, PARAMS['ATR_LENGTH'])

    # 斐波那契
    fib_h = high.rolling(PARAMS['FIB_LOOKBACK']).max()
    fib_l = low.rolling(PARAMS['FIB_LOOKBACK']).min()
    rng = fib_h - fib_l
    fib50 = fib_h - rng * 0.5; fib618 = fib_h - rng * 0.618; fib786 = fib_h - rng * 0.786

    n = len(df)
    L = n - 1
    vals = [ma50.iloc[L], vwma.iloc[L], bb_mid.iloc[L], rsi.iloc[L], atr.iloc[L],
            fib50.iloc[L], fib618.iloc[L], fib786.iloc[L]]
    if any(pd.isna(v) for v in vals):
        return None, "数据不足"

    cL = float(close.iloc[L]); aL = float(atr.iloc[L])
    if aL <= 0:
        return None, "数据不足"

    # 趋势OK (最新一根)
    trend_ok = bool((cL > float(ma50.iloc[L])) and (cL > float(vwma.iloc[L])) and (cL <= float(bb_mid.iloc[L])))

    # 斐波那契靠近 (最新一根)
    d50 = abs(cL - float(fib50.iloc[L])); d618 = abs(cL - float(fib618.iloc[L])); d786 = abs(cL - float(fib786.iloc[L]))
    fib_dist = min(d50, d618, d786)
    near_fib = bool(fib_dist <= aL * PARAMS['FIB_PROXIMITY_ATR_MULT'])
    fib_prox = round(fib_dist / aL, 2)

    # 四项底背离 (已确认枢轴 + 新鲜度窗口)
    low_arr = low.to_numpy(); rsi_arr = rsi.to_numpy(); dif_arr = dif_line.to_numpy()
    obv_arr = obv.to_numpy(); vwma_arr = vwma.to_numpy()
    pl, pr, md = PARAMS['PIVOT_LEFT'], PARAMS['PIVOT_RIGHT'], PARAMS['MIN_PIVOT_DIST']
    fw = PARAMS['DIV_FRESH_BARS']
    def _fresh(ind_arr, max_ind):
        idx = _last_div_idx(low_arr, ind_arr, pl, pr, md, max_ind)
        return (idx >= 0 and (L - idx) <= fw), idx
    rsi_div, _ = _fresh(rsi_arr, PARAMS['RSI_DIV_MAX'])
    macd_div, _ = _fresh(dif_arr, 0.0)
    obv_div, _ = _fresh(obv_arr, None)
    vwma_div, _ = _fresh(vwma_arr, None)
    div_score = int(rsi_div) + int(macd_div) + int(obv_div) + int(vwma_div)
    if PARAMS['STRICT_MODE']:
        div_ok = bool(rsi_div and macd_div and obv_div and vwma_div)
    else:
        div_ok = div_score >= PARAMS['DIVERGENCE_SCORE_THRESHOLD']

    buy = trend_ok and div_ok and near_fib
    if not buy:
        return None, "无信号"

    entry = cL
    sl = round(entry - aL * PARAMS['STOP_LOSS_ATR_MULT'], 2)
    tp = round(entry + aL * PARAMS['TAKE_PROFIT_ATR_MULT'], 2)
    sig_date = pd.to_datetime(df['date'].iloc[L]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    divs = "+".join([n for n, ok in [("RSI", rsi_div), ("MACD", macd_div), ("OBV", obv_div), ("VWMA", vwma_div)] if ok])
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(entry, 2), "信号日期": sig_date,
            "背离评分": div_score, "背离组合": divs,
            "趋势OK": bool(trend_ok), "近斐波那契": bool(near_fib), "fib_prox": fib_prox,
            "RSI": round(float(rsi.iloc[L]), 1), "ATR": round(aL, 3),
            "建议止损": sl, "建议止盈": tp,
            "score": div_score, "resonance": False, "resonance_sector": ""}, None


# ------------------ 历史双源 (需 high/low/close/volume) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    """baostock优先+东财兜底; 含 Broken pipe 自愈: 查询失败(长连接被服务端断开)时
    强制重登一次再试, 不再直接放弃, 把跑久后漏扫的票补回(提升覆盖率)。"""
    global _BS_LOGGED
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    # baostock 路径: 失败(含 Broken pipe 长连接断开)则强制重登一次再试, 自愈
    for _try in range(2):
        if not _BS_LOGGED:
            _bs_login_ok()
        if not _BS_LOGGED:
            break
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d
            break   # 登录态在但取空=非连接问题, 不重登, 直接走东财
        except Exception:
            _BS_LOGGED = False   # Broken pipe 等: 连接断了, 下轮重登重试
            continue
    return _fetch_hist_em(code, sy, ey)


def snapshot_prefilter(codes_with_prefix):
    if not PARAMS['SNAPSHOT_PRE']:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫")
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
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (宽松, 失败退化全扫)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix


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
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock国标, 本地join零接口)")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        _bs_logout()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
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
    codes = stock_df['code'].tolist()
    codes = snapshot_prefilter(codes)
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail_count = 0
    mode = "严格四项" if PARAMS['STRICT_MODE'] else f"评分≥{PARAMS['DIVERGENCE_SCORE_THRESHOLD']}/4"
    print(f"开始多重底背离扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 背离模式={mode}, 新鲜窗口={PARAMS['DIV_FRESH_BARS']}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="背离扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 背离{res['背离评分']}[{res['背离组合']}] 价={res['最新价']} RSI={res['RSI']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['score', 'fib_prox'], ascending=[False, True]).reset_index(drop=True)
    return df


# ------------------ 行业本地join + 聚类 + 风口🎯 ------------------
def enrich(df):
    targets = df.to_dict('records')
    mapped = 0
    for r in targets:
        ind = _INDUSTRY_MAP.get(r['代码'], '—')
        r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注(本地join): {mapped}/{len(targets)} 只有板块")
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"📉 底背离板块: {cluster or '无'}")
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty:
                break
        except Exception as e:
            print(f"  行业热度榜第{i+1}次失败: {e}")
        time.sleep(2 + i)
    hot = []
    if not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
        h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        hot = [(str(row['板块名称']), round(float(row['_chg']), 2)) for _, row in h.head(HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in targets:
        sec = r.get('行业', '')
        m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh):
                    m = hh; break
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 底背离遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', 'score', 'fib_prox'], ascending=[False, False, True]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**📉 多重底背离加权评分** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         "*(趋势回调×多重底背离×斐波那契支撑; 建议位=入场那刻初始位, ATR止损止盈需持仓后手动管理; 纯多头; 信号稀少属正常)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📉 **底背离板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 背离{r['背离评分']}[{r['背离组合']}] | 现价{r['最新价']} "
                f"RSI={r['RSI']} 近Fib={r['fib_prox']}ATR | 止损{r['建议止损']}/止盈{r['建议止盈']}")
    if not reso.empty:
        L.append(f"### 🎯 底背离遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    L.append(f"### 📉 全部底背离 共{len(df)}只 (按背离评分+贴斐波那契)")
    L += [line(r) for _, r in df.iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    mode = "严格四项" if PARAMS['STRICT_MODE'] else f"评分≥{PARAMS['DIVERGENCE_SCORE_THRESHOLD']}/4"
    print("=" * 70)
    print(f"📉 多重底背离加权评分 (日线扫描+初始建议位) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['lookback_days']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; 背离={mode}; 新鲜窗口={PARAMS['DIV_FRESH_BARS']}; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次扫描未发现满足 趋势回调+多重底背离+斐波那契 的信号。")
        print("三重共振+四项背离本就极严, 0命中属正常; 可调: DIVERGENCE_SCORE_THRESHOLD 降到2 / DIV_FRESH_BARS 调大")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"divergence_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"divergence_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"STRICT_MODE": PARAMS['STRICT_MODE'],
                                               "DIVERGENCE_SCORE_THRESHOLD": PARAMS['DIVERGENCE_SCORE_THRESHOLD'],
                                               "DIV_FRESH_BARS": PARAMS['DIV_FRESH_BARS']},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/divergence_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'fib_prox', 'ATR', '趋势OK', '近斐波那契'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📉 多重底背离 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_divergence<<<
