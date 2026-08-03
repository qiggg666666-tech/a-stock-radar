# -*- coding: utf-8 -*-
"""
multi_tf_glue_screener.py —— 多周期 MACD粘合 + 均线粘合 蓄势选股 全市场 · 矩阵规格
源自用户脚本(多周期MACD+均线粘合), 核心粘合检测逻辑忠实移植, 工程重写为矩阵规格。
核心: 分别检测 周/月/季/年 线的 MACD粘合(DIF≈DEA, |DIF-DEA|/close < 阈值, 动能收敛) +
  均线粘合(多条MA最大-最小 / close < 阈值), 粘合=多周期动能收敛蓄势, 往往预示变盘。
定位: 矩阵缺的"多周期长级别粘合蓄势"档; 与 first_red_wbottom/bottom_accumulation 的日线短期粘合不同。

【特性·必读】
 ① 粘合是"蓄势观察"非"买点确认": 粘合后变盘方向未定(可上可下), 不能单独作买入依据,
    需配合方向确认(均线/MACD转强)+止损。
 ② 年线MACD判断偏参考性: 年线MACD(12,26,9)的EMA26需约26根年K收敛, 可行窗口内年K仅数根~十余根,
    EMA难收敛, 故"年线粘合"可靠性低于周/月/季线(原脚本 min_bars Y=5 放宽的固有局限, 保留原逻辑)。
 ③ 数据够才算该周期: 某周期K线不足 min_bars 时, 该周期粘合视为False(跳过), 不排除整只 ->
    次新股仍可凭周/月线粘合入选(改进原脚本"任一周期不足即整只排除"的过严逻辑)。
【筛选模式 FILTER_MODE】any=任一周期粘合 / all=四周期全粘合 / major=月+季+年粘合 / W/M/Q/Y=单周期。
【矩阵化】去demo/去命令行参数; 全市场池+baostock东财双源+多进程+快照预筛+行业本地join+风口🎯+
  推送分页+存盘+收尾防护+append补丁+不拦交易日。粘合为当前状态(截面), 无历史信号日, 故不加对齐列。
⚠️ 粘合蓄势≠买入保证; 变盘方向未定, 务必方向确认+止损; 概率性观察信号, 非预测。
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
# 筛选模式: any / all / major / W / M / Q / Y
FILTER_MODE = os.environ.get('FILTER_MODE', 'any').strip().lower()
# 粘合阈值
MACD_THRESHOLD = float(os.environ.get('MACD_THRESHOLD', '0.002'))     # MACD粘合: |DIF-DEA|/close < 此
MACD_LOOKBACK = int(os.environ.get('MACD_LOOKBACK', '1'))             # 近N根都粘合才算
MA_THRESHOLD = float(os.environ.get('MA_THRESHOLD', '0.03'))          # 日线均线粘合: (max-min)/close < 此
PERIOD_MA_THRESHOLD = float(os.environ.get('PERIOD_MA_THRESHOLD', '0.03'))  # 周/月/季/年线均线粘合阈值
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS = [5, 20, 60, 120, 250]       # 日线均线
PERIOD_MA_WINDOWS = [5, 10, 20]          # 周/月/季/年线均线
USE_EMA = False
# 各周期最少K线数 (原脚本值; 数据不足该周期则粘合视为False)
MIN_BARS = {"W": 60, "M": 35, "Q": 20, "Y": 5}
# 数据/扫描
START_DATE = os.environ.get('START_DATE', '2015-01-01')   # 多周期长历史需较早起始
PARAMS = dict(
    MIN_DATA_LEN=int(os.environ.get('MIN_DATA_LEN', '250')),   # 日线至少250根(日线均线MA250需要)
    NUM_PROCESSES=int(os.environ.get('NUM_PROCESSES', '3')),
    SLEEP=float(os.environ.get('SLEEP', '0.1')),
    FETCH_TIMEOUT=int(os.environ.get('FETCH_TIMEOUT', '12')),
    SNAPSHOT_PRE=os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True'),
    PRE_AMOUNT_MIN=float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7')),
    PRE_TURNOVER_MIN=float(os.environ.get('PRE_TURNOVER_MIN', '0.3')),
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=float(os.environ.get('MIN_PRICE', '3.0')),
)
ADJUST = os.environ.get('ADJUST', 'qfq')
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '30'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无粘合": 0}
_PERIOD_CN = {"W": "周", "M": "月", "Q": "季", "Y": "年"}


# ------------------ 推送 (全发分页 + 严格检查返回) ------------------
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


def _bs_q(code, fields, sd, ed, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed, adjustflag="2").get_data()
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


# ==================== 指标 (忠于原脚本) ====================
def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def is_macd_glued(dif, dea, close, threshold=0.002, lookback=1):
    if len(dif) < lookback:
        return False
    spread = (dif.iloc[-lookback:] - dea.iloc[-lookback:]).abs()
    return bool((spread / close.iloc[-lookback:] < threshold).all())


def calc_ma(close, windows, use_ema=False):
    df = pd.DataFrame(index=close.index)
    for w in windows:
        if use_ema:
            df[f"MA{w}"] = close.ewm(span=w, adjust=False).mean()
        else:
            df[f"MA{w}"] = close.rolling(window=w, min_periods=w).mean()
    return df


def is_ma_glued(ma_df, close, threshold=0.03, lookback=1):
    if ma_df.empty or len(ma_df) < lookback:
        return False
    recent = ma_df.iloc[-lookback:]
    spread = (recent.max(axis=1) - recent.min(axis=1)) / close.iloc[-lookback:]
    return bool((spread < threshold).all())


def ma_arrangement(ma_df):
    if ma_df.empty:
        return "数据不足"
    last = ma_df.iloc[-1].dropna()
    if len(last) < 2:
        return "数据不足"
    vals = last.values
    if np.all(np.diff(vals) < 0):
        return "多头排列"
    if np.all(np.diff(vals) > 0):
        return "空头排列"
    if (vals.max() - vals.min()) / vals.mean() < 0.03:
        return "纠缠/粘合"
    return "交叉混乱"


def resample_ohlc(df, rule):
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ohlc = {k: v for k, v in ohlc.items() if k in df.columns}
    if not ohlc:
        return pd.DataFrame()
    try:
        return df.resample(rule).agg(ohlc).dropna(subset=["close"])
    except Exception:
        return pd.DataFrame()


# ------------------ 历史双源 (长历史, START_DATE 起) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust=ADJUST, timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = START_DATE
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d
        except Exception:
            pass
    return _fetch_hist_em(code, sy, ey)


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


def snapshot_prefilter(codes_with_prefix):
    if not PARAMS['SNAPSHOT_PRE']:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫"); return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PRE_AMOUNT_MIN if False else spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (宽松, 失败退化全扫)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes_with_prefix


# ==================== 策略内核: 多周期粘合检测 (忠于原 analyze) ====================
def _pass_filter(macd_glued):
    if FILTER_MODE == 'any':
        return any(macd_glued.values())
    if FILTER_MODE == 'all':
        return all(macd_glued.values())
    if FILTER_MODE == 'major':
        return bool(macd_glued['M'] and macd_glued['Q'] and macd_glued['Y'])
    if FILTER_MODE in ('w', 'm', 'q', 'y'):
        return bool(macd_glued[FILTER_MODE.upper()])
    return any(macd_glued.values())


def check_one_stock(df):
    """返回 (命中dict 或 None, 失败原因 或 None)。数据够的周期才算粘合, 不足则该周期False不排除整只。"""
    if df is None or len(df) < PARAMS['MIN_DATA_LEN']:
        return None, "数据不足"
    if 'close' not in df.columns:
        return None, "数据不足"
    close = df['close'].astype(float)

    # 日线均线粘合 + 排列
    daily_ma = calc_ma(close, MA_WINDOWS, USE_EMA)
    daily_ma_glued = is_ma_glued(daily_ma, close, MA_THRESHOLD)
    arrangement = ma_arrangement(daily_ma)

    # resample 需 datetime index
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date']).set_index('date').sort_index()
    weekly = resample_ohlc(d, 'W-FRI')
    monthly = resample_ohlc(d, 'ME')
    quarterly = resample_ohlc(d, 'QE')
    yearly = resample_ohlc(d, 'YE')

    macd_glued = {"W": False, "M": False, "Q": False, "Y": False}
    period_ma_glued = {"W": False, "M": False, "Q": False, "Y": False}
    for tag, pdf, min_b in [("W", weekly, MIN_BARS["W"]), ("M", monthly, MIN_BARS["M"]),
                            ("Q", quarterly, MIN_BARS["Q"]), ("Y", yearly, MIN_BARS["Y"])]:
        if pdf is not None and not pdf.empty and len(pdf) >= min_b and 'close' in pdf.columns:
            pc = pdf['close'].astype(float)
            dif, dea, _ = calc_macd(pc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            macd_glued[tag] = is_macd_glued(dif, dea, pc, MACD_THRESHOLD, MACD_LOOKBACK)
            pma = calc_ma(pc, PERIOD_MA_WINDOWS, USE_EMA)
            period_ma_glued[tag] = is_ma_glued(pma, pc, PERIOD_MA_THRESHOLD)

    if not _pass_filter(macd_glued):
        return None, "无粘合"

    # 评分 (忠于原 score 逻辑)
    score = 0
    if macd_glued['W']: score += 2
    if macd_glued['M']: score += 2
    if macd_glued['Q']: score += 2
    if macd_glued['Y']: score += 3
    if daily_ma_glued: score += 2
    if period_ma_glued['W']: score += 1
    if period_ma_glued['M']: score += 1
    if period_ma_glued['Q']: score += 1
    if period_ma_glued['Y']: score += 1

    glued_tags = [t for t in "WMQY" if macd_glued[t]]
    glued_label = "+".join(_PERIOD_CN[t] for t in glued_tags)
    ma_glued_tags = [t for t in "WMQY" if period_ma_glued[t]]
    ma_glued_label = "+".join(_PERIOD_CN[t] for t in ma_glued_tags)

    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(float(close.iloc[-1]), 2),
            "信号日期": datetime.now().strftime('%Y-%m-%d'),
            "粘合周期": glued_label, "周期数": len(glued_tags),
            "均线粘合": ma_glued_label or "—", "排列": arrangement,
            "日线均线粘合": bool(daily_ma_glued),
            "score": score, "resonance": False, "resonance_sector": ""}, None


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < PARAMS['MIN_DATA_LEN']:
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
    print(f"开始多周期粘合扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 起始{START_DATE}, 模式{FILTER_MODE}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="粘合扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 粘合[{res['粘合周期']}] {res['排列']} 分{res['score']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['score', '周期数'], ascending=[False, False]).reset_index(drop=True)
    return df


# ------------------ 行业 + 聚类 + 风口 ------------------
def enrich(df):
    targets = df.to_dict('records')
    mapped = 0
    for r in targets:
        ind = _INDUSTRY_MAP.get(r['代码'], '—'); r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注(本地join): {mapped}/{len(targets)} 只有板块")
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🌀 粘合板块: {cluster or '无'}")
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
        sec = r.get('行业', ''); m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh):
                    m = hh; break
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 粘合遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', 'score', '周期数'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**🌀 多周期MACD+均线粘合 蓄势选股** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         f"*(粘合=周/月/季/年动能收敛蓄势, 变盘方向未定非买点; 模式{FILTER_MODE}; 年线粘合偏参考; 需方向确认+止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🌀 **粘合板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        ma_note = f" 均线粘合[{r['均线粘合']}]" if r.get('均线粘合') not in (None, '', '—') else ""
        daily_note = " 日线粘合" if r.get('日线均线粘合') else ""
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 粘合[{r['粘合周期']}] {r['排列']}"
                f"{ma_note}{daily_note} 分{r['score']} 价{r['最新价']}")
    if not reso.empty:
        L.append(f"### 🎯 粘合遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    L.append(f"### 🌀 粘合蓄势 共{len(df)}只 (按评分)")
    L += [line(r) for _, r in df.head(PUSH_TOP * 2).iterrows()]
    if len(df) > PUSH_TOP * 2:
        L.append(f"\n*…另有{len(df)-PUSH_TOP*2}只, 见output*")
    return "\n".join(L)


# ------------------ 主程序 ------------------
def main():
    print("=" * 70)
    print(f"🌀 多周期MACD+均线粘合 蓄势选股 | {datetime.now():%Y-%m-%d %H:%M} | 起始{START_DATE} 复权{ADJUST}")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 多进程{PARAMS['NUM_PROCESSES']}; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; 不拦交易日; 推送全列+分页")
    print(f"模式{FILTER_MODE}; MACD粘合<{MACD_THRESHOLD}; 均线粘合<{MA_THRESHOLD}; 粘合=蓄势非买点, 变盘方向未定")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次未发现满足 多周期粘合 的股票 (粘合本就稀少, 0命中属正常)。")
        print("可调: FILTER_MODE=any(放宽) / MACD_THRESHOLD调大 / MA_THRESHOLD调大 / MACD_LOOKBACK=1")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"multi_tf_glue_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"multi_tf_glue_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"FILTER_MODE": FILTER_MODE, "MACD_THRESHOLD": MACD_THRESHOLD,
                       "MA_THRESHOLD": MA_THRESHOLD, "PERIOD_MA_THRESHOLD": PERIOD_MA_THRESHOLD,
                       "START_DATE": START_DATE, "ADJUST": ADJUST},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/multi_tf_glue_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}"); traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"🌀 多周期粘合蓄势 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_multi_tf_glue<<<
