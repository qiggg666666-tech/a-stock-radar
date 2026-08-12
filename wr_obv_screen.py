# -*- coding: utf-8 -*-
"""
wr_obv_screen.py —— Williams %R + OBV 超卖反转选股 全市场 · 矩阵规格
收盘后选股(移植自 Pine Script): W%R快线上穿慢线(交叉)或低位拐头(反转) + OBV多头 + 非涨停封板。
定位: 超卖反弹/左侧偏右信号, 与粘合蓄势/趋势启动类脚本互补。
【防接飞刀(ANTI_KNIFE, 默认开)】①企稳K线 ②非崩盘 ③非跌停; 被剔票计入"防飞刀过滤"。
【矩阵化】output/存盘+baostock东财双源+query_stock_basic股票池+快照预筛+行业本地join+风口🎯+
  推送serverchan_sdk/requests双通道分页+失败统计+收尾防护+append补丁+参数env可调。
【本版修复】run_scan() 内 _BS_LOGGED 先用后声明 global 导致的 SyntaxError(并入开头global声明)。
【矩阵接入】新增 SCAN_OFFSET 环境变量支持，配合 SCAN_LIMIT 实现分段扫描。
⚠️ 超卖反弹≠必涨, 防飞刀只降概率不消除, 务必止损+仓位控制; 概率性信号, 非预测。
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
WR_LENGTH = int(os.environ.get('WR_LENGTH', '34'))
WR_SMOOTH = int(os.environ.get('WR_SMOOTH', '5'))
WR_SLOW = int(os.environ.get('WR_SLOW', '13'))
OBV_LENGTH = int(os.environ.get('OBV_LENGTH', '13'))
REV_THRESHOLD = float(os.environ.get('REV_THRESHOLD', '-30'))
# 防接飞刀 (env 可调)
ANTI_KNIFE = os.environ.get('ANTI_KNIFE', '1').strip() in ('1', 'true', 'True')
KNIFE_REQUIRE_STABILIZE = os.environ.get('KNIFE_REQUIRE_STABILIZE', '1').strip() in ('1', 'true', 'True')
KNIFE_LOWER_SHADOW = float(os.environ.get('KNIFE_LOWER_SHADOW', '0.3'))
KNIFE_MAX_5D_DROP = float(os.environ.get('KNIFE_MAX_5D_DROP', '-0.28'))
PARAMS = dict(
    LOOKBACK_DAYS=int(os.environ.get('LOOKBACK_DAYS', '180')),
    MIN_DATA_LEN=int(os.environ.get('MIN_DATA_LEN', '60')),
    NUM_PROCESSES=int(os.environ.get('NUM_PROCESSES', '3')),
    SLEEP=float(os.environ.get('SLEEP', '0.1')),
    FETCH_TIMEOUT=int(os.environ.get('FETCH_TIMEOUT', '15')),
    SNAPSHOT_PRE=os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True'),
    PRE_AMOUNT_MIN=float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7')),
    PRE_TURNOVER_MIN=float(os.environ.get('PRE_TURNOVER_MIN', '0.3')),
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=float(os.environ.get('MIN_PRICE', '3.0')),
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SCAN_OFFSET = int(os.environ.get('SCAN_OFFSET', '0'))  # 矩阵新增: 分段扫描偏移量
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0, "防飞刀过滤": 0}

# ------------------ 推送 (双通道 + 分页) ------------------
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
    return ok

# ------------------ baostock 登录/超时 ------------------
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

# ------------------ 涨跌停幅度 (对应 Pine board_type) ------------------
def get_limit_pct(code, name=""):
    if "ST" in str(name).upper():
        return 5.0
    c = str(code).split('.')[-1].zfill(6)
    if c.startswith(("688", "300", "301")):
        return 20.0
    if str(code).startswith("bj.") or c.startswith("8"):
        return 30.0
    return 10.0

# ------------------ 指标 (忠于原 Pine 逻辑) ------------------
def wma(series, length):
    weights = pd.Series(range(1, length + 1))
    return series.rolling(length).apply(lambda x: (x * weights.values).sum() / weights.sum(), raw=True)

def compute_signals(df, limit_pct):
    if len(df) < max(WR_LENGTH + WR_SMOOTH + WR_SLOW + 5, PARAMS['MIN_DATA_LEN']):
        return None
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    wr_upper = high.rolling(WR_LENGTH).max()
    wr_lower = low.rolling(WR_LENGTH).min()
    wr_range = (wr_upper - wr_lower).replace(0, np.nan)
    wr_output = ((close - wr_upper) / wr_range * 100).fillna(0)
    wr_fast = wma(wr_output, WR_SMOOTH)
    wr_slow = wr_output.ewm(span=WR_SLOW, adjust=False).mean()
    obv_dir = np.sign(close.diff()).fillna(0)
    obv_cum = (obv_dir * volume).cumsum()
    obv_ema = obv_cum.ewm(span=OBV_LENGTH, adjust=False).mean()
    if wr_fast.isna().iloc[-3:].any() or wr_slow.isna().iloc[-2:].any():
        return None
    f_now, f_prev, f_prev2 = wr_fast.iloc[-1], wr_fast.iloc[-2], wr_fast.iloc[-3]
    s_now, s_prev = wr_slow.iloc[-1], wr_slow.iloc[-2]
    bull_cross = f_prev <= s_prev and f_now > s_now
    bull_reverse = (f_prev2 > f_prev) and (f_now > f_prev) and (f_now < REV_THRESHOLD)
    obv_trend_up = obv_cum.iloc[-1] > obv_ema.iloc[-1]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    is_limit_up = last_close >= prev_close * (1 + limit_pct / 100) * 0.9985
    if not ((bull_cross or bull_reverse) and obv_trend_up and not is_limit_up):
        return None

    # ===== 防接飞刀过滤 (ANTI_KNIFE=1 时启用) =====
    if ANTI_KNIFE:
        o_last = float(df['open'].iloc[-1])
        h_last = float(high.iloc[-1])
        l_last = float(low.iloc[-1])
        bar_range = h_last - l_last
        lower_shadow = min(o_last, last_close) - l_last
        lower_ratio = (lower_shadow / bar_range) if bar_range > 0 else 0.0
        stabilized = (last_close >= o_last) or (lower_ratio >= KNIFE_LOWER_SHADOW)
        if KNIFE_REQUIRE_STABILIZE and not stabilized:
            return {"__knife__": "无企稳K线"}
        if len(close) >= 6:
            ret_5d = last_close / float(close.iloc[-6]) - 1.0
            if ret_5d < KNIFE_MAX_5D_DROP:
                return {"__knife__": "崩盘式下跌"}
        day_return = last_close / prev_close - 1.0
        if day_return <= -(limit_pct / 100) * 0.95:
            return {"__knife__": "当日跌停"}

    signal_type = "交叉" if bull_cross else "反转"
    wr_score = min(max(-float(f_now), 0.0), 100.0) * 0.5
    score = round(wr_score + (20.0 if bull_cross else 30.0) + 15.0, 1)
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(last_close, 2), "信号日期": datetime.now().strftime('%Y-%m-%d'),
            "WR快线": round(float(f_now), 2), "WR慢线": round(float(s_now), 2),
            "OBV趋势": "多头" if obv_trend_up else "空头", "信号类型": signal_type,
            "score": score, "resonance": False, "resonance_sector": ""}

# ------------------ 数据双源 (含 open) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None

def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['LOOKBACK_DAYS'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']); d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
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

def snapshot_prefilter(codes):
    if not PARAMS['SNAPSHOT_PRE']:
        return codes
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            return codes
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes if c[3:] in keep]
        return out if out else codes
    except Exception:
        return codes

def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < PARAMS['MIN_DATA_LEN']:
            return {"__fail__": "数据不足"}
        time.sleep(PARAMS['SLEEP'])
        info = compute_signals(df, get_limit_pct(code, name))
        if info is None:
            return {"__fail__": "无信号"}
        if "__knife__" in info:
            return {"__fail__": "防飞刀过滤"}
        info["代码"] = code; info["名称"] = name
        return info
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}

def run_scan():
    global _INDUSTRY_MAP, FAIL_STATS, _BS_LOGGED
    FAIL_STATS = {k: 0 for k in FAIL_STATS}
    _bs_login_ok()
    stock_df = pd.DataFrame()
    if _BS_LOGGED:
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
                for _, row in ind.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception:
            stock_df = pd.DataFrame()
        try:
            bs.logout()
        except Exception:
            pass
        _BS_LOGGED = False
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        stock_df = _fetch_list_akshare()
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = snapshot_prefilter(stock_df['code'].tolist())
    
    # 矩阵新增: 分段扫描逻辑 (先 offset，再 limit)
    if SCAN_OFFSET and len(codes) > SCAN_OFFSET:
        codes = codes[SCAN_OFFSET:]
        print(f"  分段扫描: 跳过前 {SCAN_OFFSET} 只, 本段剩余 {len(codes)} 只")
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
        
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail_count = 0
    print(f"开始 W%R+OBV 扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 防飞刀={'开' if ANTI_KNIFE else '关'}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="W%R+OBV", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['信号类型']} WR快{res['WR快线']} 分{res['score']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df

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

def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    rev = df[df['信号类型'] == '反转'] if '信号类型' in df.columns else pd.DataFrame()
    cross = df[df['信号类型'] == '交叉'] if '信号类型' in df.columns else pd.DataFrame()
    L = [f"**📉➡️ W%R+OBV 超卖反转选股(防飞刀)** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         f"*(W%R快线上穿/低位拐头+OBV多头+非涨停+防飞刀; 超卖反弹非必涨, 必止损; 非预测)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📈 **信号板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] {r['信号类型']} 收{r['最新价']} "
                f"WR快{r['WR快线']} OBV{r['OBV趋势']} 分{r['score']}")
    if not reso.empty:
        L.append(f"### 🎯 信号遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not rev.empty:
        L.append(f"### 🔥 反转信号(低位拐头) 共{len(rev)}只")
        L += [line(r) for _, r in rev.head(PUSH_TOP).iterrows()]; L.append("")
    if not cross.empty:
        L.append(f"### ✚ 交叉信号(快上穿慢) 共{len(cross)}只")
        L += [line(r) for _, r in cross.head(PUSH_TOP).iterrows()]
    return "\n".join(L)

def main():
    print("=" * 70)
    print(f"📉➡️ W%R+OBV 超卖反转选股(防飞刀) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"WR({WR_LENGTH},{WR_SMOOTH},{WR_SLOW}) OBV{OBV_LENGTH} 反转阈值{REV_THRESHOLD}; 防飞刀={'开' if ANTI_KNIFE else '关'} | Offset={SCAN_OFFSET} Limit={SCAN_LIMIT}")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n今日无符合 W%R+OBV 信号的股票 (0命中属正常)。")
        print(f"失败统计: {FAIL_STATS}")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"wr_obv_signals_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"wr_obv_signals_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"WR_LENGTH": WR_LENGTH, "WR_SMOOTH": WR_SMOOTH,
                       "WR_SLOW": WR_SLOW, "OBV_LENGTH": OBV_LENGTH, "REV_THRESHOLD": REV_THRESHOLD,
                       "ANTI_KNIFE": ANTI_KNIFE, "KNIFE_LOWER_SHADOW": KNIFE_LOWER_SHADOW,
                       "KNIFE_MAX_5D_DROP": KNIFE_MAX_5D_DROP},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/wr_obv_signals_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📉➡️ W%R+OBV 超卖反转(防飞刀) 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_wr_obv_offset<<<
