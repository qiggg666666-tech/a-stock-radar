# -*- coding: utf-8 -*-
"""
multi_tf_glue_screener.py —— 多周期 MACD粘合+均线粘合 蓄势选股 全市场 · 矩阵规格
含日线(D); 复合共振(季年双粘合/日周双粘合/多头+日粘合)既加分又作入选门槛(REQUIRE_RESONANCE)。
粘合=动能收敛蓄势, 变盘方向未定, 非买点; 年线粘合偏参考; 务必方向确认+止损。
"""
import os, re, sys, json, time, random, warnings, traceback, requests
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
FILTER_MODE = os.environ.get('FILTER_MODE', 'any').strip().lower()
REQUIRE_RESONANCE = os.environ.get('REQUIRE_RESONANCE', '1').strip() in ('1', 'true', 'True')  # 复合共振作入选门槛(0=只加分)
MACD_THRESHOLD = float(os.environ.get('MACD_THRESHOLD', '0.002'))
MACD_LOOKBACK = int(os.environ.get('MACD_LOOKBACK', '1'))
MA_THRESHOLD = float(os.environ.get('MA_THRESHOLD', '0.03'))
PERIOD_MA_THRESHOLD = float(os.environ.get('PERIOD_MA_THRESHOLD', '0.03'))
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS = [5, 20, 60, 120, 250]
PERIOD_MA_WINDOWS = [5, 10, 20]
USE_EMA = False
MIN_BARS = {"W": 60, "M": 35, "Q": 20, "Y": 5}
START_DATE = os.environ.get('START_DATE', '2015-01-01')
PARAMS = dict(
    MIN_DATA_LEN=int(os.environ.get('MIN_DATA_LEN', '250')),
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无粘合": 0, "无复合共振": 0}
_PERIOD_CN = {"D": "日", "W": "周", "M": "月", "Q": "季", "Y": "年"}

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

def _init_worker():
    global _BS_LOGGED
    time.sleep(random.uniform(0, 2)); _BS_LOGGED = False
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
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

# ==================== 指标 ====================
def calc_macd(close, fast=12, slow=26, signal=9):
    e1 = close.ewm(span=fast, adjust=False).mean(); e2 = close.ewm(span=slow, adjust=False).mean()
    dif = e1 - e2; dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2

def is_macd_glued(dif, dea, close, threshold=0.002, lookback=1):
    if len(dif) < lookback:
        return False
    spread = (dif.iloc[-lookback:] - dea.iloc[-lookback:]).abs()
    return bool((spread / close.iloc[-lookback:] < threshold).all())

def calc_ma(close, windows, use_ema=False):
    df = pd.DataFrame(index=close.index)
    for w in windows:
        df[f"MA{w}"] = close.ewm(span=w, adjust=False).mean() if use_ema else close.rolling(window=w, min_periods=w).mean()
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
    ohlc = {k: v for k, v in {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}.items() if k in df.columns}
    if not ohlc:
        return pd.DataFrame()
    try:
        return df.resample(rule).agg(ohlc).dropna(subset=["close"])
    except Exception:
        return pd.DataFrame()

# ------------------ 数据双源 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust=ADJUST, timeout=AK_TIMEOUT)
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
    sd = START_DATE; ed = datetime.now().strftime('%Y-%m-%d')
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

# ==================== 策略内核 ====================
def _pass_filter(macd_glued):
    if FILTER_MODE == 'any':
        return any(macd_glued.values())
    if FILTER_MODE == 'all':
        return all(macd_glued.values())
    if FILTER_MODE == 'major':
        return bool(macd_glued['M'] and macd_glued['Q'] and macd_glued['Y'])
    if FILTER_MODE in ('d', 'w', 'm', 'q', 'y'):
        return bool(macd_glued[FILTER_MODE.upper()])
    return any(macd_glued.values())

def _resonance_tags(macd_glued, daily_macd_glued, arrangement, daily_ma_glued):
    """复合共振标签: 季年双粘合 / 日周双粘合 / 多头+日粘合"""
    tags = []
    if macd_glued['Q'] and macd_glued['Y']:
        tags.append("季年双粘合")
    if daily_macd_glued and macd_glued['W']:
        tags.append("日周双粘合")
    if arrangement == "多头排列" and daily_ma_glued:
        tags.append("多头+日粘合")
    return tags

def check_one_stock(df):
    if df is None or len(df) < PARAMS['MIN_DATA_LEN']:
        return None, "数据不足"
    if 'close' not in df.columns:
        return None, "数据不足"
    close = df['close'].astype(float)

    daily_ma = calc_ma(close, MA_WINDOWS, USE_EMA)
    daily_ma_glued = is_ma_glued(daily_ma, close, MA_THRESHOLD)
    arrangement = ma_arrangement(daily_ma)
    dif_d, dea_d, _ = calc_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    daily_macd_glued = is_macd_glued(dif_d, dea_d, close, MACD_THRESHOLD, MACD_LOOKBACK)

    d = df.copy(); d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date']).set_index('date').sort_index()
    weekly = resample_ohlc(d, 'W-FRI'); monthly = resample_ohlc(d, 'ME')
    quarterly = resample_ohlc(d, 'QE'); yearly = resample_ohlc(d, 'YE')

    macd_glued = {"D": bool(daily_macd_glued), "W": False, "M": False, "Q": False, "Y": False}
    period_ma_glued = {"D": bool(daily_ma_glued), "W": False, "M": False, "Q": False, "Y": False}
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

    # 【新增】复合共振参与"选不选": 至少满足一条才入选(REQUIRE_RESONANCE=1时)
    reso_tags = _resonance_tags(macd_glued, daily_macd_glued, arrangement, daily_ma_glued)
    if REQUIRE_RESONANCE and not reso_tags:
        return None, "无复合共振"

    # 评分 (基础 + 单周期 + 复合共振加分)
    score = 0
    if macd_glued['D']: score += 2
    if macd_glued['W']: score += 2
    if macd_glued['M']: score += 2
    if macd_glued['Q']: score += 2
    if macd_glued['Y']: score += 3
    if daily_ma_glued: score += 2
    for t in "WMQY":
        if period_ma_glued[t]: score += 1
    if macd_glued['Q'] and macd_glued['Y']: score += 5
    if daily_macd_glued and macd_glued['W']: score += 3
    if arrangement == "多头排列" and daily_ma_glued: score += 3

    glued_tags = [t for t in "DWMQY" if macd_glued[t]]
    glued_label = "+".join(_PERIOD_CN[t] for t in glued_tags)
    ma_glued_tags = [t for t in "WMQY" if period_ma_glued[t]]
    ma_glued_label = "+".join(_PERIOD_CN[t] for t in ma_glued_tags)

    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(float(close.iloc[-1]), 2),
            "信号日期": datetime.now().strftime('%Y-%m-%d'),
            "粘合周期": glued_label, "周期数": len(glued_tags),
            "均线粘合": ma_glued_label or "—", "排列": arrangement,
            "日线均线粘合": bool(daily_ma_glued), "日线MACD粘合": bool(daily_macd_glued),
            "复合共振": "+".join(reso_tags) if reso_tags else "—",
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

def run_scan():
    global _INDUSTRY_MAP, FAIL_STATS
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
        global _BS_LOGGED
        _BS_LOGGED = False
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        stock_df = _fetch_list_akshare()
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
    print(f"开始多周期粘合扫描 {len(tasks)} 只（模式{FILTER_MODE}, 复合共振门槛={'开' if REQUIRE_RESONANCE else '关'}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="粘合扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 粘合[{res['粘合周期']}] 共振[{res['复合共振']}] 分{res['score']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['score', '周期数'], ascending=[False, False]).reset_index(drop=True)
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
    df2 = df2.sort_values(['resonance', 'score', '周期数'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**🌀 多周期粘合蓄势(含日线+复合共振门槛)** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         f"*(粘合=动能收敛蓄势, 变盘方向未定非买点; 模式{FILTER_MODE}; 复合共振门槛={'开' if REQUIRE_RESONANCE else '关'}; 需方向确认+止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🌀 **粘合板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        ma_note = f" 均线粘合[{r['均线粘合']}]" if r.get('均线粘合') not in (None, '', '—') else ""
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 粘合[{r['粘合周期']}] 共振[{r['复合共振']}] {r['排列']}"
                f"{ma_note} 分{r['score']} 价{r['最新价']}")
    if not reso.empty:
        L.append(f"### 🎯 粘合遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    L.append(f"### 🌀 粘合蓄势 共{len(df)}只 (按评分)")
    L += [line(r) for _, r in df.head(PUSH_TOP * 2).iterrows()]
    if len(df) > PUSH_TOP * 2:
        L.append(f"\n*…另有{len(df)-PUSH_TOP*2}只, 见output*")
    return "\n".join(L)

def main():
    print("=" * 70)
    print(f"🌀 多周期粘合蓄势(含日线+复合共振门槛) | {datetime.now():%Y-%m-%d %H:%M} | 起始{START_DATE} 复权{ADJUST}")
    print(f"模式{FILTER_MODE}; 复合共振门槛={'开' if REQUIRE_RESONANCE else '关'}; 粘合=蓄势非买点")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次未发现满足 多周期粘合+复合共振 的股票。")
        print("可调: REQUIRE_RESONANCE=0(退回只加分不卡门槛) / FILTER_MODE=any / MACD_THRESHOLD调大")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"multi_tf_glue_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"multi_tf_glue_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"FILTER_MODE": FILTER_MODE, "REQUIRE_RESONANCE": REQUIRE_RESONANCE,
                       "MACD_THRESHOLD": MACD_THRESHOLD, "MA_THRESHOLD": MA_THRESHOLD, "START_DATE": START_DATE, "ADJUST": ADJUST},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/multi_tf_glue_{tag}.*")
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
            send_serverchan(f"🌀 多周期粘合(含日线+共振门槛) 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_multi_tf_glue<<<
