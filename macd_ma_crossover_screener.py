# -*- coding: utf-8 -*-
"""
macd_ma_crossover_screener.py —— MACD金叉红柱+站上年线/季线+零轴上 趋势启动确认 · 矩阵规格
入选=四条件AND: ①站上MA(年250/季60) ②红柱 ③DIF+DEA零轴上 ④DIF上穿DEA金叉。
【方案B·FRESH_ONLY】只推"最新一根K线刚成型"的信号(金叉在最新根), 不推几天前旧信号, 彻底防接盘;
  判断用"数据最新根"而非日历今天, baostock延迟一天也不误杀。FRESH_ONLY=0退回旧行为。
【实时价+复核】腾讯实时价刷新"最新价"(盘后=当日收盘), 信号日收盘存"信号价"; 实时复核剔除破位/追高。
【增强】CROSS_LOOKBACK(默认3)/MAX_RET20(默认0.4) env可调。复权默认qfq。
⚠️ 右侧趋势启动≠买入保证, 金叉后仍可能假突破, 必止损; 只推当日信号=>多数天会空, 空=无新鲜进场点。
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

# ==================== 参数 ====================
SCAN_BOTH = os.environ.get('SCAN_BOTH', '1').strip() in ('1', 'true', 'True')
MA_PERIOD = int(os.environ.get('MA_PERIOD', '250'))
REQUIRE_ZERO_AXIS = os.environ.get('REQUIRE_ZERO_AXIS', '1').strip() in ('1', 'true', 'True')
CROSS_LOOKBACK = int(os.environ.get('CROSS_LOOKBACK', '3'))
MAX_RET20 = float(os.environ.get('MAX_RET20', '0.4'))
ADJUST = os.environ.get('ADJUST', 'qfq')
FRESH_ONLY = os.environ.get('FRESH_ONLY', '1').strip() in ('1', 'true', 'True')   # 方案B: 只推最新根信号
REALTIME_RECHECK = os.environ.get('REALTIME_RECHECK', '1').strip() in ('1', 'true', 'True')
CHASE_MAX = float(os.environ.get('CHASE_MAX', '0.15'))
PARAMS = dict(
    LOOKBACK_DAYS=int(os.environ.get('LOOKBACK_DAYS', '800')),
    MIN_DATA_LEN=int(os.environ.get('MIN_DATA_LEN', '260')),
    NUM_PROCESSES=int(os.environ.get('NUM_PROCESSES', '3')),
    SLEEP=float(os.environ.get('SLEEP', '0.1')),
    FETCH_TIMEOUT=int(os.environ.get('FETCH_TIMEOUT', '12')),
    SNAPSHOT_PRE=os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True'),
    PRE_AMOUNT_MIN=float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7')),
    PRE_TURNOVER_MIN=float(os.environ.get('PRE_TURNOVER_MIN', '0.3')),
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=float(os.environ.get('MIN_PRICE', '3.0')),
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0, "追高过高": 0, "非当日金叉": 0}

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

def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

# ------------------ 历史双源 ------------------
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
            m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes_with_prefix

# ------------------ 策略内核 ------------------
def _ma_name(p):
    return "年线" if p == 250 else ("季线" if p == 60 else f"MA{p}")

def check_one_stock(df, ma_period, require_zero_axis):
    if df is None or len(df) < PARAMS['MIN_DATA_LEN']:
        return None, "数据不足"
    if 'close' not in df.columns or 'volume' not in df.columns:
        return None, "数据不足"
    close = df['close'].astype(float); volume = df['volume'].astype(float)
    n = len(close); L = n - 1
    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    if pd.isna(ma.iloc[L]):
        return None, "数据不足"
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26; dea = dif.ewm(span=9, adjust=False).mean(); bar = (dif - dea) * 2
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    cL = float(close.iloc[L])
    ret20 = (cL / float(close.iloc[L - 20]) - 1.0) if (L >= 20 and close.iloc[L - 20] > 0) else 0.0
    if ret20 > MAX_RET20:
        return None, "追高过高"
    cond1 = cL > float(ma.iloc[L])
    cond2 = float(bar.iloc[L]) > 0
    cond3 = (float(dif.iloc[L]) > 0 and float(dea.iloc[L]) > 0) if require_zero_axis else True
    cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    tail = cross.iloc[-CROSS_LOOKBACK:]
    cond4 = bool(tail.any())
    if not (cond1 and cond2 and cond3 and cond4):
        return None, "无信号"
    pos_in_tail = np.where(tail.to_numpy())[0]
    abs_pos = (n - CROSS_LOOKBACK) + int(pos_in_tail[-1])
    abs_pos = max(0, min(L, abs_pos))
    days_since = L - abs_pos
    # 【方案B】只推最新根刚成型的金叉
    if FRESH_ONLY and days_since > 0:
        return None, "非当日金叉"
    sig_date = pd.to_datetime(df['date'].iloc[abs_pos]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    sig_price = float(close.iloc[abs_pos])
    dif_pct = float(dif.iloc[L]) / cL * 100 if cL > 0 else 0
    vol_ratio = float(volume.iloc[L]) / float(vol_ma20.iloc[L]) if (pd.notna(vol_ma20.iloc[L]) and vol_ma20.iloc[L] > 0) else 0.0
    dev = (cL - float(ma.iloc[L])) / float(ma.iloc[L]) * 100 if ma.iloc[L] > 0 else 0
    score = round(25 * _clip(dif_pct / 3.0) + 25 * _clip((vol_ratio - 0.5) / 1.5)
                  + 25 * (1 - _clip(days_since / max(CROSS_LOOKBACK, 1))) + 25 * _clip(1 - dev / 15.0), 1)
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(sig_price, 2), "信号价": round(sig_price, 2), "信号日期": sig_date, "距今天数": int(days_since),
            "ma_period": ma_period, "ma_name": _ma_name(ma_period),
            "level": "A" if ma_period == 250 else "B",
            "MA值": round(float(ma.iloc[L]), 2),
            "DIF": round(float(dif.iloc[L]), 4), "DEA": round(float(dea.iloc[L]), 4),
            "MACD柱": round(float(bar.iloc[L]), 4),
            "量比": round(vol_ratio, 2), "近20日涨幅%": round(ret20 * 100, 1),
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
        if SCAN_BOTH:
            i250, r250 = check_one_stock(df, 250, REQUIRE_ZERO_AXIS)
            i60, r60 = check_one_stock(df, 60, REQUIRE_ZERO_AXIS)
            info = i250 or i60
            if info is None:
                reason = next((r for r in (r250, r60) if r not in ("无信号",)), "无信号")
                return {"__fail__": reason}
        else:
            info, reason = check_one_stock(df, MA_PERIOD, REQUIRE_ZERO_AXIS)
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
    mode = "年线+季线" if SCAN_BOTH else _ma_name(MA_PERIOD)
    print(f"开始MACD金叉+站上{mode}扫描 {len(tasks)} 只（FRESH_ONLY={'开' if FRESH_ONLY else '关'}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="金叉扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} [{res['ma_name']}{res['level']}] 金叉{res['距今天数']}天前 分{res['score']}")
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
        print("  实时价刷新: 腾讯未取到, 沿用信号日收盘")
    return df, rt

def _realtime_recheck(df):
    if not REALTIME_RECHECK or df is None or df.empty:
        return df
    keep = []
    for r in df.to_dict('records'):
        px = r.get('最新价'); ma = r.get('MA值'); sig = r.get('信号价', r.get('最新价'))
        if px is None or pd.isna(px):
            keep.append(r); continue
        if ma is not None and not pd.isna(ma) and px <= ma:
            continue
        if sig and not pd.isna(sig) and sig > 0 and (px / sig - 1) > CHASE_MAX:
            continue
        keep.append(r)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame()

def _fetch_spot_now():
    try:
        d = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if d is None or d.empty or '代码' not in d.columns:
            return {}
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        if '最新价' in d.columns:
            d['最新价'] = pd.to_numeric(d['最新价'], errors='coerce')
        return {r['代码']: float(r['最新价']) for _, r in d.iterrows() if pd.notna(r.get('最新价'))}
    except Exception:
        return {}

def _align_suffix(r, spot_now):
    sig_price = r.get('信号价', r.get('最新价')); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price):
        return ""
    head = f"🕒金叉日{sig_price}"
    if sig_date and not pd.isna(sig_date):
        sd = str(sig_date)[:10]; head += f"@{sd[-5:]}"
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return f" | {head} → 现价{now}@run({chg:+.1f}%)"
        except Exception:
            return f" | {head}"
    return f" | {head}"

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    year = df[df['ma_name'] == '年线'] if 'ma_name' in df.columns else pd.DataFrame()
    quarter = df[df['ma_name'] == '季线'] if 'ma_name' in df.columns else pd.DataFrame()
    mode = "年线+季线" if SCAN_BOTH else _ma_name(MA_PERIOD)
    L = [f"**📈 MACD金叉+站上{mode}+零轴上 趋势启动** | 命中{len(df)}只 🎯风口{len(reso)} (只推当日信号, 现价=实时价)",
         f"*(右侧趋势启动; 只推最新根金叉; 追高≤{MAX_RET20*100:.0f}%过滤; 实时复核破位/追高剔除; 必止损; 非预测)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📈 **趋势启动板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- {r['level']} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] {r['ma_name']} 现价{r['最新价']} "
                f"MA{r['ma_period']}={r['MA值']} DIF{r['DIF']} 红柱{r['MACD柱']} 量比{r['量比']} 分{r['score']}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not year.empty:
        L.append(f"### 🅰️ 年线趋势启动 共{len(year)}只")
        L += [line(r) for _, r in year.head(PUSH_TOP).iterrows()]; L.append("")
    if not quarter.empty:
        L.append(f"### 🅱️ 季线趋势启动 共{len(quarter)}只")
        L += [line(r) for _, r in quarter.head(PUSH_TOP).iterrows()]
    return "\n".join(L)

def main():
    print("=" * 70)
    mode = "年线+季线" if SCAN_BOTH else _ma_name(MA_PERIOD)
    print(f"📈 趋势启动确认 | {datetime.now():%Y-%m-%d %H:%M} | FRESH_ONLY={'开' if FRESH_ONLY else '关'} 复核={'开' if REALTIME_RECHECK else '关'}")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次无当日趋势启动信号 (只推当日, 0命中属正常)。")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    df, rt = _refresh_realtime_price(df)
    if not df.empty:
        _b = len(df); df = _realtime_recheck(df)
        print(f"  实时复核: {_b} → {len(df)}")
    tag = datetime.now().strftime("%Y%m%d")
    if df is None or df.empty:
        print("\n实时复核后无有效信号。")
        if SERVERCHAN_KEY:
            send_serverchan("📈 趋势启动 | 复核后无有效信号", "**趋势启动确认** | 当日信号经实时复核全部破位/追高, 已剔除。\n\n*(防旧信号接盘, 非故障)*")
        sys.exit(0)
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"macd_ma_crossover_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"macd_ma_crossover_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "FRESH_ONLY": FRESH_ONLY, "cluster": cluster, "n": int(len(df)),
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/macd_ma_crossover_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {e}")
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', '实时价'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = rt if rt else _fetch_spot_now()
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📈 趋势启动 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_macd_ma_cross<<<
