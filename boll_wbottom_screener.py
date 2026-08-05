# -*- coding: utf-8 -*-
"""
boll_wbottom_screener.py —— 布林带+W底 全市场选股 · 矩阵规格
形态: 突破上轨+放量 + 中轨颈线 + 两低点抬高+缩量 = W底; 只看最近信号。
【方案B·FRESH_ONLY】只推"最新一根K线刚突破"的信号(突破在数据最新根), 不推几天前旧信号, 彻底防接盘;
  用"数据最新根"判定, baostock延迟一天也不误杀。FRESH_ONLY=0退回recent_days窗口。
【实时价+复核】腾讯实时价刷新"最新价"(盘后=当日收盘), 信号日收盘存"信号价"; 实时复核剔除破颈线/追高。
⚠️ W底=突破提示非买入保证, 假突破常见, 需颈线/止损; 只推当日=>多数天会空, 空=无新鲜进场点。
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
PARAMS = dict(
    bb_period=int(os.environ.get('BB_PERIOD', '20')),
    bb_std=float(os.environ.get('BB_STD', '2.0')),
    lookback=int(os.environ.get('LOOKBACK', '75')),
    min_gap=int(os.environ.get('MIN_GAP', '8')),
    max_gap=int(os.environ.get('MAX_GAP', '45')),
    alpha=float(os.environ.get('ALPHA', '0.006')),
    volume_ma_period=int(os.environ.get('VOL_MA_PERIOD', '10')),
    volume_shrink_ratio=float(os.environ.get('VOL_SHRINK', '0.85')),
    volume_expand_ratio=float(os.environ.get('VOL_EXPAND', '1.5')),
    recent_days=int(os.environ.get('RECENT_DAYS', '10')),
    min_data_len=100, lookback_days=400,
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
FRESH_ONLY = os.environ.get('FRESH_ONLY', '1').strip() in ('1', 'true', 'True')   # 方案B: 只推最新根突破
REALTIME_RECHECK = os.environ.get('REALTIME_RECHECK', '1').strip() in ('1', 'true', 'True')
CHASE_MAX = float(os.environ.get('CHASE_MAX', '0.15'))
NECK_TOL = float(os.environ.get('NECK_TOL', '0.97'))
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无W底信号": 0, "信号过旧": 0, "非最新信号": 0}

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
        px = r.get('最新价'); mid = r.get('中轨'); sig = r.get('信号价', r.get('最新价'))
        if px is None or pd.isna(px):
            keep.append(r); continue
        if mid is not None and not pd.isna(mid) and px < float(mid) * NECK_TOL:
            continue
        if sig and not pd.isna(sig) and sig > 0 and (px / float(sig) - 1) > CHASE_MAX:
            continue
        keep.append(r)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame()

# ------------------ 布林带 ------------------
def bollinger_bands(df):
    data = df.copy()
    data['std'] = data['close'].rolling(PARAMS['bb_period'], min_periods=PARAMS['bb_period']).std()
    data['mid'] = data['close'].rolling(PARAMS['bb_period'], min_periods=PARAMS['bb_period']).mean()
    data['upper'] = data['mid'] + PARAMS['bb_std'] * data['std']
    data['lower'] = data['mid'] - PARAMS['bb_std'] * data['std']
    data['vol_ma'] = data['volume'].rolling(PARAMS['volume_ma_period'], min_periods=1).mean()
    return data

# ------------------ W底检测 ------------------
def detect_w_bottom(df):
    if len(df) < PARAMS['min_data_len']:
        return pd.DataFrame()
    data = bollinger_bands(df)
    n = len(data)
    lookback = PARAMS['lookback']; alpha = PARAMS['alpha']
    min_gap = PARAMS['min_gap']; max_gap = PARAMS['max_gap']
    shrink = PARAMS['volume_shrink_ratio']; expand = PARAMS['volume_expand_ratio']
    close = data['close'].to_numpy(float); volume = data['volume'].to_numpy(float)
    upper = data['upper'].to_numpy(float); mid = data['mid'].to_numpy(float)
    lower = data['lower'].to_numpy(float); vol_ma = data['vol_ma'].to_numpy(float)
    signal = np.zeros(n, dtype=int); coords = [''] * n
    for i in range(lookback, n):
        if close[i] > upper[i] and volume[i] >= vol_ma[i] * expand:
            found = False
            lo_i = max(i - lookback, 0)
            for j in range(i - 1, lo_i, -1):
                if abs(close[j] - mid[j]) < alpha * close[j]:
                    for k in range(j - 1, lo_i, -1):
                        if abs(close[k] - lower[k]) < alpha * close[k]:
                            threshold = close[k]
                            for m in range(i - 1, j, -1):
                                if (abs(close[m] - lower[m]) < alpha * close[m] and close[m] > lower[m] and close[m] > threshold * 0.995):
                                    gap = abs(m - k)
                                    if not (min_gap <= gap <= max_gap):
                                        continue
                                    if volume[k] <= 0 or volume[m] >= volume[k] * shrink:
                                        continue
                                    signal[i] = 1; coords[i] = f"{k},{j},{m},{i}"; found = True
                                    break
                            if found:
                                break
                if found:
                    break
    data['signal'] = signal; data['coordinates'] = coords
    return data

# ------------------ 历史双源 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily", start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                d['close'] = pd.to_numeric(d['close'], errors='coerce'); d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d[['date', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None

def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                d['close'] = pd.to_numeric(d['close'], errors='coerce'); d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
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
        if df is None or len(df) < PARAMS['min_data_len']:
            return {"__fail__": "数据不足" if df is not None else "抓取失败"}
        time.sleep(PARAMS['SLEEP'])
        signal_df = detect_w_bottom(df)
        if signal_df.empty:
            return {"__fail__": "无W底信号"}
        buy = signal_df[signal_df['signal'] == 1]
        if buy.empty:
            return {"__fail__": "无W底信号"}
        latest = buy.iloc[-1]
        signal_date = latest['date']
        days_ago = (datetime.now().date() - pd.to_datetime(signal_date).date()).days
        if days_ago > PARAMS['recent_days']:
            return {"__fail__": "信号过旧"}
        # 【方案B】只推最新根突破
        last_date = df['date'].iloc[-1]
        if FRESH_ONLY and pd.to_datetime(signal_date).date() != pd.to_datetime(last_date).date():
            return {"__fail__": "非最新信号"}
        sig_close = round(float(latest['close']), 2)
        return {"代码": code, "名称": name, "行业": "",
                "最新价": sig_close, "信号价": sig_close,
                "信号日期": pd.to_datetime(signal_date).strftime('%Y-%m-%d'),
                "距今天数": int(days_ago),
                "上轨": round(float(latest['upper']), 2), "中轨": round(float(latest['mid']), 2), "下轨": round(float(latest['lower']), 2),
                "resonance": False, "resonance_sector": ""}
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
    print(f"开始布林带+W底扫描 {len(tasks)} 只（FRESH_ONLY={'开' if FRESH_ONLY else '关'}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="w底扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 信号{res['信号日期']} 价={res['最新价']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('距今天数').reset_index(drop=True)
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
    df2 = df2.sort_values(['resonance', '距今天数'], ascending=[False, True]).reset_index(drop=True)
    return df2, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')

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

def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**📈 布林带+W底突破** | 命中{len(df)}只 🎯风口{len(reso)} (只推当日突破, 现价=实时价)",
         "*(突破上轨放量+中轨颈线+两低点抬高+缩量=W底; 只推最新根突破; 实时复核破颈线/追高剔除; 需颈线/止损; 非预测)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📈 **W底板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 现价{r['最新价']} | 上{r['上轨']}/中{r['中轨']}/下{r['下轨']}{_align_suffix(r, spot_now)}")
    if not reso.empty:
        L.append(f"### 🎯 遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    L.append(f"### 📈 全部W底 共{len(df)}只")
    L += [line(r) for _, r in df.iterrows()]
    return "\n".join(L)

def main():
    print("=" * 70)
    print(f"📈 布林带+W底 | {datetime.now():%Y-%m-%d %H:%M} | FRESH_ONLY={'开' if FRESH_ONLY else '关'} 复核={'开' if REALTIME_RECHECK else '关'} alpha={PARAMS['alpha']}")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次无当日W底突破信号 (只推当日, 0命中属正常)。")
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
            send_serverchan("📈 布林带+W底 | 复核后无有效信号", "**布林带+W底** | 当日突破经实时复核全部破颈线/追高, 已剔除。\n\n*(防旧信号接盘, 非故障)*")
        sys.exit(0)
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"boll_wbottom_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"boll_wbottom_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "FRESH_ONLY": FRESH_ONLY, "cluster": cluster, "n": int(len(df)),
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/boll_wbottom_{tag}.*")
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
            spot_now = rt
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📈 布林带+W底 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_boll_wbottom<<<
