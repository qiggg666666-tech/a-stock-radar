# -*- coding: utf-8 -*-
"""
ma5_sideways.py —— MA5横盘蓄势 + 量价背离(OBV/CMF/VPT) 选股
横盘(MA5走平+窄幅+回踩MA) + 量缩 + 量价背离(价平量/指标走强) = 蓄势待涨。
策略判定逻辑保留; 量价背离打分做 z-score 标准化(修复跨股票不可比)。

【本版修复】
 1 append 兼容补丁(核心): 原版缺此, 致 baostock 的 get_data()(行业表/列表/历史)在 pandas2.x
   全报错被静默吞, "双源"实际只剩 akshare 一条腿; akshare 列表一限流即空跑(图3行18/24)。
   补上后 baostock 列表+行业表+历史双源真正复活, 抗限流大增, 消"取行业表 append"红字。
 2 去掉交易日拦截: 原版"仅 schedule 拦", 按不限交易日诉求删除, 周末 cron 也用周五数据跑。
 3 软超时主动收工: ma5 无边扫边记抢救机制, 撞 timeout 会白干; 到点 break 保已命中正常出结果拿绿勾。
 4 akshare 列表兜底加 None/empty 检查(限流返回 None 时原代码会崩)。
【历史修复保留】z-score 标准化(跨股票可比)/双源/超时/推送/行业join/🔥板块聚类/json留痕/收尾防护。
⚠️ MA5横盘=方向未定等突破, 量缩+背离只提高向上突破概率, 非保证; 突破确认再动。
"""
import os
import re
import sys
import json
import time
import random
import traceback
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错(原版缺此, 致 baostock 双源失效)
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
from scipy.stats import linregress
import akshare as ak
import baostock as bs
from tqdm import tqdm

# ------------------ 参数 (env 可调) ------------------
MAX_DEV = float(os.environ.get('MAX_DEV', '0.05'))
MIN_WINDOW = int(os.environ.get('MIN_WINDOW', '8'))
MAX_WINDOW = int(os.environ.get('MAX_WINDOW', '20'))
MA5_SLOPE_MAX = float(os.environ.get('MA5_SLOPE_MAX', '0.008'))
PRICE_RANGE_MAX = float(os.environ.get('PRICE_RANGE_MAX', '0.12'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '60'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = float(os.environ.get('SLEEP_PER_STOCK', '0.15'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SOFT_TIMEOUT = int(os.environ.get('SOFT_TIMEOUT', '8000'))   # 【本版】扫描软超时(秒, 默认133分钟<150 timeout)

VOLUME_SHRINK_THRESHOLD = float(os.environ.get('VOLUME_SHRINK_THRESHOLD', '0.5'))
VOLUME_CHECK_WEIGHT = float(os.environ.get('VOLUME_CHECK_WEIGHT', '0.6'))
DIV_SCALE = float(os.environ.get('DIV_SCALE', '0.3'))
VOL_SLOPE_K = float(os.environ.get('VOL_SLOPE_K', '2.0'))
Z_PRICE_FLAT = float(os.environ.get('Z_PRICE_FLAT', '0.05'))

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}


# ------------------ 推送 / 交易日 / 登录 / 超时 ------------------
def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content); print("📲 推送成功"); return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(f"https://sctapi.ftqq.com/{key}.send",
                             data={"title": title, "desp": content}, timeout=10).json().get("code") == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def is_trading_day():
    """保留备用; 【本版】main 不再调用, 即不限交易日"""
    try:
        d = ak.tool_trade_date_hist_sina()
        return datetime.now().strftime('%Y-%m-%d') in set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  交易日历失败, 默认继续: {e}"); return True


def _bs_login_ok(retries=5):
    global _BS_LOGGED
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
    time.sleep(random.uniform(0, 2))
    _bs_login_ok()


def _bs_q(code, fields, sd, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


def _clean(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")


# ------------------ z-score 标准化 (跨股票可比的核心) ------------------
def _zscore(arr):
    a = np.asarray(arr, dtype=float)
    std = np.nanstd(a)
    if std == 0 or np.isnan(std):
        return np.zeros_like(a)
    return (a - np.nanmean(a)) / std


# ------------------ 量价指标 (内核, 一字未动) ------------------
def calculate_obv(df):
    df = df.copy()
    df['obv'] = (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum()
    return df

def calculate_cmf(df, period=21):
    df = df.copy()
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'].replace(0, 1e-8))
    mfv = mfm * df['volume']
    df['cmf'] = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return df

def calculate_vpt(df):
    df = df.copy()
    df['vpt'] = (df['close'].pct_change().fillna(0) * df['volume']).cumsum()
    return df


def detect_divergence(recent, indicator_col='obv', price_flat=None):
    if price_flat is None:
        price_flat = Z_PRICE_FLAT
    if len(recent) < 8:
        return 0.0, False
    x = np.arange(len(recent))
    pz = _zscore(recent['close'].values)
    iz = _zscore(recent[indicator_col].values)
    price_slope = linregress(x, pz).slope
    ind_slope = linregress(x, iz).slope
    if price_slope <= price_flat and ind_slope > 0:
        return round(float(ind_slope), 4), True
    return 0.0, False


# ------------------ 核心检测 (横盘判定保留; 量缩/背离改 z-score) ------------------
def detect_ma5_sideways(df):
    if len(df) < 50 or 'volume' not in df.columns:
        return False, 0, 0.0, {}

    df = df.copy()
    df['MA5'] = df['close'].rolling(5).mean()
    df['dev'] = (df['close'] - df['MA5']).abs() / df['MA5']
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    best_window = 0
    best_score = 0
    best_metrics = {}

    for window in range(MIN_WINDOW, MAX_WINDOW + 1):
        recent = df.iloc[-window:].reset_index(drop=True)
        if not (recent['dev'] <= MAX_DEV).all():
            continue

        ma5_slope = recent['MA5'].pct_change().abs().mean()
        price_range = (recent['close'].max() - recent['close'].min()) / recent['close'].mean()
        rebound = recent['close'].iloc[-1] >= recent['MA5'].iloc[-1] * 0.985

        if not (ma5_slope < MA5_SLOPE_MAX and price_range < PRICE_RANGE_MAX and rebound):
            continue

        vols = recent['volume'].values
        if vols.sum() <= 0:
            continue
        x = np.arange(len(vols))
        vol_slope_z = linregress(x, _zscore(vols)).slope
        third = max(window // 3, 1)
        vol_ratio = vols[-third:].mean() / (vols[:third].mean() + 1e-8)
        shrink_score = min(1.0, (1 - vol_ratio) * 1.5 + max(0.0, -vol_slope_z * VOL_SLOPE_K))

        obv_df = calculate_obv(recent)
        cmf_df = calculate_cmf(recent)
        vpt_df = calculate_vpt(recent)
        obv_score, _ = detect_divergence(obv_df, 'obv')
        cmf_score, _ = detect_divergence(cmf_df, 'cmf')
        vpt_score, _ = detect_divergence(vpt_df, 'vpt')
        total_div = (obv_score * 0.5 + cmf_score * 0.3 + vpt_score * 0.2)
        div_norm = min(1.0, total_div / DIV_SCALE)

        combined = shrink_score * 0.55 + div_norm * 0.45

        if combined >= VOLUME_SHRINK_THRESHOLD:
            score = window * (1 - ma5_slope) * (1 - price_range) * (1 + combined * VOLUME_CHECK_WEIGHT)
            if score > best_score:
                best_score = score
                best_window = window
                best_metrics = {
                    'shrink_score': round(shrink_score, 3),
                    'div_score': round(total_div, 3),
                    'div_norm': round(div_norm, 3),
                    'obv_score': obv_score,
                    'vol_ratio': round(vol_ratio, 3),
                    'combined': round(combined, 3),
                }

    return best_window >= MIN_WINDOW, best_window, df['dev'].iloc[-1], best_metrics


# ------------------ 数据获取 (双源, 含 high/low 供指标) ------------------
def _fetch_hist(code):
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(code, "date,high,low,close,volume", start_dash)
            if d is not None and not d.empty:
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 50:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=datetime.now().strftime("%Y%m%d"),
                                   adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume'})
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 50:
                    return d
        except Exception:
            time.sleep(1 + attempt)
    return None


def _process_one(args):
    code, name = args
    df = _fetch_hist(code)
    if df is None or len(df) < 50 or df['close'].iloc[-1] < MIN_PRICE:
        return None
    is_match, days, dev, metrics = detect_ma5_sideways(df)
    if not is_match:
        return None
    time.sleep(SLEEP_PER_STOCK)
    return {
        "代码": code, "名称": name, "行业": "",
        "最新价": round(float(df['close'].iloc[-1]), 2),
        "横盘天数": days,
        "当前偏差%": round(float(dev) * 100, 2),
        "量缩显著性": metrics.get('shrink_score', 0),
        "量价背离分": metrics.get('div_score', 0),
        "背离归一": metrics.get('div_norm', 0),
        "蓄势综合": metrics.get('combined', 0),
        "OBV背离": metrics.get('obv_score', 0),
        "量缩比例": metrics.get('vol_ratio', 1.0),
    }


# ------------------ 主扫描 (baostock列表优先 + akshare兜底; 软超时收工) ------------------
def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业表 + 子进程登录）...")
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                for _, r in ind.iterrows():
                    _INDUSTRY_MAP[r['code']] = _clean(r.get('industry', ''))
                print(f"  行业表 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        bs.logout()

    print("取股票列表...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception:
            stock_df = pd.DataFrame()
        bs.logout()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        try:
            d = ak.stock_info_a_code_name()
            # 【本版】None/empty 检查(限流返回 None 时原代码会崩)
            if d is None or d.empty or 'code' not in d.columns:
                print("  akshare 列表为空(限流), 本次跳过"); return pd.DataFrame()
            d['code'] = d['code'].astype(str).str.zfill(6)
            d['code'] = d['code'].apply(lambda c: ('sh.' if c[0] in '69' else 'sz.') + c)
            d['type'] = '1'; d['status'] = '1'
            if 'name' in d.columns:
                d = d.rename(columns={'name': 'code_name'})
            stock_df = d
        except Exception as e:
            print(f"  取列表失败: {e}")
            return pd.DataFrame()

    if 'type' in stock_df.columns:
        stock_df = stock_df[(stock_df['type'] == '1') & (stock_df['status'] == '1')]
    name_col = 'code_name' if 'code_name' in stock_df.columns else 'name'
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.'))]
    stock_df = stock_df[~stock_df[name_col].astype(str).str.contains('ST|退', na=False, regex=True)]
    if stock_df.empty:
        print("⚠️ 过滤后无股票"); return pd.DataFrame()

    codes = stock_df['code'].tolist()
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df[name_col]))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail = 0
    print(f"开始检测 {len(tasks)} 只（{NUM_PROCESSES} 进程, 双源 baostock+东财, 背离z-score标准化, 软超时{SOFT_TIMEOUT//60}分钟）...")
    _t0 = time.time()
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                results.append(res)
                pbar.write(f"  🔥 {res['代码']} {res['名称']} 横盘{res['横盘天数']}天 蓄势{res['蓄势综合']} 背离{res['量价背离分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results))
            # 【本版】软超时主动收工, 保已命中正常出结果(拿绿勾), 防撞 timeout 白干
            if time.time() - _t0 > SOFT_TIMEOUT:
                print(f"\n⏱️ 软超时{SOFT_TIMEOUT//60}分钟到, 主动收工(已扫{pbar.n}/{len(tasks)}), 保住已命中{len(results)}只")
                break
    print(f"扫描完成 命中{len(results)}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["量价背离分", "横盘天数"], ascending=False).reset_index(drop=True)
    return df


# ------------------ 行业join + 横盘板块聚类 ------------------
def enrich(df):
    for _, r in df.iterrows():
        df.loc[df['代码'] == r['代码'], '行业'] = _INDUSTRY_MAP.get(r['代码'], '—')
    lab = df[df['行业'].isin([x for x in df['行业'] if x not in ('—', '')])]
    cluster = [(n, int(c)) for n, c in lab['行业'].value_counts().head(CLUSTER_TOP).items()] if not lab.empty else []
    print(f"🔥 横盘蓄势板块: {cluster or '无'}")
    return df, cluster


def build_push(df, cluster):
    P = PUSH_TOP
    L = [f"**🔥 MA5横盘蓄势(量缩+量价背离·z-score标准化)** | 命中{len(df)}只",
         "*(MA5走平+量缩+OBV/CMF/VPT背离=蓄势待涨; 横盘后方向未定, 突破确认再动; 非预测)*", ""]
    if cluster:
        L.append("🔥 **横盘蓄势板块**: " + "、".join(f"{n}({c})" for n, c in cluster))
        L.append("")
    L.append(f"### 🔥 蓄势命中 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        L.append(f"- **{r['名称']}({r['代码']})** [{r['行业']}] 现价{r['最新价']} 横盘{r['横盘天数']}天 "
                 f"偏差{r['当前偏差%']}% | 量缩{r['量缩显著性']} 背离{r['量价背离分']} 蓄势{r['蓄势综合']}")
    if len(df) > P:
        L.append(f"\n*…另有{len(df)-P}只, 见output*")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"🔥 MA5横盘蓄势(量缩+量价背离) | {datetime.now():%Y-%m-%d %H:%M} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 进程{NUM_PROCESSES}; 背离z-score标准化(跨股票可比); 不拦交易日; 软超时{SOFT_TIMEOUT//60}分钟")
    print("=" * 70)
    # 【本版】不再拦交易日(周末用周五数据复盘横盘蓄势票)
    df = run_scan()
    if df is None or df.empty:
        print("本次未找到符合条件的股票(横盘+量缩+背离门槛较严, 或z-score阈值需微调)"); sys.exit(0)
    df, cluster = enrich(df)
    df = df.sort_values(["量价背离分", "横盘天数"], ascending=False).reset_index(drop=True)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"ma5_volume_div_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"ma5_volume_div_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "cluster": cluster, "n": int(len(df)),
                       "zscore_params": {"Z_PRICE_FLAT": Z_PRICE_FLAT, "DIV_SCALE": DIV_SCALE,
                                         "VOL_SLOPE_K": VOL_SLOPE_K, "VOLUME_SHRINK_THRESHOLD": VOLUME_SHRINK_THRESHOLD},
                       "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/ma5_volume_div_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, "板块", disp["行业"])
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            send_serverchan(f"🔥 MA5横盘蓄势 命中{len(df)}只 🔥板块{len(cluster)}", build_push(df, cluster))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_ma5<<<
