import os
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
import numpy as np
from scipy.stats import linregress
import akshare as ak
import baostock as bs
from tqdm import tqdm

# ------------------ 参数 ------------------
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

VOLUME_SHRINK_THRESHOLD = float(os.environ.get('VOLUME_SHRINK_THRESHOLD', '0.65'))
VOLUME_CHECK_WEIGHT = float(os.environ.get('VOLUME_CHECK_WEIGHT', '0.6'))

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
    """子进程各自登录 baostock (baostock 非进程安全)"""
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
    import re
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")


# ------------------ 量价指标函数 (内核, 一字未动) ------------------
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

def detect_divergence(recent, indicator_col='obv'):
    if len(recent) < 8:
        return 0.0, False
    x = np.arange(len(recent))
    price_slope = linregress(x, recent['close']).slope
    ind_slope = linregress(x, recent[indicator_col]).slope
    score = 0.0
    is_bullish = False
    if price_slope <= 0.0008 and ind_slope > 0:
        score = ind_slope / (abs(price_slope) + 1e-6)
        is_bullish = True
    elif abs(price_slope) < 0.001 and ind_slope > 0:
        score = ind_slope * 800
        is_bullish = True
    return round(score, 3), is_bullish


# ------------------ 核心检测 (内核, 一字未动) ------------------
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
        rebound = recent['close'].iloc[-1] >= recent['MA5'].iloc[-1] * 0.98

        if not (ma5_slope < MA5_SLOPE_MAX and price_range < PRICE_RANGE_MAX and rebound):
            continue

        # 量缩
        vols = recent['volume'].values
        if vols.mean() < 500000:
            continue
        x = np.arange(len(vols))
        vol_slope = linregress(x, vols).slope
        third = max(window // 3, 1)
        vol_ratio = vols[-third:].mean() / (vols[:third].mean() + 1e-8)
        shrink_score = min(1.0, (1 - vol_ratio) * 1.5 + max(0, -vol_slope * 5000))

        # 量价背离
        obv_df = calculate_obv(recent)
        cmf_df = calculate_cmf(recent)
        vpt_df = calculate_vpt(recent)
        obv_score, _ = detect_divergence(obv_df, 'obv')
        cmf_score, _ = detect_divergence(cmf_df, 'cmf')
        vpt_score, _ = detect_divergence(vpt_df, 'vpt')
        total_div = (obv_score * 0.5 + cmf_score * 0.3 + vpt_score * 0.2)

        combined = shrink_score * 0.55 + total_div * 0.45

        if combined >= VOLUME_SHRINK_THRESHOLD:
            score = window * (1 - ma5_slope) * (1 - price_range) * (1 + combined * VOLUME_CHECK_WEIGHT)
            if score > best_score:
                best_score = score
                best_window = window
                best_metrics = {
                    'shrink_score': round(shrink_score, 3),
                    'div_score': round(total_div, 3),
                    'obv_score': obv_score,
                    'vol_ratio': round(vol_ratio, 3)
                }

    return best_window >= MIN_WINDOW, best_window, df['dev'].iloc[-1], best_metrics


# ------------------ 数据获取 (双源: baostock优先 + 东财兜底, 均带超时) ------------------
def _fetch_hist(code):
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace('-', '')
    # 路径1: baostock (子进程已登录)
    if _BS_LOGGED:
        try:
            d = _bs_q(code, "date,open,high,low,close,volume", start_dash)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 40:
                    return d
        except Exception:
            pass
    # 路径2: 东财兜底 (超时+重试)
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 40:
                    return d
        except Exception:
            time.sleep(1 + attempt)
    return None


def _process_one(args):
    code, name = args
    df = _fetch_hist(code)
    if df is None or len(df) < 40 or df['close'].iloc[-1] < MIN_PRICE:
        return None

    is_match, days, dev, metrics = detect_ma5_sideways(df)
    if not is_match:
        return None

    time.sleep(SLEEP_PER_STOCK)
    shrink = metrics.get('shrink_score', 0)
    div = metrics.get('div_score', 0)
    combined = round(shrink * 0.55 + div * 0.45, 3)   # 复用内核综合公式
    grade = "🟢强蓄势" if combined >= 0.8 else "🟡蓄势"   # 启发式分级(阈值可调)
    return {
        "代码": code, "名称": name, "行业": "",
        "最新价": round(float(df['close'].iloc[-1]), 2),
        "横盘天数": days,
        "当前偏差%": round(float(dev) * 100, 2),
        "量缩显著性": shrink,
        "量价背离分": div,
        "OBV背离": metrics.get('obv_score', 0),
        "量缩比例": metrics.get('vol_ratio', 1.0),
        "蓄势强度": combined,
        "分级": grade,
    }


# ------------------ 主扫描 (baostock列表优先 + akshare兜底; 修原 \~ 语法错) ------------------
def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业表 + 列表 + 子进程登录）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                for _, r in ind.iterrows():
                    _INDUSTRY_MAP[r['code']] = _clean(r.get('industry', ''))
                print(f"  行业表 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception:
            stock_df = pd.DataFrame()
        bs.logout()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        try:
            d = ak.stock_info_a_code_name()
            d['code'] = d['code'].astype(str).str.zfill(6)
            d['code'] = d['code'].apply(lambda c: ('sh.' if c[0] in '69' else 'sz.') + c)
            d['type'] = '1'; d['status'] = '1'
            if 'name' in d.columns:
                d = d.rename(columns={'name': 'code_name'})
            stock_df = d
        except Exception as e:
            print(f"  取列表失败: {e}")
            return pd.DataFrame()

    # 过滤 (修原 \~ 语法错; 兼容 baostock/akshare 两种列名)
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
    print(f"开始检测 {len(tasks)} 只（{NUM_PROCESSES} 进程, 双源 baostock+东财）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                results.append(res)
                pbar.write(f"  🔥 {res['代码']} {res['名称']} {res['分级']} 横盘{res['横盘天数']}天 强度{res['蓄势强度']}")
            pbar.update(1); pbar.set_postfix(命中=len(results))
    print(f"扫描完成 命中{len(results)}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["蓄势强度", "横盘天数"], ascending=False).reset_index(drop=True)
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
    L = [f"**🔥 MA5横盘蓄势(量缩+量价背离)** | 命中{len(df)}只",
         "*(MA5走平+量缩+OBV/CMF/VPT背离=蓄势待涨; 横盘后方向未定, 突破确认再动; 非预测)*", ""]
    if cluster:
        L.append("🔥 **横盘蓄势板块**: " + "、".join(f"{n}({c})" for n, c in cluster))
        L.append("")
    L.append(f"### 🔥 蓄势命中 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        L.append(f"- {r['分级']} **{r['名称']}({r['代码']})** [{r['行业']}] 现价{r['最新价']} 横盘{r['横盘天数']}天 "
                 f"偏差{r['当前偏差%']}% | 量缩{r['量缩显著性']} 背离{r['量价背离分']} 强度{r['蓄势强度']}")
    if len(df) > P:
        L.append(f"\n*…另有{len(df)-P}只, 见output*")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"🔥 MA5横盘蓄势(量缩+量价背离) | {datetime.now():%Y-%m-%d %H:%M} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 进程{NUM_PROCESSES}; 双源baostock+东财")
    print("=" * 70)
    if not is_trading_day():
        print("非交易日, 跳过"); sys.exit(0)
    df = run_scan()
    if df is None or df.empty:
        print("本次未找到符合条件的股票"); sys.exit(0)
    # ---- 收尾全部包防护: 扫描已成功, 任何收尾IO/推送异常都不应让job误红 ----
    df, cluster = enrich(df)
    df = df.sort_values(["蓄势强度", "横盘天数"], ascending=False).reset_index(drop=True)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(f"{OUTPUT_DIR}/ma5_volume_div_{tag}.csv", index=False, encoding='utf-8-sig')
        with open(f"{OUTPUT_DIR}/ma5_volume_div_{tag}.json", 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "cluster": cluster, "n": int(len(df)),
                       "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/ma5_volume_div_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', disp['行业'])
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            send_serverchan(f"🔥 MA5横盘蓄势 命中{len(df)}只 🔥板块{len(cluster)}", build_push(df, cluster))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)   # 扫描已成功, 显式成功退出
