import os
import time
import random
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ------------------ 参数 ------------------
MAX_DEV = float(os.environ.get('MAX_DEV', '0.05'))         # 相对MA5的最大偏差 5%
MIN_WINDOW = int(os.environ.get('MIN_WINDOW', '8'))        # 横盘最短天数
MAX_WINDOW = int(os.environ.get('MAX_WINDOW', '20'))       # 横盘最长天数（用于搜索最佳窗口）
MA5_SLOPE_MAX = float(os.environ.get('MA5_SLOPE_MAX', '0.008'))  # MA5走平判定：斜率均值上限
PRICE_RANGE_MAX = float(os.environ.get('PRICE_RANGE_MAX', '0.12'))  # 横盘期振幅上限
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '60'))  # 只需最近约2个月数据
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 15
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))        # 0=全扫(横盘不宜只扫活跃股)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
LABEL_TOP = int(os.environ.get('LABEL_TOP', '300'))        # 补行业上限(横盘命中一般<此值, 全补)
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))      # 蓄势板块聚类展示数

# 蓄势遇风口(横盘版"共振": 风口里的滞涨补涨候选; 记号🔥, 与cox/mtf的⭐区分)
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 交易日 ------------------
def send_serverchan(title, content, sendkey=""):
    """Server酱推送: serverchan-sdk 软导入优先, requests 兜底"""
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content)
        print("📲 serverchan-sdk 推送成功")
        return True
    except Exception as e:
        print(f"  serverchan-sdk 失败, 回退 requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=10)
        return r.json().get("code") == 0
    except Exception as e:
        print(f"  requests 推送失败: {e}")
        return False


def is_trading_day():
    try:
        d = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        print(f"  交易日历获取失败, 默认继续: {e}")
        return True


# ------------------ baostock 登录重试 ------------------
def _bs_login_ok(retries=5):
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _init_worker():
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求将走东财兜底")


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 数据兜底 ------------------
def _fetch_hist_em(sym, start_y):
    """东财 K 线兜底(前复权); 返回 date/close 或 None; 子进程内调用(HTTP 无状态, 进程安全)"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close'})
            if 'close' not in d.columns:
                return None
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            return d[['date', 'close']] if len(d) >= 30 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_list_akshare():
    """akshare 兜底取股票列表; 构造与 baostock 同结构(code 带 sh./sz. 前缀, 含 type/status)"""
    for attempt in range(3):
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                name_col = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', name_col]].copy()
                d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'
                d['status'] = '1'
                return d
        except Exception as e:
            print(f"  akshare 股票列表第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])


# ------------------ 行业热度 / 风口 / 匹配 (蓄势遇风口用) ------------------
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
    h = heat.copy()
    h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
    h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
    return [(str(row['板块名称']), round(float(row['_chg']), 2))
            for _, row in h.head(HOT_SECTOR_TOP).iterrows()]


def match_sector(sector, hot_names):
    if not sector or sector in ('—', '未知') or not hot_names:
        return ""
    s = sector.strip()
    for h in hot_names:
        if h and h == s:
            return h
    for h in hot_names:
        if h and (h in s or s in h):
            return h
    return ""


def fetch_industry(symbol):
    """东财个股所属行业; 失败返回 —"""
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


def sec_tag(r):
    """展示用板块标记: 蓄势遇风口标🔥, 否则标行业名"""
    return ('🔥' + r.get('hot_sector', '')) if r.get('hot_meet') else (r.get('行业') or '—')


# ------------------ 策略内核 (一字未动) ------------------
def detect_ma5_sideways(df, max_dev=MAX_DEV, min_window=MIN_WINDOW, max_window=MAX_WINDOW):
    """
    检测：股价在MA5附近±max_dev内横盘 min_window-max_window 天。
    用 pandas.rolling 代替 talib.SMA，避免在 GitHub Actions 上编译 talib 的麻烦。
    返回: (是否符合, 最佳横盘天数, 当前偏差)
    """
    if len(df) < 30:
        return False, 0, 0.0

    df = df.copy()
    df['MA5'] = df['close'].rolling(5).mean()
    df['dev'] = (df['close'] - df['MA5']).abs() / df['MA5']

    best_window = 0
    best_score = 0

    for window in range(min_window, max_window + 1):
        recent = df.iloc[-window:]

        if not (recent['dev'] <= max_dev).all():
            continue

        ma5_slope = recent['MA5'].pct_change().abs().mean()
        slope_ok = ma5_slope < MA5_SLOPE_MAX

        price_range = (recent['close'].max() - recent['close'].min()) / recent['close'].mean()
        range_ok = price_range < PRICE_RANGE_MAX

        rebound = recent['close'].iloc[-1] >= recent['MA5'].iloc[-1] * 0.98

        if slope_ok and range_ok and rebound:
            score = window * (1 - ma5_slope) * (1 - price_range)
            if score > best_score:
                best_score = score
                best_window = window

    current_dev = df['dev'].iloc[-1]
    is_match = best_window >= min_window

    return is_match, best_window, current_dev


# ------------------ 单只处理 (K线双源) ------------------
def _process_one(args):
    """单只股票的抓取+判断逻辑，运行在子进程里"""
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录)
    try:
        df = _query_with_timeout(code, "date,close", start_dash)
        if df is None or df.empty or len(df) < 30:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时空/异常 -> 子进程内重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= 30:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None:
        return {"__error__": f"{code} 双源均无数据, 已跳过"}

    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        if df.empty or df['close'].iloc[-1] < MIN_PRICE:
            return None
        time.sleep(SLEEP_PER_STOCK)

        is_match, days, dev = detect_ma5_sideways(df)
        if not is_match:
            return None

        return {
            "代码": code, "名称": name, "行业": "",
            "最新价": round(float(df['close'].iloc[-1]), 2),
            "横盘天数": days,
            "当前偏差%": round(float(dev) * 100, 2),
            "hot_meet": False, "hot_sector": "",
        }
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_ma5_sideways_scan(limit=SCAN_LIMIT):
    # 1) 取股票列表: baostock 优先, akshare 兜底 (修复 login failed -> KeyError 崩溃)
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        bs.logout()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切换 akshare 兜底取列表 ...")
        stock_df = _fetch_list_akshare()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("⚠️ 双源均无法获取股票列表, 本次跳过")
        return pd.DataFrame()

    # 2) 过滤: 沪深股票 + 正常上市 + 剔 ST/退
    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    if stock_df.empty:
        print("⚠️ 过滤后无股票, 本次跳过")
        return pd.DataFrame()

    codes = stock_df['code'].tolist()
    if limit and len(codes) > limit:   # 横盘默认全扫(limit=0); 仅显式设限才截断
        codes = codes[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    # 3) 多进程扫描 (保留原架构: 每子进程独立登录 baostock + 硬超时 + 东财兜底)
    results = []
    fail_count = 0
    print(f"开始MA5横盘扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行, K线=baostock+东财双源）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（横盘{res['横盘天数']}天）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("横盘天数", ascending=False).reset_index(drop=True)
    return result_df


# ------------------ 行业标注 + 蓄势聚类 + 蓄势遇风口 ------------------
def enrich(results):
    """补行业(并发) -> 蓄势板块聚类(本地) -> 蓄势遇风口标记(热度榜1次)"""
    if not results:
        return pd.DataFrame(), []

    # 补行业 (并发, 容错; 横盘命中一般不多, 全补; 超 LABEL_TOP 截断)
    targets = results[:LABEL_TOP]
    print(f"为 {len(targets)} 只命中标的补行业 ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    # 蓄势板块聚类: 横盘票的行业分布 (纯本地 groupby, 零接口)
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items() if cnt >= 2]
    print(f"蓄势板块聚类(横盘≥2只): {cluster or '无'}")

    # 蓄势遇风口: 风口里的滞涨横盘=补涨候选 (热度榜1次)
    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    meet_cnt = 0
    for r in results:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['hot_meet'] = True
            r['hot_sector'] = m
            meet_cnt += 1
    print(f"🔥 蓄势遇风口 {meet_cnt} 只 (风口滞涨补涨候选)")

    # 终排序: 蓄势遇风口优先, 再按横盘天数
    results.sort(key=lambda r: (1 if r.get('hot_meet') else 0, r['横盘天数']), reverse=True)
    return pd.DataFrame(results), cluster, hot


def build_push_content(df, cluster, hot):
    P = PUSH_TOP
    lines = []
    if hot:
        lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
        lines.append("")
    if cluster:
        lines.append("🧱 **蓄势板块**(横盘扎堆, 或集体启动): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    meet = df[df['hot_meet'] == True] if 'hot_meet' in df.columns else pd.DataFrame()
    if not meet.empty:
        lines.append(f"### 🔥 蓄势遇风口 Top{min(len(meet), P)} (风口滞涨补涨)")
        for _, r in meet.head(P).iterrows():
            lines.append(f"- {r['名称']}（{r['代码']}）[🔥{r['hot_sector']}] 最新价{r['最新价']} "
                         f"| 横盘{r['横盘天数']}天 | 偏差{r['当前偏差%']}%")
        lines.append("")
    lines.append(f"### 📦 全部横盘 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        lines.append(f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}] 最新价{r['最新价']} "
                     f"| 横盘{r['横盘天数']}天 | 偏差{r['当前偏差%']}%")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output 报告*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"MA5横盘筛选 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print("=" * 70)

    if not is_trading_day():
        print("今日非A股交易日, 跳过扫描")
        raise SystemExit(0)

    df = run_ma5_sideways_scan(limit=SCAN_LIMIT)

    if df is not None and not df.empty:
        df, cluster, hot = enrich(df.to_dict('records'))
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        csv_path = f"{OUTPUT_DIR}/ma5_sideways_{datetime.now().strftime('%Y%m%d')}.csv"
        json_path = f"{OUTPUT_DIR}/ma5_sideways_{datetime.now().strftime('%Y%m%d')}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"\n结果已保存: {csv_path}")

        # 控制台 (带板块标记)
        disp = df.copy()
        disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业', 'hot_meet', 'hot_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))

        if SERVERCHAN_KEY:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            meet_n = int(df['hot_meet'].sum()) if 'hot_meet' in df.columns else 0
            title = f"MA5横盘 命中{len(df)}只 🔥遇风口{meet_n}"
            content = f"扫描时间：{now}\n\n" + build_push_content(df, cluster, hot)
            send_serverchan(title, content)
    else:
        print("本次未找到符合条件的股票")
