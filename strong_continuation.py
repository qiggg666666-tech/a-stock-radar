import os
import sys
import time
import random
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
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
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
LOOKBACK_DAYS = 550
QUERY_TIMEOUT_SEC = 15
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))        # 0=全扫
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
LABEL_TOP = int(os.environ.get('LABEL_TOP', '200'))        # 补行业上限(强势命中少, 一般全补)
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))      # 强势板块聚类展示数

# 强势+风口主线确认(记号✅, 与cox/mtf的⭐、ma5的🔥区分; 强势票已领涨, 故不做"补涨"语义)
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))

NEW_HIGH_RATIO = 0.97
MOMENTUM_DAYS = 20
MOMENTUM_THRESHOLD = 0.15
MA50_SLOPE_LOOKBACK = 10
BREAK_MA20_CHECK_DAYS = 10
MIN_PRICE = 3
MAX_RSI = 80
MIN_VOLUME_RATIO = 0.8

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
    """子进程初始化: 各自登录 baostock (baostock 非线程/进程安全, 必须每进程独立登录)"""
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))


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
    """东财 K 线兜底(前复权 qfq); 子进程内调用(HTTP 无状态, 进程安全)"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
            if 'close' not in d.columns:
                return None
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            if 'volume' in d.columns:
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'close', 'volume'] if c in d.columns]
            return d[cols] if len(d) >= 260 else None
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


def _rank_by_amount(codes_with_prefix):
    """按成交额降序排 codes(解决 baostock 顺序截断漏深市); 失败返回原序"""
    try:
        d = ak.stock_zh_a_spot_em()
        if d is None or d.empty or '代码' not in d.columns:
            return codes_with_prefix
        amt = pd.to_numeric(d.get('成交额') if '成交额' in d.columns else pd.Series(), errors='coerce')
        d = d.assign(_amt=amt)
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        rank = dict(zip(d['代码'], d['_amt']))
        return sorted(codes_with_prefix, key=lambda c: -rank.get(c[3:], -1))
    except Exception as e:
        print(f"  成交额排序失败, 按原序截断: {e}")
        return codes_with_prefix


# ------------------ 行业 / 风口 / 匹配 (强势板块确认用) ------------------
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
    """展示用板块标记: 强势+风口主线确认标✅, 否则标行业名"""
    return ('✅' + r.get('hot_sector', '')) if r.get('hot_meet') else (r.get('行业') or '—')


# ------------------ 指标 / 策略 (核心, 一字未动) ------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def strategy_strong_continuation(df):
    """
    强势股延续（保守版）：
    1. 现价逼近/创近250日新高（97%以上）
    2. MA10 > MA20 > MA50 > MA200，长期趋势也确认向上（不只是中短期反弹）
    3. 近20日涨幅达标（15%以上），确认动能扎实
    4. 近10日没有跌破MA20（趋势保持时间更久）
    5. RSI < 80，排除过热风险
    6. 近5日均量不低于近20日均量的80%，排除"上涨但量能已萎缩"的情况
    """
    try:
        if len(df) < 260:
            return None

        df = df.copy()
        df['close'] = df['close'].astype(float)
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        latest = df.iloc[-1]
        if pd.isna(latest['ma200']):
            return None

        recent_high = df['close'].iloc[-250:].max()
        if latest['close'] < recent_high * NEW_HIGH_RATIO:
            return None

        if not (latest['ma10'] > latest['ma20'] > latest['ma50'] > latest['ma200']):
            return None
        ma50_now = df['ma50'].iloc[-1]
        ma50_before = df['ma50'].iloc[-1 - MA50_SLOPE_LOOKBACK]
        if ma50_now <= ma50_before:
            return None

        price_20d_ago = df['close'].iloc[-1 - MOMENTUM_DAYS]
        momentum = (latest['close'] - price_20d_ago) / price_20d_ago
        if momentum < MOMENTUM_THRESHOLD:
            return None

        recent_closes = df['close'].iloc[-BREAK_MA20_CHECK_DAYS:]
        recent_ma20 = df['ma20'].iloc[-BREAK_MA20_CHECK_DAYS:]
        if (recent_closes < recent_ma20).any():
            return None

        rsi = calculate_rsi(df['close'])
        if rsi is not None and rsi > MAX_RSI:
            return None

        if 'volume' in df.columns:
            vol_5d = df['volume'].iloc[-5:].mean()
            vol_20d = df['volume'].iloc[-20:].mean()
            if vol_20d > 0 and (vol_5d / vol_20d) < MIN_VOLUME_RATIO:
                return None

        if latest['close'] < MIN_PRICE:
            return None

        return {
            "close": latest['close'],
            "距高点比例": round(latest['close'] / recent_high * 100, 1),
            "近20日涨幅%": round(momentum * 100, 1),
            "RSI": round(rsi, 1) if rsi is not None else None
        }
    except Exception:
        return None


# ------------------ 单只处理 (K线双源) ------------------
def _process_one(args):
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录)
    try:
        df = _query_with_timeout(code, "date,close,volume", start_dash)
        if df is None or df.empty or len(df) < 260:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时的空/异常 -> 子进程内重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close,volume", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= 260:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None:
        return {"__error__": f"{code} 双源均无数据, 已跳过"}

    time.sleep(SLEEP_PER_STOCK)
    res = strategy_strong_continuation(df)
    if res:
        return {
            "代码": code, "名称": name, "行业": "",
            "最新价": round(float(res["close"]), 2),
            "距高点比例%": res["距高点比例"],
            "近20日涨幅%": res["近20日涨幅%"],
            "RSI": res["RSI"],
            "hot_meet": False, "hot_sector": "",
        }
    return None


# ------------------ 主扫描 ------------------
def run_strong_continuation_scan(limit=SCAN_LIMIT):
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

    # 3) 按成交额降序截断 (解决原 [:limit] 偏沪市漏深市; 失败退化原序)
    codes = stock_df['code'].tolist()
    if limit and len(codes) > limit:
        codes = _rank_by_amount(codes)[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    # 4) 多进程扫描 (保留原架构: 每子进程独立登录 baostock + 硬超时 + 东财兜底)
    results = []
    fail_count = 0
    print(f"开始检测 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行, K线=baostock+东财双源）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（近20日涨幅{res['近20日涨幅%']}%）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("近20日涨幅%", ascending=False).reset_index(drop=True)
    return result_df


# ------------------ 行业标注 + 强势板块聚类 + 强势+风口确认 ------------------
def enrich(results):
    """补行业(并发) -> 强势板块聚类(本地, 命中票行业分布=已兑现强势) -> 强势+风口主线确认(热度榜1次)"""
    if not results:
        return pd.DataFrame(), [], []

    # 补行业 (并发, 容错; 强势命中少, 全补; 超 LABEL_TOP 截断)
    targets = results[:LABEL_TOP]
    print(f"为 {len(targets)} 只命中标的补行业 ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    # 强势板块聚类: 命中票的行业分布 (纯本地 groupby, 零接口; 不卡阈值, 强势票每只都珍贵)
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"🏆 强势板块(领涨确认, 命中票行业分布): {cluster or '无'}")

    # 强势+风口主线确认: 领涨且行业在风口=主线双重确认 (热度榜1次)
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
    print(f"✅ 强势+风口主线确认 {meet_cnt} 只")

    # 终排序: 主线确认优先, 再按近20日涨幅
    results.sort(key=lambda r: (1 if r.get('hot_meet') else 0, r['近20日涨幅%']), reverse=True)
    return pd.DataFrame(results), cluster, hot


def build_push_content(df, cluster, hot):
    P = PUSH_TOP
    lines = []
    if hot:
        lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
        lines.append("")
    if cluster:
        lines.append("🏆 **强势板块**(领涨确认, 命中票行业分布): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    meet = df[df['hot_meet'] == True] if 'hot_meet' in df.columns else pd.DataFrame()
    if not meet.empty:
        lines.append(f"### ✅ 强势+风口主线确认 Top{min(len(meet), P)} (领涨且在风口)")
        for _, r in meet.head(P).iterrows():
            lines.append(f"- {r['名称']}（{r['代码']}）[✅{r['hot_sector']}] 现价{r['最新价']} "
                         f"| 近20日涨{r['近20日涨幅%']}% | 距高点{r['距高点比例%']}% | RSI{r['RSI']}")
        lines.append("")
    lines.append(f"### 📈 全部强势 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        lines.append(f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}] 现价{r['最新价']} "
                     f"| 近20日涨{r['近20日涨幅%']}% | 距高点{r['距高点比例%']}% | RSI{r['RSI']}")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output 报告*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"强势股延续扫描 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print("=" * 70)

    if not is_trading_day():
        print("今日非A股交易日, 跳过扫描")
        sys.exit(0)

    df = run_strong_continuation_scan(limit=SCAN_LIMIT)

    if df is not None and not df.empty:
        df, cluster, hot = enrich(df.to_dict('records'))

        csv_path = f"{OUTPUT_DIR}/strong_continuation_{datetime.now().strftime('%Y%m%d')}.csv"
        json_path = f"{OUTPUT_DIR}/strong_continuation_{datetime.now().strftime('%Y%m%d')}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"\n结果已保存: {csv_path}")

        # 控制台 (带板块标记 + 聚类 + 风口)
        disp = df.copy()
        disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业', 'hot_meet', 'hot_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))

        if SERVERCHAN_KEY:
            meet_n = int(df['hot_meet'].sum()) if 'hot_meet' in df.columns else 0
            title = f"强势延续 命中{len(df)}只 🏆板块{len(cluster)} ✅确认{meet_n}"
            content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster, hot)
            send_serverchan(title, content)
    else:
        print("本次未找到符合条件的股票")
