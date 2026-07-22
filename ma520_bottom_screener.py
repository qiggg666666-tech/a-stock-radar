import os
import sys
import time
import random
import json
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


# ------------------ 策略参数 (全部 env 可调) ------------------
MA5_DEV = float(os.environ.get('MA5_DEV', '0.05'))
MIN_SIDEWAYS = int(os.environ.get('MIN_SIDEWAYS', '8'))
MAX_SIDEWAYS = int(os.environ.get('MAX_SIDEWAYS', '20'))
LOW_THRESHOLD = float(os.environ.get('LOW_THRESHOLD', '8.0'))
MA520_THRESHOLD = float(os.environ.get('MA520_THRESHOLD', '5.0'))
MIN_AVG_TURNOVER = float(os.environ.get('MIN_AVG_TURNOVER', '1.4'))  # baostock的turn本身就是百分比
MIN_VOLUME_RATIO = float(os.environ.get('MIN_VOLUME_RATIO', '1.4'))
MIN_AVG_AMPLITUDE = float(os.environ.get('MIN_AVG_AMPLITUDE', '4.5'))

# 数据长度要求：MA520需520天 + 判断MA520低位需再往前100天 + 价格低位需300天 ≈ 950+交易日
MIN_REQUIRED_ROWS = 950
LOOKBACK_DAYS = 1600  # 约4.4个日历年，确保有950+个交易日

MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '20'))  # 4年数据, 超时给宽松
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))        # 0=全扫
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
LABEL_TOP = int(os.environ.get('LABEL_TOP', '200'))        # 补行业上限(命中少, 一般全补)
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))

# 筑底遇风口(深度底部横盘+板块催化=价值挖掘提示; 记号⛏️, 与ma5的🔥区分深浅)
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 交易日 / 容错 ------------------
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
    _bs_login_ok(retries=5)


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
    """东财 K 线兜底(前复权); 关键: rename 含 '换手率'->turn, 否则兜底路径换手率恒假筛不出票"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                                  '收盘': 'close', '成交量': 'volume', '换手率': 'turn'})
            if 'close' not in d.columns:
                return None
            for c in ['open', 'high', 'low', 'close', 'volume', 'turn']:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'turn'] if c in d.columns]
            return d[cols] if len(d) >= MIN_REQUIRED_ROWS else None
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


# ------------------ 行业 / 风口 / 匹配 (筑底遇风口用) ------------------
def fetch_industry(symbol):
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
    """展示用板块标记: 筑底遇风口标⛏️, 否则标行业名"""
    return ('⛏️' + r.get('hot_sector', '')) if r.get('hot_meet') else (r.get('行业') or '—')


# ------------------ 策略内核 (一字未动) ------------------
def _safe_pct_position(current, history_series):
    """
    计算current在history_series历史区间内的位置百分比：0=历史最低，100=历史最高。
    对"历史区间内最大值=最小值"（除零）和空数据做了防护，返回None表示无法计算。
    """
    if history_series is None or len(history_series) == 0:
        return None
    hist_min = history_series.min()
    hist_max = history_series.max()
    if pd.isna(hist_min) or pd.isna(hist_max) or hist_max == hist_min:
        return None
    return (current - hist_min) / (hist_max - hist_min) * 100


def detect_full_strategy_with_vr(
    df,
    ma5_dev=MA5_DEV,
    min_sideways=MIN_SIDEWAYS,
    max_sideways=MAX_SIDEWAYS,
    low_threshold=LOW_THRESHOLD,
    ma520_threshold=MA520_THRESHOLD,
    min_avg_turnover=MIN_AVG_TURNOVER,
    min_volume_ratio=MIN_VOLUME_RATIO,
    min_avg_amplitude=MIN_AVG_AMPLITUDE,
):
    """
    MA5横盘 + 价格低位 + 520日线低位 + 换手率 + 振幅 + 量比 综合筛选。
    用 pandas.rolling 代替 talib.SMA，避免在 GitHub Actions 上编译 talib 的麻烦。
    """
    if len(df) < MIN_REQUIRED_ROWS:
        return False, {}, None

    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['close', 'high', 'low', 'volume']).reset_index(drop=True)
    if len(df) < MIN_REQUIRED_ROWS:
        return False, {}, None

    df['MA5'] = df['close'].rolling(5).mean()
    df['dev_ma5'] = (df['close'] - df['MA5']).abs() / df['MA5']
    df['amplitude'] = ((df['high'] - df['low']) / df['close']) * 100

    current_price = df['close'].iloc[-1]
    current_ma5 = df['MA5'].iloc[-1]

    # 价格/MA5 是否处于近300天(排除最近5天)的历史低位
    history = df.iloc[-300:-5]
    price_low_pct = _safe_pct_position(current_price, history['close'])
    ma5_low_pct = _safe_pct_position(current_ma5, history['MA5'])
    if price_low_pct is None or ma5_low_pct is None:
        return False, {}, current_ma5

    # MA5横盘天数
    sideways_days = 0
    for w in range(min_sideways, max_sideways + 1):
        recent = df.iloc[-w:]
        if (recent['dev_ma5'] <= ma5_dev).all():
            if recent['MA5'].pct_change().abs().mean() < 0.008 and \
               (recent['close'].max() - recent['close'].min()) / recent['close'].mean() < 0.12:
                sideways_days = w
                break

    # 520日年线是否处于历史低位
    df['MA520'] = df['close'].rolling(520).mean()
    recent_ma520 = df['MA520'].iloc[-1]
    history_ma520 = df['MA520'].dropna().iloc[:-100] if len(df['MA520'].dropna()) > 100 else pd.Series(dtype=float)
    ma520_low_pct = _safe_pct_position(recent_ma520, history_ma520)
    if ma520_low_pct is None:
        return False, {}, current_ma5

    # 换手率：baostock的turn字段已经是百分比数值(如1.23代表1.23%)，不要再乘100
    if 'turn' in df.columns:
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        avg_turnover = df['turn'].iloc[-5:].mean()
    else:
        avg_turnover = 0
    if pd.isna(avg_turnover):
        avg_turnover = 0

    avg_amplitude = df['amplitude'].iloc[-5:].mean()

    # 量比
    df['avg_vol_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['avg_vol_5']
    latest_vr = df['volume_ratio'].iloc[-1]
    if pd.isna(latest_vr):
        latest_vr = 0

    is_match = (
        price_low_pct < low_threshold and
        ma5_low_pct < low_threshold and
        sideways_days >= min_sideways and
        ma520_low_pct <= ma520_threshold and
        avg_turnover >= min_avg_turnover and
        latest_vr >= min_volume_ratio and
        avg_amplitude >= min_avg_amplitude and
        current_price >= MIN_PRICE
    )

    info = {
        'sideways_days': sideways_days,
        'price_low_%': round(price_low_pct, 2),
        'ma5_low_%': round(ma5_low_pct, 2),
        'ma520_low_%': round(ma520_low_pct, 2),
        'avg_turnover_%': round(avg_turnover, 2),
        'volume_ratio': round(latest_vr, 2),
        'avg_amplitude_%': round(avg_amplitude, 2),
    }

    return is_match, info, current_ma5


# ------------------ 单只处理 (K线双源, 含换手率) ------------------
def _process_one(args):
    """单只股票的抓取+判断逻辑，运行在子进程里"""
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录; fields 含 turn 换手率)
    try:
        df = _query_with_timeout(code, "date,open,high,low,close,volume,turn", start_dash)
        if df is None or df.empty or len(df) < MIN_REQUIRED_ROWS:
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
                df2 = _query_with_timeout(code, "date,open,high,low,close,volume,turn", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= MIN_REQUIRED_ROWS:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底 (rename 含换手率->turn, 保证换手率条件不恒假)
    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None or len(df) < MIN_REQUIRED_ROWS:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        is_match, info, ma5 = detect_full_strategy_with_vr(df)
        if not is_match:
            return None

        return {
            "代码": code, "名称": name, "行业": "",
            **info,
            "current_ma5": round(float(ma5), 2) if ma5 is not None else None,
            "hot_meet": False, "hot_sector": "",
        }
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_ma520_bottom_scan(limit=SCAN_LIMIT):
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
    if limit and len(codes) > limit:   # 默认全扫(limit=0); 仅显式设限才截断
        codes = codes[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    # 3) 多进程扫描 (保留原架构: 每子进程独立登录 baostock + 硬超时 + 东财兜底)
    results = []
    fail_count = 0
    print(f"开始520日线低位筛选 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行, K线=baostock+东财双源, 含换手率）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（量比{res['volume_ratio']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("volume_ratio", ascending=False).reset_index(drop=True)
    return result_df


# ------------------ 行业标注 + 筑底聚类 + 筑底遇风口 ------------------
def enrich(results):
    """补行业(并发) -> 筑底板块聚类(本地, 命中票行业分布=板块级深度筑底) -> 筑底遇风口(热度榜1次)"""
    if not results:
        return pd.DataFrame(), [], []

    # 补行业 (并发, 容错; 命中少, 全补; 超 LABEL_TOP 截断)
    targets = results[:LABEL_TOP]
    print(f"为 {len(targets)} 只命中标的补行业 ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    # 筑底板块聚类: 命中票的行业分布 (纯本地 groupby, 零接口; 不卡阈值, 命中每只都珍贵)
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"🪨 筑底板块(年线低位+横盘扎堆, 板块级深度筑底): {cluster or '无'}")

    # 筑底遇风口: 深度底部横盘 + 行业在风口 = 价值挖掘 (热度榜1次)
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
    print(f"⛏️ 筑底遇风口 {meet_cnt} 只 (深度底部+板块催化, 价值挖掘提示)")

    # 终排序: 遇风口优先, 再按量比(保留原排序意图)
    results.sort(key=lambda r: (1 if r.get('hot_meet') else 0, r['volume_ratio']), reverse=True)
    return pd.DataFrame(results), cluster, hot


def build_push_content(df, cluster, hot):
    P = PUSH_TOP
    lines = []
    if hot:
        lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
        lines.append("")
    if cluster:
        lines.append("🪨 **筑底板块**(年线低位+横盘扎堆, 板块级深度筑底): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    meet = df[df['hot_meet'] == True] if 'hot_meet' in df.columns else pd.DataFrame()
    if not meet.empty:
        lines.append(f"### ⛏️ 筑底遇风口 Top{min(len(meet), P)} (深度底部+板块催化, 价值挖掘)")
        for _, r in meet.head(P).iterrows():
            lines.append(f"- {r['名称']}（{r['代码']}）[⛏️{r['hot_sector']}] 横盘{r['sideways_days']}天 "
                         f"| 量比{r['volume_ratio']} | 换手{r['avg_turnover_%']}% | 振幅{r['avg_amplitude_%']}% "
                         f"| 价低{r['price_low_%']}% 年线低{r['ma520_low_%']}%")
        lines.append("")
    lines.append(f"### 📋 全部筑底 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        lines.append(f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}] 横盘{r['sideways_days']}天 "
                     f"| 量比{r['volume_ratio']} | 换手{r['avg_turnover_%']}% | 振幅{r['avg_amplitude_%']}% "
                     f"| 价低{r['price_low_%']}% 年线低{r['ma520_low_%']}%")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output 报告*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"520日线低位筛选 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print(f"K线双源(baostock+东财, 均含换手率); 条件极严, 命中可能极少属正常")
    print("=" * 70)

    if not is_trading_day():
        print("今日非A股交易日, 跳过扫描")
        sys.exit(0)

    df = run_ma520_bottom_scan(limit=SCAN_LIMIT)

    if df is not None and not df.empty:
        df, cluster, hot = enrich(df.to_dict('records'))

        csv_path = f"{OUTPUT_DIR}/ma520_bottom_{datetime.now().strftime('%Y%m%d')}.csv"
        json_path = f"{OUTPUT_DIR}/ma520_bottom_{datetime.now().strftime('%Y%m%d')}.json"
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
            title = f"520筑底 命中{len(df)}只 🪨筑底{len(cluster)} ⛏️挖掘{meet_n}"
            content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster, hot)
            send_serverchan(title, content)
    else:
        print("本次未找到符合条件的股票 (条件极严: 年线低位+横盘+换手+振幅+量比全过, 命中0只属正常)")
