import os
import re
import sys
import time
import random
import json
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


# ------------------ 阈值参数 ------------------
LOW_WINDOW = int(os.environ.get('LOW_WINDOW', '520'))
VOL_WINDOW = int(os.environ.get('VOL_WINDOW', '20'))
NEW_LOW_TOLERANCE = float(os.environ.get('NEW_LOW_TOLERANCE', '1.02'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
QUERY_TIMEOUT_SEC = 15
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))        # 0=全扫
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '10'))     # 见底板块聚类展示数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模块级行业映射: run 阶段主进程登录时一次拿全(baostock国标), enrich 本地 join, 零逐只接口零限流
_INDUSTRY_MAP = {}


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
    """东财 K 线兜底(前复权); 返回 date/open/high/low/close/volume 或 None; 子进程内调用(进程安全)"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                  '最低': 'low', '收盘': 'close', '成交量': 'volume'})
            if 'close' not in d.columns:
                return None
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in d.columns]
            return d[cols] if len(d) >= 60 else None
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


def _clean_industry(s):
    """清洗 baostock 国标行业名: 去掉 'C39 ' 这类字母+数字前缀, 留可读行业名"""
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


# ------------------ 策略内核 (一字未动) ------------------
def detect_first_red_to_520_low(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['520_low'] = df['low'].rolling(LOW_WINDOW).min()
    df['5_low'] = df['low'].rolling(5).min()  # 近5日最低价，作为短期支撑位参考
    df['avg_vol'] = df['volume'].rolling(VOL_WINDOW).mean()
    df['is_red'] = df['close'] > df['open']
    df['made_new_low_recently'] = df['low'] <= df['520_low'].shift(1) * NEW_LOW_TOLERANCE

    signals = []
    in_low_zone = False
    for i in range(LOW_WINDOW, len(df)):
        if df['made_new_low_recently'].iloc[i]:
            in_low_zone = True
        if in_low_zone and df['is_red'].iloc[i]:
            vol_ratio = (
                df['volume'].iloc[i] / df['avg_vol'].iloc[i]
                if df['avg_vol'].iloc[i] > 0 else 0
            )
            distance_pct = (df['close'].iloc[i] - df['520_low'].iloc[i]) / df['520_low'].iloc[i] * 100

            five_day_low = df['5_low'].iloc[i]
            distance_pct_5 = (
                (df['close'].iloc[i] - five_day_low) / five_day_low * 100
                if pd.notna(five_day_low) and five_day_low > 0 else None
            )

            signals.append({
                'date': df['date'].iloc[i],
                'close': df['close'].iloc[i],
                'vol_ratio': vol_ratio,
                'distance_pct': round(distance_pct, 2),
                '5日最低价': round(float(five_day_low), 2) if pd.notna(five_day_low) else None,
                '距5日低点%': round(distance_pct_5, 2) if distance_pct_5 is not None else None,
            })
            in_low_zone = False

    return pd.DataFrame(signals)


# ------------------ 单只处理 (K线双源) ------------------
def _process_one(args):
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=int(LOW_WINDOW * 1.6))).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录)
    try:
        df = _query_with_timeout(code, "date,open,high,low,close,volume", start_dash)
        if df is None or df.empty or len(df) < LOW_WINDOW:
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
                df2 = _query_with_timeout(code, "date,open,high,low,close,volume", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= LOW_WINDOW:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None or len(df) < LOW_WINDOW:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        signals = detect_first_red_to_520_low(df)
        if signals.empty:
            return None

        latest = signals.iloc[-1]
        if (df['date'].iloc[-1] != latest['date']) and \
           (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(latest['date'])).days > 7:
            return None

        return {
            "代码": code, "名称": name, "行业": "",   # 行业在 enrich 用 _INDUSTRY_MAP 本地填, 不逐只调接口
            "信号日期": latest['date'],
            "最新价": round(float(latest['close']), 2),
            "量比": round(float(latest['vol_ratio']), 2),
            "距520日低点%": latest['distance_pct'],
            "5日最低价": latest['5日最低价'],
            "距5日低点%": latest['距5日低点%'],
        }
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_first_red_520_scan(limit=SCAN_LIMIT):
    global _INDUSTRY_MAP
    # 1) 取股票列表 + 行业映射: baostock 优先, akshare 兜底 (修复 login failed -> KeyError 崩溃)
    print("正在连接 Baostock（主进程，用于取股票列表+行业映射）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        # 顺手一次拿全市场行业映射(同一登录态, 零额外登录, 零逐只接口 -> 317只也无限流)
        try:
            ind_df = bs.query_stock_industry().get_data()
            if ind_df is not None and not ind_df.empty and 'code' in ind_df.columns and 'industry' in ind_df.columns:
                for _, row in ind_df.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock国标行业, 一次拿全, 命中再多也不限流)")
        except Exception as e:
            print(f"  baostock 取行业表异常: {e}")
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
    if limit and len(codes) > limit:   # 首红默认全扫(limit=0); 仅显式设限才截断
        codes = codes[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    # 3) 多进程扫描 (保留原架构: 每子进程独立登录 baostock + 硬超时 + 东财兜底)
    results = []
    fail_count = 0
    print(f"开始520天首红扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行, K线=baostock+东财双源）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（量比 {res['量比']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("量比", ascending=False).reset_index(drop=True)
    return result_df


# ------------------ 行业标注(本地join) + 见底聚类 ------------------
def enrich(results):
    """用 _INDUSTRY_MAP 本地填行业(零接口, 317只也瞬间完成) -> 见底板块聚类(本地 groupby)"""
    if not results:
        return pd.DataFrame(), []

    # 本地 join 行业: 不调任何接口, 不会限流, 命中再多也全覆盖
    mapped = 0
    for r in results:
        ind = _INDUSTRY_MAP.get(r['代码'], '—')
        r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注完成: {mapped}/{len(results)} 只命中行业映射 (baostock国标, 本地join)")

    # 见底板块聚类: 命中票的行业分布 (纯本地 groupby, 零接口; 不卡阈值, 首红票每只都珍贵)
    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"🔻 见底板块(首红扎堆, 板块级反转信号): {cluster or '无'}")

    # 终排序: 按量比(保留原排序意图)
    results.sort(key=lambda r: r['量比'], reverse=True)
    return pd.DataFrame(results), cluster


def sec_tag(r):
    """展示用板块标记: 行业名"""
    return r.get('行业') or '—'


def build_push_content(df, cluster):
    P = PUSH_TOP
    lines = []
    if cluster:
        lines.append("🔻 **见底板块**(首红扎堆, 板块级反转): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    lines.append(f"### 📋 520天首红 Top{min(len(df), P)} (含板块)")
    for _, r in df.head(P).iterrows():
        five = f" | 5日最低{r['5日最低价']}（距5日低点{r['距5日低点%']}%）" if pd.notna(r.get('5日最低价')) else ""
        lines.append(f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}] {r['信号日期']} 最新价 {r['最新价']} "
                     f"| 量比 {r['量比']} | 距520日低点 {r['距520日低点%']}%{five}")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output 报告*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"520天首红扫描 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print(f"K线双源(baostock+东财); 板块=baostock国标行业一次拿全+本地join(命中再多不限流)")
    print("=" * 70)

    if not is_trading_day():
        print("今日非A股交易日, 跳过扫描")
        sys.exit(0)

    df = run_first_red_520_scan(limit=SCAN_LIMIT)

    if df is not None and not df.empty:
        df, cluster = enrich(df.to_dict('records'))

        csv_path = f"{OUTPUT_DIR}/first_red_520_{datetime.now().strftime('%Y%m%d')}.csv"
        json_path = f"{OUTPUT_DIR}/first_red_520_{datetime.now().strftime('%Y%m%d')}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"\n结果已保存: {csv_path}")

        # 控制台 (带板块标记 + 聚类)
        disp = df.copy()
        disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))

        if SERVERCHAN_KEY:
            title = f"520天首红信号 命中 {len(df)} 只（含板块）"
            content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster)
            send_serverchan(title, content)
    else:
        print("本次未找到符合条件的股票")
