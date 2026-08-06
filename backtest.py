import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错(兼容任意 pandas 版本)
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
import akshare as ak
import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from tqdm import tqdm

# ------------------ 参数（与 main.py 保持一致，改动请两边同步）------------------
WEEK_THRESHOLD = 0.008
MONTH_THRESHOLD = 0.012
YEAR_THRESHOLD = 0.018
MIN_PRICE = 5
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = 3
QUERY_TIMEOUT_SEC = 20   # 回测要拉10年数据，单次请求重，超时给宽松一点

FORWARD_DAYS = 20
BACKTEST_START = "2015-01-01"
STOCK_LIMIT = 500

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ baostock 登录重试 + 股票列表 akshare 兜底 ------------------
def _bs_login_ok(retries=5):
    for attempt in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {attempt + 1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {attempt + 1}/{retries}")
        time.sleep(2 * (attempt + 1))
    return False


def _fetch_list_akshare():
    """baostock 列表失败时的 akshare 兜底, 返回 code/code_name/type/status 同构表。"""
    for attempt in range(3):
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                nc = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'; d['status'] = '1'
                return d
        except Exception as e:
            print(f"  akshare列表第{attempt + 1}次失败: {e}")
        time.sleep(2 + attempt)
    return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])


def backtest_single_stock(df, code, name):
    records = []
    try:
        if len(df) < 260 + FORWARD_DAYS:
            return records

        df = df.copy()
        df['close'] = df['close'].astype(float)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()

        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('ME', on='date')['close'].last().dropna()
        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()

        month_series = pd.DataFrame({"m_ma5": m_ma5, "m_ma20": m_ma20}).dropna()
        daily_indexed = df.set_index('date')[['ma20', 'ma250']].dropna()

        for i in range(20, len(w_ma20)):
            week_date = df_week.index[i]
            if pd.isna(w_ma5.iloc[i]) or pd.isna(w_ma20.iloc[i]):
                continue

            w_gap = (w_ma20.iloc[i] - w_ma5.iloc[i]) / w_ma20.iloc[i]
            if not (0 < w_gap < WEEK_THRESHOLD):
                continue

            m_asof = month_series[month_series.index <= week_date]
            if m_asof.empty:
                continue
            m_gap = (m_asof['m_ma20'].iloc[-1] - m_asof['m_ma5'].iloc[-1]) / m_asof['m_ma20'].iloc[-1]
            if not (0 < m_gap < MONTH_THRESHOLD):
                continue

            d_asof = daily_indexed[daily_indexed.index <= week_date]
            if d_asof.empty:
                continue
            y_gap = (d_asof['ma250'].iloc[-1] - d_asof['ma20'].iloc[-1]) / d_asof['ma250'].iloc[-1]
            if not (0 < y_gap < YEAR_THRESHOLD):
                continue

            trigger_rows = df[df['date'] <= week_date]
            if trigger_rows.empty:
                continue
            trigger_idx = trigger_rows.index[-1]
            trigger_close = df['close'].iloc[trigger_idx]

            if trigger_close < MIN_PRICE:
                continue

            future_idx = trigger_idx + FORWARD_DAYS
            if future_idx >= len(df):
                continue

            future_close = df['close'].iloc[future_idx]
            ret_pct = (future_close - trigger_close) / trigger_close * 100

            records.append({
                "代码": code, "名称": name,
                "触发日期": df['date'].iloc[trigger_idx].strftime("%Y-%m-%d"),
                "触发价": round(trigger_close, 2),
                f"{FORWARD_DAYS}日后价": round(future_close, 2),
                "涨跌幅%": round(ret_pct, 2),
                "是否上涨": ret_pct > 0
            })

    except Exception:
        pass

    return records


def _init_worker():
    import random
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程 baostock 登录失败, 该进程将走 akshare 兜底")


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次 baostock 查询包硬超时, 防网络卡顿导致进程池假死。"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def _fetch_hist_akshare(code):
    """baostock 拿不到历史时的 akshare 兜底, 返回 date/close 两列(前复权)。"""
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    sd = BACKTEST_START.replace('-', '')
    ed = datetime.now().strftime('%Y%m%d')
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=sd, end_date=ed, adjust="qfq")
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '收盘': 'close'})
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if not d.empty:
                    return d[['date', 'close']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _process_one(args):
    code, name = args
    try:
        # 1) baostock(带硬超时)  2) 失败/超时 -> akshare 兜底
        df = None
        try:
            d = _query_with_timeout(code, "date,close", BACKTEST_START)
            if d is not None and not d.empty and 'close' in d.columns:
                df = d[['date', 'close']]
        except FutureTimeoutError:
            df = None
        except Exception:
            df = None
        if df is None or df.empty:
            df = _fetch_hist_akshare(code)
        if df is None or df.empty:
            return [{"__error__": f"{code} 双源数据均失败"}]
        time.sleep(SLEEP_PER_STOCK)
        return backtest_single_stock(df, code, name)
    except Exception as e:
        return [{"__error__": f"{code} 处理失败: {e}"}]


def run_backtest():
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            rs = bs.query_stock_basic()
            stock_df = rs.get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        try:
            bs.logout()
        except Exception:
            pass

    # 【修复】登录失败/列表为空 -> akshare 兜底, 杜绝 KeyError:'code'
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        stock_df = _fetch_list_akshare()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("⚠️ 股票列表获取失败(baostock 与 akshare 均失败), 无法回测")
        return pd.DataFrame()

    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ]

    target_stocks = stock_df['code'].tolist()[:STOCK_LIMIT] if STOCK_LIMIT else stock_df['code'].tolist()
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(code, code_to_name.get(code, "")) for code in target_stocks]

    all_records = []
    fail_count = 0
    print(f"开始回测 {len(tasks)} 只股票，起始日期 {BACKTEST_START}，往后看 {FORWARD_DAYS} 个交易日...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="回测进度", unit="只")
        for records in pool.imap_unordered(_process_one, tasks):
            for r in records:
                if "__error__" in r:
                    fail_count += 1
                else:
                    all_records.append(r)
            pbar.update(1)
            pbar.set_postfix(触发次数=len(all_records), 失败=fail_count)

    print(f"回测完成，共失败 {fail_count} 只")
    return pd.DataFrame(all_records)


def summarize(result_df):
    if result_df.empty:
        print("\n历史上没有找到任何触发记录（可能阈值太严格，或回测起点太晚）")
        return

    total = len(result_df)
    win_rate = result_df["是否上涨"].mean() * 100
    avg_ret = result_df["涨跌幅%"].mean()
    median_ret = result_df["涨跌幅%"].median()
    best = result_df["涨跌幅%"].max()
    worst = result_df["涨跌幅%"].min()

    print("\n" + "=" * 50)
    print(f"回测统计（{FORWARD_DAYS}个交易日后）")
    print("=" * 50)
    print(f"历史触发次数：{total} 次")
    print(f"上涨概率：{win_rate:.1f}%")
    print(f"平均涨跌幅：{avg_ret:+.2f}%")
    print(f"涨跌幅中位数：{median_ret:+.2f}%")
    print(f"最佳单次：{best:+.2f}%   最差单次：{worst:+.2f}%")
    print("=" * 50)
    print("\n⚠️ 提醒：以上是历史统计，不代表未来表现；样本量少于30次时统计意义有限。")


if __name__ == "__main__":
    df = run_backtest()
    summarize(df)
    if not df.empty:
        df = df.sort_values("触发日期")
        out_csv = os.path.join(OUTPUT_DIR, "backtest_result.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n明细已保存到 {out_csv}")
