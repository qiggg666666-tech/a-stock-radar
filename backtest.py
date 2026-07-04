import pandas as pd
# 补丁：解决 Pandas 2.0+ 环境下 baostock 调用 append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = pd.DataFrame._append

import baostock as bs
import os
import time
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# ------------------ 参数（与 main.py 保持一致，改动请两边同步）------------------
WEEK_THRESHOLD = 0.008
MONTH_THRESHOLD = 0.012
YEAR_THRESHOLD = 0.018
MIN_PRICE = 5
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = 3

FORWARD_DAYS = 20        # 触发后往后看多少个交易日的涨跌（默认20，约1个月）
BACKTEST_START = "2015-01-01"  # 回测起点，越早样本越多，但拉取和计算耗时也越久
STOCK_LIMIT = 500        # 先用跟main.py一样的500只做验证，跑通后可以去掉limit测全市场


def backtest_single_stock(df, code, name):
    """
    对单只股票的完整历史做回测：
    找出历史上每一个"三线即将金叉"触发点，记录触发后FORWARD_DAYS个交易日的涨跌幅
    """
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
        df_month = df.resample('ME', on='date')['close'].last().dropna()  # pandas 2.2+ 用ME替代已废弃的M
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
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求可能持续报错")


def _process_one(args):
    code, name = args
    try:
        k_rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date=BACKTEST_START,
            adjustflag="2"
        )
        df = k_rs.get_data()
        time.sleep(SLEEP_PER_STOCK)
        return backtest_single_stock(df, code, name)
    except Exception as e:
        return [{"__error__": f"{code} 处理失败: {e}"}]


def run_backtest():
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.'))]
    bs.logout()

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
        df.to_csv("backtest_result.csv", index=False, encoding="utf-8-sig")
        print("\n明细已保存到 backtest_result.csv")
