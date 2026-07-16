import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
from serverchan_sdk import sc_send
import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 阈值参数 ------------------
LOW_WINDOW = 520
VOL_WINDOW = 20
NEW_LOW_TOLERANCE = 1.02
MIN_PRICE = 5
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = 3
QUERY_TIMEOUT_SEC = 15  # 单次查询硬超时，防止网络卡死拖垮整个job


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


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def _process_one(args):
    code, name = args
    try:
        start_date = (datetime.now() - timedelta(days=int(LOW_WINDOW * 1.6))).strftime('%Y-%m-%d')
        df = _query_with_timeout(code, "date,open,high,low,close,volume", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if len(df) < LOW_WINDOW:
            return None

        signals = detect_first_red_to_520_low(df)
        if signals.empty:
            return None

        latest = signals.iloc[-1]
        if (df['date'].iloc[-1] != latest['date']) and \
           (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(latest['date'])).days > 7:
            return None

        return {
            "代码": code, "名称": name,
            "信号日期": latest['date'],
            "最新价": round(float(latest['close']), 2),
            "量比": round(float(latest['vol_ratio']), 2),
            "距520日低点%": latest['distance_pct'],
            "5日最低价": latest['5日最低价'],
            "距5日低点%": latest['距5日低点%'],
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_first_red_520_scan(limit=None):
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ]
    bs.logout()

    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(code, code_to_name.get(code, "")) for code in target_stocks]

    results = []
    fail_count = 0
    print(f"开始520天首红扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

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


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        five_low_str = (
            f" | 5日最低{row['5日最低价']}（距5日低点{row['距5日低点%']}%）"
            if pd.notna(row.get('5日最低价')) else ""
        )
        lines.append(
            f"- {row['名称']}（{row['代码']}）{row['信号日期']} 最新价 {row['最新价']} "
            f"| 量比 {row['量比']} | 距520日低点 {row['距520日低点%']}%{five_low_str}"
        )
    return "\n".join(lines)


def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}，message={response.get('message')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


if __name__ == "__main__":
    df = run_first_red_520_scan(limit=None)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"520天首红信号 命中 {len(df)} 只"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
