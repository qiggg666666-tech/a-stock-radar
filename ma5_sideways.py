import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
import baostock as bs
from serverchan_sdk import sc_send
import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 参数 ------------------
MAX_DEV = 0.05        # 相对MA5的最大偏差 5%
MIN_WINDOW = 8         # 横盘最短天数
MAX_WINDOW = 20        # 横盘最长天数（用于搜索最佳窗口）
MA5_SLOPE_MAX = 0.008  # MA5走平判定：斜率均值上限
PRICE_RANGE_MAX = 0.12 # 横盘期振幅上限
MIN_PRICE = 5
LOOKBACK_DAYS = 60      # 只需最近约2个月数据，足够覆盖最长20天窗口+MA5计算
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 15  # 单次查询硬超时，防止网络卡死拖垮整个job


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


def _init_worker():
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
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
    """单只股票的抓取+判断逻辑，运行在子进程里"""
    code, name = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        df = _query_with_timeout(code, "date,close", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        if df.empty or df['close'].iloc[-1] < MIN_PRICE:
            return None

        is_match, days, dev = detect_ma5_sideways(df)
        if not is_match:
            return None

        return {
            "代码": code, "名称": name,
            "最新价": round(float(df['close'].iloc[-1]), 2),
            "横盘天数": days,
            "当前偏差%": round(float(dev) * 100, 2),
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_ma5_sideways_scan(limit=None):
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
    print(f"开始MA5横盘扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

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


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- {row['名称']}（{row['代码']}）最新价 {row['最新价']} "
            f"| 横盘{row['横盘天数']}天 | 当前偏差{row['当前偏差%']}%"
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
    df = run_ma5_sideways_scan(limit=None)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"MA5横盘筛选 命中 {len(df)} 只"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
