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

# ------------------ 策略参数 ------------------
MA5_DEV = 0.05
MIN_SIDEWAYS = 8
MAX_SIDEWAYS = 20
LOW_THRESHOLD = 8.0
MA520_THRESHOLD = 5.0
MIN_AVG_TURNOVER = 1.4     # baostock的turn字段本身就是百分比(如1.23代表1.23%)，不要再乘100
MIN_VOLUME_RATIO = 1.4
MIN_AVG_AMPLITUDE = 4.5

# 数据长度要求：
# MA520需要520天历史；判断MA520是否处于历史低位，还需要再往前看至少100天的MA520历史区间；
# 判断价格/MA5是否处于低位，还需要再往前看300天价格历史。
# 三者合起来，实际至少需要 520 + 100 + 一定缓冲 ≈ 900天以上交易日数据，
# 原代码只要求550天，会导致MA520历史区间几乎总是空的，条件恒假，筛不出任何股票——这是本次重写修复的核心bug。
MIN_REQUIRED_ROWS = 950
LOOKBACK_DAYS = 1600  # 约4.4个日历年，确保有950+个交易日

MIN_PRICE = 5
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 20  # 这个策略要拉4年多数据，单次请求比其他脚本重，超时给宽松一点


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
        df = _query_with_timeout(code, "date,open,high,low,close,volume,turn", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None

        is_match, info, ma5 = detect_full_strategy_with_vr(df)
        if not is_match:
            return None

        return {
            "代码": code, "名称": name,
            **info,
            "current_ma5": round(float(ma5), 2) if ma5 is not None else None,
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_ma520_bottom_scan(limit=None):
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
    print(f"开始520日线低位筛选 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

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


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- {row['名称']}（{row['代码']}）横盘{row['sideways_days']}天 "
            f"| 量比{row['volume_ratio']} | 换手{row['avg_turnover_%']}% | 振幅{row['avg_amplitude_%']}%"
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
    df = run_ma520_bottom_scan(limit=None)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"520日线低位筛选 命中 {len(df)} 只"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
