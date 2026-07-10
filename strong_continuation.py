import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
# 不依赖 pandas 内部的 _append（该私有方法在 pandas 3.0+ 也被移除了），
# 直接用 pd.concat 重新实现，兼容任意 pandas 版本
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 参数 ------------------
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
LOOKBACK_DAYS = 550
QUERY_TIMEOUT_SEC = 15    # 单次查询硬超时，防止网络卡死拖垮整个job

NEW_HIGH_RATIO = 0.97     # 现价 >= 近250日最高价 * 该比例，提高到97%，只认最贴近真实新高的
MOMENTUM_DAYS = 20         # 近多少个交易日的涨幅用于判断动能
MOMENTUM_THRESHOLD = 0.15  # 近20日涨幅门槛从10%提高到15%，过滤掉动能不够扎实的
MA50_SLOPE_LOOKBACK = 10   # 判断MA50是否持续向上的回看天数
BREAK_MA20_CHECK_DAYS = 10 # 从5天延长到10天，要求趋势保持更久，减少刚起势就被算进来的情况
MIN_PRICE = 3
MAX_RSI = 80               # 新增：RSI超过80视为过热，即使趋势强劲也排除，规避追高后立即回调的风险
MIN_VOLUME_RATIO = 0.8     # 新增：近5日均量 / 近20日均量，不能低于此值，避免"上涨但量能已萎缩"的情况


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


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


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

        # 条件1：逼近/创近250日新高
        recent_high = df['close'].iloc[-250:].max()
        if latest['close'] < recent_high * NEW_HIGH_RATIO:
            return None

        # 条件2：多头排列 + 长期趋势(MA200)也确认向上
        if not (latest['ma10'] > latest['ma20'] > latest['ma50'] > latest['ma200']):
            return None
        ma50_now = df['ma50'].iloc[-1]
        ma50_before = df['ma50'].iloc[-1 - MA50_SLOPE_LOOKBACK]
        if ma50_now <= ma50_before:
            return None

        # 条件3：近20日涨幅达标
        price_20d_ago = df['close'].iloc[-1 - MOMENTUM_DAYS]
        momentum = (latest['close'] - price_20d_ago) / price_20d_ago
        if momentum < MOMENTUM_THRESHOLD:
            return None

        # 条件4：近10日没有跌破MA20
        recent_closes = df['close'].iloc[-BREAK_MA20_CHECK_DAYS:]
        recent_ma20 = df['ma20'].iloc[-BREAK_MA20_CHECK_DAYS:]
        if (recent_closes < recent_ma20).any():
            return None

        # 条件5：RSI排除过热
        rsi = calculate_rsi(df['close'])
        if rsi is not None and rsi > MAX_RSI:
            return None

        # 条件6：量能未萎缩（如果有volume数据才检查）
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


def _process_one(args):
    code, name = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        df = _query_with_timeout(code, "date,close,volume", start_date)
        time.sleep(SLEEP_PER_STOCK)
        res = strategy_strong_continuation(df)
        if res:
            return {
                "代码": code, "名称": name,
                "最新价": round(float(res["close"]), 2),
                "距高点比例%": res["距高点比例"],
                "近20日涨幅%": res["近20日涨幅%"],
                "RSI": res["RSI"]
            }
        return None
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_strong_continuation_scan(limit=500):
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
    print(f"开始检测 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

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


if __name__ == "__main__":
    df = run_strong_continuation_scan(limit=500)
    if not df.empty:
        print("\n" + df.to_string(index=False))
        df.to_csv("strong_continuation_result.csv", index=False, encoding="utf-8-sig")
        print("\n结果已保存到 strong_continuation_result.csv")
    else:
        print("本次未找到符合条件的股票")
