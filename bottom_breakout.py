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
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 参数 ------------------
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
LOOKBACK_DAYS = 500

LOW_POSITION_RATIO = 0.6
MA_SLOPE_LOOKBACK = 10
VOLUME_RATIO_THRESHOLD = 1.8
MIN_PRICE = 3


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


def strategy_bottom_breakout(df):
    """
    底部强势突破：
    1. 现价处于近期高点的低位区域（避免追高）
    2. MA50 斜率由走平/向下转为向上（底部拐头）
    3. 短中期均线理顺：MA10 > MA20 > MA50（初步多头排列）
    4. 当日放量确认（量比 > 阈值）
    """
    try:
        if len(df) < 260:
            return None

        df = df.copy()
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        latest = df.iloc[-1]
        if pd.isna(latest['ma50']) or pd.isna(latest['vol_ma20']):
            return None

        recent_high = df['close'].iloc[-250:].max()
        if latest['close'] > recent_high * LOW_POSITION_RATIO:
            return None

        ma50_now = df['ma50'].iloc[-1]
        ma50_before = df['ma50'].iloc[-1 - MA_SLOPE_LOOKBACK]
        ma50_earlier = df['ma50'].iloc[-1 - MA_SLOPE_LOOKBACK * 2]
        turning_up = (ma50_now > ma50_before) and (ma50_before <= ma50_earlier * 1.002)
        if not turning_up:
            return None

        if not (latest['ma10'] > latest['ma20'] > latest['ma50']):
            return None

        volume_ratio = latest['volume'] / latest['vol_ma20']
        if volume_ratio < VOLUME_RATIO_THRESHOLD:
            return None

        if latest['close'] < MIN_PRICE:
            return None

        return {
            "close": latest['close'],
            "距高点比例": round(latest['close'] / recent_high * 100, 1),
            "量比": round(volume_ratio, 2)
        }
    except Exception:
        return None


def _process_one(args):
    code, name = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        k_rs = bs.query_history_k_data_plus(
            code, "date,close,volume",
            start_date=start_date,
            adjustflag="2"
        )
        df = k_rs.get_data()
        time.sleep(SLEEP_PER_STOCK)
        res = strategy_bottom_breakout(df)
        if res:
            return {
                "代码": code, "名称": name,
                "最新价": round(float(res["close"]), 2),
                "距高点比例%": res["距高点比例"],
                "量比": res["量比"]
            }
        return None
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_bottom_breakout_scan(limit=500):
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
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（量比{res['量比']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("量比", ascending=False).reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    df = run_bottom_breakout_scan(limit=500)
    if not df.empty:
        print("\n" + df.to_string(index=False))
        df.to_csv("bottom_breakout_result.csv", index=False, encoding="utf-8-sig")
        print("\n结果已保存到 bottom_breakout_result.csv")
    else:
        print("本次未找到符合条件的股票")
