import akshare as ak
import pandas as pd
import random
import time
import concurrent.futures as cf
from datetime import datetime, timedelta

# --- 配置参数 ---
MIN_GREEN_WEEKS = 2
MAX_WORKERS = 4  # 建议调小以防被限流
RETRY = 2
REVERSAL_MIN_PCT_CHANGE = 7.0
REVERSAL_MAX_PRICE_RATIO = 0.5
REVERSAL_LOOKBACK_WEEKS = 12
REVERSAL_LOW_MUST_BE_WITHIN = 8

# --- 策略 1: 周线首红 ---
def _check_first_red(symbol: str) -> dict | None:
    for attempt in range(RETRY + 1):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            end = datetime.now()
            start = end - timedelta(weeks=MIN_GREEN_WEEKS + 8)
            df = ak.stock_zh_a_hist(symbol=symbol, period="weekly", 
                                    start_date=start.strftime("%Y%m%d"), 
                                    end_date=end.strftime("%Y%m%d"), adjust="qfq")
            if df is None or len(df) < MIN_GREEN_WEEKS + 1: return None
            df = df.tail(MIN_GREEN_WEEKS + 1).reset_index(drop=True)
            latest = df.iloc[-1]
            if not (latest["收盘"] > latest["开盘"]): return None
            prior = df.iloc[:-1]
            if not (prior["收盘"] < prior["开盘"]).all(): return None
            return {"代码": symbol, "本周涨幅%": round((latest["收盘"]-latest["开盘"])/latest["开盘"]*100, 2)}
        except: time.sleep(2)
    return None

def screen_first_red(limit=None):
    stock_list = ak.stock_info_a_code_name()
    if limit: stock_list = stock_list.head(limit)
    name_map = dict(zip(stock_list["code"], stock_list["name"]))
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_check_first_red, code): code for code in stock_list["code"]}
        for future in cf.as_completed(futures):
            res = future.result()
            if res:
                res["名称"] = name_map.get(res["代码"], "")
                results.append(res)
    return pd.DataFrame(results)

# --- 策略 2: 长期反转 ---
def _check_long_term_reversal(code, name, current_price, today_pct) -> dict | None:
    try:
        hist = ak.stock_zh_a_hist(symbol=code, period="weekly", adjust="qfq", 
                                  start_date="19900101", end_date=datetime.now().strftime("%Y%m%d"))
        if hist is None or len(hist) < REVERSAL_LOOKBACK_WEEKS: return None
        all_time_high = hist["收盘"].max()
        if all_time_high <= 0 or current_price > all_time_high * REVERSAL_MAX_PRICE_RATIO: return None
        recent = hist.tail(REVERSAL_LOOKBACK_WEEKS).reset_index(drop=True)
        low_idx = int(recent["收盘"].idxmin())
        if low_idx > REVERSAL_LOW_MUST_BE_WITHIN: return None
        low_price = float(recent["收盘"].iloc[low_idx])
        return {
            "代码": code, "名称": name, "现价": current_price, "今日涨幅%": today_pct,
            "距历史高点回撤%": round((1 - current_price / all_time_high) * 100, 1)
        }
    except: return None

def screen_long_term_reversal(min_pct_change=REVERSAL_MIN_PCT_CHANGE, limit=None):
    spot = ak.stock_zh_a_spot_em()
    candidates = spot[spot["涨跌幅"] >= min_pct_change].copy()
    if limit: candidates = candidates.head(limit)
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_check_long_term_reversal, r["代码"], r["名称"], r["最新价"], r["涨跌幅"]): r["代码"] for _, r in candidates.iterrows()}
        for future in cf.as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return pd.DataFrame(results).sort_values("距历史高点回撤%", ascending=False) if results else pd.DataFrame()

# --- 统一调度器 ---
def screen_all(strategy="first_red", limit=None):
    if strategy == "first_red": return screen_first_red(limit=limit)
    elif strategy == "long_term_reversal": return screen_long_term_reversal(limit=limit)
    return pd.DataFrame()
