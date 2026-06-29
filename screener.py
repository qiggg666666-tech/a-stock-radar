import akshare as ak
import pandas as pd
import random
import time
import concurrent.futures as cf
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 全局优化配置 ---
MAX_WORKERS = 2  
RETRY = 3        

# 创建 Session 用于连接复用
session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=RETRY, backoff_factor=1, status_forcelist=[500, 502, 503, 504]))
session.mount('https://', adapter)
session.mount('http://', adapter)

# --- 策略 1: 周线首红 ---
def _check_first_red(symbol: str) -> dict | None:
    try:
        time.sleep(random.uniform(0.5, 1.0))
        end = datetime.now()
        start = end - timedelta(weeks=12)
        df = ak.stock_zh_a_hist(symbol=symbol, period="weekly", 
                                start_date=start.strftime("%Y%m%d"), 
                                end_date=end.strftime("%Y%m%d"), adjust="qfq")
        if df is None or len(df) < 3: return None
        df = df.tail(3).reset_index(drop=True)
        latest = df.iloc[-1]
        if not (latest["收盘"] > latest["开盘"]): return None
        prior = df.iloc[:-1]
        if not (prior["收盘"] < prior["开盘"]).all(): return None
        return {"代码": symbol, "本周涨幅%": round((latest["收盘"]-latest["开盘"])/latest["开盘"]*100, 2)}
    except: return None

def screen_first_red(limit=None):
    # 【优化点】：改为从行情接口获取代码列表，避开官网爬取导致的 Network Unreachable
    try:
        df_spot = ak.stock_zh_a_spot_em()
        stock_list = df_spot[['代码', '名称']].copy()
        stock_list.columns = ['code', 'name']
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()

    if limit: stock_list = stock_list.head(limit)
    name_map = dict(zip(stock_list["code"], stock_list["name"]))
    results = []
    
    batch_size = 50
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list.iloc[i:i+batch_size]
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_check_first_red, code): code for code in batch["code"]}
            for future in cf.as_completed(futures):
                res = future.result()
                if res:
                    res["名称"] = name_map.get(res["代码"], "")
                    results.append(res)
        time.sleep(2)
    return pd.DataFrame(results)

# --- 策略 2: 长期反转 ---
def _check_long_term_reversal(code, name, current_price, today_pct) -> dict | None:
    try:
        time.sleep(random.uniform(0.5, 1.0))
        hist = ak.stock_zh_a_hist(symbol=code, period="weekly", adjust="qfq", 
                                  start_date="19900101", end_date=datetime.now().strftime("%Y%m%d"))
        if hist is None or len(hist) < 12: return None
        all_time_high = hist["收盘"].max()
        if all_time_high <= 0 or current_price > all_time_high * 0.5: return None
        recent = hist.tail(12).reset_index(drop=True)
        low_idx = int(recent["收盘"].idxmin())
        if low_idx > 8: return None
        low_price = float(recent["收盘"].iloc[low_idx])
        return {
            "代码": code, "名称": name, "现价": current_price, "今日涨幅%": today_pct,
            "距历史高点回撤%": round((1 - current_price / all_time_high) * 100, 1)
        }
    except: return None

def screen_long_term_reversal(min_pct_change=7.0, limit=None):
    try:
        spot = ak.stock_zh_a_spot_em()
        candidates = spot[spot["涨跌幅"] >= min_pct_change].copy()
    except: return pd.DataFrame()
    
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
