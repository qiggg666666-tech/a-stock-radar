import baostock as bs
import pandas as pd
import random
import time
import akshare as ak
from datetime import datetime, timedelta
import concurrent.futures as cf

# --- 基础工具函数 ---
_bs_logged_in = False
def _ensure_bs_login():
    global _bs_logged_in
    if not _bs_logged_in:
        result = bs.login()
        if result.error_code != '0': raise RuntimeError(f"登录失败: {result.error_msg}")
        _bs_logged_in = True

def _to_bs_code(code: str) -> str:
    if code.startswith("6"): return f"sh.{code}"
    elif code.startswith(("8", "4")): return f"bj.{code}"
    else: return f"sz.{code}"

def _fetch_daily(code: str, days: int) -> pd.DataFrame | None:
    _ensure_bs_login()
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.6))
    try:
        rs = bs.query_history_k_data_plus(_to_bs_code(code), "date,open,high,low,close,volume,amount,turn,pctChg", 
                                          start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"), 
                                          frequency="d", adjustflag="2")
        rows = []
        while rs.next(): rows.append(rs.get_row_data())
        if not rows: return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", 
                                "volume": "成交量", "amount": "成交额", "turn": "换手率", "pctChg": "涨跌幅"})
        return df.sort_values("日期").reset_index(drop=True)
    except: return None

def _run_strategy(check_fn, min_bars, candidates, strategy_name):
    results = []
    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(check_fn, r["代码"], r["名称"], min_bars): r["代码"] for _, r in candidates.iterrows()}
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if (i + 1) % 50 == 0: time.sleep(1)
    return pd.DataFrame(results)

def _default_candidates(limit=None):
    try:
        df = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
        return df.head(limit) if limit else df
    except: return pd.DataFrame({"代码": ["600519"], "名称": ["贵州茅台"]})

# --- 策略定义 ---

def _check_ma_volume(code, name, min_bars):
    df = _fetch_daily(code, 30)
    if df is None or len(df) < min_bars: return None
    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()
    df["量MA20"] = df["成交量"].rolling(20).mean()
    if df["MA5"].iloc[-2] < df["MA20"].iloc[-2] and df["MA5"].iloc[-1] > df["MA20"].iloc[-1]:
        if df["成交量"].iloc[-1] > df["量MA20"].iloc[-1] * 1.5:
            return {"代码": code, "名称": name}
    return None

def screen_ma_volume_cross(limit=None):
    return _run_strategy(_check_ma_volume, 20, _default_candidates(limit), "均线金叉")

# 请务必在此处补全你main.py中提到的所有其他 screen_ 函数
# 例如：screen_turtle_breakout, screen_high_tight_flag 等...
