import pandas as pd
import akshare as ak
import time
import concurrent.futures as cf
from datetime import datetime, timedelta

# --- 1. 基础工具与数据获取 ---
def _fetch_history(code: str) -> pd.DataFrame:
    """获取足够覆盖所有策略的历史数据 (至少130天以满足RPS 120和MA60)"""
    try:
        start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        # 前复权，数据包含: 日期,开盘,收盘,最高,最低,成交量,成交额
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        # 重命名以对齐策略逻辑
        df = df.rename(columns={"开盘": "开盘", "收盘": "收盘", "最高": "最高", "最低": "最低", "成交量": "成交量", "成交额": "成交额"})
        return df
    except: return None

# --- 2. 核心策略逻辑 (通达信公式转Python) ---
def _check_strategies(code, name):
    df = _fetch_history(code)
    if df is None or len(df) < 120: return None
    
    # 预计算通用指标
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['MA60'] = df['收盘'].rolling(60).mean()
    df['VOL_MA20'] = df['成交量'].rolling(20).mean()
    
    # 定义结果集
    res = {"代码": code, "名称": name}
    
    # 1. 海龟突破
    if df['收盘'].iloc[-1] > df['最高'].shift(1).rolling(20).max().iloc[-1] and \
       df['成交额'].iloc[-1] > 1e8 and df['收盘'].iloc[-1] > df['开盘'].iloc[-1]:
        res['策略_海龟突破'] = True
        
    # 2. 均线金叉
    if df['MA5'].iloc[-2] < df['MA20'].iloc[-2] and df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and \
       df['成交量'].iloc[-1] > df['VOL_MA20'].iloc[-1] * 1.5:
        res['策略_均线金叉'] = True
        
    # 3. 高窄旗形
    high40 = df['最高'].rolling(40).max().iloc[-1]
    low40 = df['最低'].rolling(40).min().iloc[-1]
    high10 = df['最高'].rolling(10).max().iloc[-1]
    low10 = df['最低'].rolling(10).min().iloc[-1]
    if (high40/low40 > 1.6) and (high10/low10 < 1.15) and (low10 >= high40*0.8) and \
       (df['成交量'].iloc[-1] < df['VOL_MA20'].iloc[-1] * 0.6):
        res['策略_高窄旗形'] = True
        
    # 4. 涨停洗盘
    if (df['收盘'].iloc[-2] >= df['收盘'].iloc[-3] * 1.095) and (df['收盘'].iloc[-1] < df['开盘'].iloc[-1]) and \
       (df['成交量'].iloc[-1] > df['成交量'].iloc[-2] * 2.0) and (df['最低'].iloc[-1] >= df['收盘'].iloc[-2]):
        res['策略_涨停洗盘'] = True
        
    # 5. 上升趋势跌停反包
    if (df['MA20'].iloc[-2] > df['MA60'].iloc[-2]) and (df['收盘'].iloc[-1] <= df['收盘'].iloc[-2] * 0.905) and \
       (df['成交量'].iloc[-1] > df['VOL_MA20'].iloc[-1] * 2.0):
        res['策略_跌停反包'] = True

    # 6. RPS近似
    if ((df['收盘'].iloc[-1] - df['收盘'].iloc[-120])/df['收盘'].iloc[-120] > 0.5) and \
       (df['收盘'].iloc[-1] >= df['最高'].rolling(120).max().iloc[-1] * 0.9):
        res['策略_RPS突破'] = True

    return res if len(res) > 2 else None

# --- 3. 自动化扫描引擎 ---
def run_all_strategies(limit=None):
    print("开始全市场策略扫描...")
    stocks = ak.stock_info_a_code_name()
    if limit: stocks = stocks.head(limit)
    
    results = []
    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_check_strategies, r["code"], r["name"]): r["code"] for _, r in stocks.iterrows()}
        for future in cf.as_completed(futures):
            if future.result(): results.append(future.result())
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_all_strategies(limit=100) # 调试模式
    print(df)
