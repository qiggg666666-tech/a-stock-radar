import baostock as bs
import pandas as pd
import requests
import os
import time
from datetime import datetime

# 核心策略：年月周金叉
def strategy_triple_cross(df):
    try:
        if len(df) < 260: return False
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()
        df['date'] = pd.to_datetime(df['date'])
        
        # 重采样为周线与月线
        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('M', on='date')['close'].last().dropna()
        
        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()
        
        # 金叉判定逻辑
        周即将 = (w_ma5.iloc[-1] > w_ma20.iloc[-1]) and (abs(w_ma5.iloc[-1] - w_ma20.iloc[-1]) / w_ma20.iloc[-1] < 0.008)
        月即将 = (m_ma5.iloc[-1] > m_ma20.iloc[-1]) and (abs(m_ma5.iloc[-1] - m_ma20.iloc[-1]) / m_ma20.iloc[-1] < 0.012)
        年即将 = (df['ma250'].iloc[-1] > df['ma20'].iloc[-1]) and (abs(df['ma250'].iloc[-1] - df['ma20'].iloc[-1]) / df['ma20'].iloc[-1] < 0.018)
        
        return 周即将 and 月即将 and 年即将 and (df['close'].iloc[-1] > 5)
    except:
        return False

# 获取数据并筛选
def run_all_strategies(limit=None):
    print("正在连接 Baostock...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    # 筛选A股
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.'))]
    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()
    
    results = []
    print(f"开始检测 {len(target_stocks)} 只股票...")
    
    for idx, code in enumerate(target_stocks):
        try:
            k_rs = bs.query_history_k_data_plus(code, "date,close", start_date="2020-01-01")
            df = k_rs.get_data()
            if strategy_triple_cross(df):
                name = stock_df[stock_df['code'] == code]['code_name'].values[0]
                results.append({"代码": code, "名称": name, "最新价": round(float(df['close'].iloc[-1]), 2)})
                print(f"✅ 命中: {code} {name}")
        except: continue
        if idx % 50 == 0: time.sleep(1)
        
    bs.logout()
    return pd.DataFrame(results)

# 消息推送
def send_to_serverchan(sendkey, title, desp):
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    requests.post(url, data={"title": title, "desp": desp}, timeout=15)

if __name__ == "__main__":
    df = run_all_strategies(limit=500)
    if not df.empty:
        df.to_csv('results.csv', index=False, encoding='utf-8-sig')
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            for _, row in df.iterrows():
                send_to_serverchan(sendkey, f"命中: {row['名称']}", f"代码：{row['代码']}\n价：{row['最新价']}\n时间：{now}")
    else:
        print("未找到符合条件的股票")
