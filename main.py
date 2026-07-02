import akshare as ak
import pandas as pd
import requests
import os
import time
from datetime import datetime

def strategy_triple_cross(df_day):
    try:
        if len(df_day) < 260:
            return False
        df_day['ma20'] = df_day['close'].rolling(20).mean()
        df_day['ma250'] = df_day['close'].rolling(250).mean()
        df_day['date'] = pd.to_datetime(df_day['date'])
        
        # 使用 'M' 替代 'ME' 以兼容旧版本Pandas
        df_week = df_day.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df_day.resample('M', on='date')['close'].last().dropna()
        
        w_ma5 = df_week.rolling(5).mean()
        w_ma20 = df_week.rolling(20).mean()
        m_ma5 = df_month.rolling(5).mean()
        m_ma20 = df_month.rolling(20).mean()
        
        周即将 = (w_ma5.iloc[-1] > w_ma20.iloc[-1]) and (abs(w_ma5.iloc[-1] - w_ma20.iloc[-1]) / w_ma20.iloc[-1] < 0.008)
        月即将 = (m_ma5.iloc[-1] > m_ma20.iloc[-1]) and (abs(m_ma5.iloc[-1] - m_ma20.iloc[-1]) / m_ma20.iloc[-1] < 0.012)
        年即将 = (df_day['ma250'].iloc[-1] > df_day['ma20'].iloc[-1]) and (abs(df_day['ma250'].iloc[-1] - df_day['ma20'].iloc[-1]) / df_day['ma20'].iloc[-1] < 0.018)
        
        return 周即将 and 月即将 and 年即将 and (df_day['close'].iloc[-1] > 5)
    except:
        return False

def run_all_strategies(limit=None):
    print("正在获取A股列表...")
    stock_info = ak.stock_zh_a_spot_em()
    target_stocks = stock_info['代码'].tolist()[:limit] if limit else stock_info['代码'].tolist()
    results = []
    
    for idx, code in enumerate(target_stocks):
        try:
            if idx % 100 == 0 and idx > 0:
                print(f"进度: {idx}/{len(target_stocks)}")
                time.sleep(2)
            
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20200101")
            df = df.rename(columns={'日期': 'date', '收盘': 'close'})
            df['close'] = df['close'].astype(float)
            df['date'] = pd.to_datetime(df['date'])
            
            if strategy_triple_cross(df):
                name = stock_info[stock_info['代码'] == code]['名称'].values[0]
                results.append({"代码": code, "名称": name, "最新价": round(df['close'].iloc[-1], 2)})
        except:
            continue
    return pd.DataFrame(results)

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
                send_to_serverchan(sendkey, f"命中: {row['名称']}", f"代码：{row['代码']}\n最新价：{row['最新价']}\n时间：{now}")
