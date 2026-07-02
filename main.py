import akshare as ak
import pandas as pd
import requests
import os
from datetime import datetime


def strategy_triple_cross(df_day):
    try:
        if len(df_day) < 260:
            return False
        df_day['ma20'] = df_day['close'].rolling(20).mean()
        df_day['ma250'] = df_day['close'].rolling(250).mean()
        df_day['date'] = pd.to_datetime(df_day['date'])
        df_week = df_day.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df_day.resample('ME', on='date')['close'].last().dropna()
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
    target_stocks = stock_info['代码'].tolist()
    if limit:
        target_stocks = target_stocks[:limit]
    print(f"共 {len(target_stocks)} 只股票待检测...")
    results = []
    for idx, code in enumerate(target_stocks):
        try:
            if idx % 50 == 0:
                print(f"进度: {idx}/{len(target_stocks)}")
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20200101")
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover'})
            df['close'] = df['close'].astype(float)
            df['date'] = pd.to_datetime(df['date'])
            if strategy_triple_cross(df):
                name = stock_info[stock_info['代码'] == code]['名称'].values[0]
                price = round(df['close'].iloc[-1], 2)
                results.append({"代码": code, "名称": name, "最新价": price})
        except:
            continue
    df_results = pd.DataFrame(results)
    print(f"\n找到 {len(df_results)} 只符合策略的股票！")
    return df_results


def send_to_serverchan(sendkey, title, desp=""):
    if not sendkey:
        return False
    url = f"https://sct.ftqq.com/{sendkey}.send"
    try:
        requests.post(url, data={"title": title, "desp": desp}, timeout=10)
        print(f"✅ 推送成功: {title}")
        return True
    except:
        return False


if __name__ == "__main__":
    df = run_all_strategies(limit=1000)
    if not df.empty:
        print(df)
        df.to_csv('results.csv', index=False, encoding='utf-8-sig')
        sendkey = os.getenv("SERVERCHAN_SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            for _, row in df.iterrows():
                title = f"【Triple Cross】{row['代码']} {row['名称']}"
                desp = f"代码：{row['代码']}\n名称：{row['名称']}\n最新价：{row['最新价']} 元\n时间：{now}"
                send_to_serverchan(sendkey, title, desp)
    else:
        print("未找到符合条件的股票")
