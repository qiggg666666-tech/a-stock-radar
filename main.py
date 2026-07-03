import pandas as pd
# 补丁：解决 Pandas 2.0+ 环境下 baostock 调用 append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = pd.DataFrame._append

import baostock as bs
from serverchan_sdk import sc_send
import os
import time
from datetime import datetime
from tqdm import tqdm

# ------------------ 阈值参数 ------------------
WEEK_THRESHOLD = 0.008   # 周线 MA5/MA20 差距阈值 0.8%
MONTH_THRESHOLD = 0.012  # 月线 MA5/MA20 差距阈值 1.2%
YEAR_THRESHOLD = 0.018   # 日线 MA20/MA250 差距阈值 1.8%
MIN_PRICE = 5            # 过滤低价股
SLEEP_PER_STOCK = 0.15   # 每只股票请求间隔，降低被限流风险


# 核心策略：年月周即将金叉（三线均未金叉，但差距收窄到阈值内）
def strategy_triple_cross(df):
    try:
        if len(df) < 260:
            return False
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()
        df['date'] = pd.to_datetime(df['date'])

        # 重采样
        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('M', on='date')['close'].last().dropna()

        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()

        if len(w_ma20.dropna()) == 0 or len(m_ma20.dropna()) == 0:
            return False

        # 统一语义：三条线都还未金叉（短均线仍在长均线下方），但差距已收窄到阈值内
        # ——即"即将金叉"，而不是"刚金叉不久"
        周即将 = (w_ma5.iloc[-1] < w_ma20.iloc[-1]) and \
                (abs(w_ma5.iloc[-1] - w_ma20.iloc[-1]) / w_ma20.iloc[-1] < WEEK_THRESHOLD)
        月即将 = (m_ma5.iloc[-1] < m_ma20.iloc[-1]) and \
                (abs(m_ma5.iloc[-1] - m_ma20.iloc[-1]) / m_ma20.iloc[-1] < MONTH_THRESHOLD)
        年即将 = (df['ma20'].iloc[-1] < df['ma250'].iloc[-1]) and \
                (abs(df['ma250'].iloc[-1] - df['ma20'].iloc[-1]) / df['ma250'].iloc[-1] < YEAR_THRESHOLD)

        return 周即将 and 月即将 and 年即将 and (df['close'].iloc[-1] > MIN_PRICE)
    except Exception:
        return False


# 获取并筛选
def run_all_strategies(limit=None):
    print("正在连接 Baostock...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.'))]
    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()

    results = []
    fail_count = 0
    print(f"开始检测 {len(target_stocks)} 只股票...")

    pbar = tqdm(target_stocks, desc="扫描进度", unit="只")
    for code in pbar:
        try:
            # adjustflag="2" 前复权，避免除权除息造成的价格跳空扭曲均线
            k_rs = bs.query_history_k_data_plus(
                code, "date,close",
                start_date="2020-01-01",
                adjustflag="2"
            )
            df = k_rs.get_data()
            if strategy_triple_cross(df):
                name = stock_df[stock_df['code'] == code]['code_name'].values[0]
                results.append({"代码": code, "名称": name, "最新价": round(float(df['close'].iloc[-1]), 2)})
                pbar.write(f"✅ 命中: {code} {name}")
        except Exception as e:
            fail_count += 1
            pbar.write(f"⚠️ {code} 处理失败: {e}")
            continue

        # 每只股票间隔小睡，降低被限流/封IP的风险
        time.sleep(SLEEP_PER_STOCK)
        pbar.set_postfix(命中=len(results), 失败=fail_count)

    bs.logout()
    print(f"扫描完成，共失败 {fail_count} 只")
    return pd.DataFrame(results)


# 推送（合并为一条消息，避免超出 Server酱 免费额度）
def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        # 常见返回：{'code': 0, 'message': '', ...} code=0 才是真正成功
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}，message={response.get('message')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(f"- {row['名称']}（{row['代码']}）最新价 {row['最新价']}")
    return "\n".join(lines)


if __name__ == "__main__":
    df = run_all_strategies(limit=500)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"年月周即将金叉 命中 {len(df)} 只"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
