import akshare as ak
import os
import requests
import google.generativeai as genai

# 配置 Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_watchlist():
    print("正在抓取市场热点与活跃个股...")
    # 获取热点和活跃个股数据
    df_hot = ak.stock_hot_rank_em()
    df_spot = ak.stock_zh_a_spot_em()
    # 筛选成交额大于5亿且涨幅在2%-7%之间的股票
    candidates = df_spot[(df_spot['成交额'] > 500000000) & (df_spot['涨跌幅'] > 2) & (df_spot['涨跌幅'] < 7)]
    
    prompt = f"你是专业量化分析师。热点板块: {df_hot.head(5).to_string()}。活跃股票池: {candidates.head(10).to_string()}。请选出3只最具潜力的股票，仅返回格式: 代码1 名称1, 代码2 名称2, 代码3 名称3"
    res = model.generate_content(prompt)
    return res.text.strip()

def push_to_wechat(title, content):
    # 使用环境变量读取 SENDKEY
    url = f"https://sctapi.ftqq.com/{os.getenv('SENDKEY')}.send"
    requests.post(url, data={"title": title, "desp": content})

if __name__ == "__main__":
    try:
        codes = get_ai_watchlist()
        report = model.generate_content(f"分析股票: {codes}。分析逻辑及风险，输出Markdown研报。").text
        push_to_wechat("今日AI选股研报", report)
        print("推送成功")
    except Exception as e:
        push_to_wechat("运行错误", str(e))
        print(f"错误: {e}")
