import akshare as ak
import os
import requests
import google.generativeai as genai
import pandas as pd

# 1. 基础配置
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_watchlist():
    print("正在抓取市场热点与活跃个股...")
    # 获取热点排行榜和个股实时数据
    df_hot = ak.stock_hot_rank_em()
    df_spot = ak.stock_zh_a_spot_em()
    
    # 筛选成交额大于5亿，涨幅在2%-7%之间的活跃股
    candidates = df_spot[(df_spot['成交额'] > 500000000) & (df_spot['涨跌幅'] > 2) & (df_spot['涨跌幅'] < 7)]
    
    if candidates.empty:
        return None
    
    # 构建分析提示词
    prompt = f"你是专业量化分析师。热点板块: {df_hot.head(5).to_string()}。活跃股票池: {candidates.head(10).to_string()}。请选出3只最具潜力的股票，仅返回格式: 代码1 名称1, 代码2 名称2, 代码3 名称3"
    res = model.generate_content(prompt)
    return res.text.strip()

def push_to_wechat(title, content):
    # 推送接口
    url = f"https://sctapi.ftqq.com/{os.getenv('PUSH_KEY')}.send"
    res = requests.post(url, data={"title": title, "desp": content})
    print(f"微信推送状态码: {res.status_code}")

if __name__ == "__main__":
    try:
        print("开始量化分析流程...")
        codes = get_ai_watchlist()
        if codes:
            # 生成详细研报
            analysis = model.generate_content(f"针对股票: {codes}，请结合近期A股题材逻辑，分析上涨逻辑、催化剂及核心风险。输出 Markdown 格式研报。").text
            push_to_wechat("今日AI选股研报", analysis)
            print("分析完成并已推送")
        else:
            push_to_wechat("今日提示", "暂无符合筛选条件的股票。")
            print("未找到候选股票")
    except Exception as e:
        print(f"运行发生错误: {str(e)}")
        push_to_wechat("任务运行异常", str(e))
