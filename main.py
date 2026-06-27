import akshare as ak
import os
import requests
import google.generativeai as genai

# 配置 Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_watchlist():
    try:
        # 获取热点和活跃个股
        df_hot = ak.stock_hot_rank_em()
        df_spot = ak.stock_zh_a_spot_em()
        candidates = df_spot[(df_spot['成交额'] > 500000000) & (df_spot['涨跌幅'] > 2) & (df_spot['涨跌幅'] < 7)]
        
        prompt = f"""
        你是一名专业量化分析师。请结合市场热点板块和活跃股票，选出3只最具成长潜力的股票。
        热点板块: \n{df_hot.head(10).to_string()}
        活跃个股池: \n{candidates.head(20).to_string()}
        
        请只返回格式: "代码1 名称1, 代码2 名称2, 代码3 名称3"
        """
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"Error in screening: {str(e)}"

def push_to_wechat(content):
    url = f"https://sctapi.ftqq.com/{os.getenv('SENDKEY')}.send"
    requests.post(url, data={"title": "今日 AI 选股与热点分析", "desp": content})

if __name__ == "__main__":
    codes = get_ai_watchlist()
    analysis_prompt = f"""
    针对这些股票: {codes}。
    请结合近期A股题材逻辑，分析它们的上涨逻辑、催化剂及核心风险。
    请输出一份详细的Markdown格式研报。
    """
    report = model.generate_content(analysis_prompt).text
    push_to_wechat(report)
