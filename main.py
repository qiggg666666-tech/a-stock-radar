import akshare as ak
import os
import requests
import google.generativeai as genai
import pandas as pd # 显式引入 pandas

# 配置 API
api_key = os.getenv("GEMINI_API_KEY")
push_key = os.getenv("PUSH_KEY") # 确保变量名与你 GitHub 设置的 Secret 一致

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def run_analysis():
    try:
        print("正在抓取市场热点...")
        # 增加数据获取的异常捕获
        df_hot = ak.stock_hot_rank_em()
        df_spot = ak.stock_zh_a_spot_em()
        
        # 筛选逻辑：增加空值处理，防止筛选结果为空导致报错
        candidates = df_spot[(df_spot['成交额'] > 500000000) & (df_spot['涨跌幅'] > 2)]
        
        if candidates.empty:
            print("未筛选到符合条件的个股。")
            return

        # AI 分析
        print("正在进行 AI 分析...")
        prompt = f"分析热点: {df_hot.head(5).to_string()}。候选个股: {candidates.head(10).to_string()}。请结合当前A股逻辑，输出Markdown格式的个股研报。"
        res = model.generate_content(prompt)
        
        # 推送：增加对推送结果的判断
        print("正在推送至微信...")
        url = f"https://sctapi.ftqq.com/{push_key}.send"
        response = requests.post(url, data={"title": "今日AI选股研报", "desp": res.text})
        
        if response.status_code == 200:
            print("推送完成，状态码: 200")
        else:
            print(f"推送失败，返回信息: {response.text}")

    except Exception as e:
        print(f"运行出错: {str(e)}")
        # 出错时也尝试推送到微信，方便你在手机上看到报错原因
        requests.post(f"https://sctapi.ftqq.com/{push_key}.send", data={"title": "任务报错", "desp": str(e)})

if __name__ == "__main__":
    run_analysis()
