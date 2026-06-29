import argparse
import os
import google.generativeai as genai
from screener import screen_all # 导入你整合后的策略模块
from push import push_to_wechat

# 配置 Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def run(strategy: str, mode: str):
    print(f"-> 正在执行策略: {strategy}...")
    
    # 1. 执行选股
    df = screen_all(strategy=strategy, limit=None)
    
    if df.empty:
        print("-> 未筛选到符合条件的股票，结束。")
        return

    # 2. 生成研报 (将筛选出的股票数据喂给 AI)
    print("-> 正在生成分析研报...")
    prompt = f"你是一个量化分析师。以下是筛选出的 {strategy} 策略股票数据：{df.to_string()}。请分析它们的趋势并总结。"
    res = model.generate_content(prompt)
    
    # 3. 推送
    push_to_wechat(f"AI 策略选股-{strategy}", res.text)
    print("-> 推送完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["first_red", "long_term_reversal"], default="first_red")
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    args = parser.parse_args()
    
    run(strategy=args.strategy, mode=args.mode)
