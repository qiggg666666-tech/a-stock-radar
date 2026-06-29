import screener
import push
import pandas as pd
import os
import time

def run():
    print("--- 启动量化监控任务 ---")
    
    # 1. 检查环境变量
    sendkey = os.environ.get("SENDKEY")
    if not sendkey:
        print("警告：未检测到 SENDKEY，无法执行微信推送。")
    
    # 2. 策略执行与数据筛选
    all_results = pd.DataFrame()

    print("正在执行策略：周线首红...")
    res_red = screener.screen_all(strategy="first_red", limit=20)
    print(f"策略 'first_red' 筛选结果数量: {len(res_red) if res_red is not None else 0}")
    if res_red is not None and not res_red.empty:
        all_results = pd.concat([all_results, res_red])

    print("正在执行策略：长期反转...")
    res_rev = screener.screen_all(strategy="long_term_reversal", limit=20)
    print(f"策略 'long_term_reversal' 筛选结果数量: {len(res_rev) if res_rev is not None else 0}")
    if res_rev is not None and not res_rev.empty:
        all_results = pd.concat([all_results, res_rev])

    # 3. 最终推送
    if not all_results.empty:
        print("发现目标，正在准备推送...")
        # 调用你在 push.py 中定义的函数名
        push.push_to_wechat(title="量化策略提醒", content=all_results.to_markdown())
        print("微信推送执行完成。")
    else:
        print("今日无符合策略的股票，跳过推送。")

    print("--- 任务全部执行完毕 ---")

if __name__ == "__main__":
    run()
