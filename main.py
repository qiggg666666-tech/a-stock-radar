import screener
import push
import pandas as pd
import os

def run():
    print("--- 启动量化监控任务 ---")
    
    # 1. 策略一：周线首红
    print("正在执行策略：周线首红...")
    try:
        res_red = screener.screen_all(strategy="first_red", limit=20)
        print(f"策略 'first_red' 筛选结果数量: {len(res_red) if res_red is not None else 0}")
        if res_red is not None and not res_red.empty:
            push.send(res_red)
            print("周线首红推送完成。")
    except Exception as e:
        print(f"策略 'first_red' 运行出错: {e}")

    # 2. 策略二：长期反转
    print("正在执行策略：长期反转...")
    try:
        res_rev = screener.screen_all(strategy="long_term_reversal", limit=20)
        print(f"策略 'long_term_reversal' 筛选结果数量: {len(res_rev) if res_rev is not None else 0}")
        if res_rev is not None and not res_rev.empty:
            push.send(res_rev)
            print("长期反转推送完成。")
    except Exception as e:
        print(f"策略 'long_term_reversal' 运行出错: {e}")

    print("--- 任务全部执行完毕 ---")

if __name__ == "__main__":
    run()
