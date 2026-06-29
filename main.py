import screener
import push
import pandas as pd
import os

def run():
    print("--- 启动量化监控任务 ---")
    
    # 打印环境变量检查 Key 是否存在
    sendkey = os.environ.get("SENDKEY")
    print(f"检查 SENDKEY 是否配置: {'已配置' if sendkey else '未配置'}")
    
    # 1. 策略一：周线首红
    res_red = screener.screen_all(strategy="first_red", limit=20)
    print(f"策略 'first_red' 筛选结果数量: {len(res_red) if res_red is not None else 0}")
    
    # 2. 策略二：长期反转
    res_rev = screener.screen_all(strategy="long_term_reversal", limit=20)
    print(f"策略 'long_term_reversal' 筛选结果数量: {len(res_rev) if res_rev is not None else 0}")

    # 3. 强制测试推送（这一行是为了帮你确认推送通道是否通畅）
    test_df = pd.DataFrame({"测试": ["通道检测"], "状态": ["正常"]})
    print("正在尝试发送测试通知...")
    push.send(test_df)
    print("测试通知发送指令已执行。")

    print("--- 任务全部执行完毕 ---")

if __name__ == "__main__":
    run()
