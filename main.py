import pandas as pd
from datetime import datetime
import screener  # 假设上面的代码保存在 screener.py

def main():
    # 1. 策略列表：将策略函数名存入列表，方便后续循环执行
    strategies = [
        {"func": screener.screen_turtle_breakout, "name": "海龟突破"},
        {"func": screener.screen_ma_volume_cross, "name": "均线金叉"},
        {"func": screener.screen_high_tight_flag, "name": "高窄旗形"},
        {"func": screener.screen_limit_up_shakeout_v2, "name": "涨停洗盘v2"},
        {"func": screener.screen_uptrend_limit_down, "name": "上升趋势跌停反包"},
        {"func": screener.screen_rps_breakout, "name": "RPS突破"},
        {"func": screener.screen_new_stock_rebound, "name": "次新股反弹"}
    ]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行每日选股任务...")

    # 2. 依次执行策略，并保存结果
    final_report = []
    
    for item in strategies:
        try:
            print(f"\n>>> 正在运行策略: {item['name']}")
            # 如果是调试阶段，可以在这里设置 limit=50，避免全市场扫描耗时过长
            df = item["func"](limit=None) 
            
            if not df.empty:
                # 记录结果
                df.to_csv(f"result_{item['name']}.csv", index=False, encoding="utf_8_sig")
                print(f"[成功] {item['name']} 选出 {len(df)} 只股票，已保存 CSV")
                final_report.append({"策略": item["name"], "数量": len(df)})
            else:
                print(f"[提示] {item['name']} 未选出符合条件的股票")
                
        except Exception as e:
            print(f"[错误] 执行 {item['name']} 失败: {e}")
            continue

    # 3. 输出汇总简报
    print("\n" + "="*30)
    print("选股任务执行完毕，汇总如下：")
    report_df = pd.DataFrame(final_report)
    print(report_df)
    print("="*30)

if __name__ == "__main__":
    main()
