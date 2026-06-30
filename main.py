import screener
import push
import pandas as pd

def run():
    print("--- 启动量化监控任务 (多策略版) ---")
    
    # 建立一个策略执行映射列表
    # 格式: (函数引用, 策略名称)
    strategies = [
        (screener.screen_turtle_breakout, "海龟突破"),
        (screener.screen_ma_volume_cross, "均线金叉"),
        (screener.screen_high_tight_flag, "高窄旗形"),
        (screener.screen_limit_up_shakeout_v2, "涨停洗盘v2"),
        (screener.screen_uptrend_limit_down, "上升趋势跌停反包"),
        (screener.screen_rps_breakout, "RPS突破"),
        (screener.screen_private_placement, "定增监控")
    ]
    
    all_results = pd.DataFrame()
    
    for func, name in strategies:
        print(f"正在执行策略: {name}...")
        try:
            # 策略调用，RPS和定增可能需要特殊参数，这里默认跑
            res = func(limit=200) 
            if res is not None and not res.empty:
                # 标记该结果来源哪个策略
                res["策略"] = name
                all_results = pd.concat([all_results, res])
                print(f"  -> {name} 选出 {len(res)} 只股票")
        except Exception as e:
            print(f"  -> 策略 {name} 执行出错: {e}")

    # 最终推送
    if not all_results.empty:
        print(f"本次任务共选出 {len(all_results)} 只股票，准备推送...")
        push.push_to_wechat(title="量化策略提醒", content=all_results.to_markdown())
    else:
        print("今日无符合任何策略的股票，跳过推送。")

if __name__ == "__main__":
    run()
