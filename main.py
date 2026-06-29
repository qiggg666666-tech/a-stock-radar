import screener
import push

def run():
    print("开始执行全自动量化监控...")
    
    # 策略 1：周线首红
    print("正在运行策略: 周线首红...")
    results_red = screener.screen_all(strategy="first_red", limit=20)
    if not results_red.empty:
        push.send(results_red)
        
    # 策略 2：长期反转
    print("正在运行策略: 长期反转...")
    results_rev = screener.screen_all(strategy="long_term_reversal", limit=20)
    if not results_rev.empty:
        push.send(results_rev)
        
    print("全部任务执行完毕。")

if __name__ == "__main__":
    run()
