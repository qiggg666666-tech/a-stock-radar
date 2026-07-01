import pandas as pd
import screener
from datetime import datetime
import os

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务开始执行...")
    
    # 1. 执行全策略扫描
    # 在 GitHub Actions 环境下，建议 limit=None 跑全市场，如果你发现超时，可改为 2000
    try:
        df = screener.run_all_strategies(limit=None)
    except Exception as e:
        print(f"[错误] 扫描过程出错: {e}")
        return

    if df.empty:
        print("未扫描到符合任何策略的股票。")
        return

    # 2. 保存结果
    output_file = "选股结果.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[完成] 结果已保存至 {output_file}")

    # 3. 生成并打印简报
    print("\n" + "="*30)
    print("策略筛选汇总：")
    # 提取所有策略列（以"策略_"开头）
    strategy_cols = [col for col in df.columns if col.startswith("策略_")]
    
    summary = {}
    for col in strategy_cols:
        count = df[col].sum()
        summary[col.replace("策略_", "")] = int(count)
        print(f"{col.replace('策略_', '')}: {int(count)} 只")
    print("="*30)

    # 4. 可选：如果筛选结果非常多，可以在这里打印出具体的股票清单
    if not df.empty:
        print("\n部分筛选结果预览:")
        print(df.head(10)[["代码", "名称"] + strategy_cols])

if __name__ == "__main__":
    main()
