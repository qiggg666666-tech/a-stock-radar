import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import akshare as ak
from serverchan_sdk import sc_send
import os
from datetime import datetime

"""
foreign_holder_monitor.py

外资 + 香港中央结算 十大流通股东监控。
⚠️ 单次运行架构（不是 while+schedule 常驻进程），靠 GitHub Actions 的 cron 每周定时触发一次
   （十大流通股东是季度更新数据，每周检查足够，没必要每天跑）。
⚠️ 正确函数是 stock_gdfx_free_top_10_em(symbol=股票代码)（不带sh./sz.前缀，直接6位数字），
   原代码里的 stock_gdfx_free_holding_statistics_em 在akshare里不存在。
"""

FOREIGN_KEYWORDS = [
    "BARCLAYS BANK", "J. P. Morgan", "UBS AG", "高盛国际",
    "Morgan", "HSBC", "Citigroup", "BlackRock",
    "香港中央结算", "HKSCC", "Central Clearing"
]

WATCH_STOCKS = ["603619"]  # 可以加多个股票代码（6位数字，不带前缀）


def check_stock_foreign_holders(stock_code):
    """检查单只股票的十大流通股东里有没有外资/香港中央结算"""
    try:
        holders = ak.stock_gdfx_free_top_10_em(symbol=stock_code)
        if holders is None or holders.empty:
            print(f"{stock_code}: 未获取到股东数据")
            return None

        print(f"{stock_code} 十大流通股东数据列名: {list(holders.columns)}")

        name_col = None
        for col in ['股东名称', '名称']:
            if col in holders.columns:
                name_col = col
                break

        if name_col is None:
            print(f"⚠️ {stock_code}: 未能识别出股东名称列，跳过匹配")
            return None

        print(f"{stock_code} 前十大股东概览:")
        print(holders.head(10).to_string(index=False))

        foreign = holders[holders[name_col].astype(str).str.contains(
            '|'.join(FOREIGN_KEYWORDS), na=False, case=False
        )]

        if not foreign.empty:
            print(f"\n【{stock_code} 检测到外资/中央结算】:")
            print(foreign.to_string(index=False))
            return foreign
        else:
            print(f"{stock_code}: 本次未检测到重点外资/中央结算")
            return None

    except Exception as e:
        print(f"⚠️ {stock_code} 获取失败: {e}")
        return None


def build_alert_content(results):
    lines = []
    for code, foreign_df in results.items():
        lines.append(f"【{code}】检测到外资/中央结算持股：")
        for _, row in foreign_df.iterrows():
            lines.append(f"  - {dict(row)}")
    return "\n".join(lines)


def send_alert(title, content):
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        print("未配置SENDKEY，仅打印不推送")
        return
    try:
        response = sc_send(sendkey, title, content)
        print(f"推送结果: {response}")
    except Exception as e:
        print(f"推送失败: {e}")


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== 外资+香港中央结算监控 {now} ===")

    hit_results = {}
    for code in WATCH_STOCKS:
        result = check_stock_foreign_holders(code)
        if result is not None:
            hit_results[code] = result

    if hit_results:
        content = f"检查时间：{now}\n\n" + build_alert_content(hit_results)
        print("\n" + content)
        send_alert(f"外资/中央结算持股预警 {now}", content)
    else:
        print("\n本次检查未在监控名单中检测到外资/中央结算")
