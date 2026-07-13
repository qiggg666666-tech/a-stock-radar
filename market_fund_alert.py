import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import akshare as ak
import baostock as bs
from serverchan_sdk import sc_send
import os
from datetime import datetime, timedelta

"""
market_fund_alert.py

市场资金异动监控：北向资金 + 重点个股成交量放大。
每天收盘后跑一次，检查的是当天的北向资金净流入和个股量比。
"""

THRESHOLD_NORTH = 50       # 北向净流入异动阈值（亿元）
THRESHOLD_VOLUME = 1.5     # 成交量放大倍数阈值
ALERT_STOCKS = {           # 重点监控个股，baostock格式代码
    "sh.600519": "贵州茅台",
    "sz.000001": "平安银行",
}


def check_north_flow():
    """
    北向资金净流入异动检查。
    akshare接口字段名可能随版本变化，这里做了动态识别数值列的兜底，
    并把原始列名打印出来，方便第一次运行时人工核对。
    """
    alerts = []
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is None or df.empty:
            print("⚠️ 北向资金接口返回空数据")
            return alerts

        print(f"北向资金接口返回列名: {list(df.columns)}")
        latest = df.iloc[-1]

        net_flow = None
        for col in ['value', '净流入', '当日净流入', 'net_flow']:
            if col in df.columns:
                net_flow = latest[col]
                break
        if net_flow is None:
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                net_flow = latest[numeric_cols[-1]]

        if net_flow is None:
            print("⚠️ 未能从北向资金数据中识别出净流入数值列，跳过本项检查")
            return alerts

        net_flow = float(net_flow)
        print(f"北向资金最新净流入: {net_flow:.2f}亿元")
        if abs(net_flow) > THRESHOLD_NORTH:
            alerts.append(f"🌊 北向资金大异动: {net_flow:+.1f}亿元（阈值{THRESHOLD_NORTH}亿）")

    except Exception as e:
        print(f"⚠️ 北向资金检查失败: {e}")

    return alerts


def check_stock_volume():
    """重点个股成交量放大检查，用baostock（与仓库其他脚本数据源一致）"""
    alerts = []
    try:
        bs.login()
        for code, name in ALERT_STOCKS.items():
            try:
                start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
                rs = bs.query_history_k_data_plus(
                    code, "date,close,volume",
                    start_date=start_date, adjustflag="2"
                )
                df = rs.get_data()
                if df.empty or len(df) < 6:
                    continue
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                latest_vol = df['volume'].iloc[-1]
                avg_vol = df['volume'].iloc[-5:-1].mean()
                if avg_vol <= 0:
                    continue
                vol_ratio = latest_vol / avg_vol
                print(f"{name}({code}) 量比: {vol_ratio:.2f}")
                if vol_ratio > THRESHOLD_VOLUME:
                    alerts.append(f"📊 {name}({code}) 成交量放大: {vol_ratio:.2f}倍（阈值{THRESHOLD_VOLUME}倍）")
            except Exception as e:
                print(f"⚠️ {code} 检查失败: {e}")
        bs.logout()
    except Exception as e:
        print(f"⚠️ baostock登录失败: {e}")

    return alerts


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
    print(f"=== 资金异动检查 {now} ===")

    all_alerts = []
    all_alerts += check_north_flow()
    all_alerts += check_stock_volume()

    if all_alerts:
        content = f"检查时间：{now}\n\n" + "\n".join(all_alerts)
        print("\n" + content)
        send_alert(f"资金异动预警 {now}", content)
    else:
        print("本次检查未触发任何异动阈值")
