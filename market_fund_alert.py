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
⚠️ 单次运行架构（不是 while+schedule 常驻进程），靠 GitHub Actions 的 cron 定时重复触发。
"""

THRESHOLD_NORTH = 50       # 北向净流入异动阈值（单位待第一次运行后核对，见check_north_flow说明）
THRESHOLD_VOLUME = 1.5     # 成交量放大倍数阈值
ALERT_STOCKS = {           # 重点监控个股，baostock格式代码
    "sh.600519": "贵州茅台",
    "sz.000001": "平安银行",
}


def check_north_flow():
    """
    北向资金净流入异动检查。
    正确函数是 stock_hsgt_fund_flow_summary_em()（不传symbol参数）。
    返回结构可能是"沪股通/深股通/港股通"分行的表格，具体列名和数值单位尚未100%确认，
    这里第一次运行会把完整数据打印出来，供人工核对后再精确处理列名匹配和单位换算逻辑。
    """
    alerts = []
    try:
        print(f"当前akshare版本: {ak.__version__}")
        if not hasattr(ak, "stock_hsgt_fund_flow_summary_em"):
            print("⚠️ 当前akshare版本没有 stock_hsgt_fund_flow_summary_em 函数，"
                  "需要在 requirements.txt 里把 akshare 升级到较新版本")
            return alerts

        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            print("⚠️ 沪深港通资金流向接口返回空数据")
            return alerts

        print(f"沪深港通资金流向接口返回列名: {list(df.columns)}")
        print("完整数据（用于第一次核对列名/结构/单位）:")
        print(df.to_string(index=False))

        # 尝试识别"类型/板块"这类标识列，过滤出沪股通+深股通（=北向，港资买A股）
        # 港股通是南向（内地资金买港股），不计入北向异动
        id_col = None
        for col in ['类型', '板块', '资金方向', '名称']:
            if col in df.columns:
                id_col = col
                break

        value_col = None
        for col in ['成交净买额', '净买额', '净流入', 'value', '当日资金流入']:
            if col in df.columns:
                value_col = col
                break

        if id_col is None or value_col is None:
            print("⚠️ 未能自动识别出'类型'列或'净流入金额'列，本次仅打印数据供人工核对，不做阈值判断")
            return alerts

        north_rows = df[df[id_col].astype(str).str.contains('沪股通|深股通|北向', na=False)]
        if north_rows.empty:
            print(f"⚠️ 在 {id_col} 列中未匹配到沪股通/深股通/北向相关行，本次不做阈值判断")
            return alerts

        net_flow = pd.to_numeric(north_rows[value_col], errors='coerce').sum()
        # 注意：单位可能是"元"而非"亿元"，第一次运行后需要人工核对是否要除以1e8
        print(f"北向资金(沪股通+深股通)净流入合计: {net_flow}（单位待人工核对，可能需要换算）")

        if abs(net_flow) > THRESHOLD_NORTH:
            alerts.append(f"🌊 北向资金大异动: {net_flow:+.1f}（阈值{THRESHOLD_NORTH}，注意单位待核对）")

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
