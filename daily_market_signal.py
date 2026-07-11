import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
import os
from serverchan_sdk import sc_send
from datetime import datetime, timedelta
from market_signal_utils import INDEX_CODE, BANK_STOCKS, calculate_score

CALIBRATION_CSV = "calibration_table.csv"


def fetch_latest(code):
    """只拉最近几天数据，取最新一条，轻量快速"""
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    rs = bs.query_history_k_data_plus(
        code, "date,close,pctChg",
        start_date=start_date,
        frequency="d", adjustflag="2"
    )
    df = rs.get_data()
    df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['pctChg']).sort_values('date').reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{code} 未获取到有效数据")
    return df.iloc[-1]


def estimate_probability(score, calibration):
    """在校准表里找到得分所在的分箱，返回历史经验频率"""
    for _, row in calibration.iterrows():
        if row['bin_left'] <= score <= row['bin_right']:
            return row['次日上涨频率'] * 100, row['样本数']
    calibration = calibration.copy()
    calibration['mid'] = (calibration['bin_left'] + calibration['bin_right']) / 2
    nearest = calibration.iloc[(calibration['mid'] - score).abs().argmin()]
    return nearest['次日上涨频率'] * 100, nearest['样本数']


def run_daily_signal():
    if not os.path.exists(CALIBRATION_CSV):
        print(f"❌ 找不到 {CALIBRATION_CSV}，请先跑一次 build_calibration.py 生成校准表")
        return

    calibration = pd.read_csv(CALIBRATION_CSV)

    bs.login()
    sz = fetch_latest(INDEX_CODE)
    banks = {name: fetch_latest(code) for name, code in BANK_STOCKS.items()}
    bs.logout()

    jh, gh, zh = banks["建设银行"], banks["工商银行"], banks["招商银行"]

    print("=== 📊 大盘综合信号（历史校准版）===")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    print(f"上证指数 : {sz['close']} ({sz['pctChg']:.2f}%)")
    print(f"建设银行 : {jh['close']} ({jh['pctChg']:.2f}%)")

    score, reasons = calculate_score(sz['pctChg'], jh['pctChg'], gh['pctChg'], zh['pctChg'])

    print("\n【信号解读】")
    for r in reasons:
        print(r)
    if not reasons:
        print("（今日无明显信号触发）")

    prob, sample_size = estimate_probability(score, calibration)

    print(f"\n【大盘短期预测】")
    print(f"次日上涨的历史经验概率: {prob:.1f}%（基于 {int(sample_size)} 个历史相似样本，非未来保证）")

    if prob >= 65:
        verdict = "🚀 历史上类似信号后，大盘次日上涨占多数"
    elif prob >= 52:
        verdict = "🟢 历史上类似信号后，大盘次日略偏上涨"
    elif prob >= 48:
        verdict = "➡️ 历史上类似信号后，涨跌接近五五开，无明显方向"
    else:
        verdict = "⚠️ 历史上类似信号后，大盘次日下跌占多数"
    print(verdict)
    print("\n（该概率来自历史统计校准，样本量有限，仅供参考，不构成投资建议）")

    sendkey = os.getenv("SENDKEY")
    if sendkey:
        title = f"大盘信号 {datetime.now().strftime('%m-%d')} | 上涨经验概率 {prob:.0f}%"
        content = (
            f"上证指数: {sz['close']} ({sz['pctChg']:.2f}%)\n"
            f"建设银行: {jh['close']} ({jh['pctChg']:.2f}%)\n\n"
            + "\n".join(reasons or ["（今日无明显信号触发）"])
            + f"\n\n{verdict}\n\n历史样本数: {int(sample_size)}（仅供参考，不构成投资建议）"
        )
        try:
            sc_send(sendkey, title, content)
        except Exception as e:
            print(f"⚠️ 推送失败: {e}")


if __name__ == "__main__":
    run_daily_signal()
