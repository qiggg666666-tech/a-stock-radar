"""
market_signal_utils.py
大盘信号打分逻辑，被 build_calibration.py 和 daily_market_signal.py 共用。
"""

INDEX_CODE = "sh.000001"
BANK_STOCKS = {
    "建设银行": "sh.601939",
    "工商银行": "sh.601398",
    "招商银行": "sh.600036",
}


def calculate_score(sz_chg, jh_chg, gh_chg, zh_chg):
    score = 0.0
    reasons = []

    rel_strength = jh_chg - sz_chg
    if rel_strength > 0.8:
        score += 2.5
        reasons.append("🟢 建行显著强于大盘")
    elif rel_strength > 0:
        score += 1.0
        reasons.append("🟡 建行略强于大盘")

    if jh_chg > 1.2:
        score += 2.0
        reasons.append("🟢 建行强势领涨")

    bank_avg = (jh_chg + gh_chg + zh_chg) / 3
    if bank_avg > 0.8:
        score += 1.5
        reasons.append("🟢 银行板块整体走强")

    return score, reasons
