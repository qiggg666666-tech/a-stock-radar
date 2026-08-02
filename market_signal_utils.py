"""
market_signal_utils.py
大盘信号打分逻辑，被 build_calibration.py 和 daily_market_signal.py 共用。

【v2 重写说明 —— 针对"打分不准确"的根因修复】
原逻辑缺陷: ① 只有看多加分、无看空扣分 -> score 永远>=0 -> 校准概率系统性偏高;
  ② 阶跃离散(0.01->-0.01 分数从+1跳0, 弱0.01%与弱2%同分) -> 丢失连续信息且看空无区分;
  ③ reasons 单向(看空时为空, "无信号"与"银行在跌"混淆); ④ 未利用大盘自身涨跌;
  ⑤ 无 NaN 防御(数据缺失被静默当"中性")。
修复: 四维度全部改为"连续分段线性 + 看多看空对称", score 范围约 [-7.5, +7.5] 可正可负;
  reasons 双向(▲看多/▼看空/⚪中性, 箭头记号矩阵内未用过, 零撞色且表达方向贡献);
  输入 NaN/None 显式标记"数据缺失"(区别于真中性)。
接口不变: calculate_score 签名与返回结构不变 -> daily_market_signal.py / build_calibration.py
  的调用代码无需修改。模块保持零外部依赖(仅 import math), 不强制依赖 pandas。

⚠️⚠️ 连锁后果(必读): 分数分布从 [0,~6] 变为 [-7.5,7.5], 旧 calibration_table.csv 的分箱
  与新分数对不上 -> 改完本文件后【必须重跑 build_calibration.py 重新生成校准表】,
  否则 daily_market_signal 的概率会比改之前更不准! 另: 若 build_calibration.py 内部
  硬编码了分箱边界(如 bins=[0,1,2,...]), 重跑前需把边界改为覆盖 [-8, 8], 否则负分仍落错箱
  (该文件未在此提供, 需自行核对)。
"""

import math

# ------------------ 共享常量 (被 daily_market_signal / build_calibration import) ------------------
INDEX_CODE = "sh.000001"
BANK_STOCKS = {
    "建设银行": "sh.601939",
    "工商银行": "sh.601398",
    "招商银行": "sh.600036",
}

# 新 score 理论范围约 [-7.5, +7.5]; build_calibration 的分箱需覆盖此区间(建议 [-8, 8])
SCORE_MIN, SCORE_MAX = -7.5, 7.5


# ------------------ 工具: 零依赖的数值/线性映射 ------------------
def _to_float(v):
    """转 float; None/非数/NaN 返回 None (用于 NaN 防御, 不依赖 pandas)"""
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _lin(x, x_lo, x_hi, y_lo, y_hi):
    """分段线性映射: x<=x_lo->y_lo, x>=x_hi->y_hi, 中间线性插值; x 为 None 返回 0(中性贡献)"""
    if x is None:
        return 0.0
    if x_hi <= x_lo:
        return 0.0
    if x <= x_lo:
        return float(y_lo)
    if x >= x_hi:
        return float(y_hi)
    t = (x - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


# ------------------ 打分 (连续/对称/双向/NaN防御; 签名不变) ------------------
def calculate_score(sz_chg, jh_chg, gh_chg, zh_chg):
    """
    大盘综合打分。四个维度均连续对称、可正可负:
      A 建行相对大盘强弱 ([-3,+3])  B 建行自身涨跌 ([-2,+2])
      C 银行板块整体     ([-1.5,+1.5])  D 大盘自身今日涨跌 ([-1,+1], 权重小)
    返回 (score, reasons); reasons 用 ▲/▼/⚪ 标注每维度的方向贡献。
    """
    sz = _to_float(sz_chg)
    jh = _to_float(jh_chg)
    gh = _to_float(gh_chg)
    zh = _to_float(zh_chg)

    # 整体数据缺失防御: 关键输入(上证/建行)缺失 -> 明确标记, 不与"真中性"混淆
    if sz is None or jh is None:
        return 0.0, ["⚠️ 上证或建行数据缺失，本次无法打分（非中性，是数据不可用）"]

    # 银行平均: 用可用者计算, 全缺则该项中性
    banks = [b for b in (jh, gh, zh) if b is not None]
    bank_avg = sum(banks) / len(banks) if banks else 0.0

    score = 0.0
    reasons = []

    # 维度A: 建行相对大盘强弱 (连续对称, 看空扣分)
    rel = jh - sz
    score += _lin(rel, -2.0, 2.0, -3.0, 3.0)
    if rel >= 0.8:
        reasons.append(f"▲ 建行显著强于大盘（相对{rel:+.2f}%）")
    elif rel > 0.1:
        reasons.append(f"▲ 建行略强于大盘（相对{rel:+.2f}%）")
    elif rel > -0.1:
        reasons.append(f"⚪ 建行与大盘同步（相对{rel:+.2f}%）")
    elif rel > -0.8:
        reasons.append(f"▼ 建行略弱于大盘（相对{rel:+.2f}%）")
    else:
        reasons.append(f"▼ 建行显著弱于大盘（相对{rel:+.2f}%）")

    # 维度B: 建行自身涨跌 (连续对称)
    score += _lin(jh, -2.0, 2.0, -2.0, 2.0)
    if jh >= 1.2:
        reasons.append(f"▲ 建行强势领涨（{jh:+.2f}%）")
    elif jh <= -1.2:
        reasons.append(f"▼ 建行领跌（{jh:+.2f}%）")

    # 维度C: 银行板块整体 (连续对称)
    score += _lin(bank_avg, -1.5, 1.5, -1.5, 1.5)
    if bank_avg >= 0.8:
        reasons.append(f"▲ 银行板块整体走强（均{bank_avg:+.2f}%）")
    elif bank_avg <= -0.8:
        reasons.append(f"▼ 银行板块整体走弱（均{bank_avg:+.2f}%）")

    # 维度D: 大盘自身今日涨跌 (新增, 不扩大数据依赖; 权重小, 避免与"预测次日"过度耦合)
    score += _lin(sz, -3.0, 3.0, -1.0, 1.0)
    if sz <= -1.5:
        reasons.append(f"▼ 大盘今日大跌（{sz:+.2f}%）")
    elif sz >= 1.5:
        reasons.append(f"▲ 大盘今日大涨（{sz:+.2f}%）")

    score = round(max(SCORE_MIN, min(SCORE_MAX, score)), 2)
    if not reasons:
        reasons.append("⚪ 各维度均处中性区间，无明显方向")

    return score, reasons
