import os
import time
import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import baostock as bs

# 核心打分逻辑与常量来自本地模块 market_signal_utils (本脚本强依赖, 其健康需单独确认)
from market_signal_utils import INDEX_CODE, BANK_STOCKS, calculate_score

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

"""
daily_market_signal.py

大盘综合信号（历史校准版）：上证指数 + 三大银行股涨跌幅 -> calculate_score 综合得分
-> calibration_table.csv 校准表映射为"次日上涨的历史经验概率" -> 🚀/🟢/➡️/⚠️ 判断。

⚠️ 依赖关系（务必了解）：
- 核心打分 calculate_score 与常量 INDEX_CODE/BANK_STOCKS 来自 market_signal_utils.py
  （本脚本未含其源码, 该文件健康需单独确认; 若其有 baostock 裸登录等问题会连带影响本脚本）。
- 校准表 calibration_table.csv 由 build_calibration.py 生成 (每月1号 cron 或手动跑)。

【v2 升级说明】
- 修复 fetch_latest 无硬超时(原代码 baostock 卡住会无限挂起到 Actions 强杀); 加 _query_with_timeout。
- 修复登录失败直接崩(原代码裸 bs.login 失败 -> fetch_latest raise ValueError -> 无 try/except
  -> 未捕获异常崩退出); 加 _bs_login_ok 重试 + try/except 优雅处理 + 登录失败走东财兜底。
- 加东财兜底(区分指数/个股: sh.000xxx/sz.399xxx 走 index_zh_a_hist, 其余个股走 stock_zh_a_hist;
  均取最近一条 close+pctChg, 全功能不降级)。
- 修复 sc_send 硬导入(软导入+requests兜底); 结果存 output/ json 留痕。
- 加交易日判断: 本脚本是"每日预测次日上涨概率", 时效性强, 非交易日推送基于旧数据的"次日概率"
  会误导(次日在周末无意义), 故非交易日跳过。这与 index_divergence(背离为历史形态, 非交易日仍有效)
  的"不加"是有原则的区分。
- 保留 verdict 渐变记号体系 🚀>🟢>➡️>⚠️ (有内在语义逻辑: 概率从高到低渐变; 虽与别处 emoji 有
  复用, 但改任一即破坏渐变, 故保留; 跨脚本 emoji 靠标题区分)。
- 保留 market_signal_utils 的 import 与 calculate_score 调用、estimate_probability 校准逻辑、
  calibration_table 缺失防御。总是推送(指数层状态型)。不加多进程(仅4标的)。
"""

# ------------------ 参数 (env 可调) ------------------
CALIBRATION_CSV = os.environ.get('CALIBRATION_CSV', 'calibration_table.csv')
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '20'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False


# ------------------ 推送 (软导入) / 交易日 / 登录重试 / 工具 ------------------
def send_serverchan(title, content, sendkey=""):
    """Server酱推送: serverchan-sdk 软导入优先, requests 兜底"""
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content)
        print("📲 serverchan-sdk 推送成功")
        return True
    except Exception as e:
        print(f"  serverchan-sdk 失败, 回退 requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=10)
        return r.json().get("code") == 0
    except Exception as e:
        print(f"  requests 推送失败: {e}")
        return False


def is_trading_day():
    """本脚本为每日预测信号, 非交易日跳过(避免推送基于旧数据的'次日概率'误导)"""
    try:
        import akshare as ak
        d = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        print(f"  交易日历获取失败, 默认继续: {e}")
        return True


def _bs_login_ok(retries=5):
    global _BS_LOGGED
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _six(code_pref):
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


def _is_index(code_pref):
    """判断指数/个股: sh.000xxx(上证系列) 或 sz.399xxx(深证系列) 为指数, 其余为个股。
    注意 sz.000001 是平安银行(个股), 故 sz.000 不算指数。"""
    code6 = _six(code_pref)
    if code_pref.startswith('sh.') and code6.startswith('000'):
        return True
    if code_pref.startswith('sz.') and code6.startswith('399'):
        return True
    return False


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致脚本无限挂起"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date,
                                          frequency="d", adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 东财兜底 (区分指数/个股, 取最近一条 close+pctChg) ------------------
def _fetch_latest_em(code_pref):
    """东财兜底; 指数走 index_zh_a_hist, 个股走 stock_zh_a_hist; 返回最后一行 Series 或 None"""
    code6 = _six(code_pref)
    is_idx = _is_index(code_pref)
    end_y = datetime.now().strftime("%Y%m%d")
    start_y = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    import akshare as ak
    for attempt in range(2):
        try:
            if is_idx:
                d = ak.index_zh_a_hist(symbol=code6, period="daily", start_date=start_y, end_date=end_y)
            else:
                d = ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start_y,
                                       end_date=end_y, adjust="qfq")
            if d is None or d.empty or '收盘' not in d.columns:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close', '涨跌幅': 'pctChg'})
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['pctChg'] = pd.to_numeric(d['pctChg'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['pctChg']).sort_values('date')
            if d.empty:
                return None
            return d.iloc[-1]
        except Exception as e:
            print(f"    东财 {code_pref} 第{attempt+1}次失败: {e}")
            time.sleep(1 + attempt)
    return None


# ------------------ 取最新一条 (双源: baostock 优先 + 东财兜底; 含硬超时) ------------------
def fetch_latest(code):
    """只拉最近几天数据，取最新一条，轻量快速。返回 Series 或 None。"""
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    df = None

    # 路径1: baostock (主进程登录态; 含硬超时)
    if _BS_LOGGED:
        try:
            df = _query_with_timeout(code, "date,close,pctChg", start_date)
            if df is not None and not df.empty:
                df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['pctChg']).sort_values('date').reset_index(drop=True)
                if df.empty:
                    df = None
        except FutureTimeoutError:
            df = None
        except Exception:
            df = None

    # 路径2: 东财兜底
    if df is None:
        return _fetch_latest_em(code)

    return df.iloc[-1]


# ------------------ 校准概率 (原逻辑保留) ------------------
def estimate_probability(score, calibration):
    """在校准表里找到得分所在的分箱，返回历史经验频率"""
    for _, row in calibration.iterrows():
        if row['bin_left'] <= score <= row['bin_right']:
            return row['次日上涨频率'] * 100, row['样本数']
    calibration = calibration.copy()
    calibration['mid'] = (calibration['bin_left'] + calibration['bin_right']) / 2
    nearest = calibration.iloc[(calibration['mid'] - score).abs().argmin()]
    return nearest['次日上涨频率'] * 100, nearest['样本数']


# ------------------ 主程序 ------------------
def run_daily_signal():
    # 交易日判断 (每日预测信号, 非交易日跳过避免误导)
    if not is_trading_day():
        print("今日非A股交易日, 跳过每日信号 (避免推送基于旧数据的'次日概率')")
        return None

    if not os.path.exists(CALIBRATION_CSV):
        print(f"❌ 找不到 {CALIBRATION_CSV}，请先跑一次 build_calibration.py 生成校准表")
        return None

    calibration = pd.read_csv(CALIBRATION_CSV)

    # 主进程登录检查 (修复裸登录 -> 失败直接崩); 失败也继续, 走东财兜底
    if not _bs_login_ok():
        print("⚠️ baostock 主进程登录失败, 各标的将走东财兜底")

    # 拉数据 (try/except 包住, 修复原代码未捕获 ValueError 直接崩)
    try:
        sz = fetch_latest(INDEX_CODE)
        banks = {name: fetch_latest(code) for name, code in BANK_STOCKS.items()}
    except Exception as e:
        print(f"⚠️ 数据获取失败: {e}")
        sz, banks = None, {}

    if _BS_LOGGED:
        try:
            bs.logout()
        except Exception:
            pass

    # 防御性检查: 任一标的缺数据则优雅退出 (避免后续 KeyError/None 崩溃)
    need = ["建设银行", "工商银行", "招商银行"]
    if sz is None:
        print(f"⚠️ 上证指数({INDEX_CODE}) 双源均无数据, 本次跳过")
        return None
    missing = [n for n in need if banks.get(n) is None]
    if missing:
        print(f"⚠️ 银行股缺数据: {missing}, 本次跳过")
        return None

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

    # 留痕 (含 score/prob/verdict/各股数据, 审计/汇总用)
    tag = datetime.now().strftime('%Y%m%d')
    record = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bs_logged": _BS_LOGGED,
        "index": {"code": INDEX_CODE, "close": float(sz['close']), "pctChg": float(sz['pctChg'])},
        "banks": {n: {"close": float(banks[n]['close']), "pctChg": float(banks[n]['pctChg'])} for n in need},
        "score": float(score),
        "reasons": list(reasons),
        "prob_up_pct": round(float(prob), 1),
        "sample_size": int(sample_size),
        "verdict": verdict,
    }
    json_path = f"{OUTPUT_DIR}/daily_market_signal_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 结果已保存: {json_path}")

    # 总是推送 (指数层状态型: 每日大盘信号都有参考价值)
    if SERVERCHAN_KEY:
        title = f"大盘信号 {datetime.now().strftime('%m-%d')} | 上涨经验概率 {prob:.0f}%"
        content = (
            f"上证指数: {sz['close']} ({sz['pctChg']:.2f}%)\n"
            f"建设银行: {jh['close']} ({jh['pctChg']:.2f}%)\n\n"
            + "\n".join(reasons or ["（今日无明显信号触发）"])
            + f"\n\n{verdict}\n\n历史样本数: {int(sample_size)}（仅供参考，不构成投资建议）"
        )
        send_serverchan(title, content)

    return record


if __name__ == "__main__":
    run_daily_signal()
