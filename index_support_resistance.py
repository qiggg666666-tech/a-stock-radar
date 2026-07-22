import os
import time
import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.signal import find_peaks        # 核心算法依赖(找峰谷), 硬导入正确: 缺它不能跑, fail-fast
from sklearn.cluster import KMeans         # 核心算法依赖(价位聚类), 硬导入正确: 缺它不能跑, fail-fast
import baostock as bs

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

"""
index_support_resistance.py

上证指数 成交量加权支撑/阻力位 + 斐波那契回撤位。

⚠️ 相比原始版本的修复（原作者已修, v2 全部保留不回退）：
1. 数据源从 Ashare（第三方小库，维护程度不明、无超时保护）换成 baostock，
   与仓库其他脚本数据源一致。
2. 修了一个会导致崩溃的bug：原代码 fib_ratios 变量只在"检测到峰值和谷值"的分支里定义，
   如果某次没检测到峰谷（数据窗口短/参数偏严格都可能发生），走else分支后
   fib_ratios 根本不存在，后面打印斐波那契位时会直接 NameError 崩溃。
   现在 fib_ratios 统一在最外层定义，不再依赖分支条件。
3. merge_weighted 合并相近价位时用成交量分数做加权平均，如果两个待合并点的
   成交量分数恰好都是0会除零崩溃，加了防护。
4. 去掉了 mplfinance 画图和CSV文件输出，改成纯文字+数值，通过Server酱推送，
   与仓库其他模块风格一致。

⚠️ 依赖提醒：这个脚本用到了 scikit-learn（KMeans聚类），
   如果 requirements.txt 里还没有，需要加一行 scikit-learn。

【v2 升级说明】
- 修复主进程裸登录(登录失败原代码静默无产出, 误以为"无关键位"实为"无数据"); 加 _bs_login_ok
  重试, 且登录失败也继续 -> 走东财指数日线兜底, 不空跑。
- 加东财指数日线兜底 index_zh_a_hist (有 close/high/low/volume 四列, find_peaks/KMeans/
  成交量加权全可用, 故东财路径全功能不降级; 指数代码 baostock 'sh.000001' <-> 东财 6位 自动转换)。
- 修复 sc_send 硬导入(软导入+requests兜底); fetch_index_data 的 df.empty 对 None 的
  AttributeError 防御; 结果存 output/ (json 含关键位+斐波那契+数据范围 + md)。
- 关键位记号 🏆 -> 📏 (避免与 strong_continuation 的🏆强势板块撞色; 斐波那契📐/★近期价保留)。
  注: 跨脚本 emoji 不保证全局唯一, 靠推送标题/脚本上下文区分。
- 保留 scipy/sklearn 硬导入: find_peaks/KMeans 是核心算法依赖, 非"锦上添花", 硬导入正确。
- 保留原作者全部修复(fib_ratios/除零/换数据源/去画图)与全部算法逻辑, 一字未动。
- 本脚本属"指数层"(关键价位地图), 非选股/选板块, 故不并入选股记号体系。
- 总是推送(指数层状态型: 每日关键位地图都有参考价值)。不加交易日判断(历史价格统计,
  非交易日有效)。不加多进程(仅1指数)。单指数保持, 加 INDEX_CODE 旋钮可调。
"""

# ------------------ 参数 (全部 env 可调) ------------------
INDEX_CODE = os.environ.get('INDEX_CODE', 'sh.000001')   # 默认上证指数; 可改如 sh.000300(沪深300)
INDEX_NAME = os.environ.get('INDEX_NAME', '上证指数')

PERIODS = int(os.environ.get('PERIODS', '800'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '1200'))
MIN_DISTANCE = int(os.environ.get('MIN_DISTANCE', '10'))
PROMINENCE = float(os.environ.get('PROMINENCE', '35'))
MERGE_THRESHOLD = float(os.environ.get('MERGE_THRESHOLD', '20'))
VOLUME_WINDOW = int(os.environ.get('VOLUME_WINDOW', '5'))
N_CLUSTERS = int(os.environ.get('N_CLUSTERS', '10'))
TOP_N_LEVELS = int(os.environ.get('TOP_N_LEVELS', '15'))
NEAR_CURRENT_THRESHOLD = float(os.environ.get('NEAR_CURRENT_THRESHOLD', '80'))
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '20'))

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False   # 主进程 baostock 登录态标志


# ------------------ 推送 (软导入) / 登录重试 / 工具 ------------------
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
    """baostock 带前缀 -> 6位 (东财用)"""
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致脚本卡死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 东财指数日线兜底 (close/high/low/volume 全有, 全功能不降级) ------------------
def _fetch_index_em(sym6, start_y):
    """东财指数日线兜底; 返回 date/close/high/low/volume df 或 None; 指数代码用6位"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak_index_zh_a_hist(sym6, start_y, end_y)
            if d is None or d.empty or '收盘' not in d.columns:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close', '最高': 'high',
                                  '最低': 'low', '成交量': 'volume'})
            for c in ['close', 'high', 'low', 'volume']:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close', 'high', 'low', 'volume']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'close', 'high', 'low', 'volume'] if c in d.columns]
            return d[cols] if len(d) >= 50 else None
        except Exception as e:
            print(f"    东财指数 {sym6} 第{attempt+1}次失败: {e}")
            time.sleep(1 + attempt)
    return None


def ak_index_zh_a_hist(sym6, start_y, end_y):
    """隔离 akshare 调用, 便于版本差异时单点调整"""
    import akshare as ak
    return ak.index_zh_a_hist(symbol=sym6, period="daily", start_date=start_y, end_date=end_y)


# ------------------ 取指数日线 (双源: baostock 优先 + 东财兜底) ------------------
def fetch_index_data(code, periods=PERIODS):
    sym6 = _six(code)
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None

    # 路径1: baostock (主进程登录态; 未登录则跳过此路径)
    if _BS_LOGGED:
        try:
            df = _query_with_timeout(code, "date,close,high,low,volume", start_dash)
            if df is None or df.empty:
                df = None
        except FutureTimeoutError:
            df = None   # 超时不冒泡, 走东财兜底
        except Exception:
            df = None

    # 路径2: 东财兜底 (全功能)
    if df is None or (hasattr(df, 'empty') and df.empty):
        df = _fetch_index_em(sym6, start_y)

    if df is None or df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    for col in ['close', 'high', 'low', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'high', 'low', 'volume']).sort_values('date').reset_index(drop=True)

    if len(df) > periods:
        df = df.iloc[-periods:].reset_index(drop=True)
    return df


# ------------------ 算法逻辑 (原作者全部修复保留, 一字未动) ------------------
def volume_weighted_score(price_idx, volume_series, window=VOLUME_WINDOW):
    start = max(0, price_idx - window)
    end = min(len(volume_series), price_idx + window + 1)
    return volume_series[start:end].sum()


def get_volume_weighted_kmeans(df, n_clusters=N_CLUSTERS):
    prices = df['close'].values.reshape(-1, 1)
    actual_clusters = min(n_clusters, len(np.unique(prices)))
    if actual_clusters < 2:
        return []

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(prices)
    centers = kmeans.cluster_centers_.flatten()

    weighted = []
    for i, center in enumerate(centers):
        cluster_mask = labels == i
        cluster_vol = df['volume'][cluster_mask].sum()
        weighted.append((round(float(center), 2), float(cluster_vol)))
    return sorted(weighted, key=lambda x: x[1], reverse=True)


def merge_weighted(levels, thresh=MERGE_THRESHOLD):
    """按价格排序后合并相近价位，用成交量分数做加权平均；两边权重都为0时退化为简单平均，避免除零"""
    levels = sorted(levels, key=lambda x: x[0])
    merged = []
    for lvl in levels:
        if not merged or abs(lvl[0] - merged[-1][0]) > thresh:
            merged.append(list(lvl))
        else:
            total_weight = merged[-1][1] + lvl[1]
            if total_weight > 0:
                merged[-1][0] = (merged[-1][0] * merged[-1][1] + lvl[0] * lvl[1]) / total_weight
            else:
                merged[-1][0] = (merged[-1][0] + lvl[0]) / 2
            merged[-1][1] += lvl[1]
    return sorted(merged, key=lambda x: x[1], reverse=True)


def analyze():
    df = fetch_index_data(INDEX_CODE)
    if df.empty or len(df) < 50:
        print(f"⚠️ {INDEX_NAME} 数据不足，无法分析")
        return None

    print(f"数据范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    current_close = float(df['close'].iloc[-1])
    print(f"最新收盘: {current_close:.2f}")

    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    peaks, _ = find_peaks(high, distance=MIN_DISTANCE, prominence=PROMINENCE)
    valleys, _ = find_peaks(-low, distance=MIN_DISTANCE, prominence=PROMINENCE)

    swing_levels = []
    for p in peaks:
        score = volume_weighted_score(p, volume)
        swing_levels.append((float(high[p]), float(score), 'Resistance'))
    for v in valleys:
        score = volume_weighted_score(v, volume)
        swing_levels.append((float(low[v]), float(score), 'Support'))

    kmeans_weighted = get_volume_weighted_kmeans(df)

    # fib_ratios 统一在最外层定义 (原作者修复: 避免无峰谷时 NameError)
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    fib_levels = []
    if len(peaks) > 0 and len(valleys) > 0:
        sh = float(high[peaks[-1]])
        sl = float(low[valleys[-1]])
        diff = sh - sl
        fib_levels = [round(sh - diff * r, 2) for r in fib_ratios]

    all_levels = list(swing_levels)
    for price, score in kmeans_weighted:
        all_levels.append((price, score, 'KMeans'))

    final_levels = merge_weighted(all_levels) if all_levels else []

    return {
        'current_close': round(current_close, 2),
        'data_start': df['date'].iloc[0].date(),
        'data_end': df['date'].iloc[-1].date(),
        'levels': final_levels[:TOP_N_LEVELS],
        'fib_ratios': fib_ratios,
        'fib_levels': fib_levels,
    }


# ------------------ 报告 (关键位 🏆->📏; 斐波那契📐/★近期价保留) ------------------
def build_report(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"{INDEX_NAME} 成交量加权支撑阻力位 - {now}", ""]
    lines.append(f"数据范围: {results['data_start']} ~ {results['data_end']}")
    lines.append(f"最新收盘: {results['current_close']}")
    lines.append("")

    current = results['current_close']
    lines.append("📏 成交量加权关键位（分数越高越重要）：")
    if results['levels']:
        for price, score, typ in results['levels']:
            marker = " ★近期价" if abs(price - current) < NEAR_CURRENT_THRESHOLD else ""
            lines.append(f"- {price:.2f} | 分数 {score:,.0f} | {typ}{marker}")
    else:
        lines.append("本次未检测到明显的关键位")

    lines.append("")
    lines.append("📐 斐波那契回撤位：")
    if results['fib_levels']:
        for r, p in zip(results['fib_ratios'], results['fib_levels']):
            lines.append(f"- {r * 100:.1f}% → {p:.2f}")
    else:
        lines.append("本次数据窗口内未检测到明显的峰谷，无法计算斐波那契位")

    lines.append("")
    lines.append("⚠️ 支撑/阻力位基于历史价格及成交量统计，不代表未来必然在此止跌/止涨，仅供参考，不构成投资建议")
    lines.append("*注: 关键位记号📏(原🏆, 避免与强势板块🏆撞色); 跨脚本emoji靠标题区分。*")

    return "\n".join(lines)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"{INDEX_NAME} 成交量加权支撑阻力位 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"双源(baostock+东财指数日线); 总是推送(指数层状态型); 不加交易日判断")
    print("=" * 70)

    # 主进程登录检查 (修复裸登录 -> 静默无产出); 失败也继续, 走东财兜底
    if not _bs_login_ok():
        print(f"⚠️ baostock 主进程登录失败, {INDEX_NAME} 将走东财指数日线兜底")

    try:
        results = analyze()
    except FutureTimeoutError:
        print(f"⚠️ {INDEX_NAME} 查询超时")
        results = None
    except Exception as e:
        print(f"⚠️ 分析失败: {e}")
        results = None

    if _BS_LOGGED:
        try:
            bs.logout()
        except Exception:
            pass

    # 留痕 (含关键位+斐波那契+数据范围, 审计/汇总用)
    tag = datetime.now().strftime('%Y%m%d')
    record = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "index_name": INDEX_NAME,
        "index_code": INDEX_CODE,
        "bs_logged": _BS_LOGGED,
        "has_data": results is not None,
        "result": results,
    }
    json_path = f"{OUTPUT_DIR}/index_support_resistance_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    if results:
        report = build_report(results)
        print("\n" + report)
        with open(f"{OUTPUT_DIR}/index_support_resistance_{tag}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📁 结果已保存: {OUTPUT_DIR}/index_support_resistance_*_{tag}.*")

        # 总是推送 (指数层状态型: 每日关键位地图都有参考价值)
        if SERVERCHAN_KEY:
            now = datetime.now().strftime("%m-%d")
            send_serverchan(f"{INDEX_NAME}支撑阻力位 {now}", report)
    else:
        print("本次未能生成分析结果 (双源均失败或数据不足)")
