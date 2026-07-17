import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import baostock as bs
from serverchan_sdk import sc_send
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

"""
index_support_resistance.py

上证指数 成交量加权支撑/阻力位 + 斐波那契回撤位。

⚠️ 相比原始版本的修复：
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
"""

INDEX_CODE = 'sh.000001'
INDEX_NAME = '上证指数'

PERIODS = 800
LOOKBACK_DAYS = 1200
MIN_DISTANCE = 10
PROMINENCE = 35
MERGE_THRESHOLD = 20
VOLUME_WINDOW = 5
N_CLUSTERS = 10
TOP_N_LEVELS = 15
NEAR_CURRENT_THRESHOLD = 80
QUERY_TIMEOUT_SEC = 20


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致脚本卡死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def fetch_index_data(code, periods=PERIODS):
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    df = _query_with_timeout(code, "date,close,high,low,volume", start_date)
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    for col in ['close', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'high', 'low', 'volume']).sort_values('date').reset_index(drop=True)

    if len(df) > periods:
        df = df.iloc[-periods:].reset_index(drop=True)
    return df


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


def build_report(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"{INDEX_NAME} 成交量加权支撑阻力位 - {now}", ""]
    lines.append(f"数据范围: {results['data_start']} ~ {results['data_end']}")
    lines.append(f"最新收盘: {results['current_close']}")
    lines.append("")

    current = results['current_close']
    lines.append("🏆 成交量加权关键位（分数越高越重要）：")
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

    return "\n".join(lines)


def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}，message={response.get('message')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


if __name__ == "__main__":
    bs.login()
    try:
        results = analyze()
    except FutureTimeoutError:
        print(f"⚠️ {INDEX_NAME} 查询超时")
        results = None
    except Exception as e:
        print(f"⚠️ 分析失败: {e}")
        results = None
    bs.logout()

    if results:
        report = build_report(results)
        print("\n" + report)

        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%m-%d")
            send_to_serverchan(sendkey, f"{INDEX_NAME}支撑阻力位 {now}", report)
    else:
        print("本次未能生成分析结果")
