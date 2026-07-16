import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
from scipy.signal import argrelextrema
import baostock as bs
from serverchan_sdk import sc_send
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

"""
index_divergence_alert.py

A股主要指数 MACD 顶/底背离监控（日线+周线+月线三个周期）。
"""

INDEX_CODES = {
    '上证指数': 'sh.000001',
    '深证成指': 'sz.399001',
    '创业板指': 'sz.399006',
    '科创50': 'sh.000688',
}

LOOKBACK_DAYS = 1500  # 约4年日线数据，够重采样出足够的周线/月线样本
QUERY_TIMEOUT_SEC = 20
RECENT_DIVERGENCE_DAYS = {'daily': 30, 'weekly': 90, 'monthly': 365}
PEAK_ORDER = {'daily': 5, 'weekly': 3, 'monthly': 2}


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致脚本卡死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def fetch_daily_data(code):
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    df = _query_with_timeout(code, "date,close", start_date)
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)


def resample_to_period(daily_df, period):
    if period == 'daily':
        return daily_df.copy()
    rule = 'W-FRI' if period == 'weekly' else 'ME'
    resampled = daily_df.set_index('date')['close'].resample(rule).last().dropna()
    return resampled.reset_index()


def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    return df


def detect_divergences(df, period):
    """用scipy找局部高低点，逐个比较相邻高点/低点，检测顶背离和底背离"""
    if df is None or len(df) < 30:
        return [], []

    df = calculate_macd(df)
    price = df['close'].values
    dif = df['DIF'].values
    dates = df['date'].values
    order = PEAK_ORDER[period]

    top_div, bottom_div = [], []

    peaks = argrelextrema(price, np.greater_equal, order=order)[0]
    for i in range(1, len(peaks)):
        p1, p2 = peaks[i - 1], peaks[i]
        if price[p2] > price[p1] and dif[p2] < dif[p1]:
            top_div.append({
                'prev_date': pd.Timestamp(dates[p1]), 'prev_price': round(float(price[p1]), 2),
                'date': pd.Timestamp(dates[p2]), 'price': round(float(price[p2]), 2),
            })

    troughs = argrelextrema(price, np.less_equal, order=order)[0]
    for i in range(1, len(troughs)):
        p1, p2 = troughs[i - 1], troughs[i]
        if price[p2] < price[p1] and dif[p2] > dif[p1]:
            bottom_div.append({
                'prev_date': pd.Timestamp(dates[p1]), 'prev_price': round(float(price[p1]), 2),
                'date': pd.Timestamp(dates[p2]), 'price': round(float(price[p2]), 2),
            })

    return top_div, bottom_div


def analyze_all():
    results = {}
    for name, code in INDEX_CODES.items():
        try:
            daily_df = fetch_daily_data(code)
            print(f"   [诊断] {name}({code}) 原始日线数据条数: {len(daily_df)}")

            if daily_df.empty:
                print(f"⚠️ {name}({code}) 完全没有获取到数据（可能baostock不支持该指数代码），跳过")
                continue
            if len(daily_df) < 30:
                print(f"⚠️ {name}({code}) 数据过少（{len(daily_df)}条），跳过")
                continue

            results[name] = {'latest_close': round(float(daily_df['close'].iloc[-1]), 2), 'periods': {}}
            latest_date = daily_df['date'].iloc[-1]

            for period in ['daily', 'weekly', 'monthly']:
                period_df = resample_to_period(daily_df, period)
                top_div, bottom_div = detect_divergences(period_df, period)

                recent_days = RECENT_DIVERGENCE_DAYS[period]
                recent_top = [d for d in top_div if (latest_date - d['date']).days < recent_days]
                recent_bottom = [d for d in bottom_div if (latest_date - d['date']).days < recent_days]

                results[name]['periods'][period] = {
                    'top_count': len(top_div), 'bottom_count': len(bottom_div),
                    'recent_top': recent_top[-1] if recent_top else None,
                    'recent_bottom': recent_bottom[-1] if recent_bottom else None,
                }

            print(f"🔍 {name}: 最新收盘 {results[name]['latest_close']}")
            for period, res in results[name]['periods'].items():
                print(f"   {period}: 顶背离{res['top_count']}次(近期{'有' if res['recent_top'] else '无'}), "
                      f"底背离{res['bottom_count']}次(近期{'有' if res['recent_bottom'] else '无'})")

        except FutureTimeoutError:
            print(f"⚠️ {name}({code}) 查询超时，跳过")
        except Exception as e:
            print(f"⚠️ {name}({code}) 分析失败: {e}")

    return results


def build_report(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"指数MACD背离监控（日/周/月） - {now}", ""]

    alerts = []
    for name, res in results.items():
        for period, p in res['periods'].items():
            if p['recent_top']:
                d = p['recent_top']
                alerts.append(
                    f"⚠️ {name}[{period}] 顶背离：{d['prev_date'].date()}→{d['date'].date()} "
                    f"价格{d['prev_price']}→{d['price']}创新高，MACD走弱"
                )
            if p['recent_bottom']:
                d = p['recent_bottom']
                alerts.append(
                    f"🟢 {name}[{period}] 底背离：{d['prev_date'].date()}→{d['date'].date()} "
                    f"价格{d['prev_price']}→{d['price']}创新低，MACD走强"
                )

    if alerts:
        lines.append(f"本次检测到 {len(alerts)} 条近期背离信号：")
        lines.extend(f"- {a}" for a in alerts)
    else:
        lines.append("本次未检测到任何近期背离信号")

    lines.append("")
    lines.append("全部指数概览：")
    for name, res in results.items():
        summary = ", ".join(
            f"{p}(顶{v['top_count']}/底{v['bottom_count']})" for p, v in res['periods'].items()
        )
        lines.append(f"- {name}：收盘{res['latest_close']} | 历史背离次数 {summary}")

    lines.append("")
    lines.append("⚠️ 背离是技术面预警信号，不代表价格必然反转，仅供参考，不构成投资建议")

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
    results = analyze_all()
    bs.logout()

    if results:
        report = build_report(results)
        print("\n" + report)

        sendkey = os.getenv("SENDKEY")
        if sendkey:
            has_alert = any(
                p['recent_top'] or p['recent_bottom']
                for res in results.values() for p in res['periods'].values()
            )
            title = "⚠️ 指数背离预警" if has_alert else "指数背离监控（无新信号）"
            send_to_serverchan(sendkey, title, report)
    else:
        print("本次未获取到任何指数的有效数据")
