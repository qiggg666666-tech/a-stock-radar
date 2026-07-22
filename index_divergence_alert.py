import os
import time
import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema   # 核心算法依赖(背离检测), 硬导入正确: 缺它脚本不能跑, fail-fast
import baostock as bs

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

"""
index_divergence_alert.py

A股主要指数 MACD 顶/底背离监控（日线+周线+月线三个周期）。

【v2 升级说明】
- 修复主进程裸登录(登录失败原代码静默全空, 误以为"无背离"实为"无数据"); 加 _bs_login_ok 重试,
  且登录失败也继续 -> 每指数走东财兜底, 不空跑。
- 加东财指数日线兜底 index_zh_a_hist (有 close, 够算 MACD 背离, 故东财路径全功能不降级;
  指数代码格式 baostock 'sh.000001' <-> 东财 6位 '000001' 自动转换)。
- 修复 sc_send 硬导入(软导入+requests兜底); _query_with_timeout 返回 None 时 df.empty 的
  AttributeError 防御; 结果存 output/ (json 含无背离事实+每周期明细+原始日线长度诊断 + md)。
- 底背离记号 🟢 -> 📈 (避免与 sector_momentum 的🟢双优撞色; 顶背离⚠️保留)。
  注: 跨脚本 emoji 不保证全局唯一, 靠推送标题/脚本上下文区分。
- 本脚本属"指数层"(盯指数拐点), 非选股/选板块/资金股东监控, 故不并入选股记号体系。
- 总是推送(状态型监控: "无背离"也是有价值的市场状态; 与事件型监控"仅触发推送"哲学不同)。
- 不加交易日判断: 背离由历史K线算, 非交易日仍有意义。不加多进程: 仅4指数, 串行最简。
- 保留 scipy 硬导入: argrelextrema 是背离算法核心, 非"锦上添花", 硬导入正确。
"""

# ------------------ 参数 (全部 env 可调) ------------------
INDEX_CODES = {           # 指数名单(名称 -> baostock格式代码); 要加指数直接改此 dict
    '上证指数': 'sh.000001',
    '深证成指': 'sz.399001',
    '创业板指': 'sz.399006',
    '科创50': 'sh.000688',
}
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '1500'))   # 约4年日线, 够重采样周/月线
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '20'))
RECENT_DIVERGENCE_DAYS = {'daily': 30, 'weekly': 90, 'monthly': 365}
PEAK_ORDER = {'daily': 5, 'weekly': 3, 'monthly': 2}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False   # 主进程 baostock 登录态标志 (供 fetch_daily_data 判断是否走 baostock 路径)


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


# ------------------ 东财指数日线兜底 (有 close, 够算背离, 全功能不降级) ------------------
def _fetch_index_em(sym6, start_y):
    """东财指数日线兜底; 返回 date/close df 或 None; 指数代码用6位"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak_index_zh_a_hist(sym6, start_y, end_y)
            if d is None or d.empty or '收盘' not in d.columns:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close'})
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            return d[['date', 'close']] if len(d) >= 30 else None
        except Exception as e:
            print(f"    东财指数 {sym6} 第{attempt+1}次失败: {e}")
            time.sleep(1 + attempt)
    return None


def ak_index_zh_a_hist(sym6, start_y, end_y):
    """隔离 akshare 调用, 便于版本差异时单点调整"""
    import akshare as ak
    return ak.index_zh_a_hist(symbol=sym6, period="daily", start_date=start_y, end_date=end_y)


# ------------------ 取指数日线 (双源: baostock 优先 + 东财兜底) ------------------
def fetch_daily_data(code):
    sym6 = _six(code)
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None

    # 路径1: baostock (主进程登录态; 未登录则跳过此路径)
    if _BS_LOGGED:
        try:
            df = _query_with_timeout(code, "date,close", start_dash)
            if df is None or df.empty:
                df = None
        except FutureTimeoutError:
            df = None   # 超时不冒泡, 走东财兜底 (比"跳过整个指数"更鲁棒)
        except Exception:
            df = None

    # 路径2: 东财兜底 (全功能)
    if df is None or (hasattr(df, 'empty') and df.empty):
        df = _fetch_index_em(sym6, start_y)

    if df is None or df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)


# ------------------ 重采样 / MACD / 背离检测 (逻辑一字未动) ------------------
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


# ------------------ 全指数分析 ------------------
def analyze_all():
    results = {}
    for name, code in INDEX_CODES.items():
        try:
            daily_df = fetch_daily_data(code)
            print(f"   [诊断] {name}({code}) 原始日线数据条数: {len(daily_df)}")

            if daily_df.empty:
                print(f"⚠️ {name}({code}) 双源均未获取到数据，跳过")
                continue
            if len(daily_df) < 30:
                print(f"⚠️ {name}({code}) 数据过少（{len(daily_df)}条），跳过")
                continue

            results[name] = {'latest_close': round(float(daily_df['close'].iloc[-1]), 2),
                             'daily_rows': len(daily_df), 'periods': {}}
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


# ------------------ 报告 (顶⚠️保留, 底🟢->📈) ------------------
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
                    f"📈 {name}[{period}] 底背离：{d['prev_date'].date()}→{d['date'].date()} "
                    f"价格{d['prev_price']}→{d['price']}创新低，MACD走强"
                )

    if alerts:
        lines.append(f"本次检测到 {len(alerts)} 条近期背离信号：")
        lines.extend(f"- {a}" for a in alerts)
    else:
        lines.append("本次未检测到任何近期背离信号（趋势延续、无拐点预警）")

    lines.append("")
    lines.append("全部指数概览：")
    for name, res in results.items():
        summary = ", ".join(
            f"{p}(顶{v['top_count']}/底{v['bottom_count']})" for p, v in res['periods'].items()
        )
        lines.append(f"- {name}：收盘{res['latest_close']} | 历史背离次数 {summary}")

    lines.append("")
    lines.append("⚠️ 背离是技术面预警信号，不代表价格必然反转，仅供参考，不构成投资建议")
    lines.append("*注: 底背离记号📈(原🟢, 避免与板块双优🟢撞色); 跨脚本emoji靠标题区分。*")

    return "\n".join(lines)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"指数MACD背离监控 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 指数{len(INDEX_CODES)}个 | 周期 日/周/月")
    print(f"双源(baostock+东财指数日线); 总是推送(状态型监控); 不加交易日判断")
    print("=" * 70)

    # 主进程登录检查 (修复裸登录 -> 静默全空); 失败也继续, 走东财兜底
    if not _bs_login_ok():
        print("⚠️ baostock 主进程登录失败, 各指数将走东财指数日线兜底")

    results = analyze_all()

    if _BS_LOGGED:
        try:
            bs.logout()
        except Exception:
            pass

    # 留痕 (含"无背离"事实 + 每周期明细 + 原始日线长度诊断, 审计/汇总用)
    tag = datetime.now().strftime('%Y%m%d')
    has_alert = any(
        p['recent_top'] or p['recent_bottom']
        for res in results.values() for p in res['periods'].values()
    ) if results else False
    record = {
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "has_alert": has_alert,
        "bs_logged": _BS_LOGGED,
        "per_index": results,
    }
    json_path = f"{OUTPUT_DIR}/index_divergence_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    if results:
        report = build_report(results)
        print("\n" + report)
        with open(f"{OUTPUT_DIR}/index_divergence_{tag}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📁 结果已保存: {OUTPUT_DIR}/index_divergence_*_{tag}.* (has_alert={has_alert})")

        # 总是推送 (状态型监控: 无背离也是有价值的市场状态)
        if SERVERCHAN_KEY:
            title = "⚠️ 指数背离预警" if has_alert else "指数背离监控（无新信号·趋势延续）"
            send_serverchan(title, report)
    else:
        print("本次未获取到任何指数的有效数据 (双源均失败)")
