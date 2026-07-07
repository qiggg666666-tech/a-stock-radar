import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
# 不依赖 pandas 内部的 _append（该私有方法在 pandas 3.0+ 也被移除了），
# 直接用 pd.concat 重新实现，兼容任意 pandas 版本
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
from serverchan_sdk import sc_send
import os
import time
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# ------------------ 阈值参数 ------------------
WEEK_THRESHOLD = 0.008   # 周线 MA5/MA20 差距阈值 0.8%
MONTH_THRESHOLD = 0.012  # 月线 MA5/MA20 差距阈值 1.2%
YEAR_THRESHOLD = 0.018   # 日线 MA20/MA250 差距阈值 1.8%
MIN_PRICE = 5            # 过滤低价股
SLEEP_PER_STOCK = 0.15   # 每只股票请求间隔，降低被限流风险
NUM_PROCESSES = 3        # 进程数。GitHub Actions 通常只有2核CPU，开太多进程无法真正并行，
                          # 反而会因为对baostock并发请求过多触发限流，3是相对稳妥的值
WEEKLY_ONLY_THRESHOLD = 0.015  # 宽松版：只看周线MA5/MA20，差距阈值放宽到1.5%（对应原来0.985的写法）


# 核心策略：年月周即将金叉（三线均未金叉，但差距收窄到阈值内）
# 现在返回详情字典（而非单纯bool），供后续打分使用
def strategy_triple_cross(df):
    try:
        if len(df) < 260:
            return None
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()
        df['date'] = pd.to_datetime(df['date'])

        # 重采样
        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('ME', on='date')['close'].last().dropna()  # pandas 2.2+ 用ME替代已废弃的M

        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()

        if len(w_ma20.dropna()) == 0 or len(m_ma20.dropna()) == 0:
            return None

        w_gap = (w_ma20.iloc[-1] - w_ma5.iloc[-1]) / w_ma20.iloc[-1]
        m_gap = (m_ma20.iloc[-1] - m_ma5.iloc[-1]) / m_ma20.iloc[-1]
        y_gap = (df['ma250'].iloc[-1] - df['ma20'].iloc[-1]) / df['ma250'].iloc[-1]

        # 统一语义：三条线都还未金叉（短均线仍在长均线下方），但差距已收窄到阈值内
        周即将 = (w_gap > 0) and (w_gap < WEEK_THRESHOLD)
        月即将 = (m_gap > 0) and (m_gap < MONTH_THRESHOLD)
        年即将 = (y_gap > 0) and (y_gap < YEAR_THRESHOLD)

        if not (周即将 and 月即将 and 年即将 and (df['close'].iloc[-1] > MIN_PRICE)):
            return None

        return {"w_gap": w_gap, "m_gap": m_gap, "y_gap": y_gap, "close": df['close'].iloc[-1]}
    except Exception:
        return None


def strategy_weekly_only(df):
    """
    补充信号：宽松版单周线即将金叉。
    只看周线 MA5/MA20（不要求月线、年线同步），且 MA5 正在抬头（比上一周更接近甚至反超）。
    门槛比"三线共振"低很多，触发会更频繁，仅作为辅助参考，不代表严格确认。
    """
    try:
        if len(df) < 150:
            return None
        d = df.copy()
        d['close'] = d['close'].astype(float)
        d['date'] = pd.to_datetime(d['date'])
        df_week = d.resample('W-FRI', on='date')['close'].last().dropna()
        w_ma5 = df_week.rolling(5).mean().dropna()
        w_ma20 = df_week.rolling(20).mean().dropna()

        if len(w_ma5) < 2 or len(w_ma20) < 2:
            return None

        latest_w5, prev_w5 = w_ma5.iloc[-1], w_ma5.iloc[-2]
        latest_w20 = w_ma20.iloc[-1]
        gap = (latest_w20 - latest_w5) / latest_w20

        if latest_w5 < latest_w20 and 0 <= gap < WEEKLY_ONLY_THRESHOLD and latest_w5 > prev_w5:
            return {"gap": gap, "close": d['close'].iloc[-1]}
        return None
    except Exception:
        return None


def calculate_rsi(series, period=14):
    """标准RSI计算，返回最新一期的RSI值"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_signal_score(gaps, df):
    """
    综合打分（0-100）：
    - 三线临界程度各占权重：差距越接近0（越快要金叉），分越高
    - RSI 30-55 区间（弱势企稳、尚未过热）加分；RSI>70（已经涨多了）减分
    """
    score = 0.0
    score += max(0, (1 - gaps["w_gap"] / WEEK_THRESHOLD)) * 30    # 周线，满分30
    score += max(0, (1 - gaps["m_gap"] / MONTH_THRESHOLD)) * 30   # 月线，满分30
    score += max(0, (1 - gaps["y_gap"] / YEAR_THRESHOLD)) * 20    # 年线，满分20

    rsi = calculate_rsi(df['close'])
    if rsi is not None:
        if 30 <= rsi <= 55:
            score += 20
        elif rsi > 70:
            score -= 10
        elif rsi < 20:
            score += 5

    return round(max(0, min(100, score)), 1)


# ------------------ 多进程扫描 ------------------
def _init_worker():
    """每个子进程启动时执行一次：独立登录baostock，带重试+错开延迟避免并发登录冲击服务器"""
    import random
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求可能持续报错")


def _process_one(args):
    """单只股票的抓取+判断逻辑，运行在子进程里"""
    code, name = args
    try:
        k_rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date="2020-01-01",
            adjustflag="2"
        )
        df = k_rs.get_data()
        time.sleep(SLEEP_PER_STOCK)

        signals = []
        hit = {"代码": code, "名称": name, "评分": None, "周线宽松评分": None}

        gaps = strategy_triple_cross(df)
        if gaps:
            signals.append("三线共振")
            hit["评分"] = calculate_signal_score(gaps, df)
            hit["最新价"] = round(float(gaps["close"]), 2)

        weekly_res = strategy_weekly_only(df)
        if weekly_res:
            signals.append("周线宽松")
            hit["周线宽松评分"] = round(max(0, (1 - weekly_res["gap"] / WEEKLY_ONLY_THRESHOLD)) * 100, 1)
            hit.setdefault("最新价", round(float(weekly_res["close"]), 2))

        if not signals:
            return None

        hit["信号"] = "+".join(signals)
        hit["_排序权重"] = hit["评分"] if hit["评分"] is not None else (hit["周线宽松评分"] or 0) * 0.5
        return hit
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_all_strategies(limit=None):
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ]
    bs.logout()

    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(code, code_to_name.get(code, "")) for code in target_stocks]

    results = []
    fail_count = 0
    print(f"开始检测 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（{res['信号']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("_排序权重", ascending=False).reset_index(drop=True)
        result_df = result_df.drop(columns=["_排序权重"])
    return result_df


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


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        parts = [f"- {row['名称']}（{row['代码']}）最新价 {row['最新价']} | 信号: {row['信号']}"]
        if row["评分"] is not None:
            parts.append(f"三线评分 {row['评分']}")
        if row["周线宽松评分"] is not None:
            parts.append(f"周线宽松评分 {row['周线宽松评分']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    df = run_all_strategies(limit=500)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"选股信号 命中 {len(df)} 只（三线共振/周线宽松）"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
