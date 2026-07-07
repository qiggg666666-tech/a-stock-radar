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
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 板块选择参数 ------------------
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
SECTOR_STOCK_LIMIT = 800   # 参与板块打分的股票池大小
TOP_N_SECTORS = 5
MIN_SECTOR_SCORE = 55
MIN_STOCKS_IN_SECTOR = 5

# ------------------ 个股信号参数（与main.py保持一致）------------------
WEEK_THRESHOLD = 0.008
MONTH_THRESHOLD = 0.012
YEAR_THRESHOLD = 0.018
WEEKLY_ONLY_THRESHOLD = 0.015
MIN_PRICE = 5


def _init_worker():
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
    print("⚠️ 子进程登录多次重试后仍失败")


# ================== 第一阶段：板块打分 + 自动选择 ==================

def _get_stock_status(args):
    code, name = args
    try:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        k_rs = bs.query_history_k_data_plus(
            code, "date,close,pctChg",
            start_date=start_date,
            adjustflag="2"
        )
        df = k_rs.get_data()
        time.sleep(SLEEP_PER_STOCK)
        if df.empty or len(df) < 20:
            return None
        df['close'] = df['close'].astype(float)
        df['pctChg'] = df['pctChg'].astype(float)
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        latest_close = df['close'].iloc[-1]
        latest_chg = df['pctChg'].iloc[-1]
        return {"代码": code, "名称": name, "涨跌幅%": latest_chg, "站上MA20": latest_close > ma20}
    except Exception:
        return None


def score_and_select_sectors():
    print("=" * 60)
    print("第一阶段：板块打分 + 自动选择强势板块")
    print("=" * 60)

    bs.login()
    rs = bs.query_stock_industry()
    industry_df = rs.get_data()
    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]
    bs.logout()

    if SECTOR_STOCK_LIMIT:
        industry_df = industry_df.iloc[:SECTOR_STOCK_LIMIT]

    tasks = [(row['code'], row['code_name']) for _, row in industry_df.iterrows()]
    code_to_industry = dict(zip(industry_df['code'], industry_df['industry']))

    print(f"开始获取 {len(tasks)} 只股票的状态数据...")
    stock_status = []
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="板块打分进度", unit="只")
        for res in pool.imap_unordered(_get_stock_status, tasks):
            if res:
                res["行业"] = code_to_industry.get(res["代码"], "未知")
                stock_status.append(res)
            pbar.update(1)

    if not stock_status:
        print("未获取到有效数据")
        return pd.DataFrame(), []

    status_df = pd.DataFrame(stock_status)

    grouped = status_df.groupby("行业").agg(
        股票数=("代码", "count"),
        站上MA20占比=("站上MA20", "mean"),
        平均涨跌幅=("涨跌幅%", "mean")
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]

    max_chg = grouped["平均涨跌幅"].abs().max() or 1
    grouped["板块强度分"] = (
        grouped["站上MA20占比"] * 60 +
        (grouped["平均涨跌幅"] / max_chg) * 40 + 40
    ).clip(0, 100).round(1)
    grouped = grouped.sort_values("板块强度分", ascending=False).reset_index(drop=True)

    print("\n全部板块强度排行：")
    print(grouped[["行业", "股票数", "板块强度分"]].to_string(index=False))

    top = grouped.head(TOP_N_SECTORS)
    selected = top[top["板块强度分"] >= MIN_SECTOR_SCORE]

    if selected.empty:
        print(f"\n没有板块强度分达到{MIN_SECTOR_SCORE}分门槛，本轮不选任何板块，跳过个股筛选")
        return grouped, []

    selected_names = selected["行业"].tolist()
    print(f"\n自动选中板块：{selected_names}")

    candidate_codes = status_df[status_df["行业"].isin(selected_names)]
    candidates = list(zip(candidate_codes["代码"], candidate_codes["名称"]))
    print(f"候选股票池：{len(candidates)} 只（来自 {len(selected_names)} 个强势板块）")

    return grouped, candidates


# ================== 第二阶段：个股信号筛选（复用main.py的策略）==================

def strategy_triple_cross(df):
    try:
        if len(df) < 260:
            return None
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()
        df['date'] = pd.to_datetime(df['date'])
        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('ME', on='date')['close'].last().dropna()
        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()
        if len(w_ma20.dropna()) == 0 or len(m_ma20.dropna()) == 0:
            return None
        w_gap = (w_ma20.iloc[-1] - w_ma5.iloc[-1]) / w_ma20.iloc[-1]
        m_gap = (m_ma20.iloc[-1] - m_ma5.iloc[-1]) / m_ma20.iloc[-1]
        y_gap = (df['ma250'].iloc[-1] - df['ma20'].iloc[-1]) / df['ma250'].iloc[-1]
        周即将 = (w_gap > 0) and (w_gap < WEEK_THRESHOLD)
        月即将 = (m_gap > 0) and (m_gap < MONTH_THRESHOLD)
        年即将 = (y_gap > 0) and (y_gap < YEAR_THRESHOLD)
        if not (周即将 and 月即将 and 年即将 and (df['close'].iloc[-1] > MIN_PRICE)):
            return None
        return {"w_gap": w_gap, "m_gap": m_gap, "y_gap": y_gap, "close": df['close'].iloc[-1]}
    except Exception:
        return None


def strategy_weekly_only(df):
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
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_signal_score(gaps, df):
    score = 0.0
    score += max(0, (1 - gaps["w_gap"] / WEEK_THRESHOLD)) * 30
    score += max(0, (1 - gaps["m_gap"] / MONTH_THRESHOLD)) * 30
    score += max(0, (1 - gaps["y_gap"] / YEAR_THRESHOLD)) * 20
    rsi = calculate_rsi(df['close'])
    if rsi is not None:
        if 30 <= rsi <= 55:
            score += 20
        elif rsi > 70:
            score -= 10
        elif rsi < 20:
            score += 5
    return round(max(0, min(100, score)), 1)


def _process_signal(args):
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


def screen_candidates(candidates):
    print("\n" + "=" * 60)
    print(f"第二阶段：在 {len(candidates)} 只候选股票中筛选三线共振/周线宽松信号")
    print("=" * 60)

    if not candidates:
        return pd.DataFrame()

    results = []
    fail_count = 0
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(candidates), desc="信号筛选进度", unit="只")
        for res in pool.imap_unordered(_process_signal, candidates):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（{res['信号']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"筛选完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("_排序权重", ascending=False).reset_index(drop=True)
        result_df = result_df.drop(columns=["_排序权重"])
    return result_df


# ================== 推送 ==================

def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


def build_push_content(sector_df, signal_df):
    lines = []
    if not sector_df.empty:
        lines.append("【强势板块TOP5】")
        for _, row in sector_df.head(5).iterrows():
            lines.append(f"- {row['行业']}：强度分 {row['板块强度分']}")
        lines.append("")
    lines.append("【个股信号】")
    if signal_df.empty:
        lines.append("本次候选池内未筛出符合条件的股票")
    else:
        for _, row in signal_df.iterrows():
            parts = [f"- {row['名称']}（{row['代码']}）最新价 {row['最新价']} | 信号: {row['信号']}"]
            if row["评分"] is not None:
                parts.append(f"三线评分{row['评分']}")
            if row["周线宽松评分"] is not None:
                parts.append(f"周线宽松评分{row['周线宽松评分']}")
            lines.append(" | ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    sector_df, candidates = score_and_select_sectors()
    signal_df = screen_candidates(candidates)

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    if not signal_df.empty:
        print(signal_df.to_string(index=False))
    else:
        print("本次候选池内未筛出符合条件的股票")

    sendkey = os.getenv("SENDKEY")
    if sendkey:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"板块优选+信号筛选 命中{len(signal_df)}只" if not signal_df.empty else "板块优选+信号筛选 本轮无命中"
        content = f"扫描时间：{now}\n\n" + build_push_content(sector_df, signal_df)
        send_to_serverchan(sendkey, title, content)

    if not signal_df.empty:
        signal_df.to_csv("pipeline_result.csv", index=False, encoding="utf-8-sig")
        print("\n结果已保存到 pipeline_result.csv")
