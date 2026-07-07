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
import time
import multiprocessing as mp
from datetime import datetime, timedelta
from tqdm import tqdm

# ------------------ 参数 ------------------
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
LOOKBACK_DAYS = 400

MAJOR_INDEXES = {
    "sh.000001": "上证指数",
    "sz.399001": "深证成指",
    "sz.399006": "创业板指",
    "sh.000300": "沪深300",
}


def analyze_market():
    """分析核心指数的多空趋势"""
    print("\n" + "=" * 60)
    print("大盘分析")
    print("=" * 60)

    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    results = []

    for code, name in MAJOR_INDEXES.items():
        try:
            k_rs = bs.query_history_k_data_plus(
                code, "date,close,pctChg",
                start_date=start_date,
                frequency="d"
            )
            df = k_rs.get_data()
            if df.empty or len(df) < 60:
                print(f"⚠️ {name}({code}) 数据不足，跳过")
                continue

            df['close'] = df['close'].astype(float)
            df['pctChg'] = df['pctChg'].astype(float)
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma60 = df['close'].rolling(60).mean().iloc[-1]
            latest_close = df['close'].iloc[-1]
            latest_chg = df['pctChg'].iloc[-1]

            if latest_close > ma20 > ma60:
                trend = "多头排列（强势）"
            elif latest_close < ma20 < ma60:
                trend = "空头排列（弱势）"
            elif latest_close > ma20:
                trend = "站上MA20，短线偏多"
            else:
                trend = "MA20下方，短线偏弱"

            results.append({
                "指数": name, "代码": code,
                "最新点位": round(latest_close, 2),
                "涨跌幅%": round(latest_chg, 2),
                "MA20": round(ma20, 2),
                "MA60": round(ma60, 2),
                "趋势判断": trend
            })
        except Exception as e:
            print(f"⚠️ {name}({code}) 处理失败: {e}")

    df_result = pd.DataFrame(results)
    if not df_result.empty:
        print(df_result.to_string(index=False))
        bullish_count = sum(1 for r in results if "多头" in r["趋势判断"] or "偏多" in r["趋势判断"])
        print(f"\n{bullish_count}/{len(results)} 个核心指数处于MA20上方（短线偏多）")

    return df_result


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
        return {
            "代码": code, "名称": name,
            "涨跌幅%": latest_chg,
            "站上MA20": latest_close > ma20
        }
    except Exception:
        return None


def score_sectors(stock_limit=800):
    """
    板块（申万行业）打分：
    - 该行业里"站上MA20"股票的占比（趋势强度）
    - 该行业当日平均涨跌幅（短期动能）
    """
    print("\n" + "=" * 60)
    print("板块打分（申万行业分类）")
    print("=" * 60)

    print("正在获取行业分类...")
    rs = bs.query_stock_industry()
    industry_df = rs.get_data()
    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]

    if stock_limit:
        industry_df = industry_df.iloc[:stock_limit]

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
        return pd.DataFrame()

    status_df = pd.DataFrame(stock_status)

    grouped = status_df.groupby("行业").agg(
        股票数=("代码", "count"),
        站上MA20占比=("站上MA20", "mean"),
        平均涨跌幅=("涨跌幅%", "mean")
    ).reset_index()

    grouped = grouped[grouped["股票数"] >= 5]

    max_chg = grouped["平均涨跌幅"].abs().max() or 1
    grouped["板块强度分"] = (
        grouped["站上MA20占比"] * 60 +
        (grouped["平均涨跌幅"] / max_chg) * 40 + 40
    ).clip(0, 100).round(1)

    grouped["站上MA20占比"] = (grouped["站上MA20占比"] * 100).round(1).astype(str) + "%"
    grouped["平均涨跌幅"] = grouped["平均涨跌幅"].round(2)

    grouped = grouped.sort_values("板块强度分", ascending=False).reset_index(drop=True)
    return grouped


if __name__ == "__main__":
    bs.login()
    market_df = analyze_market()
    bs.logout()

    sector_df = score_sectors(stock_limit=800)
    if not sector_df.empty:
        print("\n" + "=" * 60)
        print("板块强度排行（前15）")
        print("=" * 60)
        print(sector_df.head(15).to_string(index=False))
        sector_df.to_csv("sector_score_result.csv", index=False, encoding="utf-8-sig")
        print("\n完整结果已保存到 sector_score_result.csv")
