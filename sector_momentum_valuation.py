import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from tqdm import tqdm

"""
sector_momentum_valuation.py

板块"动量 + 估值分位"量化初筛。

⚠️ 重要局限（务必读完再用）：
板块上涨的完整依据通常有5层——价格动量、估值位置、订单/业绩数据、
一级市场融资/BD交易、政策催化。本脚本只能覆盖前两层（技术面+估值面），
因为baostock只提供价格和PE/PB这类结构化行情数据。
订单、融资、政策这三层依据藏在财报文字、新闻、研报里，不是结构化数字，
没有免费API能自动抓取，需要人工查证或让AI实时联网搜索核实。

本脚本输出的"动量强""估值低"只是技术面初筛信号，
不代表该板块有真实的基本面支撑，不构成投资建议。
"""

# ------------------ 参数 ------------------
SECTOR_STOCK_LIMIT = 1500   # 参与打分的股票池大小（越大越全面，也越慢）
LOOKBACK_DAYS = 950         # 约3年历史，用于计算估值历史分位
MOMENTUM_DAYS = 20          # 动量：最近N个交易日涨跌幅
MIN_STOCKS_IN_SECTOR = 5    # 板块内样本数太少则该板块不参与排名
TOP_N_SECTORS = 10
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 20      # 单次查询硬超时，防止网络卡死拖垮整个job


def _init_worker():
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
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


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def _percentile_rank(current, history_series):
    """
    计算current在history_series历史序列中的分位：0=历史最低，100=历史最高。
    对空数据和"历史全同值"（除零）做防护，返回None表示无法计算。
    """
    valid = history_series.dropna()
    if len(valid) < 20 or pd.isna(current):
        return None
    return float((valid < current).mean() * 100)


def _process_one(args):
    """单只股票：拉取~3年数据，同时算动量和估值历史分位，一次查询搞定，运行在子进程里"""
    code, name, industry = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        df = _query_with_timeout(code, "date,close,peTTM,pbMRQ", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
        df['pbMRQ'] = pd.to_numeric(df['pbMRQ'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)

        if len(df) < MOMENTUM_DAYS + 1:
            return None

        latest_close = df['close'].iloc[-1]
        price_n_ago = df['close'].iloc[-1 - MOMENTUM_DAYS]
        momentum_pct = (latest_close - price_n_ago) / price_n_ago * 100 if price_n_ago > 0 else None

        ma20 = df['close'].rolling(20).mean().iloc[-1]
        above_ma20 = bool(latest_close > ma20) if pd.notna(ma20) else None

        current_pe = df['peTTM'].iloc[-1]
        current_pb = df['pbMRQ'].iloc[-1]
        pe_percentile = _percentile_rank(current_pe, df['peTTM'])
        pb_percentile = _percentile_rank(current_pb, df['pbMRQ'])

        if momentum_pct is None:
            return None

        return {
            "代码": code, "名称": name, "行业": industry,
            "动量%": momentum_pct,
            "站上MA20": above_ma20,
            "PE历史分位": pe_percentile,
            "PB历史分位": pb_percentile,
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_sector_scan(limit=SECTOR_STOCK_LIMIT):
    print("正在连接 Baostock（主进程，用于取行业分类列表）...")
    bs.login()
    rs = bs.query_stock_industry()
    industry_df = rs.get_data()
    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]
    bs.logout()

    if limit:
        industry_df = industry_df.iloc[:limit]

    tasks = [(row['code'], row['code_name'], row['industry']) for _, row in industry_df.iterrows()]

    results = []
    fail_count = 0
    print(f"开始计算 {len(tasks)} 只股票的动量+估值分位（{NUM_PROCESSES} 个进程并行）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
            pbar.update(1)
            pbar.set_postfix(样本=len(results), 失败=fail_count)

    print(f"个股数据抓取完成，共失败 {fail_count} 只")

    stock_df = pd.DataFrame(results)
    if stock_df.empty:
        print("未获取到有效数据")
        return pd.DataFrame()

    # 按行业聚合
    grouped = stock_df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均动量=("动量%", "mean"),
        站上MA20占比=("站上MA20", "mean"),
        平均PE分位=("PE历史分位", "mean"),
        平均PB分位=("PB历史分位", "mean"),
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]

    grouped["平均动量"] = grouped["平均动量"].round(2)
    grouped["站上MA20占比"] = (grouped["站上MA20占比"] * 100).round(1)
    grouped["平均PE分位"] = grouped["平均PE分位"].round(1)
    grouped["平均PB分位"] = grouped["平均PB分位"].round(1)

    grouped = grouped.sort_values("平均动量", ascending=False).reset_index(drop=True)

    return grouped


def build_report(grouped):
    lines = []
    lines.append("【板块动量+估值分位 初筛】")
    lines.append("⚠️ 仅覆盖技术面(动量)和估值面(历史分位)，不含订单/融资/政策等基本面依据，仅供技术面参考")
    lines.append("")

    top_momentum = grouped.head(TOP_N_SECTORS)
    lines.append(f"【动量最强TOP{TOP_N_SECTORS}】")
    for _, row in top_momentum.iterrows():
        valuation_note = ""
        if pd.notna(row["平均PE分位"]):
            if row["平均PE分位"] < 30:
                valuation_note = "（估值处于自身历史低位，动量+低估值同时具备）"
            elif row["平均PE分位"] > 70:
                valuation_note = "（估值处于自身历史高位，注意追高风险）"
        lines.append(
            f"- {row['行业']}：{MOMENTUM_DAYS}日动量{row['平均动量']}% "
            f"| 站上MA20占比{row['站上MA20占比']}% "
            f"| PE历史分位{row['平均PE分位']} | PB历史分位{row['平均PB分位']}"
            f"{valuation_note}"
        )

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
    grouped = run_sector_scan()
    if not grouped.empty:
        report = build_report(grouped)
        print("\n" + report)

        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"板块动量+估值分位初筛 {now}"
            send_to_serverchan(sendkey, title, f"扫描时间：{now}\n\n" + report)

        grouped.to_csv("sector_momentum_valuation.csv", index=False, encoding="utf-8-sig")
        print("\n结果已保存到 sector_momentum_valuation.csv")
    else:
        print("本次未获取到有效板块数据")
