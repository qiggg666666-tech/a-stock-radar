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
sector_rotation_alert.py

板块轮动预警：看多/看空打分 + 潜力接力板块 + 热度TOP3。
参考了 oficcejo/aiagents-stock 仓库"智策板块"模块的报告结构
（板块多空/潜力接力板块/热度TOP3），但这是纯量化版本：

⚠️ 重要区别：原仓库的"核心机会"文字点评是调用AI(DeepSeek等)生成的自然语言分析，
本脚本没有接LLM，输出的是结构化数字和打分，不是AI生成的叙述性判断。
"潜力接力板块"的识别逻辑也是自己定义的技术面规则（短期动量加速），
不是原仓库可能包含的资金链/游资动向等更复杂的判断维度。
"""

# ------------------ 参数 ------------------
SECTOR_STOCK_LIMIT = 1500      # 参与打分的股票池大小
LOOKBACK_DAYS = 60             # 拉取约2个月数据，够算20日动量+5日动量+换手率
LONG_MOMENTUM_DAYS = 20        # 长动量窗口
SHORT_MOMENTUM_DAYS = 5        # 短动量窗口（用于识别"接力"：短期是否在加速）
MIN_STOCKS_IN_SECTOR = 5
BULLISH_SCORE_THRESHOLD = 65   # 板块强度分 >= 此值算"看多"
BEARISH_SCORE_THRESHOLD = 35   # 板块强度分 <= 此值算"看空"
TOP_N_HOT = 3                  # 热度榜显示前N个
TOP_N_POTENTIAL = 5            # 潜力接力板块显示前N个
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 15


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


def _process_one(args):
    """单只股票：算长/短动量、是否站上MA20、换手率，运行在子进程里"""
    code, name, industry = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        df = _query_with_timeout(code, "date,close,volume,turn", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)

        if len(df) < LONG_MOMENTUM_DAYS + 1:
            return None

        latest_close = df['close'].iloc[-1]

        price_long_ago = df['close'].iloc[-1 - LONG_MOMENTUM_DAYS]
        long_momentum = (latest_close - price_long_ago) / price_long_ago * 100 if price_long_ago > 0 else None

        price_short_ago = df['close'].iloc[-1 - SHORT_MOMENTUM_DAYS] if len(df) > SHORT_MOMENTUM_DAYS else None
        short_momentum = (
            (latest_close - price_short_ago) / price_short_ago * 100
            if price_short_ago is not None and price_short_ago > 0 else None
        )

        ma20 = df['close'].rolling(20).mean().iloc[-1]
        above_ma20 = bool(latest_close > ma20) if pd.notna(ma20) else None

        avg_turnover = df['turn'].iloc[-5:].mean() if df['turn'].notna().any() else None

        if long_momentum is None:
            return None

        return {
            "代码": code, "名称": name, "行业": industry,
            "长动量%": long_momentum,
            "短动量%": short_momentum,
            "站上MA20": above_ma20,
            "换手率%": avg_turnover,
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
    # 过滤掉行业字段为空的股票（新上市/未分类等），避免聚合出一个"无名板块"
    industry_df = industry_df[industry_df['industry'].astype(str).str.strip() != ""]
    bs.logout()

    if limit:
        industry_df = industry_df.iloc[:limit]

    tasks = [(row['code'], row['code_name'], row['industry']) for _, row in industry_df.iterrows()]

    results = []
    fail_count = 0
    print(f"开始计算 {len(tasks)} 只股票的板块轮动数据（{NUM_PROCESSES} 个进程并行）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                else:
                    results.append(res)
            pbar.update(1)
            pbar.set_postfix(样本=len(results), 失败=fail_count)

    print(f"个股数据抓取完成，共失败 {fail_count} 只")

    stock_df = pd.DataFrame(results)
    if stock_df.empty:
        print("未获取到有效数据")
        return pd.DataFrame()

    # 双重保险：即使前面漏网，这里再过滤一次空行业名
    stock_df = stock_df[stock_df["行业"].astype(str).str.strip() != ""]

    grouped = stock_df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均长动量=("长动量%", "mean"),
        平均短动量=("短动量%", "mean"),
        站上MA20占比=("站上MA20", "mean"),
        平均换手率=("换手率%", "mean"),
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]

    max_abs_long = grouped["平均长动量"].abs().max() or 1
    grouped["强度分"] = (
        grouped["站上MA20占比"] * 60 +
        (grouped["平均长动量"] / max_abs_long) * 40 + 40
    ).clip(0, 100).round(1)

    grouped["短期加速度"] = (
        (grouped["平均短动量"] / SHORT_MOMENTUM_DAYS) - (grouped["平均长动量"] / LONG_MOMENTUM_DAYS)
    ).round(3)

    grouped["平均长动量"] = grouped["平均长动量"].round(2)
    grouped["平均短动量"] = grouped["平均短动量"].round(2)
    grouped["站上MA20占比"] = (grouped["站上MA20占比"] * 100).round(1)
    grouped["平均换手率"] = grouped["平均换手率"].round(2)

    return grouped.sort_values("强度分", ascending=False).reset_index(drop=True)


def build_report(grouped):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"板块轮动预警报告 - {now}")
    lines.append("⚠️ 纯量化版本，不含AI生成的文字点评，仅供技术面参考，不构成投资建议")
    lines.append("")

    bullish = grouped[grouped["强度分"] >= BULLISH_SCORE_THRESHOLD].sort_values("强度分", ascending=False)
    bearish = grouped[grouped["强度分"] <= BEARISH_SCORE_THRESHOLD].sort_values("强度分", ascending=True)

    lines.append("📊 板块多空")
    if not bullish.empty:
        bull_str = "、".join(f"{r['行业']}({r['强度分']:.0f}分)" for _, r in bullish.head(8).iterrows())
        lines.append(f"【看多】{bull_str}")
    else:
        lines.append("【看多】本次无板块达到看多阈值")
    if not bearish.empty:
        bear_str = "、".join(f"{r['行业']}({r['强度分']:.0f}分)" for _, r in bearish.head(8).iterrows())
        lines.append(f"【看空】{bear_str}")
    else:
        lines.append("【看空】本次无板块达到看空阈值")
    lines.append("")

    mid_range = grouped[(grouped["强度分"] > BEARISH_SCORE_THRESHOLD) & (grouped["强度分"] < BULLISH_SCORE_THRESHOLD)]
    potential = mid_range[mid_range["短期加速度"] > 0].sort_values("短期加速度", ascending=False).head(TOP_N_POTENTIAL)

    lines.append("🔄 潜力接力板块（短期动量正在加速，尚未进入强势区间）")
    if not potential.empty:
        for _, row in potential.iterrows():
            lines.append(f"• {row['行业']}：强度分{row['强度分']:.0f}，短期加速度+{row['短期加速度']:.3f}")
    else:
        lines.append("本次未识别到明显的潜力接力板块")
    lines.append("")

    hot = grouped.dropna(subset=["平均换手率"]).sort_values("平均换手率", ascending=False).head(TOP_N_HOT)
    lines.append(f"🌡️ 热度TOP{TOP_N_HOT}（按平均换手率）")
    for i, (_, row) in enumerate(hot.iterrows(), 1):
        lines.append(f"{i}. {row['行业']} - 换手率{row['平均换手率']:.2f}% | 强度分{row['强度分']:.0f}")

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
            send_to_serverchan(sendkey, f"板块轮动预警 {now}", report)

        grouped.to_csv("sector_rotation_alert.csv", index=False, encoding="utf-8-sig")
        print("\n结果已保存到 sector_rotation_alert.csv")
    else:
        print("本次未获取到有效板块数据")
