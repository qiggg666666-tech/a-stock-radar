"""
A股周线首红 全市场扫描模块

"首红"定义（可按需调整）：
  最近一周收盘价 > 开盘价（红/涨），且此前至少 MIN_GREEN_WEEKS 周
  连续收盘价 < 开盘价（绿/跌） —— 即下跌趋势中第一根转涨的周线，潜在反转信号。

⚠️ 注意：
- 全市场约5000只股票逐个请求历史数据，即使开多线程也需要几分钟到几十分钟，
  请勿设置过高并发（容易被东方财富接口限流/封IP）。
- 建议每个交易日收盘后跑一次（如15:30之后），而非交易时段内反复跑。
- 本模块仅做技术形态筛选，不构成投资建议。
"""
import akshare as ak
import pandas as pd
import random
import time
import concurrent.futures as cf
from datetime import datetime, timedelta

MIN_GREEN_WEEKS = 2   # 首红之前至少要有几周连续下跌
MAX_WORKERS = 8        # 并发线程数，过高容易被限流
RETRY = 2              # 单只股票请求失败的重试次数


def get_stock_list() -> pd.DataFrame:
    """获取全部A股代码和名称"""
    return ak.stock_info_a_code_name()


def _check_first_red(symbol: str) -> dict | None:
    """对单只股票判断是否满足'周线首红'条件"""
    for attempt in range(RETRY + 1):
        try:
            time.sleep(random.uniform(0.5, 1.5))  # 随机延时，比固定sleep更不易被识别为机械请求
            end = datetime.now()
            start = end - timedelta(weeks=MIN_GREEN_WEEKS + 8)  # 多取几周保证数据充足
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="weekly",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
            if df is None or len(df) < MIN_GREEN_WEEKS + 1:
                return None

            df = df.tail(MIN_GREEN_WEEKS + 1).reset_index(drop=True)
            latest = df.iloc[-1]
            is_red = latest["收盘"] > latest["开盘"]
            if not is_red:
                return None

            prior = df.iloc[:-1]
            all_green = (prior["收盘"] < prior["开盘"]).all()
            if not all_green:
                return None

            change_pct = (latest["收盘"] - latest["开盘"]) / latest["开盘"] * 100
            return {
                "代码": symbol,
                "最新收盘": float(latest["收盘"]),
                "本周涨幅%": round(float(change_pct), 2),
            }
        except Exception as e:
            if attempt == RETRY:
                print(f"[筛选] {symbol} 获取失败: {e}")
                return None
            time.sleep(random.uniform(1.0, 2.5))
    return None


def screen_first_red(limit: int | None = None) -> pd.DataFrame:
    """
    扫描全市场，返回符合'周线首红'条件的股票
    limit: 仅用于调试，限制扫描的股票数量；正式运行时设为 None 扫全市场
    """
    stock_list = get_stock_list()
    if limit:
        stock_list = stock_list.head(limit)

    name_map = dict(zip(stock_list["code"], stock_list["name"]))
    results = []
    total = len(stock_list)

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 使用线程池并发执行，这是提高速度的关键
        futures = {executor.submit(_check_first_red, code): code for code in stock_list["code"]}
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res:
                res["名称"] = name_map.get(res["代码"], "")
                results.append(res)
            if (i + 1) % 200 == 0:
                print(f"[筛选] 已扫描 {i + 1}/{total}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 单独测试本模块时，先用 limit 小范围跑一下确认逻辑没问题，再去掉 limit 跑全市场
    df = screen_first_red(limit=200)
    print(df)


# ============================================================
# 选股策略二：远离历史高点 + 近期放量启动
# 思路：先用一次全市场实时行情筛出"今日涨幅够大"的候选（几百只），
#      再只对这一小批做历史高点和周线形态的精筛，避免对全市场5000多只
#      都跑一遍历史K线（那样既慢又容易被限流）。
# 本策略仅做形态归纳，不是基于统计验证的规律，不构成投资建议。
# ============================================================

REVERSAL_MIN_PCT_CHANGE = 7.0       # 今日涨幅门槛(%)，可调到接近涨停(9.9/19.9)收窄候选池
REVERSAL_MAX_PRICE_RATIO = 0.5      # 现价 不高于 历史最高价 的这个比例，才算"深度回撤"
REVERSAL_LOOKBACK_WEEKS = 12        # 观察最近多少周来判断"是否刚从底部启动"
REVERSAL_LOW_MUST_BE_WITHIN = 8     # 这12周里的最低点，必须发生在前几周内(而不是这一两周还在探底)


def _check_long_term_reversal(code: str, name: str, current_price: float, today_pct: float) -> dict | None:
    """对单只候选股票，检查是否满足'远离历史高点+近期触底启动'"""
    for attempt in range(RETRY + 1):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            hist = ak.stock_zh_a_hist(
                symbol=code, period="weekly", adjust="qfq",
                start_date="19900101", end_date=datetime.now().strftime("%Y%m%d"),
            )
            if hist is None or len(hist) < REVERSAL_LOOKBACK_WEEKS:
                return None

            all_time_high = hist["收盘"].max()
            if all_time_high <= 0 or current_price > all_time_high * REVERSAL_MAX_PRICE_RATIO:
                return None  # 离历史高点不够远

            recent = hist.tail(REVERSAL_LOOKBACK_WEEKS).reset_index(drop=True)
            low_idx = int(recent["收盘"].idxmin())
            if low_idx > REVERSAL_LOW_MUST_BE_WITHIN:
                return None  # 最低点太靠近现在，说明可能还在探底，不算企稳反转

            low_price = float(recent["收盘"].iloc[low_idx])
            bounce_pct = (current_price - low_price) / low_price * 100 if low_price > 0 else 0

            return {
                "代码": code, "名称": name,
                "现价": round(current_price, 2),
                "今日涨幅%": round(today_pct, 2),
                "历史最高价": round(float(all_time_high), 2),
                "距历史高点回撤%": round((1 - current_price / all_time_high) * 100, 1),
                f"近{REVERSAL_LOOKBACK_WEEKS}周最低点反弹%": round(bounce_pct, 1),
            }
        except Exception as e:
            if attempt == RETRY:
                print(f"[长期反转筛选] {code} 获取失败: {e}")
                return None
            time.sleep(random.uniform(1.0, 2.5))
    return None


def screen_long_term_reversal(
    min_pct_change: float = REVERSAL_MIN_PCT_CHANGE,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    筛选条件：
      1. 当日涨幅 >= min_pct_change（一次性拉全市场实时行情，不用逐个请求）
      2. 现价 <= 历史最高价 * REVERSAL_MAX_PRICE_RATIO（深度回撤）
      3. 最近 REVERSAL_LOOKBACK_WEEKS 周里的最低点，发生在较早的几周（说明已经企稳启动，不是还在探底）
    limit: 仅调试用，限制进入精筛环节的候选数量
    """
    print("第1步: 获取全市场当日行情...")
    spot = ak.stock_zh_a_spot_em()
    candidates = spot[spot["涨跌幅"] >= min_pct_change].copy()
    if limit:
        candidates = candidates.head(limit)
    print(f"   今日涨幅>={min_pct_change}%的有 {len(candidates)} 只，进入第2步精筛(历史高点+周线形态)")

    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_check_long_term_reversal, row["代码"], row["名称"], row["最新价"], row["涨跌幅"]): row["代码"]
            for _, row in candidates.iterrows()
        }
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            if (i + 1) % 50 == 0:
                print(f"[长期反转筛选] 已精筛 {i + 1}/{len(candidates)}")

    return pd.DataFrame(results).sort_values("距历史高点回撤%", ascending=False) if results else pd.DataFrame(results
