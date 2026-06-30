"""
移植自开源项目 Sequoia-X (github.com/sngyai/Sequoia-X) 的6个选股策略。

原项目数据架构：本地SQLite(baostock批量灌库) + 全市场DataFrame向量化计算，英文列名。
本项目数据架构：实时akshare逐只请求 + 线程池并发，中文列名(收盘/最高/最低/成交量等)。

移植原则：
- 保留每个策略的核心选股条件不变
- 把"全市场一次性向量化计算"改写成"单只股票计算+线程池并发"，复用 screener.py
  里已有的随机延时/重试/并发框架，避免对akshare接口造成集中冲击
- 返回格式从原版的 list[str] 改成 DataFrame（带现价等字段），与本项目其余策略一致，
  方便后续生成研报时引用具体数值
- 海龟策略原版用baostock反查流通市值排序，这里直接用akshare实时行情自带的"总市值"
  字段代替，省掉额外数据源依赖

⚠️ 以上条件均为技术形态归纳，不构成投资建议。
"""
import akshare as ak
import pandas as pd
import random
import time
from datetime import datetime, timedelta
import concurrent.futures as cf

MAX_WORKERS = 8
RETRY = 2


def _fetch_daily(code: str, days: int) -> pd.DataFrame | None:
    """统一的日线数据获取，带随机延时和重试，供下面各策略复用"""
    for attempt in range(RETRY + 1):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            end = datetime.now()
            start = end - timedelta(days=int(days * 1.6))
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily", adjust="qfq",
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return None
            return df.sort_values("日期").reset_index(drop=True)
        except Exception as e:
            if attempt == RETRY:
                print(f"[Sequoia策略] {code} 获取日线失败: {e}")
                return None
            time.sleep(random.uniform(1.0, 2.5))
    return None


def _run_strategy(check_fn, min_bars: int, candidates: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """通用执行框架：并发跑某个check_fn，统一打印进度"""
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(check_fn, row["代码"], row["名称"], min_bars): row["代码"]
            for _, row in candidates.iterrows()
        }
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            if (i + 1) % 100 == 0:
                print(f"[{strategy_name}] 已扫描 {i + 1}/{len(candidates)}")
    print(f"[{strategy_name}] 选出 {len(results)} 只股票")
    return pd.DataFrame(results)


def _default_candidates(limit: int | None) -> pd.DataFrame:
    """大多数策略默认扫全市场，limit仅调试用"""
    stock_list = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
    return stock_list.head(limit) if limit else stock_list


# ============================================================
# 策略一：海龟突破（移植自 TurtleTradeStrategy）
# 20日新高突破 + 成交额过亿 + 阳线防诱多
# ============================================================

def _check_turtle(code: str, name: str, min_bars: int) -> dict | None:
    df = _fetch_daily(code, 35)
    if df is None or len(df) < min_bars:
        return None
    df["前20日最高"] = df["最高"].shift(1).rolling(20).max()
    last, prev = df.iloc[-1], df.iloc[-2]
    if pd.isna(last["前20日最高"]):
        return None

    breakout = last["收盘"] > last["前20日最高"]
    liquid = last["成交额"] > 100_000_000
    is_yang = last["收盘"] > last["开盘"]
    is_up = last["收盘"] > prev["收盘"]

    if breakout and liquid and is_yang and is_up:
        return {"代码": code, "名称": name, "现价": round(float(last["收盘"]), 2),
                "前20日最高": round(float(last["前20日最高"]), 2), "成交额(万)": round(last["成交额"] / 10000, 1)}
    return None


def screen_turtle_breakout(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    df = _run_strategy(_check_turtle, 21, candidates, "海龟突破")
    if df.empty:
        return df
    # 用akshare实时行情自带的总市值代替原版baostock反查，按市值降序排
    try:
        spot = ak.stock_zh_a_spot_em()[["代码", "总市值"]]
        df = df.merge(spot, on="代码", how="left").sort_values("总市值", ascending=False)
    except Exception as e:
        print(f"[海龟突破] 市值排序失败，跳过排序: {e}")
    return df


# ============================================================
# 策略二：均线+成交量金叉（移植自 MaVolumeStrategy）
# 5日均线上穿20日均线 + 当日放量1.5倍
# ============================================================

def _check_ma_volume(code: str, name: str, min_bars: int) -> dict | None:
    df = _fetch_daily(code, 30)
    if df is None or len(df) < min_bars:
        return None
    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()
    df["量MA20"] = df["成交量"].rolling(20).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    if pd.isna(prev["MA5"]) or pd.isna(prev["MA20"]) or pd.isna(last["量MA20"]):
        return None

    golden_cross = prev["MA5"] < prev["MA20"] and last["MA5"] > last["MA20"]
    volume_surge = last["成交量"] > last["量MA20"] * 1.5

    if golden_cross and volume_surge:
        return {"代码": code, "名称": name, "现价": round(float(last["收盘"]), 2),
                "MA5": round(float(last["MA5"]), 2), "MA20": round(float(last["MA20"]), 2)}
    return None


def screen_ma_volume_cross(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    return _run_strategy(_check_ma_volume, 20, candidates, "均线金叉")


# ============================================================
# 策略三：高而窄的旗形整理（移植自 HighTightFlagStrategy）
# 40日涨幅>60% + 近10日振幅<15% + 高位抗跌 + 缩量
# ============================================================

def _check_high_tight_flag(code: str, name: str, min_bars: int) -> dict | None:
    df = _fetch_daily(code, 45)
    if df is None or len(df) < min_bars:
        return None
    tail40, tail10 = df.tail(40), df.tail(10)
    high40, low40 = tail40["最高"].max(), tail40["最低"].min()
    high10, low10 = tail10["最高"].max(), tail10["最低"].min()
    if low40 == 0 or low10 == 0:
        return None

    momentum = high40 / low40 > 1.6
    consolidation = high10 / low10 < 1.15
    high_level = low10 >= high40 * 0.8
    vol_ma20 = df["成交量"].iloc[-21:-1].mean()
    shrink = df["成交量"].iloc[-1] < vol_ma20 * 0.6

    if momentum and consolidation and high_level and shrink:
        return {"代码": code, "名称": name, "现价": round(float(df["收盘"].iloc[-1]), 2),
                "40日涨幅%": round((high40 / low40 - 1) * 100, 1), "近10日振幅%": round((high10 / low10 - 1) * 100, 1)}
    return None


def screen_high_tight_flag(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    return _run_strategy(_check_high_tight_flag, 40, candidates, "高窄旗形")


# ============================================================
# 策略四：涨停洗盘（移植自 LimitUpShakeoutStrategy，与你之前自创版逻辑不同，更短周期更严格）
# 昨日涨停 + 今日收阴放量2倍 + 支撑不破(今日最低>=昨日收盘)
# ============================================================

def _check_limit_up_shakeout_v2(code: str, name: str, min_bars: int) -> dict | None:
    df = _fetch_daily(code, 10)
    if df is None or len(df) < min_bars:
        return None
    prev2, prev1, today = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    limit_up_yesterday = prev1["收盘"] >= prev2["收盘"] * 1.095
    bearish_today = today["收盘"] < today["开盘"]
    volume_surge = today["成交量"] > prev1["成交量"] * 2.0
    support_hold = today["最低"] >= prev1["收盘"]

    if limit_up_yesterday and bearish_today and volume_surge and support_hold:
        return {"代码": code, "名称": name, "现价": round(float(today["收盘"]), 2),
                "昨日涨停价": round(float(prev1["收盘"]), 2), "支撑位(昨收)": round(float(prev1["收盘"]), 2)}
    return None


def screen_limit_up_shakeout_v2(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    return _run_strategy(_check_limit_up_shakeout_v2, 3, candidates, "涨停洗盘v2")


# ============================================================
# 策略五：上升趋势中的跌停反包（移植自 UptrendLimitDownStrategy）
# 20日线>60日线(多头排列) + 放量跌停(跌幅>=9.5% + 量>20日均量2倍)
# ============================================================

def _check_uptrend_limit_down(code: str, name: str, min_bars: int) -> dict | None:
    df = _fetch_daily(code, 70)
    if df is None or len(df) < min_bars:
        return None
    df["MA20"] = df["收盘"].rolling(20).mean()
    df["MA60"] = df["收盘"].rolling(60).mean()
    df["量MA20"] = df["成交量"].rolling(20).mean()
    prev, today = df.iloc[-2], df.iloc[-1]
    if pd.isna(prev["MA20"]) or pd.isna(prev["MA60"]) or pd.isna(today["量MA20"]):
        return None

    uptrend = prev["MA20"] > prev["MA60"]
    limit_down = today["收盘"] <= prev["收盘"] * 0.905
    volume_surge = today["成交量"] > today["量MA20"] * 2.0

    if uptrend and limit_down and volume_surge:
        return {"代码": code, "名称": name, "现价": round(float(today["收盘"]), 2),
                "今日跌幅%": round((today["收盘"] / prev["收盘"] - 1) * 100, 1)}
    return None


def screen_uptrend_limit_down(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    return _run_strategy(_check_uptrend_limit_down, 60, candidates, "上升趋势跌停反包")


# ============================================================
# 策略六：RPS相对强度突破（移植自 RpsBreakoutStrategy）
# 120日涨幅在全市场排名前10% + 现价达120日最高价90%以上
# 原版用SQL一次性拉全市场数据做groupby计算；这里没有本地数据库，
# 改成先全市场扫一遍120日涨幅(单次请求每只股票)，逻辑等价但网络开销更大，
# 建议加大 limit 控制候选范围，或只在不忙的时段跑。
# ============================================================

RPS_PERIOD = 120
RPS_THRESHOLD_PCT = 90  # 全市场涨幅百分位门槛


def _get_rps_raw(code: str, name: str) -> dict | None:
    df = _fetch_daily(code, RPS_PERIOD + 20)
    if df is None or len(df) < RPS_PERIOD:
        return None
    close_now = float(df["收盘"].iloc[-1])
    close_then = float(df["收盘"].iloc[-RPS_PERIOD])
    if close_then <= 0:
        return None
    roll_high = float(df["最高"].tail(RPS_PERIOD).max())
    pct_change = (close_now - close_then) / close_then
    return {"代码": code, "名称": name, "现价": close_now, "120日涨幅": pct_change, "120日最高": roll_high}


def screen_rps_breakout(limit: int | None = None) -> pd.DataFrame:
    candidates = _default_candidates(limit)
    print("第1步: 并发计算全市场120日涨幅(耗时较长)...")
    raw_results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_get_rps_raw, row["代码"], row["名称"]): row["代码"] for _, row in candidates.iterrows()}
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res:
                raw_results.append(res)
            if (i + 1) % 200 == 0:
                print(f"[RPS突破] 已计算 {i + 1}/{len(candidates)}")

    if not raw_results:
        return pd.DataFrame()

    df = pd.DataFrame(raw_results)
    df["RPS百分位"] = df["120日涨幅"].rank(pct=True) * 100
    strong = df[df["RPS百分位"] >= RPS_THRESHOLD_PCT].copy()
    strong = strong[strong["现价"] >= strong["120日最高"] * 0.90]
    strong["120日涨幅%"] = (strong["120日涨幅"] * 100).round(1)
    strong["RPS百分位"] = strong["RPS百分位"].round(1)

    print(f"[RPS突破] 选出 {len(strong)} 只股票")
    return strong[["代码", "名称", "现价", "120日涨幅%", "RPS百分位"]].sort_values("RPS百分位", ascending=False)


# ============================================================
# 策略七（额外）：定增公告监控（移植自 PrivatePlacementStrategy）
# 不需要逐只请求历史K线，直接用akshare的全市场增发接口，速度很快
# ============================================================

def screen_private_placement(lookback_days: int = 7) -> pd.DataFrame:
    """筛选最近 lookback_days 天内发布的定向增发公告"""
    try:
        df = ak.stock_qbzf_em()
    except Exception as e:
        print(f"[定增监控] 获取数据失败: {e}")
        return pd.DataFrame()

    if df is None or df.empty or "发行方式" not in df.columns:
        return pd.DataFrame()

    df = df[df["发行方式"] == "定向增发"].copy()
    if df.empty:
        return df

    df["发行日期"] = pd.to_datetime(df["发行日期"], errors="coerce")
    df = df.dropna(subset=["发行日期"])
    cutoff = datetime.now().date() - timedelta(days=lookback_days)
    df = df[df["发行日期"].dt.date >= cutoff]
    if df.empty:
        return df

    return df.sort_values("发行日期", ascending=False).drop_duplicates(subset=["股票代码"])
