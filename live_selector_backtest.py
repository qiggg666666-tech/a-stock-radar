#!/usr/bin/env python3
"""
实战选股逻辑回测 —— 对齐 live_stock_selector.py
================================================
硬过滤（与实盘脚本一致）：
  红筹码占优 + 获利盘 40%~85% + MA20 溢价 ≤ 18%

两路打分（与实盘脚本一致）：
  A路 宽幅启动：wide_score / year_first_red / VwapClose / HighOpen / CloseLow / Turnover
  B路 红筹尖峰：profit_pct / inv_wide / HighOpen / VwapClose / CloseLow / Turnover

时点规则（关键，消除前视）：
  每个交易日只用「当日及之前」K 线调用 analyze 与因子；
  按综合分取当日排名前 TOP_N（默认 2）只，等权持有至下一交易日。

择时（可选）：
  调仓日用多指数 RSRS + 昨日跌幅规则；不允许新开仓则当日空仓。

注意：
  日频 + 全市场 + 每日 analyze 非常耗时，建议先短区间或单分片试跑。

用法：
  python live_selector_backtest.py --shard 1 --start 2024-01-01 --end 2024-12-31
  python live_selector_backtest.py --shard all --start 2024-01-01 --end 2024-12-31
  python live_selector_backtest.py --backtest
  python live_selector_backtest.py --backtest --top 2 --cost 0.002 --notify
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import akshare as ak
import numpy as np
import pandas as pd
import requests

try:
    from final_chip_research import analyze, fetch_ohlcv, is_beijing_stock

    FULL_CHIP_AVAILABLE = True
    print("已加载完整 final_chip_research 模块")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：未找到 final_chip_research.py，回测无法做筹码过滤")

CACHE_DIR = Path("./live_selector_bt_cache")
CACHE_DIR.mkdir(exist_ok=True)

SHARD_MAP = {
    1: ["000", "001"],
    2: ["002"],
    3: ["300"],
    4: ["600"],
    5: ["601"],
    6: ["603"],
    7: ["605", "688"],
    8: ["8", "4", "9"],
}

# ----- 与 live_stock_selector.py 对齐的阈值 -----
PROFIT_MIN = 40.0
PROFIT_MAX = 85.0
TURN_OVER_CAP = 0.15
MIN_BARS = 100
MA20_MAX_PREMIUM = 0.18
PEAK_PROFIT_MIN = 70.0

WEIGHTS_WIDE = {
    "wide_score": 0.35,
    "year_first_red": 0.20,
    "VwapClose": 0.15,
    "HighOpen": 0.10,
    "CloseLow": 0.10,
    "Turnover": 0.10,
}
WEIGHTS_PEAK = {
    "profit_pct": 0.30,
    "inv_wide": 0.25,
    "HighOpen": 0.15,
    "VwapClose": 0.15,
    "CloseLow": 0.10,
    "Turnover": 0.05,
}

RSRS_N = 18
RSRS_M = 600
RSRS_BUY_THRESHOLD = 0.7
INDEX_CRASH_THRESHOLD = -0.03
INDEX_LIST = [
    ("上证综指", "sh000001"),
    ("深证成指", "sz399001"),
    ("创业板指", "sz399006"),
    ("沪深300", "sh000300"),
]

TOP_N_DEFAULT = 2  # 只回测排名前两只
ROUND_TRIP_COST_DEFAULT = 0.0


# ========================= 数据 =========================
def get_all_a_stocks() -> list[str]:
    codes: list[str] = []
    try:
        import baostock as bs

        lg = bs.login()
        if lg.error_code == "0":
            rs = bs.query_stock_basic()
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                code = row[0]
                stock_type = row[4] if len(row) > 4 else "1"
                if stock_type == "1":
                    pure = code.split(".")[-1] if "." in code else code
                    codes.append(pure.zfill(6))
            bs.logout()
            if codes:
                codes = sorted(set(codes))
                print(f"baostock 股票数 {len(codes)}")
                return codes
    except Exception as e:
        print(f"baostock 列表失败: {e}")

    try:
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        if codes:
            print(f"akshare 股票数 {len(codes)}")
            return codes
    except Exception as e:
        print(f"akshare 列表失败: {e}")
    return []


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    start_pad = (pd.Timestamp(start) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")

    if FULL_CHIP_AVAILABLE:
        try:
            df, _, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if df is not None and not df.empty:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= start_pad) & (df["date"] <= end)].copy()
                if len(df) >= MIN_BARS:
                    df["symbol"] = symbol
                    return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    cache = CACHE_DIR / f"ohlcv_{symbol}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_pad) & (df["date"] <= end)].copy()
            if len(df) >= MIN_BARS:
                return df
        except Exception:
            pass

    try:
        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_pad.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(cache, index=False)
        time.sleep(0.15)
        return df
    except Exception:
        return pd.DataFrame()


def calc_launch_factors(df: pd.DataFrame) -> pd.DataFrame:
    """与 live_stock_selector.calc_launch_factors 对齐。"""
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["CloseLow"] = df["close"] / df["low"] - 1
    amount = df["amount"] if "amount" in df.columns else 0
    df["vwap"] = np.where(
        (df["volume"] > 0) & (pd.to_numeric(amount, errors="coerce").fillna(0) > 0),
        df["amount"] / (df["volume"] * 100 + 1e-8),
        (df["high"] + df["low"] + df["close"]) / 3,
    )
    df["VwapClose"] = df["vwap"] / df["close"] - 1
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    turn = pd.to_numeric(df.get("turnover", 0), errors="coerce").fillna(0.0)
    # 百分比 → 小数
    df["Turnover"] = turn / 100.0 if turn.median(skipna=True) > 1.0 else turn

    df["ma20"] = df["close"].rolling(20, min_periods=15).mean()
    df["ma20_premium"] = df["close"] / df["ma20"] - 1

    df["ma250"] = df["close"].rolling(250, min_periods=180).mean()
    df["above_ma250"] = df["close"] > df["ma250"]
    prev_above = df["above_ma250"].shift(1).fillna(False)
    five_days_ago_below = (df["close"].shift(5) < df["ma250"].shift(5)).fillna(False)
    df["year_first_red"] = (
        df["above_ma250"] & (~prev_above) & five_days_ago_below
    ).astype(float)
    return df


def point_in_time_row(code: str, hist: pd.DataFrame) -> dict[str, Any] | None:
    """
    调仓日时点：hist 仅含 date<=t。
    硬过滤与 live apply_launch_filter 一致。
    """
    if not FULL_CHIP_AVAILABLE or hist is None or len(hist) < MIN_BARS:
        return None
    try:
        df = calc_launch_factors(hist)
        result = analyze(code, "", df)

        if not result.get("is_red_heavy_chip"):
            return None

        profit = float(result.get("profit_pct") or 0)
        if not (PROFIT_MIN <= profit <= PROFIT_MAX):
            return None

        latest = df.iloc[-1]
        premium = float(latest.get("ma20_premium") or 0)
        if not np.isfinite(premium) or premium > MA20_MAX_PREMIUM:
            return None

        return {
            "code": str(code).zfill(6),
            "date": pd.Timestamp(latest["date"]),
            "close": float(latest["close"]),
            "HighOpen": float(latest["HighOpen"]),
            "CloseLow": float(latest["CloseLow"]),
            "VwapClose": float(latest["VwapClose"]),
            "Turnover": float(latest.get("Turnover", 0) or 0),
            "profit_pct": profit,
            "ma20_premium": premium,
            "wide_score": float(result.get("wide_score") or 0),
            "is_wide_zone": bool(result.get("is_wide_zone")),
            "year_first_red": float(latest.get("year_first_red", 0) or 0),
            "ma_signal": str(result.get("ma_signal") or ""),
        }
    except Exception:
        return None


def rebalance_dates(df: pd.DataFrame, start: str, end: str, freq: str) -> list[pd.Timestamp]:
    """freq: D=每个交易日, W=每周末交易日, M=每月末交易日。"""
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        return []
    f = freq.upper()
    if f.startswith("D"):
        return [pd.Timestamp(x) for x in sorted(df["date"].unique())]
    if f.startswith("W"):
        key = df["date"].dt.to_period("W")
    else:
        key = df["date"].dt.to_period("M")
    ends = df.groupby(key, sort=True)["date"].max()
    return [pd.Timestamp(x) for x in ends.tolist()]


def build_stock_panel(code: str, df: pd.DataFrame, start: str, end: str, freq: str) -> pd.DataFrame:
    """单票：各调仓日时点过滤 + 持有至下一调仓日的收益（日频=持有至下一交易日）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").reset_index(drop=True)
    dates = rebalance_dates(df, start, end, freq)
    if len(dates) < 2:
        return pd.DataFrame()

    px = df.set_index("date")["close"].sort_index()

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(dates[:-1]):
        hist = df[df["date"] <= t]
        row = point_in_time_row(code, hist)
        if not row:
            continue
        t_next = dates[i + 1]
        c0 = px.loc[:t]
        c1 = px.loc[:t_next]
        if c0.empty or c1.empty:
            continue
        p0, p1 = float(c0.iloc[-1]), float(c1.iloc[-1])
        if p0 <= 0:
            continue
        row["next_date"] = t_next
        row["hold_ret"] = p1 / p0 - 1.0
        rows.append(row)

    return pd.DataFrame(rows)


# ========================= 择时（可选） =========================
def compute_rsrs(index_df: pd.DataFrame, n: int = RSRS_N, m: int = RSRS_M) -> float | None:
    df = index_df.sort_values("date").reset_index(drop=True)
    if len(df) < n + 30:
        return None
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    scores = np.full(len(df), np.nan)
    for i in range(n - 1, len(df)):
        y = highs[i - n + 1 : i + 1]
        x = lows[i - n + 1 : i + 1]
        if np.std(x) < 1e-8:
            continue
        x_mean, y_mean = x.mean(), y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        slope = cov / (var + 1e-12)
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = max(0.0, min(1.0, 1.0 - ss_res / (ss_tot + 1e-12)))
        scores[i] = slope * r2

    z = np.full(len(df), np.nan)
    lookback = min(m, len(df) - n)
    for i in range(n - 1, len(df)):
        start = max(0, i - lookback + 1)
        window = scores[start : i + 1]
        window = window[np.isfinite(window)]
        if len(window) < 20:
            continue
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-12:
            continue
        z[i] = (scores[i] - mu) / sigma
    last = z[-1]
    return float(last) if np.isfinite(last) else None


def load_index_history(symbol: str) -> pd.DataFrame:
    try:
        raw = ak.stock_zh_index_daily(symbol=symbol)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.rename(columns={"date": "date", "high": "high", "low": "low", "close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def timing_allow_on(date: pd.Timestamp, index_cache: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    """调仓日时点择时：与 live get_market_timing 同逻辑（用当日可得数据）。"""
    crash_any = False
    bull_cnt = 0
    valid_cnt = 0
    notes: list[str] = []

    for name, symbol in INDEX_LIST:
        df = index_cache.get(symbol)
        if df is None or df.empty:
            continue
        hist = df[df["date"] <= date].tail(800)
        if len(hist) < 2:
            continue
        ret = float(hist["close"].iloc[-1] / hist["close"].iloc[-2] - 1)
        if ret <= INDEX_CRASH_THRESHOLD:
            crash_any = True
            notes.append(f"{name}昨日{ret:.2%}")
        z = compute_rsrs(hist)
        if z is not None:
            valid_cnt += 1
            if z >= RSRS_BUY_THRESHOLD:
                bull_cnt += 1

    rsrs_ok = (valid_cnt > 0) and (bull_cnt * 2 >= valid_cnt)
    allow = (not crash_any) and (rsrs_ok if valid_cnt > 0 else True)
    if crash_any:
        reason = "指数大跌:" + ",".join(notes)
    elif valid_cnt > 0 and not rsrs_ok:
        reason = f"RSRS偏多{bull_cnt}/{valid_cnt}"
    else:
        reason = "择时偏多"
    return allow, reason


# ========================= 打分选股 =========================
def _safe_z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-8)


def score_and_split(
    panel: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """横截面打分：宽幅 top_n / 尖峰 top_n / 综合分全局 top_n。"""
    empty = panel.iloc[0:0]
    if panel.empty:
        return empty, empty, empty

    df = panel.copy()
    df["wide_z"] = _safe_z(df["wide_score"])
    df["VwapClose_z"] = _safe_z(df["VwapClose"])
    df["HighOpen_z"] = _safe_z(df["HighOpen"])
    df["CloseLow_z"] = _safe_z(df["CloseLow"])
    df["Turnover_z"] = _safe_z(df["Turnover"].clip(upper=TURN_OVER_CAP))
    df["year_first_red"] = df["year_first_red"].fillna(0.0)
    df["profit_z"] = _safe_z(df["profit_pct"])
    df["inv_wide_z"] = -df["wide_z"]

    df["score_wide"] = (
        WEIGHTS_WIDE["wide_score"] * df["wide_z"]
        + WEIGHTS_WIDE["year_first_red"] * df["year_first_red"]
        + WEIGHTS_WIDE["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_WIDE["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_WIDE["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_WIDE["Turnover"] * df["Turnover_z"]
    )
    df.loc[~df["is_wide_zone"].astype(bool), "score_wide"] *= 0.6

    df["score_peak"] = (
        WEIGHTS_PEAK["profit_pct"] * df["profit_z"]
        + WEIGHTS_PEAK["inv_wide"] * df["inv_wide_z"]
        + WEIGHTS_PEAK["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_PEAK["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_PEAK["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_PEAK["Turnover"] * df["Turnover_z"]
    )
    df.loc[df["profit_pct"] < PEAK_PROFIT_MIN, "score_peak"] *= 0.5
    df.loc[df["is_wide_zone"].astype(bool), "score_peak"] *= 0.7

    df["final_score"] = df[["score_wide", "score_peak"]].max(axis=1)
    df["main_track"] = np.where(df["score_peak"] > df["score_wide"], "尖峰", "宽幅启动")

    top_wide = (
        df[df["main_track"] == "宽幅启动"]
        .sort_values("score_wide", ascending=False)
        .head(top_n)
    )
    top_peak = (
        df[df["main_track"] == "尖峰"]
        .sort_values("score_peak", ascending=False)
        .head(top_n)
    )
    # 合并组合：按综合分全局只取前 top_n（默认 2 只），不是两路拼接
    top_both = df.sort_values("final_score", ascending=False).head(top_n)
    return top_wide, top_peak, top_both


# ========================= 分片 / 回测 =========================
def run_shard(shard_id: int, start: str, end: str, freq: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"选股回测分片 {shard_id} | {start} ~ {end} | freq={freq}")
    print(f"{'=' * 60}")

    codes = filter_shard(get_all_a_stocks(), shard_id)
    print(f"本分片 {len(codes)} 只")

    panels: list[pd.DataFrame] = []
    for i, code in enumerate(codes):
        if FULL_CHIP_AVAILABLE and is_beijing_stock(code):
            continue
        df = get_stock_data(code, start, end)
        if df.empty:
            continue
        panel = build_stock_panel(code, df, start, end, freq)
        if not panel.empty:
            panels.append(panel)
        if (i + 1) % 20 == 0:
            n = sum(len(p) for p in panels)
            print(f"  {i + 1}/{len(codes)} 命中样本累计 {n}")

    out = CACHE_DIR / f"bt_shard_{shard_id}.parquet"
    if panels:
        all_df = pd.concat(panels, ignore_index=True)
        all_df.to_parquet(out, index=False)
        print(f"保存 {len(all_df)} 行 → {out}")
    else:
        print("本分片无命中")
        (CACHE_DIR / f"bt_shard_{shard_id}_empty.txt").write_text("empty\n", encoding="utf-8")


def _perf(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    if returns is None or len(returns) == 0:
        return {
            "total": 0.0,
            "ann": 0.0,
            "sharpe": 0.0,
            "maxdd": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "n": 0,
        }
    r = returns.astype(float)
    cum = (1 + r).cumprod()
    total = float(cum.iloc[-1] - 1)
    n = max(len(r), 1)
    ann = float((1 + total) ** (periods_per_year / n) - 1)
    vol = float(r.std() + 1e-12)
    sharpe = float(r.mean() / vol * np.sqrt(periods_per_year))
    maxdd = float((cum / cum.cummax() - 1).min())
    # 胜率：组合单期收益 > 0 的占比（空仓期 return=0 不计入胜/负，单独从分母去掉可选）
    active = r[r != 0.0]
    if len(active) == 0:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    else:
        wins = active[active > 0]
        losses = active[active < 0]
        win_rate = float(len(wins) / len(active))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "total": total,
        "ann": ann,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n": n,
    }


def merge_and_backtest(
    top_n: int = TOP_N_DEFAULT,
    cost: float = ROUND_TRIP_COST_DEFAULT,
    use_timing: bool = False,
    freq: str = "D",
    do_notify: bool = False,
) -> None:
    files = list(CACHE_DIR.glob("bt_shard_*.parquet"))
    if not files:
        print("无分片结果，请先 --shard")
        return

    dataset = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "code"]).reset_index(drop=True)
    print(
        f"样本 {len(dataset)} 行 | 股票 {dataset['code'].nunique()} | "
        f"{dataset['date'].min().date()} ~ {dataset['date'].max().date()}"
    )

    index_cache: dict[str, pd.DataFrame] = {}
    if use_timing:
        for name, symbol in INDEX_LIST:
            index_cache[symbol] = load_index_history(symbol)
            print(f"  指数 {name}: {len(index_cache[symbol])} 日")

    f = freq.upper()
    if f.startswith("D"):
        periods_per_year = 252.0
    elif f.startswith("W"):
        periods_per_year = 52.0
    else:
        periods_per_year = 12.0
    dates = sorted(dataset["date"].unique())
    records_wide: list[dict[str, Any]] = []
    records_peak: list[dict[str, Any]] = []
    records_both: list[dict[str, Any]] = []

    for t in dates:
        day = dataset[dataset["date"] == t]
        if len(day) < 3:
            continue

        allow, reason = True, "未启用择时"
        if use_timing:
            allow, reason = timing_allow_on(pd.Timestamp(t), index_cache)

        if not allow:
            records_wide.append({"date": t, "return": 0.0, "n": 0, "reason": reason})
            records_peak.append({"date": t, "return": 0.0, "n": 0, "reason": reason})
            records_both.append({"date": t, "return": 0.0, "n": 0, "reason": reason})
            print(f"  {pd.Timestamp(t).date()} 空仓 | {reason}")
            continue

        top_wide, top_peak, top_both = score_and_split(day, top_n)

        def _port_ret(sub: pd.DataFrame) -> tuple[float, int]:
            if sub.empty:
                return 0.0, 0
            return float(sub["hold_ret"].mean()) - cost, len(sub)

        rw, nw = _port_ret(top_wide)
        rp, np_ = _port_ret(top_peak)
        rb, nb = _port_ret(top_both)
        records_wide.append({"date": t, "return": rw, "n": nw, "reason": reason})
        records_peak.append({"date": t, "return": rp, "n": np_, "reason": reason})
        records_both.append({"date": t, "return": rb, "n": nb, "reason": reason})
        print(
            f"  {pd.Timestamp(t).date()} 前{nb}名 {rb:.2%} | "
            f"(宽幅路{nw} {rw:.2%} / 尖峰路{np_} {rp:.2%})"
        )

    def _summary(name: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
        if not recs:
            return {"name": name, "empty": True}
        rdf = pd.DataFrame(recs).set_index("date").sort_index()
        rdf["cum"] = (1 + rdf["return"]).cumprod()
        m = _perf(rdf["return"], periods_per_year)
        path = f"live_selector_bt_{name}.csv"
        rdf.to_csv(path)
        return {
            "name": name,
            "empty": False,
            "metrics": m,
            "path": path,
            "rdf": rdf,
        }

    summaries = [
        _summary("wide", records_wide),
        _summary("peak", records_peak),
        _summary("both", records_both),
    ]

    print("\n" + "=" * 60)
    print("实盘选股逻辑回测结果（时点过滤，研究用）")
    print(
        f"过滤: 红筹码 + 获利{PROFIT_MIN}-{PROFIT_MAX}% + MA20溢价≤{MA20_MAX_PREMIUM:.0%}"
    )
    print(f"调仓: {freq} | TOP_N={top_n} | 成本={cost} | 择时={'开' if use_timing else '关'}")
    print("=" * 60)
    lines = [
        "# 实盘选股逻辑回测（时点）",
        "",
        f"- 过滤：红筹码 + 获利{PROFIT_MIN}~{PROFIT_MAX}% + MA20溢价≤{MA20_MAX_PREMIUM:.0%}",
        f"- 调仓频率：{freq}（D=日 / W=周 / M=月）",
        f"- 每日持仓：综合排名前 {top_n} 只（等权）",
        f"- 持有：当日收盘 → 下一交易日收盘",
        f"- 往返成本：{cost}",
        f"- 择时：{'多指数 RSRS + 大跌过滤' if use_timing else '关闭'}",
        f"- 筹码：调仓日时点 analyze（无末日广播）",
        "",
    ]
    for s in summaries:
        if s.get("empty"):
            print(f"{s['name']}: 无结果")
            lines.append(f"## {s['name']}\n无结果\n")
            continue
        m = s["metrics"]
        print(
            f"{s['name']}: 总收益{m['total']:.2%} 年化{m['ann']:.2%} "
            f"胜率{m.get('win_rate', 0):.1%} 夏普{m['sharpe']:.2f} "
            f"回撤{m['maxdd']:.2%} 期数{m['n']} → {s['path']}"
        )
        lines.extend(
            [
                f"## {s['name']}",
                f"- 总收益：{m['total']:.2%}",
                f"- 年化：{m['ann']:.2%}",
                f"- 胜率：{m.get('win_rate', 0):.1%}（单期收益>0占比，不含空仓）",
                f"- 平均盈利期：{m.get('avg_win', 0):.2%}",
                f"- 平均亏损期：{m.get('avg_loss', 0):.2%}",
                f"- 夏普：{m['sharpe']:.3f}",
                f"- 最大回撤：{m['maxdd']:.2%}",
                f"- 调仓期数：{m['n']}",
                "",
            ]
        )
    lines.append("仅为研究输出，不构成投资建议。")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        for s in summaries:
            if s.get("empty"):
                continue
            plt.plot(s["rdf"].index, s["rdf"]["cum"], label=s["name"])
        plt.legend()
        plt.title("Live Selector Logic Backtest (point-in-time)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("live_selector_bt_equity.png", dpi=150)
        print("净值图: live_selector_bt_equity.png")
    except Exception as e:
        print(f"画图跳过: {e}")

    if do_notify:
        title = "实盘选股逻辑回测｜" + " / ".join(
            f"{s['name']}{s['metrics']['total']:.1%}"
            for s in summaries
            if not s.get("empty")
        )
        key = os.getenv("SENDKEY", "").strip()
        if not key:
            print("未设置 SENDKEY，跳过推送")
        else:
            try:
                resp = requests.post(
                    f"https://sctapi.ftqq.com/{key}.send",
                    data={"title": title[:100], "desp": "\n".join(lines)},
                    timeout=20,
                )
                print(f"Server酱: {resp.status_code} {resp.text[:120]}")
            except Exception as e:
                print(f"推送失败: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="对齐 live_stock_selector 的时点回测")
    p.add_argument("--shard", type=str, help="1-8 / all")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--start", type=str, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, help="YYYY-MM-DD")
    p.add_argument(
        "--freq",
        type=str,
        default="D",
        help="D=日频(默认,每日选前2持有一天) W=周 M=月",
    )
    p.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="每日持仓只数,默认2")
    p.add_argument("--cost", type=float, default=ROUND_TRIP_COST_DEFAULT)
    p.add_argument("--timing", action="store_true", help="启用多指数 RSRS 择时")
    p.add_argument("--notify", action="store_true")
    args = p.parse_args()

    if args.backtest:
        merge_and_backtest(
            top_n=args.top,
            cost=args.cost,
            use_timing=args.timing,
            freq=args.freq,
            do_notify=args.notify,
        )
        return

    if not args.shard:
        p.error("请指定 --shard 或 --backtest")
    if not args.start or not args.end:
        p.error("分片需 --start 与 --end")

    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i, args.start, args.end, args.freq)
        print("\n分片完成，执行: python live_selector_backtest.py --backtest")
    else:
        run_shard(int(args.shard), args.start, args.end, args.freq)


if __name__ == "__main__":
    main()
