#!/usr/bin/env python3
"""
完整更新版（时点筹码过滤）：
  全市场分片 + HighOpen/CloseLow/VwapClose/M0/Turnover
  + 红筹码占优 ∩ 宽幅堆积区（调仓日时点 analyze，无前视）
  + 横截面回归因子加权 + 可选 SVR
  + Server酱推送

相对旧版的关键修复：
  1. 不再把「样本末日」的筹码结果广播到整段历史（消除前视）
  2. 仅在每月最后一个交易日做 analyze，决定当月是否入池
  3. 收益改为「本月末 → 下月末」持有期收益，与月度调仓一致
  4. 可选简单交易成本；样本不足时回退到因子得分选股（不硬上 SVR）
  5. 推送中写明区间、TOP_N、是否扣成本、筹码规则

用法：
  python full_factor_chip_forced_full.py --shard 1 --start 2019-01-01 --end 2024-12-31
  python full_factor_chip_forced_full.py --shard all --start 2020-01-01 --end 2025-06-30
  python full_factor_chip_forced_full.py --shard backtest
  python full_factor_chip_forced_full.py --shard backtest --use-svr
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
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ==================== 强制导入完整筹码模块 ====================
try:
    from final_chip_research import analyze, fetch_ohlcv, is_beijing_stock

    FULL_CHIP_AVAILABLE = True
    print("已加载完整 final_chip_research 模块")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：未找到 final_chip_research.py，筹码过滤将全部返回 False")

# ==================== 配置 ====================
CACHE_DIR = Path("./factor_chip_full_cache")
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

TOP_N = 20
LOOKBACK_M0 = 15
LOOKBACK_MONTHS = 12
MIN_BARS_FOR_CHIP = 80
# 单边交易成本（买入+卖出各一次约 2*cost）；默认 0 表示研究对照不计成本
ROUND_TRIP_COST = 0.0
FACTOR_COLS = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]


# ==================== 工具函数 ====================
def get_all_a_stocks() -> list[str]:
    """获取全A股列表，优先 baostock，失败再尝试 akshare。"""
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
                    pure_code = code.split(".")[-1] if "." in code else code
                    codes.append(pure_code.zfill(6))
            bs.logout()
            if codes:
                codes = sorted(list(set(codes)))
                print(f"baostock 获取到 {len(codes)} 只股票")
                return codes
    except Exception as e:
        print(f"baostock 获取股票列表失败: {e}")

    try:
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        if codes:
            print(f"akshare 获取到 {len(codes)} 只股票")
            return codes
    except Exception as e:
        print(f"akshare 获取股票列表失败: {e}")

    print("错误：无法获取全市场股票列表，请检查网络或数据源")
    return []


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    if FULL_CHIP_AVAILABLE:
        try:
            df, _source, _ = fetch_ohlcv(symbol, timeout_seconds=30, retries=2)
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            # 多留一段历史，保证最早调仓日也能做筹码
            start_ts = pd.Timestamp(start) - pd.Timedelta(days=400)
            df = df[(df["date"] >= start_ts) & (df["date"] <= end)].copy()
            if not df.empty:
                df["symbol"] = symbol
                return df
        except Exception:
            pass

    cache_file = CACHE_DIR / f"{symbol}.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            start_ts = pd.Timestamp(start) - pd.Timedelta(days=400)
            return df[(df["date"] >= start_ts) & (df["date"] <= end)].copy()
        except Exception:
            pass

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
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
        df.to_parquet(cache_file, index=False)
        time.sleep(0.3)
        return df
    except Exception:
        return pd.DataFrame()


def calc_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["CloseLow"] = df["close"] / df["low"] - 1
    df["vwap"] = np.where(
        (df["volume"] > 0) & (df["amount"] > 0),
        df["amount"] / (df["volume"] * 100 + 1e-8),
        (df["high"] + df["low"] + df["close"]) / 3,
    )
    df["VwapClose"] = df["vwap"] / df["close"] - 1
    df["intraday_ret"] = df["close"] / df["open"] - 1
    df["M0"] = df["intraday_ret"].rolling(LOOKBACK_M0, min_periods=5).sum()
    if "turnover" in df.columns:
        df["Turnover"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)
        # akshare 换手率常为百分比；若中位数>1 则按百分比转成小数
        if df["Turnover"].median(skipna=True) > 1.0:
            df["Turnover"] = df["Turnover"] / 100.0
    else:
        df["Turnover"] = 0.0

    for col in FACTOR_COLS:
        med = df[col].median()
        mad = (df[col] - med).abs().median()
        if pd.notna(mad) and mad > 0:
            df[col] = df[col].clip(med - 5 * mad, med + 5 * mad)
    return df


def chip_hit_at(code: str, hist: pd.DataFrame, name: str = "") -> tuple[bool, float, float]:
    """仅用 hist（<=调仓日）做时点筹码判定，避免前视。"""
    if not FULL_CHIP_AVAILABLE or hist is None or len(hist) < MIN_BARS_FOR_CHIP:
        return False, 0.0, 0.0
    try:
        result = analyze(code, name, hist)
        hit = bool(result.get("is_red_heavy_chip", False) and result.get("is_wide_zone", False))
        wide_score = float(result.get("wide_score", 0) or 0)
        profit_pct = float(result.get("profit_pct", 0) or 0)
        return hit, wide_score, profit_pct
    except Exception:
        return False, 0.0, 0.0


def build_month_end_panel(code: str, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    对单只股票：在每个月最后一个交易日做时点筹码过滤，
    并计算「本月末收盘 → 下月末收盘」的持有期收益。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)
    df = calc_price_factors(df)
    df["ym"] = df["date"].dt.to_period("M")

    month_ends = df.groupby("ym", sort=True).tail(1).copy()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    month_ends = month_ends[(month_ends["date"] >= start_ts) & (month_ends["date"] <= end_ts)]
    if month_ends.empty:
        return pd.DataFrame()

    # 下月末收盘价，用于持有期收益
    month_ends = month_ends.sort_values("date").reset_index(drop=True)
    month_ends["next_close"] = month_ends["close"].shift(-1)
    month_ends["next_date"] = month_ends["date"].shift(-1)
    month_ends = month_ends.dropna(subset=["next_close"])

    rows: list[dict[str, Any]] = []
    for _, row in month_ends.iterrows():
        t = row["date"]
        hist = df[df["date"] <= t]
        if len(hist) < MIN_BARS_FOR_CHIP:
            continue

        # 因子缺失则跳过
        if any(pd.isna(row.get(c)) for c in FACTOR_COLS):
            continue

        hit, wide_score, profit_pct = chip_hit_at(code, hist)
        if not hit:
            continue

        hold_ret = float(row["next_close"] / row["close"] - 1.0)
        rows.append(
            {
                "symbol": code,
                "date": t,
                "ym": str(row["ym"]),
                "close": float(row["close"]),
                "next_date": row["next_date"],
                "hold_ret": hold_ret,
                "HighOpen": float(row["HighOpen"]),
                "CloseLow": float(row["CloseLow"]),
                "VwapClose": float(row["VwapClose"]),
                "M0": float(row["M0"]),
                "Turnover": float(row["Turnover"]),
                "wide_score": wide_score,
                "profit_pct": profit_pct,
                "chip_hit": True,
            }
        )

    return pd.DataFrame(rows)


def cross_sectional_reg(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    results = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=factor_cols + ["hold_ret"])
        if len(g) < 30:
            continue
        X = (g[factor_cols] - g[factor_cols].mean()) / (g[factor_cols].std() + 1e-8)
        X = sm.add_constant(X)
        try:
            model = sm.OLS(g["hold_ret"], X).fit()
            row = {"date": dt, "r2": model.rsquared, "n": len(g)}
            for c in factor_cols:
                row[f"{c}_ret"] = model.params.get(c, np.nan)
                row[f"{c}_t"] = model.tvalues.get(c, np.nan)
            results.append(row)
        except Exception:
            continue
    return pd.DataFrame(results)


def get_factor_weights(reg_df: pd.DataFrame, lookback: int = 12) -> dict[str, float]:
    if reg_df is None or reg_df.empty:
        return {c: 1.0 / len(FACTOR_COLS) for c in FACTOR_COLS}
    recent = reg_df.tail(lookback)
    weights = {c: max(float(recent[f"{c}_ret"].mean()), 0.0) for c in FACTOR_COLS}
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return {k: 1.0 / len(FACTOR_COLS) for k in FACTOR_COLS}


def notify_serverchan(title: str, content: str) -> dict[str, Any]:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "missing_SENDKEY"}
    try:
        resp = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"title": title, "desp": content},
            timeout=20,
        )
        result = resp.json()
        if resp.ok and result.get("code") == 0:
            return {"status": "sent", "http_status": resp.status_code}
        return {"status": "failed", "http_status": resp.status_code, "code": result.get("code")}
    except Exception as e:
        return {"status": "failed", "error": f"{type(e).__name__}:{str(e)[:200]}"}


# ==================== 分片处理 ====================
def run_shard(shard_id: int, start: str, end: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
    print(f"回测区间: {start} ~ {end}")
    print("筹码规则: 每月末时点 analyze（红筹码∩宽幅），持有至下月末")
    print(f"{'=' * 60}")

    all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表，本分片退出")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")
    if not codes:
        print("本分片无匹配股票")
        return

    panels: list[pd.DataFrame] = []
    for i, code in enumerate(codes):
        if FULL_CHIP_AVAILABLE and is_beijing_stock(code):
            continue
        df = get_stock_data(code, start, end)
        if df.empty or len(df) < MIN_BARS_FOR_CHIP:
            continue
        panel = build_month_end_panel(code, df, start, end)
        if not panel.empty:
            panels.append(panel)
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(codes)}，命中月度样本累计 {sum(len(p) for p in panels)}")

    if not panels:
        debug_file = CACHE_DIR / f"shard_{shard_id}_debug_empty.txt"
        debug_file.write_text("时点过滤后无任何月度命中样本\n", encoding="utf-8")
        print(f"无有效数据，已写入 {debug_file}")
        return

    dataset = pd.concat(panels, ignore_index=True)
    shard_file = CACHE_DIR / f"shard_{shard_id}_processed.parquet"
    dataset.to_parquet(shard_file, index=False)
    print(f"时点筹码命中月度样本: {len(dataset)}  股票数: {dataset['symbol'].nunique()}")
    print(f"已保存: {shard_file}")

    reg_df = cross_sectional_reg(dataset, FACTOR_COLS)
    if not reg_df.empty:
        reg_df.to_csv(CACHE_DIR / f"reg_shard_{shard_id}.csv", index=False)
        print("横截面回归完成")

    print(f"分片 {shard_id} 完成")


# ==================== 合并回测 + 推送 ====================
def merge_and_backtest(use_svr: bool = False, round_trip_cost: float = ROUND_TRIP_COST) -> None:
    print("\n合并时点筹码样本 + 因子加权回测...")
    print(f"SVR: {'开启' if use_svr else '关闭（纯因子得分）'}  往返成本: {round_trip_cost:.4f}")

    files = list(CACHE_DIR.glob("shard_*_processed.parquet"))
    if not files:
        title = "时点筹码过滤策略回测｜无分片数据"
        content = "未找到任何分片 parquet。请先运行分片。\n\n仅为研究输出，不构成投资建议。"
        print(notify_serverchan(title, content))
        return

    dataset = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)

    print(f"命中样本: {len(dataset)}  股票数: {dataset['symbol'].nunique()}")
    print(f"日期范围: {dataset['date'].min().date()} ~ {dataset['date'].max().date()}")

    reg_files = list(CACHE_DIR.glob("reg_shard_*.csv"))
    reg_all = (
        pd.concat([pd.read_csv(f) for f in reg_files], ignore_index=True)
        if reg_files
        else pd.DataFrame()
    )
    if not reg_all.empty:
        reg_all["date"] = pd.to_datetime(reg_all["date"])
        reg_all = reg_all.sort_values("date")

    dataset["ym"] = dataset["date"].dt.to_period("M")
    months = sorted(dataset["ym"].unique())
    print(f"共有 {len(months)} 个调仓月: {months[0]} ~ {months[-1]}")

    lookback = min(LOOKBACK_MONTHS, max(3, len(months) // 3))
    print(f"因子权重回看月数: {lookback}")

    results: list[dict[str, Any]] = []
    weight_history: list[dict[str, Any]] = []

    for i in range(lookback, len(months)):
        # 训练窗：过去 lookback 个月的时点命中样本
        train = dataset[dataset["ym"].isin(months[i - lookback : i])].copy()
        # 当月调仓：本月命中且已有下月收益
        test = dataset[dataset["ym"] == months[i]].copy()
        print(f"  {months[i]}: train={len(train)} test={len(test)}")

        if len(test) < 3:
            continue

        if not reg_all.empty:
            hist_reg = reg_all[reg_all["date"] < test["date"].min()]
            weights = get_factor_weights(hist_reg, lookback=lookback)
        else:
            weights = {c: 1.0 / len(FACTOR_COLS) for c in FACTOR_COLS}

        weight_history.append({"month": str(months[i]), **weights})

        # 横截面 z-score 后加权
        for col in FACTOR_COLS:
            mu, sigma = test[col].mean(), test[col].std()
            test[col + "_z"] = (test[col] - mu) / (sigma + 1e-8)
        test["factor_score"] = 0.0
        for col in FACTOR_COLS:
            test["factor_score"] += weights[col] * test[col + "_z"]

        selected = test
        score_col = "factor_score"

        if use_svr and len(train) >= 30:
            try:
                for col in FACTOR_COLS:
                    mu, sigma = train[col].mean(), train[col].std()
                    train[col + "_z"] = (train[col] - mu) / (sigma + 1e-8)
                train["factor_score"] = 0.0
                for col in FACTOR_COLS:
                    train["factor_score"] += weights[col] * train[col + "_z"]

                feat = FACTOR_COLS + ["factor_score"]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train[feat])
                X_test = scaler.transform(test[feat])
                model = SVR(kernel="rbf", C=1.0, epsilon=0.01)
                model.fit(X_train, train["hold_ret"])
                test = test.copy()
                test["pred"] = model.predict(X_test)
                selected = test
                score_col = "pred"
            except Exception as e:
                print(f"    SVR 失败，回退因子得分: {e}")
                score_col = "factor_score"

        selected = selected.nlargest(min(TOP_N, len(selected)), score_col)
        if selected.empty:
            continue

        port_ret = float(selected["hold_ret"].mean()) - round_trip_cost
        results.append(
            {
                "date": selected["date"].max(),
                "ym": str(months[i]),
                "return": port_ret,
                "n": len(selected),
                "score_col": score_col,
            }
        )
        print(f"    → 选股 {len(selected)} 只，持有期收益 {port_ret:.4f} ({score_col})")

    if not results:
        title = "时点筹码过滤策略回测｜无有效结果"
        content = (
            "# 时点红筹码∩宽幅 + 价格因子回测\n\n"
            f"- 命中样本：{len(dataset)}\n"
            f"- 股票数：{dataset['symbol'].nunique()}\n"
            f"- 调仓月数：{len(months)}\n"
            f"- 结果：无有效回测区间\n\n"
            "筹码：每月末时点 analyze（无全历史广播）\n"
            "仅为研究输出，不构成投资建议。"
        )
        print(notify_serverchan(title, content))
        return

    ret_df = pd.DataFrame(results).set_index("date").sort_index()
    ret_df["cum"] = (1 + ret_df["return"]).cumprod()

    total = float(ret_df["cum"].iloc[-1] - 1)
    n_periods = max(len(ret_df), 1)
    ann = float((1 + total) ** (12 / n_periods) - 1)
    vol = float(ret_df["return"].std() + 1e-12)
    sharpe = float(ret_df["return"].mean() / vol * np.sqrt(12))
    maxdd = float((ret_df["cum"] / ret_df["cum"].cummax() - 1).min())

    date_min = dataset["date"].min().date()
    date_max = dataset["date"].max().date()

    print("\n" + "=" * 55)
    print("时点筹码过滤 + 因子加权回测结果")
    print("=" * 55)
    print(f"样本区间   : {date_min} ~ {date_max}")
    print(f"总收益     : {total:.2%}")
    print(f"年化收益   : {ann:.2%}")
    print(f"夏普比率   : {sharpe:.3f}")
    print(f"最大回撤   : {maxdd:.2%}")
    print(f"调仓次数   : {len(ret_df)}")
    print(f"TOP_N      : {TOP_N}")
    print(f"往返成本   : {round_trip_cost:.4f}")
    print(f"SVR        : {use_svr}")
    print("=" * 55)

    ret_df.to_csv("full_chip_forced_backtest.csv")
    if weight_history:
        pd.DataFrame(weight_history).to_csv("full_chip_forced_weights.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(ret_df.index, ret_df["cum"])
        plt.title("Point-in-time Chip Filter + Factors - Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("full_chip_forced_equity.png", dpi=150)
        print("已保存: full_chip_forced_backtest.csv / weights / equity.png")
    except Exception as e:
        print(f"画图跳过: {e}")

    title = f"时点筹码过滤回测｜总收益{total:.1%} 夏普{sharpe:.2f}"
    content_lines = [
        "# 时点红筹码∩宽幅 + 价格因子回测结果",
        "",
        f"- **样本区间**：{date_min} ~ {date_max}",
        f"- **总收益率**：{total:.2%}",
        f"- **年化收益率**：{ann:.2%}",
        f"- **夏普比率**：{sharpe:.3f}",
        f"- **最大回撤**：{maxdd:.2%}",
        f"- **调仓次数**：{len(ret_df)}",
        f"- **TOP_N**：{TOP_N}",
        f"- **持有期**：本月末 → 下月末",
        f"- **往返成本**：{round_trip_cost:.4f}",
        f"- **选股模型**：{'SVR（失败则回退因子）' if use_svr else '因子加权得分'}",
        f"- **因子回看月数**：{lookback}",
        "",
        "## 因子权重（最近一期）",
    ]
    if weight_history:
        last_w = weight_history[-1]
        for k, v in last_w.items():
            if k != "month":
                content_lines.append(f"- {k}: {v:.3f}")
    content_lines.extend(
        [
            "",
            "## 规则说明",
            "- 过滤：调仓日（每月末）时点 `analyze`，红筹码占优 ∩ 宽幅堆积区",
            "- **无**把样本末日筹码结果广播到历史（已消除此前前视）",
            "- 因子：HighOpen + CloseLow + VwapClose + M0 + Turnover",
            "",
            "仅为研究输出，不构成投资建议。",
        ]
    )
    print(f"Server酱推送结果: {notify_serverchan(title, chr(10).join(content_lines))}")


# ==================== 入口 ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="时点筹码过滤 + 价格因子全市场策略")
    parser.add_argument("--shard", type=str, required=True, help="1-8 / all / backtest")
    parser.add_argument("--start", type=str, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--use-svr", action="store_true", help="启用 SVR（默认关闭，更稳）")
    parser.add_argument(
        "--cost",
        type=float,
        default=ROUND_TRIP_COST,
        help="每次调仓往返成本，如 0.002 表示 0.2%",
    )
    args = parser.parse_args()

    if args.shard == "backtest":
        merge_and_backtest(use_svr=args.use_svr, round_trip_cost=args.cost)
        return

    if not args.start or not args.end:
        parser.error("分片运行时必须指定 --start 和 --end，例如：--start 2019-01-01 --end 2024-12-31")

    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i, args.start, args.end)
        print("\n全部分片完成，请执行: python full_factor_chip_forced_full.py --shard backtest")
    else:
        run_shard(int(args.shard), args.start, args.end)


if __name__ == "__main__":
    main()
