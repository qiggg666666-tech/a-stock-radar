#!/usr/bin/env python3
"""
修复版：全市场分片 + HighOpen/CloseLow/VwapClose/M0
+ 强制完整筹码过滤（红筹码占优 ∩ 宽幅堆积区）
+ 横截面回归因子加权 + SVR 回测 + Server酱推送

修复点（相对上一版）：
  1. TOP_N 选股去重：每月每股票只保留月末最后一行信号，再取前 N 只
  2. 因子权重按月聚合后再取最近 12 个月（原来 tail(12) 是 12 个日度截面）
  3. 训练/测试样本对齐：每月每股票仅保留月末信号行，匹配月度调仓节奏
  4. chip_hit 标注：analyze 为全区间静态计算，存在前视偏差，已显式声明
  5. 无回测结果也推送 Server酱

用法：
  python full_factor_chip_forced_full_v2.py --shard 1 --start 2019-01-01 --end 2024-12-31
  python full_factor_chip_forced_full_v2.py --shard all --start 2020-01-01 --end 2025-06-30
  python full_factor_chip_forced_full_v2.py --shard backtest
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
import matplotlib.pyplot as plt
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
FUTURE_DAYS = 10
LOOKBACK_MONTHS = 12

# 训练/测试是否仅使用每月最后一个交易日信号（推荐 True，与月度调仓一致）
MONTH_END_SIGNAL_ONLY = True


# ==================== 工具函数 ====================
def get_all_a_stocks() -> list[str]:
    """获取全A股列表，优先 baostock，失败再尝试 akshare。不做任何兜底股票池。"""
    codes = []

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
    """优先使用 final_chip 的 fetch_ohlcv，失败再降级 akshare"""
    if FULL_CHIP_AVAILABLE:
        try:
            df, source, _ = fetch_ohlcv(symbol, timeout_seconds=30, retries=2)
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
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
            return df[(df["date"] >= start) & (df["date"] <= end)].copy()
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
        df["Turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0
    else:
        df["Turnover"] = np.nan

    for col in ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]:
        if col in df.columns:
            med = df[col].median()
            mad = (df[col] - med).abs().median()
            if mad > 0:
                df[col] = df[col].clip(med - 5 * mad, med + 5 * mad)
    return df


def apply_full_chip_filter(df: pd.DataFrame, code: str, name: str = "") -> pd.DataFrame:
    """
    使用完整 analyze 打上 chip_hit 标签。

    注意（前视偏差声明）：analyze 是对传入的整段 df 做一次静态计算，
    因此 chip_hit 对该股票在整个区间是常量，等效于"用当前筹码状态筛选历史"。
    若需无偏差回测，应让 analyze 支持 as-of 日期滚动计算。
    本脚本在回测结果中已如实标注此限制。
    """
    if not FULL_CHIP_AVAILABLE or df.empty:
        df = df.copy()
        df["chip_hit"] = False
        df["wide_score"] = 0.0
        df["profit_pct"] = 0.0
        return df

    try:
        result = analyze(code, name, df)
        hit = bool(result.get("is_red_heavy_chip", False) and result.get("is_wide_zone", False))
        df = df.copy()
        df["chip_hit"] = hit
        df["wide_score"] = float(result.get("wide_score", 0) or 0)
        df["profit_pct"] = float(result.get("profit_pct", 0) or 0)
        return df
    except Exception:
        df = df.copy()
        df["chip_hit"] = False
        df["wide_score"] = 0.0
        df["profit_pct"] = 0.0
        return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"])
    df["future_ret"] = df.groupby("symbol")["close"].shift(-FUTURE_DAYS) / df["close"] - 1
    factor_cols = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]
    return df.dropna(subset=factor_cols + ["future_ret"])


def month_end_signals(df: pd.DataFrame) -> pd.DataFrame:
    """每月每股票只保留月末最后一行信号，避免同一股票占据 TOP_N 多个席位。"""
    if not MONTH_END_SIGNAL_ONLY:
        return df
    df = df.sort_values(["symbol", "date"]).copy()
    df["ym"] = df["date"].dt.to_period("M")
    out = df.groupby(["symbol", "ym"], as_index=False).tail(1)
    return out


def cross_sectional_reg(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    results = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=factor_cols + ["future_ret"])
        if len(g) < 30:
            continue
        X = (g[factor_cols] - g[factor_cols].mean()) / (g[factor_cols].std() + 1e-8)
        X = sm.add_constant(X)
        try:
            model = sm.OLS(g["future_ret"], X).fit()
            row = {"date": dt, "r2": model.rsquared, "n": len(g)}
            for c in factor_cols:
                row[f"{c}_ret"] = model.params.get(c, np.nan)
                row[f"{c}_t"] = model.tvalues.get(c, np.nan)
            results.append(row)
        except Exception:
            continue
    return pd.DataFrame(results)


def get_factor_weights(reg_df: pd.DataFrame, lookback: int = 12) -> dict[str, float]:
    """
    修复：先按月聚合每日横截面系数，再取最近 lookback 个月求均值。
    （原实现 tail(lookback) 取的是最近的日度截面，自相关严重，实际只有几天信息。）
    """
    factor_cols = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]
    if reg_df.empty:
        return {c: 1.0 / len(factor_cols) for c in factor_cols}

    r = reg_df.copy()
    r["date"] = pd.to_datetime(r["date"])
    r["ym"] = r["date"].dt.to_period("M")
    monthly = r.groupby("ym")[[f"{c}_ret" for c in factor_cols]].mean().reset_index()
    recent = monthly.tail(lookback)

    weights = {c: max(float(recent[f"{c}_ret"].mean()), 0.0) for c in factor_cols}
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(factor_cols) for k, v in weights.items()}
    return weights


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
    print(f"\n{'='*60}")
    print(f"分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
    print(f"回测区间: {start} ~ {end}")
    print(f"{'='*60}")

    all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表，本分片退出")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")

    if len(codes) == 0:
        print("本分片无匹配股票")
        return

    data_list = []
    for i, code in enumerate(codes):
        if FULL_CHIP_AVAILABLE and is_beijing_stock(code):
            continue
        df = get_stock_data(code, start, end)
        if df.empty or len(df) < 80:
            continue

        df = calc_price_factors(df)
        df = apply_full_chip_filter(df, code)
        data_list.append(df)

        if (i + 1) % 30 == 0:
            print(f"  已处理 {i+1}/{len(codes)}")

    if not data_list:
        print("无有效数据")
        return

    all_data = pd.concat(data_list, ignore_index=True)
    dataset = prepare_data(all_data)

    before = len(dataset)
    dataset = dataset[dataset["chip_hit"] == True].copy()
    print(f"完整筹码过滤后样本: {len(dataset)} / {before}")

    shard_file = CACHE_DIR / f"shard_{shard_id}_processed.parquet"
    if len(dataset) > 0:
        dataset.to_parquet(shard_file, index=False)
        print(f"已保存命中数据: {shard_file}")
    else:
        debug_file = CACHE_DIR / f"shard_{shard_id}_debug_empty.txt"
        debug_file.write_text(f"过滤后为空，原始样本数: {before}\n", encoding="utf-8")
        print(f"过滤后为空，已写入调试文件: {debug_file}")

    factor_cols = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]
    reg_df = cross_sectional_reg(dataset, factor_cols)
    if not reg_df.empty:
        reg_df.to_csv(CACHE_DIR / f"reg_shard_{shard_id}.csv", index=False)
        print("横截面回归完成")

    print(f"分片 {shard_id} 完成")


# ==================== 合并回测 + 推送 ====================
def merge_and_backtest() -> None:
    print("\n合并（已强制完整筹码过滤）+ 回归加权 + SVR 回测...")

    files = list(CACHE_DIR.glob("shard_*_processed.parquet"))
    if not files:
        print("请先运行各分片（当前无 parquet 文件）")
        title = "完整筹码过滤策略回测｜无分片数据"
        content = "未找到任何分片 parquet 文件，请先运行分片扫描。"
        print(f"Server酱推送结果: {notify_serverchan(title, content)}")
        return

    dataset = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    print(f"过滤后总样本: {len(dataset)}  股票数: {dataset['symbol'].nunique()}")

    reg_files = list(CACHE_DIR.glob("reg_shard_*.csv"))
    reg_all = (
        pd.concat([pd.read_csv(f) for f in reg_files], ignore_index=True)
        if reg_files
        else pd.DataFrame()
    )
    if not reg_all.empty:
        reg_all["date"] = pd.to_datetime(reg_all["date"])
        reg_all = reg_all.sort_values("date")

    factor_cols = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]
    dataset["ym"] = dataset["date"].dt.to_period("M")
    months = sorted(dataset["ym"].unique())

    # 修复：训练/测试统一用月末信号，确保 TOP_N 是"20 只股票"而非"20 行样本"
    signal = month_end_signals(dataset)

    results = []
    weight_history = []

    for i in range(LOOKBACK_MONTHS, len(months) - 1):
        train = signal[signal["ym"].isin(months[i - LOOKBACK_MONTHS : i])]
        test = signal[signal["ym"] == months[i]]

        if len(train) < 80 or len(test) < 5:
            continue

        if not reg_all.empty:
            hist_reg = reg_all[reg_all["date"] < test["date"].min()]
            weights = get_factor_weights(hist_reg, lookback=LOOKBACK_MONTHS)
        else:
            weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

        weight_history.append({"month": str(months[i]), **weights})

        train = train.copy()
        test = test.copy()
        train["factor_score"] = 0.0
        test["factor_score"] = 0.0

        for col in factor_cols:
            train[col + "_z"] = (train[col] - train[col].mean()) / (train[col].std() + 1e-8)
            test[col + "_z"] = (test[col] - test[col].mean()) / (test[col].std() + 1e-8)
            train["factor_score"] += weights[col] * train[col + "_z"]
            test["factor_score"] += weights[col] * test[col + "_z"]

        X_train = train[factor_cols + ["factor_score"]]
        y_train = train["future_ret"]
        X_test = test[factor_cols + ["factor_score"]]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVR(kernel="rbf", C=1.0, epsilon=0.01)
        model.fit(X_train_s, y_train)
        test["pred"] = model.predict(X_test_s)

        # 修复：test 已经每股票一行，nlargest 选出的就是 TOP_N 只不同股票
        selected = test.nlargest(min(TOP_N, len(test)), "pred")
        port_ret = selected["future_ret"].mean()
        results.append(
            {
                "date": selected["date"].max(),
                "return": port_ret,
                "n": len(selected),
                "n_unique_symbols": selected["symbol"].nunique(),
            }
        )

    if not results:
        print("无回测结果（过滤后样本可能过少）")
        title = "完整筹码过滤策略回测｜无有效结果"
        content = (
            "# 完整筹码过滤 + 价格因子回测\n\n"
            f"- 过滤后总样本：{len(dataset)}\n"
            f"- 股票数：{dataset['symbol'].nunique()}\n"
            f"- 结果：无有效回测区间（可能因筹码过滤后月度样本不足）\n\n"
            "仅为研究输出，不构成投资建议。"
        )
        notify_result = notify_serverchan(title, content)
        print(f"Server酱推送结果: {notify_result}")
        return

    ret_df = pd.DataFrame(results).set_index("date").sort_index()
    ret_df["cum"] = (1 + ret_df["return"]).cumprod()

    total = ret_df["cum"].iloc[-1] - 1
    ann = (1 + total) ** (12 / len(ret_df)) - 1
    sharpe = ret_df["return"].mean() / ret_df["return"].std() * np.sqrt(12)
    maxdd = (ret_df["cum"] / ret_df["cum"].cummax() - 1).min()

    print("\n" + "=" * 55)
    print("完整筹码过滤 + 因子加权 + SVR 回测结果（修复版 v2）")
    print("=" * 55)
    print(f"总收益     : {total:.2%}")
    print(f"年化收益   : {ann:.2%}")
    print(f"夏普比率   : {sharpe:.3f}")
    print(f"最大回撤   : {maxdd:.2%}")
    print(f"调仓次数   : {len(ret_df)}")
    print(f"平均持仓只数: {ret_df['n'].mean():.1f} (去重后)")
    print("=" * 55)

    ret_df.to_csv("full_chip_forced_backtest_v2.csv")
    if weight_history:
        pd.DataFrame(weight_history).to_csv("full_chip_forced_weights_v2.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(ret_df.index, ret_df["cum"])
    plt.title("Full Chip Forced v2 - Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("full_chip_forced_equity_v2.png", dpi=150)
    print("结果文件: full_chip_forced_backtest_v2.csv / full_chip_forced_weights_v2.csv / full_chip_forced_equity_v2.png")

    title = f"完整筹码过滤策略回测v2｜总收益{total:.1%} 夏普{sharpe:.2f}"
    content_lines = [
        "# 完整筹码过滤 + 价格因子 + M0 回测结果（修复版 v2）",
        "",
        f"- **总收益率**：{total:.2%}",
        f"- **年化收益率**：{ann:.2%}",
        f"- **夏普比率**：{sharpe:.3f}",
        f"- **最大回撤**：{maxdd:.2%}",
        f"- **调仓次数**：{len(ret_df)}",
        "",
        "## 因子权重（最近一期，按月聚合）",
    ]
    if weight_history:
        last_w = weight_history[-1]
        for k, v in last_w.items():
            if k != "month":
                content_lines.append(f"- {k}: {v:.3f}")
    content_lines.extend(
        [
            "",
            "## 修复说明",
            "- TOP_N 已按股票去重（每月每股票仅月末信号）",
            "- 因子权重按月聚合后取最近 12 个月",
            "- 已知限制：chip_hit 为全区间静态计算，存在前视偏差",
            "",
            "仅为研究输出，不构成投资建议。",
        ]
    )
    content = "\n".join(content_lines)
    notify_result = notify_serverchan(title, content)
    print(f"Server酱推送结果: {notify_result}")


# ==================== 入口 ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="完整筹码过滤 + 价格因子全市场策略 v2")
    parser.add_argument("--shard", type=str, required=True, help="1-8 / all / backtest")
    parser.add_argument("--start", type=str, help="回测开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="回测结束日期，格式 YYYY-MM-DD")
    args = parser.parse_args()

    if args.shard == "backtest":
        merge_and_backtest()
        return

    if not args.start or not args.end:
        parser.error("分片运行时必须指定 --start 和 --end 日期")

    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i, args.start, args.end)
        print("\n全部分片完成，请执行: python full_factor_chip_forced_full_v2.py --shard backtest")
    else:
        run_shard(int(args.shard), args.start, args.end)


if __name__ == "__main__":
    main()
