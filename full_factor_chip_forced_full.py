#!/usr/bin/env python3
"""
完整更新版：全市场分片 + HighOpen/CloseLow/VwapClose/M0
+ 强制完整筹码过滤（红筹码占优 ∩ 宽幅堆积区）
+ 横截面回归因子加权 + SVR 回测 + Server酱推送

修复点：
  1. 动态调整回看月数（解决只有7个月数据时无法回测的问题）
  2. 大幅放宽月度回测样本门槛（train≥30, test≥3）
  3. 打印每个月样本数量，方便排查
  4. 即使没有回测结果也会推送提示
  5. 去掉兜底股票池，日期由用户输入
  6. 股票列表优先 baostock

用法：
  python full_factor_chip_forced_full.py --shard 1 --start 2019-01-01 --end 2024-12-31
  python full_factor_chip_forced_full.py --shard all --start 2020-01-01 --end 2025-06-30
  python full_factor_chip_forced_full.py --shard backtest
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
    df["Turnover"] = df.get("turnover", 0) / 100.0

    for col in ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]:
        if col in df.columns:
            med = df[col].median()
            mad = (df[col] - med).abs().median()
            if mad > 0:
                df[col] = df[col].clip(med - 5 * mad, med + 5 * mad)
    return df


def apply_full_chip_filter(df: pd.DataFrame, code: str, name: str = "") -> pd.DataFrame:
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
    factor_cols = ["HighOpen", "CloseLow", "VwapClose", "M0", "Turnover"]
    if reg_df.empty:
        return {c: 1.0 / len(factor_cols) for c in factor_cols}
    recent = reg_df.tail(lookback)
    weights = {c: max(float(recent[f"{c}_ret"].mean()), 0.0) for c in factor_cols}
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(factor_cols) for k in factor_cols}
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
    print(f"回测区间: {start} \~ {end}")
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


# ==================== 合并回测 + 推送（动态回看 + 放宽版） ====================
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
    print(f"日期范围: {dataset['date'].min()} \~ {dataset['date'].max()}")

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
    print(f"共有 {len(months)} 个月度区间: {months[0]} \~ {months[-1]}")

    # ========== 动态调整回看月数 ==========
    lookback = min(LOOKBACK_MONTHS, max(3, len(months) // 3))
    print(f"实际使用回看月数: {lookback}（原始设置 {LOOKBACK_MONTHS}）")

    results = []
    weight_history = []

    for i in range(lookback, len(months) - 1):
        train = dataset[dataset["ym"].isin(months[i - lookback : i])]
        test = dataset[dataset["ym"] == months[i]]

        print(f"  检查 {months[i]}: train={len(train)}, test={len(test)}")

        # 大幅放宽门槛
        if len(train) < 30 or len(test) < 3:
            continue

        if not reg_all.empty:
            hist_reg = reg_all[reg_all["date"] < test["date"].min()]
            weights = get_factor_weights(hist_reg, lookback=12)
        else:
            weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

        weight_history.append({"month": str(months[i]), **weights})

        train = train.copy()
        test = test.copy()
        train["factor_score"] = 0.0
        test["factor_score"] = 0.0

        for col in factor_cols:
            train_std = train[col].std()
            test_std = test[col].std()
            train[col + "_z"] = (train[col] - train[col].mean()) / (train_std + 1e-8)
            test[col + "_z"] = (test[col] - test[col].mean()) / (test_std + 1e-8)
            train["factor_score"] += weights[col] * train[col + "_z"]
            test["factor_score"] += weights[col] * test[col + "_z"]

        try:
            X_train = train[factor_cols + ["factor_score"]]
            y_train = train["future_ret"]
            X_test = test[factor_cols + ["factor_score"]]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = SVR(kernel="rbf", C=1.0, epsilon=0.01)
            model.fit(X_train_s, y_train)
            test["pred"] = model.predict(X_test_s)

            selected = test.nlargest(min(TOP_N, len(test)), "pred")
            port_ret = selected["future_ret"].mean()
            results.append(
                {
                    "date": selected["date"].max(),
                    "return": port_ret,
                    "n": len(selected),
                }
            )
            print(f"    → 有效回测点，选股 {len(selected)} 只，收益 {port_ret:.4f}")
        except Exception as e:
            print(f"    → 本月训练失败: {e}")
            continue

    # ---------- 无回测结果也推送 ----------
    if not results:
        print("无回测结果（过滤后样本可能过少）")
        title = "完整筹码过滤策略回测｜无有效结果"
        content = (
            "# 完整筹码过滤 + 价格因子回测\n\n"
            f"- 过滤后总样本：{len(dataset)}\n"
            f"- 股票数：{dataset['symbol'].nunique()}\n"
            f"- 月份数：{len(months)}\n"
            f"- 实际回看月数：{lookback}\n"
            f"- 结果：无有效回测区间\n\n"
            "仅为研究输出，不构成投资建议。"
        )
        notify_result = notify_serverchan(title, content)
        print(f"Server酱推送结果: {notify_result}")
        return

    # ---------- 有回测结果 ----------
    ret_df = pd.DataFrame(results).set_index("date").sort_index()
    ret_df["cum"] = (1 + ret_df["return"]).cumprod()

    total = ret_df["cum"].iloc[-1] - 1
    ann = (1 + total) ** (12 / max(len(ret_df), 1)) - 1 if len(ret_df) > 0 else 0
    sharpe = ret_df["return"].mean() / (ret_df["return"].std() + 1e-8) * np.sqrt(12)
    maxdd = (ret_df["cum"] / ret_df["cum"].cummax() - 1).min()

    print("\n" + "=" * 55)
    print("完整筹码过滤 + 因子加权 + SVR 回测结果")
    print("=" * 55)
    print(f"总收益     : {total:.2%}")
    print(f"年化收益   : {ann:.2%}")
    print(f"夏普比率   : {sharpe:.3f}")
    print(f"最大回撤   : {maxdd:.2%}")
    print(f"调仓次数   : {len(ret_df)}")
    print("=" * 55)

    ret_df.to_csv("full_chip_forced_backtest.csv")
    if weight_history:
        pd.DataFrame(weight_history).to_csv("full_chip_forced_weights.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(ret_df.index, ret_df["cum"])
    plt.title("Full Chip Forced + HighOpen/CloseLow/VwapClose/M0 - Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("full_chip_forced_equity.png", dpi=150)
    print("结果文件已保存：full_chip_forced_backtest.csv / full_chip_forced_weights.csv / full_chip_forced_equity.png")

    # Server酱推送
    title = f"完整筹码过滤策略回测｜总收益{total:.1%} 夏普{sharpe:.2f}"
    content_lines = [
        "# 完整筹码过滤 + 价格因子 + M0 回测结果",
        "",
        f"- **总收益率**：{total:.2%}",
        f"- **年化收益率**：{ann:.2%}",
        f"- **夏普比率**：{sharpe:.3f}",
        f"- **最大回撤**：{maxdd:.2%}",
        f"- **调仓次数**：{len(ret_df)}",
        f"- **实际回看月数**：{lookback}",
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
            "过滤条件：红筹码占优 ∩ 宽幅堆积区（完整 analyze）",
            "因子：HighOpen + CloseLow + VwapClose + M0",
            "",
            "仅为研究输出，不构成投资建议。",
        ]
    )
    content = "\n".join(content_lines)
    notify_result = notify_serverchan(title, content)
    print(f"Server酱推送结果: {notify_result}")


# ==================== 入口 ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="完整筹码过滤 + 价格因子全市场策略")
    parser.add_argument("--shard", type=str, required=True, help="1-8 / all / backtest")
    parser.add_argument("--start", type=str, help="回测开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="回测结束日期，格式 YYYY-MM-DD")
    args = parser.parse_args()

    if args.shard == "backtest":
        merge_and_backtest()
        return

    if not args.start or not args.end:
        parser.error("分片运行时必须指定 --start 和 --end 日期，例如：--start 2019-01-01 --end 2024-12-31")

    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i, args.start, args.end)
        print("\n全部资料片完成，请执行: python full_factor_chip_forced_full.py --shard backtest")
    else:
        run_shard(int(args.shard), args.start, args.end)


if __name__ == "__main__":
    main()
