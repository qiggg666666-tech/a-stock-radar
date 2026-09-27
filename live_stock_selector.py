#!/usr/bin/env python3
"""
实战选股脚本 —— 启动版 + 年线首红
================================
目标：尽量在股票启动初期选出，降低高位接盘概率。

过滤条件：
  1. 红筹码占优 ∩ 宽幅堆积区
  2. 获利盘 profit_pct 控制在 40% \~ 78%
  3. 优先均线多头 / 金叉
  4. 年线首红加分（刚站上 MA250）

打分因子：
  - wide_score
  - CloseLow（收盘强度）
  - 年线首红加分
  - 适度 Turnover
  - 小权重 HighOpen + 均线加分

用法：
  python live_stock_selector.py --shard 1
  python live_stock_selector.py --shard all
  python live_stock_selector.py --merge --top 20 --notify
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import akshare as ak
import numpy as np
import pandas as pd
import requests

# ==================== 导入完整筹码模块 ====================
try:
    from final_chip_research import analyze, fetch_ohlcv, is_beijing_stock
    FULL_CHIP_AVAILABLE = True
    print("已加载完整 final_chip_research 模块")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：未找到 final_chip_research.py，筹码过滤将失效")

# ==================== 配置 ====================
CACHE_DIR = Path("./live_selector_cache")
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

TOP_N_DEFAULT = 20

# 启动版参数
PROFIT_MIN = 40.0
PROFIT_MAX = 78.0
TURN_OVER_CAP = 0.15

# 因子权重（启动 + 年线首红）
FACTOR_WEIGHTS = {
    "wide_score": 0.30,
    "CloseLow": 0.25,
    "year_first_red": 0.20,
    "Turnover": 0.15,
    "HighOpen": 0.05,
    "ma_bonus": 0.05,
}


# ==================== 工具函数 ====================
def get_all_a_stocks() -> list[str]:
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

    print("错误：无法获取全市场股票列表")
    return []


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def get_stock_data(symbol: str) -> pd.DataFrame:
    if FULL_CHIP_AVAILABLE:
        try:
            df, source, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if not df.empty:
                df["symbol"] = symbol
                return df
        except Exception:
            pass

    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover"
        })
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df = df.sort_values("date").reset_index(drop=True)
        time.sleep(0.25)
        return df
    except Exception:
        return pd.DataFrame()


def calc_launch_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算启动相关因子 + 年线首红"""
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["CloseLow"] = df["close"] / df["low"] - 1
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    df["Turnover"] = df.get("turnover", 0) / 100.0

    # 年线 MA250
    df["ma250"] = df["close"].rolling(250, min_periods=180).mean()
    df["above_ma250"] = df["close"] > df["ma250"]

    # 年线首红（单行写法，避免续行语法错误）
    prev_above = df["above_ma250"].shift(1).fillna(False)
    five_days_ago_below = (df["close"].shift(5) < df["ma250"].shift(5)).fillna(False)
    df["year_first_red"] = (df["above_ma250"] & (prev_above == False) & five_days_ago_below).astype(float)

    return df


def apply_launch_filter(df: pd.DataFrame, code: str) -> dict[str, Any]:
    """启动版过滤 + 年线首红特征"""
    if not FULL_CHIP_AVAILABLE or df.empty or len(df) < 200:
        return {}

    try:
        df = calc_launch_factors(df)
        result = analyze(code, "", df)

        # 1. 必须红筹码 + 宽幅
        if not (result.get("is_red_heavy_chip") and result.get("is_wide_zone")):
            return {}

        profit = float(result.get("profit_pct") or 0)

        # 2. 获利盘区间（避开过高位）
        if not (PROFIT_MIN <= profit <= PROFIT_MAX):
            return {}

        latest = df.iloc[-1]
        ma_signal = str(result.get("ma_signal") or "")

        # 均线加分
        ma_bonus = 0.0
        if "多头" in ma_signal or "金叉" in ma_signal:
            ma_bonus = 1.0
        elif "趋势健康" in ma_signal:
            ma_bonus = 0.6

        year_first_red = float(latest.get("year_first_red", 0))

        return {
            "code": str(code).zfill(6),
            "name": result.get("name", ""),
            "date": str(result.get("date", latest["date"].date())),
            "close": round(float(latest["close"]), 2),
            "HighOpen": round(float(latest["HighOpen"]), 4),
            "CloseLow": round(float(latest["CloseLow"]), 4),
            "close_pos": round(float(latest["close_pos"]), 4),
            "Turnover": round(float(latest.get("Turnover", 0)), 4),
            "profit_pct": round(profit, 2),
            "wide_score": float(result.get("wide_score") or 0),
            "wide_state": result.get("wide_state"),
            "ma_signal": ma_signal,
            "ma_bonus": ma_bonus,
            "year_first_red": year_first_red,
            "ma250": round(float(latest["ma250"]), 2) if pd.notna(latest["ma250"]) else None,
            "total_score": result.get("total_score"),
            "is_red_heavy_chip": True,
            "is_wide_zone": True,
        }
    except Exception:
        return {}


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
            return {"status": "sent"}
        return {"status": "failed", "detail": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200]}


# ==================== 分片扫描 ====================
def run_shard(shard_id: int) -> None:
    print(f"\n{'='*60}")
    print(f"启动版+年线首红 - 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
    print(f"{'='*60}")

    all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")

    records = []
    for i, code in enumerate(codes):
        if FULL_CHIP_AVAILABLE and is_beijing_stock(code):
            continue

        df = get_stock_data(code)
        if df.empty:
            continue

        row = apply_launch_filter(df, code)
        if row:
            records.append(row)

        if (i + 1) % 30 == 0:
            print(f"  已处理 {i+1}/{len(codes)}，当前命中 {len(records)} 只")

    if records:
        df_out = pd.DataFrame(records)
        df_out["code"] = df_out["code"].astype(str).str.zfill(6)
        out_file = CACHE_DIR / f"live_shard_{shard_id}.csv"
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id} 完成，命中 {len(records)} 只 → {out_file}")
    else:
        print(f"分片 {shard_id} 无命中股票")


# ==================== 合并排序 + 推送 ====================
def merge_and_select(top_n: int = 20, do_notify: bool = False) -> None:
    files = list(CACHE_DIR.glob("live_shard_*.csv"))
    if not files:
        print("没有找到任何分片结果，请先运行分片")
        return

    dfs = []
    for f in files:
        tmp = pd.read_csv(f, dtype={"code": str})
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    # 强制 6 位代码
    df["code"] = (
        df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )

    print(f"合并后共 {len(df)} 只命中股票（启动+年线过滤后）")

    if df.empty:
        print("无股票通过过滤")
        return

    # ---------- 标准化打分 ----------
    def safe_z(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / (series.std() + 1e-8)

    df["wide_z"] = safe_z(df["wide_score"]) if "wide_score" in df.columns else 0.0
    df["CloseLow_z"] = safe_z(df["CloseLow"]) if "CloseLow" in df.columns else 0.0

    if "Turnover" in df.columns:
        turn = df["Turnover"].clip(upper=TURN_OVER_CAP)
        df["Turnover_z"] = safe_z(turn)
    else:
        df["Turnover_z"] = 0.0

    df["HighOpen_z"] = safe_z(df["HighOpen"]) if "HighOpen" in df.columns else 0.0
    df["ma_bonus"] = df.get("ma_bonus", 0).fillna(0)
    df["year_first_red"] = df.get("year_first_red", 0).fillna(0)

    df["final_score"] = (
        FACTOR_WEIGHTS["wide_score"] * df["wide_z"]
        + FACTOR_WEIGHTS["CloseLow"] * df["CloseLow_z"]
        + FACTOR_WEIGHTS["year_first_red"] * df["year_first_red"]
        + FACTOR_WEIGHTS["Turnover"] * df["Turnover_z"]
        + FACTOR_WEIGHTS["HighOpen"] * df["HighOpen_z"]
        + FACTOR_WEIGHTS["ma_bonus"] * df["ma_bonus"]
    )

    df = df.sort_values(
        ["final_score", "year_first_red", "wide_score", "profit_pct"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    top_df = df.head(top_n)

    today = datetime.now().strftime("%Y%m%d")
    out_csv = f"live_select_launch_{today}.csv"
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n今日启动+年线优选前 {len(top_df)} 只已保存 → {out_csv}")

    print("\n" + "=" * 85)
    print("启动版 + 年线首红 选股结果")
    print("=" * 85)
    for i, row in top_df.iterrows():
        year_flag = "年线首红" if row.get("year_first_red", 0) > 0.5 else ""
        print(
            f"{i+1:02d}. {row['code']} {row.get('name', '')} "
            f"| 收盘 {row['close']} "
            f"| 得分 {row['final_score']:.3f} "
            f"| 获利盘 {row.get('profit_pct', '-')}% "
            f"| 宽幅分 {row.get('wide_score', '-')} "
            f"| 换手 {row.get('Turnover', 0):.3f} "
            f"| {year_flag} {row.get('ma_signal', '')}"
        )
    print("=" * 85)

    if do_notify:
        lines = [
            f"# 启动版+年线首红选股 {today}",
            "",
            "过滤：红筹码占优 ∩ 宽幅堆积区 + 获利盘 40%\~78%",
            "因子：wide_score + CloseLow + 年线首红 + 适度Turnover + 均线加分",
            f"共推荐 {len(top_df)} 只",
            "",
        ]
        for i, row in top_df.iterrows():
            year_flag = "【年线首红】" if row.get("year_first_red", 0) > 0.5 else ""
            lines.append(
                f"{i+1:02d}. {row['code']} {row.get('name', '')} "
                f"| 收盘{row['close']} | 得分{row['final_score']:.3f} "
                f"| 获利盘{row.get('profit_pct', '-')}% | 宽幅分{row.get('wide_score', '-')} "
                f"| 换手{row.get('Turnover', 0):.3f} {year_flag} | {row.get('ma_signal', '')}"
            )
        lines.append("\n仅为研究输出，不构成投资建议。")
        content = "\n".join(lines)
        title = f"启动+年线选股 {today}｜推荐{len(top_df)}只"
        result = notify_serverchan(title, content)
        print(f"Server酱推送结果: {result}")


# ==================== 入口 ====================
def main():
    parser = argparse.ArgumentParser(description="启动版+年线首红实战选股脚本")
    parser.add_argument("--shard", type=str, help="1-8 或 all")
    parser.add_argument("--merge", action="store_true", help="合并所有分片结果并选股")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="输出前 N 只")
    parser.add_argument("--notify", action="store_true", help="推送 Server酱")
    args = parser.parse_args()

    if args.merge:
        merge_and_select(top_n=args.top, do_notify=args.notify)
        return

    if not args.shard:
        parser.error("请指定 --shard 或 --merge")

    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i)
        print("\n全部分片完成，请执行：")
        print("python live_stock_selector.py --merge --top 20 --notify")
    else:
        run_shard(int(args.shard))


if __name__ == "__main__":
    main()
