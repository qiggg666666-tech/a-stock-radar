#!/usr/bin/env python3
"""
实战选股脚本（精简因子版）
========================
过滤：红筹码占优 ∩ 宽幅堆积区
因子：Turnover + VwapClose + HighOpen（去掉 CloseLow 和 M0）

修复：股票代码强制 6 位，防止 CSV 读写丢失前导零

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

# 实战推荐因子及默认权重
FACTOR_WEIGHTS = {
    "Turnover": 0.70,
    "VwapClose": 0.20,
    "HighOpen": 0.10,
}


# ==================== 工具函数 ====================
def get_all_a_stocks() -> list[str]:
    """获取全A股列表（优先 baostock）"""
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
    """获取单只股票近期数据"""
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


def calc_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """只计算实战需要的三个因子"""
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["vwap"] = np.where(
        (df["volume"] > 0) & (df["amount"] > 0),
        df["amount"] / (df["volume"] * 100 + 1e-8),
        (df["high"] + df["low"] + df["close"]) / 3,
    )
    df["VwapClose"] = df["vwap"] / df["close"] - 1
    df["Turnover"] = df.get("turnover", 0) / 100.0
    return df


def apply_full_chip_filter(df: pd.DataFrame, code: str) -> dict[str, Any]:
    """返回最新一天的筹码 + 因子结果（只保留推荐因子）"""
    if not FULL_CHIP_AVAILABLE or df.empty or len(df) < 60:
        return {}

    try:
        df = calc_price_factors(df)
        result = analyze(code, "", df)

        # 强制过滤：红筹码占优 ∩ 宽幅堆积区
        if not (result.get("is_red_heavy_chip") and result.get("is_wide_zone")):
            return {}

        latest = df.iloc[-1]
        return {
            "code": str(code).zfill(6),          # 强制 6 位
            "name": result.get("name", ""),
            "date": str(result.get("date", latest["date"].date())),
            "close": round(float(latest["close"]), 2),
            "HighOpen": round(float(latest["HighOpen"]), 4),
            "VwapClose": round(float(latest["VwapClose"]), 4),
            "Turnover": round(float(latest.get("Turnover", 0)), 4),
            "profit_pct": result.get("profit_pct"),
            "wide_score": result.get("wide_score"),
            "wide_state": result.get("wide_state"),
            "ma_signal": result.get("ma_signal"),
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
    print(f"实战选股（精简因子）- 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
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

        row = apply_full_chip_filter(df, code)
        if row:
            records.append(row)

        if (i + 1) % 30 == 0:
            print(f"  已处理 {i+1}/{len(codes)}，当前命中 {len(records)} 只")

    if records:
        df_out = pd.DataFrame(records)
        # 再次确保 code 是字符串且 6 位
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

    # 强制按字符串读取 code，防止前导零丢失
    dfs = []
    for f in files:
        tmp = pd.read_csv(f, dtype={"code": str})
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    # 统一补全为 6 位股票代码
    df["code"] = (
        df["code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    print(f"合并后共 {len(df)} 只命中股票")

    if df.empty:
        print("无股票通过过滤")
        return

    # 只使用推荐的三个因子做标准化打分
    factor_cols = ["Turnover", "VwapClose", "HighOpen"]
    for col in factor_cols:
        if col in df.columns:
            std = df[col].std()
            df[col + "_z"] = (df[col] - df[col].mean()) / (std + 1e-8)
        else:
            df[col + "_z"] = 0.0

    # 加权综合得分
    df["factor_score"] = 0.0
    for col, w in FACTOR_WEIGHTS.items():
        if col + "_z" in df.columns:
            df["factor_score"] += w * df[col + "_z"]

    # 加入宽幅得分（次要）
    if "wide_score" in df.columns:
        df["final_score"] = df["factor_score"] * 0.75 + (df["wide_score"] / 100.0) * 0.25
    else:
        df["final_score"] = df["factor_score"]

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    top_df = df.head(top_n)

    # 保存结果
    today = datetime.now().strftime("%Y%m%d")
    out_csv = f"live_select_{today}.csv"
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n今日推荐前 {len(top_df)} 只已保存 → {out_csv}")

    # 打印预览
    print("\n" + "=" * 75)
    print(f"今日实战选股（红筹码占优 ∩ 宽幅堆积区 + Turnover/VwapClose/HighOpen）")
    print("=" * 75)
    for i, row in top_df.iterrows():
        print(
            f"{i+1:02d}. {row['code']} {row.get('name', '')} "
            f"| 收盘 {row['close']} "
            f"| 得分 {row['final_score']:.3f} "
            f"| 换手 {row.get('Turnover', 0):.3f} "
            f"| 宽幅分 {row.get('wide_score', '-')} "
            f"| {row.get('ma_signal', '')}"
        )
    print("=" * 75)

    # Server酱推送
    if do_notify:
        lines = [
            f"# 实战选股 {today}（精简因子版）",
            f"",
            f"过滤：红筹码占优 ∩ 宽幅堆积区",
            f"因子：Turnover({FACTOR_WEIGHTS['Turnover']}) + VwapClose({FACTOR_WEIGHTS['VwapClose']}) + HighOpen({FACTOR_WEIGHTS['HighOpen']})",
            f"共推荐 {len(top_df)} 只",
            f"",
        ]
        for i, row in top_df.iterrows():
            lines.append(
                f"{i+1:02d}. {row['code']} {row.get('name', '')} "
                f"| 收盘{row['close']} | 得分{row['final_score']:.3f} "
                f"| 换手{row.get('Turnover', 0):.3f} | 宽幅分{row.get('wide_score', '-')} | {row.get('ma_signal', '')}"
            )
        lines.append("\n仅为研究输出，不构成投资建议。")
        content = "\n".join(lines)
        title = f"实战选股 {today}｜推荐{len(top_df)}只（精简因子）"
        result = notify_serverchan(title, content)
        print(f"Server酱推送结果: {result}")


# ==================== 入口 ====================
def main():
    parser = argparse.ArgumentParser(description="实战选股脚本（精简因子版）")
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
