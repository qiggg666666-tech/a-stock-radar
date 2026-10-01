#!/usr/bin/env python3
"""
多周期共振股票池（独立第1步）
================================
只做周/日/月共振筛选，不做筹码与打分。
输出 CSV，供 live_stock_selector.py --universe 使用。

共振条件（与 live_stock_selector.pass_mtf_filter(mode=resonate) 一致）：
  1. 周线收盘在周 MA20 上方
  2. 日线 MA20 溢价 >= -3%
  3. 排除：月线近一年极高位 + 日线已弱
  4. 排除：相对年内低点拉伸过大 + 日线已弱

用法：
  python mtf_resonate_universe.py --shard 1
  python mtf_resonate_universe.py --shard all
  python mtf_resonate_universe.py --shard all --out mtf_resonate_universe.csv

然后：
  python live_stock_selector.py --shard all --universe mtf_resonate_universe.csv
  python live_stock_selector.py --merge --top 15 --notify
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 复用选股脚本的数据与多周期逻辑，避免两套规则漂移
from live_stock_selector import (
    SHARD_MAP,
    MIN_BARS,
    calc_launch_factors,
    calc_multi_timeframe,
    filter_shard,
    get_all_a_stocks,
    get_stock_data,
    pass_mtf_filter,
)

try:
    from final_chip_research import is_beijing_stock

    HAS_BJ = True
except ImportError:

    def is_beijing_stock(code: str) -> bool:
        return str(code).startswith(("8", "4", "9"))

    HAS_BJ = False

OUT_DIR = Path("./mtf_universe_cache")
OUT_DIR.mkdir(exist_ok=True)


def scan_shard(shard_id: int) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"共振宇宙 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
    print(f"{'=' * 60}")

    all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表")
        return pd.DataFrame()

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")

    rows: list[dict[str, Any]] = []
    n_ok = 0
    for i, code in enumerate(codes):
        if is_beijing_stock(code):
            continue
        df = get_stock_data(code)
        if df is None or df.empty or len(df) < MIN_BARS:
            continue
        n_ok += 1
        try:
            df2 = calc_launch_factors(df)
            prem = float(df2.iloc[-1].get("ma20_premium") or 0)
            if not np.isfinite(prem):
                continue
            mtf = calc_multi_timeframe(df2)
            if not pass_mtf_filter(mtf, prem, "resonate"):
                continue
            latest = df2.iloc[-1]
            rows.append(
                {
                    "code": str(code).zfill(6),
                    "date": str(
                        latest["date"].date()
                        if hasattr(latest["date"], "date")
                        else latest["date"]
                    ),
                    "close": round(float(latest["close"]), 2),
                    "ma20_premium": round(prem, 4),
                    "w_ma20": mtf.get("w_ma20"),
                    "w_ma20_premium": mtf.get("w_ma20_premium"),
                    "w_above_ma20": mtf.get("w_above_ma20"),
                    "m_pos_12": mtf.get("m_pos_12"),
                    "m_stretch": mtf.get("m_stretch"),
                    "mtf_label": mtf.get("mtf_label") or "",
                    "mtf_risk": mtf.get("mtf_risk") or "",
                }
            )
        except Exception:
            continue

        if (i + 1) % 40 == 0:
            print(f"  已处理 {i + 1}/{len(codes)} 有效{n_ok} 共振{len(rows)}")

    print(f"分片 {shard_id} 完成: 有效K线{n_ok} 共振命中{len(rows)}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="多周期共振股票池")
    p.add_argument("--shard", type=str, required=True, help="1-8 或 all")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 CSV 路径（默认 mtf_resonate_universe_YYYYMMDD.csv）",
    )
    args = p.parse_args()

    frames: list[pd.DataFrame] = []
    shard_arg = args.shard
    if shard_arg == "all":
        for i in range(1, 9):
            frames.append(scan_shard(i))
    else:
        frames.append(scan_shard(int(shard_arg)))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        print("无共振股票")
        # 分片并行时写空文件，避免 artifact 缺失
        if shard_arg != "all":
            empty = OUT_DIR / f"resonate_shard_{shard_arg}.csv"
            empty.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["code"]).to_csv(empty, index=False, encoding="utf-8-sig")
            print(f"已写空分片: {empty}")
        return

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.drop_duplicates("code").sort_values("code").reset_index(drop=True)

    today = datetime.now().strftime("%Y%m%d")

    # 单分片：只写分片文件，避免并行覆盖 mtf_resonate_universe.csv
    if shard_arg != "all":
        shard_out = Path(args.out) if args.out else (OUT_DIR / f"resonate_shard_{shard_arg}.csv")
        shard_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(shard_out, index=False, encoding="utf-8-sig")
        print(f"\n分片 {shard_arg} 共振 {len(df)} 只 → {shard_out}")
        return

    out = Path(args.out) if args.out else Path(f"mtf_resonate_universe_{today}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    latest = Path("mtf_resonate_universe.csv")
    df.to_csv(latest, index=False, encoding="utf-8-sig")
    shard_cache = OUT_DIR / f"universe_{today}.csv"
    df.to_csv(shard_cache, index=False, encoding="utf-8-sig")

    print(f"\n共振池共 {len(df)} 只")
    print(f"  → {out}")
    print(f"  → {latest}")
    print(f"  → {shard_cache}")
    print(
        "\n下一步:\n"
        f"  python live_stock_selector.py --shard all --universe {latest}\n"
        "  python live_stock_selector.py --merge --top 15 --notify"
    )


if __name__ == "__main__":
    main()
