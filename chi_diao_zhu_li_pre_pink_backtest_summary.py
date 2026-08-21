#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并严格预备首粉无前视回测分片，并生成唯一总览产物。"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from chi_diao_zhu_li_pre_pink_backtest import TRADE_COLUMNS, summarize_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="严格预备首粉20交易日回测汇总器")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for path in sorted(args.input_dir.rglob("pre_pink_backtest_*.state.json")):
        try: states.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception: pass
    for path in sorted(args.input_dir.rglob("pre_pink_backtest_*.trades.csv")):
        try: frames.append(pd.read_csv(path, dtype={"代码": str}))
        except Exception: pass
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRADE_COLUMNS)
    if not trades.empty:
        trades = trades.drop_duplicates(["代码", "信号类型", "信号日期"], keep="first").sort_values(["信号日期", "信号类型", "净收益%"], ascending=[True, True, False])
    trades.to_csv(args.output_dir / "pre_pink_20d_backtest_trades.csv", index=False, encoding="utf-8-sig")
    web_dir = args.output_dir / "web_pages"
    web_dir.mkdir(parents=True, exist_ok=True)
    page_size = 50
    page_count = int(math.ceil(len(trades) / page_size)) if not trades.empty else 0
    page_columns = ["代码", "名称", "信号类型", "信号日期", "入场日期", "退出日期", "净收益%", "最大不利波动%", "最大有利波动%", "信号评分", "预备首粉评分", "多周期评分", "多周期通过数", "多周期缺失条件"]
    for page_number in range(1, page_count + 1):
        start, end = (page_number - 1) * page_size, page_number * page_size
        page_frame = trades.iloc[start:end][[column for column in page_columns if column in trades.columns]].copy()
        page_frame = page_frame.replace([float("inf"), float("-inf")], np.nan).astype(object).where(lambda value: pd.notna(value), None)
        rows = page_frame.to_dict("records")
        payload = {"page": page_number, "page_size": page_size, "total_rows": int(len(trades)), "rows": rows}
        (web_dir / f"page-{page_number:03d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    by_type = {kind: summarize_trades(group) for kind, group in trades.groupby("信号类型", dropna=False)} if not trades.empty else {}
    processed = sum(int(item.get("processed_codes") or 0) for item in states)
    errors = sum(len(item.get("errors") or []) for item in states)
    skipped: dict[str, int] = {}
    for item in states:
        for key, value in (item.get("skipped") or {}).items(): skipped[key] = skipped.get(key, 0) + int(value)
    first_period = (states[0].get("period") or {}) if states else {}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "state": "completed" if states else "missing_input", "shards": len(states), "processed_codes": processed, "universe_count": int(states[0].get("universe_count") or 0) if states else 0, "errors": errors, "skipped": skipped, "period": first_period, "summary": summarize_trades(trades), "by_signal_type": by_type, "web_pages": {"page_size": page_size, "page_count": page_count, "total_rows": int(len(trades)), "path": "web_snapshots/pre_pink_20d_backtest_pages/page-###.json"}, "disclosure": "信号仅使用信号日及此前日线，入场为下一交易日开盘，退出为入场后第20个交易日收盘；当前股票池回放有存续股票偏差，结果仅作研究。"}
    (args.output_dir / "pre_pink_20d_backtest_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 严格预备首粉：20交易日无前视回测", "", f"- 状态：`{report['state']}`", f"- 分片：{report['shards']}；已处理代码：{processed}；错误：{errors}", f"- 逐笔样本：{report['summary']['trades']}", f"- 胜率：{report['summary']['win_rate_pct']}", f"- 平均净收益%：{report['summary']['mean_net_return_pct']}", f"- 中位净收益%：{report['summary']['median_net_return_pct']}", f"- 最差单笔最大不利波动%：{report['summary']['worst_trade_mae_pct']}", f"- 最佳单笔最大有利波动%：{report['summary']['best_trade_mfe_pct']}", "", "> 信号日后下一交易日开盘入场，第20个交易日收盘退出；不使用未来日线，但当前股票池回放存在存续股票偏差。仅供研究，不构成投资建议。"]
    (args.output_dir / "pre_pink_20d_backtest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
