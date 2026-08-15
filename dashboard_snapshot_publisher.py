#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从单个策略的分片artifact生成公开、可追溯的仪表盘快照，不填充任何虚构候选。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from strategy_shard_notify_summary import SPECS, find_candidate_files, read_candidates, shard_labels


DISPLAY = {
    "bottom-accumulation": {"score_label": "综合评分", "tag_column": "阶段", "extras": ["最新价", "MACD状态", "数据源"]},
    "pattern-breakout": {"score_label": "策略评分", "tag_column": "形态", "extras": ["最新价", "突破状态", "数据源"]},
    "smallcap-trend": {"score_label": "信号评分", "tag_column": "信号类型", "extras": ["流通市值_亿", "最新价", "数据源"]},
    "vcp-fast": {"score_label": "VCP评分", "tag_column": "Breakout", "extras": ["Trend_Score", "RS_Percentile", "最新价"]},
    "bull-confirm": {"score_label": "多头评分", "tag_column": "综合倾向", "extras": ["最新价", "拐点数", "tags"]},
    "yearline-limitup-v5": {"score_label": "综合评分", "tag_column": "", "extras": ["最新价", "涨跌幅", "突破幅度", "板块"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将真实策略artifact生成前端仪表盘快照")
    parser.add_argument("--strategy", choices=sorted(SPECS), required=True)
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-date", default="")
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def row_view(row: dict[str, Any], key: str) -> dict[str, Any]:
    spec = SPECS[key]
    display = DISPLAY[key]
    raw_code = str(row.get("代码", "")).replace(".0", "")
    entry: dict[str, Any] = {
        "code": raw_code.zfill(6) if raw_code.isdigit() else raw_code,
        "name": str(row.get("名称", "未知")),
        "score": json_value(row.get(spec.score_column)),
        "tag": json_value(row.get(display["tag_column"])) if display["tag_column"] else None,
    }
    extras = {column: json_value(row.get(column)) for column in display["extras"] if column in row}
    entry["extras"] = extras
    return entry


def build_snapshot(key: str, input_root: Path, run_date: str, top: int) -> dict[str, Any]:
    spec = SPECS[key]
    paths = find_candidate_files(input_root, spec)
    frame, errors = read_candidates(paths, spec)
    tz = ZoneInfo("Asia/Shanghai")
    generated_at = datetime.now(tz).isoformat(timespec="seconds")
    display = DISPLAY[key]
    return {
        "schema_version": "1.0",
        "strategy": {
            "key": key,
            "label": spec.label,
            "score_label": display["score_label"],
        },
        "run_date": run_date or datetime.now(tz).strftime("%Y-%m-%d"),
        "generated_at": generated_at,
        "status": "ready" if paths else "missing_artifact",
        "candidate_count": int(len(frame)),
        "shards": shard_labels(paths),
        "source_files": [path.name for path in paths],
        "read_errors": errors,
        "candidates": [row_view(row, key) for row in frame.head(max(top, 0)).to_dict("records")],
        "disclaimer": "仅展示自动化筛选产物及其生成时点，不构成投资建议。",
    }


def main() -> int:
    args = parse_args()
    snapshot = build_snapshot(args.strategy, Path(args.input_root), args.run_date, args.top)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"DASHBOARD_SNAPSHOT_READY strategy={args.strategy} candidates={snapshot['candidate_count']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
