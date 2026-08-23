#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总DistilledQuant核心研究排序与520低位首红观察层的A-D分片输出。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import quant_distilled_core_research as core


def find_shard_directory(root: Path, shard_index: int) -> Path | None:
    direct = root / f"shard-{shard_index}"
    if direct.exists():
        return direct
    matches = sorted(path for path in root.rglob("status.json") if path.parent.name in {f"shard-{shard_index}", f"shard_{shard_index}"})
    return matches[0].parent if matches else None


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="DistilledQuant低位首红协同全截面汇总")
    parser.add_argument("--input-root", type=Path, default=Path("collected"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("summary_output"))
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()
    if args.shard_count < 1 or args.top_n < 1:
        parser.error("shard-count and top-n must be positive")
    raw_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    shard_statuses: dict[str, Any] = {}
    missing: list[int] = []
    for shard_index in range(args.shard_count):
        location = find_shard_directory(args.input_root, shard_index)
        if location is None:
            missing.append(shard_index)
            shard_statuses[str(shard_index)] = {"status": "missing", "error_count": None}
            continue
        status_path = location / "status.json"
        if not status_path.exists():
            missing.append(shard_index)
            shard_statuses[str(shard_index)] = {"status": "missing_status", "error_count": None}
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        shard_statuses[str(shard_index)] = status
        raw = read_csv_or_empty(location / "raw_records.csv")
        if not raw.empty:
            raw["source_shard"] = shard_index
            raw_frames.append(raw)
        errors = read_csv_or_empty(location / "errors.csv")
        if not errors.empty:
            errors["source_shard"] = shard_index
            error_frames.append(errors)
    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(
        columns=["code", "name", "stage", "error_type", "error_message", "attempts", "source_shard"]
    )
    if not raw_all.empty:
        raw_all = raw_all.drop_duplicates("code", keep="first")
        scored = core.score_cross_section(raw_all)
        core_candidates = scored.head(args.top_n).copy()
        lowrise = scored.loc[scored["lowrise_observation"].fillna(False).astype(bool)].copy()
        lowrise = lowrise.sort_values(["distance_to_520_low_pct", "volume_ratio_previous_5", "distilled_research_score"], ascending=[True, False, False]).reset_index(drop=True)
        lowrise["lowrise_priority_rank"] = range(1, len(lowrise) + 1)
    else:
        scored = pd.DataFrame()
        core_candidates = pd.DataFrame()
        lowrise = pd.DataFrame()
    degraded = [index for index, payload in shard_statuses.items() if payload.get("status") != "ready"]
    if missing:
        final_status = "partial"
        reasons = ["missing_shards"]
    elif degraded or not errors_all.empty:
        final_status = "degraded"
        reasons = ["degraded_shards_or_data_errors"]
    elif scored.empty:
        final_status = "unavailable"
        reasons = ["no_valid_records"]
    else:
        final_status = "ready"
        reasons = []
    signal_dates = sorted({str(value) for value in raw_all.get("data_last_date", pd.Series(dtype=str)).dropna().unique()})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    core_candidates.to_csv(output / "distilled_quant_core_candidates.csv", index=False, encoding="utf-8-sig")
    lowrise.to_csv(output / "lowrise_520_observations.csv", index=False, encoding="utf-8-sig")
    errors_all.to_csv(output / "joint_errors.csv", index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": "distilled-quant-lowrise-summary/v1",
        "status": final_status,
        "reasons": reasons,
        "shard_count_expected": args.shard_count,
        "shard_count_received": len(shard_statuses),
        "missing_shards": missing,
        "degraded_shards": degraded,
        "valid_record_count": int(len(scored)),
        "error_count": int(len(errors_all)),
        "core_candidate_count": int(len(core_candidates)),
        "lowrise_observation_count": int(len(lowrise)),
        "data_last_dates_seen": signal_dates,
        "fixed_factor_weights": core.FACTOR_WEIGHTS,
        "disclosure": "DistilledQuant为真实OHLCV横截面研究排序；520低位首红为观察条件。两者均不构成收益预测、交易建议或仓位建议。",
        "shard_statuses": shard_statuses,
    }
    (output / "joint_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# DistilledQuant + 520低位首红协同研究汇总",
        "",
        f"- 状态：`{final_status}`",
        f"- 分片：{len(shard_statuses)}/{args.shard_count}，缺失：{missing or '无'}`",
        f"- 有效研究记录：{len(scored)}，错误记录：{len(errors_all)}",
        f"- DistilledQuant研究候选：{len(core_candidates)}",
        f"- 520低位首红观察记录：{len(lowrise)}",
        "",
        "> 任何缺片或源错误均在JSON/CSV中保留；研究排序和观察条件均不构成个性化投资建议。",
    ]
    (output / "joint_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
