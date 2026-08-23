#!/usr/bin/env python3
"""独立DistilledQuant核心全截面汇总器；不读取低位首红结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import quant_distilled_core_research as core


def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="独立DistilledQuant核心汇总")
    parser.add_argument("--input-root", type=Path, default=Path("collected"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("distilled_core_summary"))
    args = parser.parse_args()
    frames, error_frames, statuses, missing = [], [], {}, []
    for index in range(args.shard_count):
        candidates = list(args.input_root.rglob(f"shard-{index}/status.json"))
        if not candidates:
            missing.append(index); statuses[str(index)] = {"status": "missing"}; continue
        folder = candidates[0].parent
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        statuses[str(index)] = status
        raw = safe_csv(folder / "raw_records.csv")
        if not raw.empty: frames.append(raw.assign(source_shard=index))
        errors = safe_csv(folder / "errors.csv")
        if not errors.empty: error_frames.append(errors.assign(source_shard=index))
    raw_all = pd.concat(frames, ignore_index=True).drop_duplicates("code") if frames else pd.DataFrame()
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=["code", "name", "stage", "error_type", "error_message", "attempts", "source_shard"])
    scored = core.score_cross_section(raw_all).head(args.top_n) if not raw_all.empty else pd.DataFrame()
    degraded = [index for index, value in statuses.items() if value.get("status") != "ready"]
    final = "partial" if missing else "degraded" if degraded or not errors_all.empty else "unavailable" if scored.empty else "ready"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "distilled_quant_core_candidates.csv", index=False, encoding="utf-8-sig")
    errors_all.to_csv(args.output_dir / "distilled_quant_core_errors.csv", index=False, encoding="utf-8-sig")
    summary = {"schema_version": "distilled-quant-core-summary/v1", "status": final, "missing_shards": missing, "degraded_shards": degraded, "valid_record_count": len(raw_all), "candidate_count": len(scored), "error_count": len(errors_all), "fixed_factor_weights": core.FACTOR_WEIGHTS, "shard_statuses": statuses, "disclosure": "独立真实OHLCV研究排序；不与520低位首红任务共享输入或输出。"}
    (args.output_dir / "distilled_quant_core_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "distilled_quant_core_summary.md").write_text(f"# DistilledQuant Core\n\n- 状态：`{final}`\n- 候选：{len(scored)}\n- 错误：{len(errors_all)}\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
