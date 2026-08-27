#!/usr/bin/env python3
"""FINAL Chip专属汇总器；只在有完成分片时尝试单条通知。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() and path.stat().st_size else pd.DataFrame()


def notify(title: str, body: str) -> dict[str, object]:
    key = os.getenv("SENDKEY", "").strip()
    if not key: return {"status": "skipped", "reason": "missing_sendkey"}
    try:
        import requests
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body}, timeout=20); result = response.json()
        return {"status": "sent", "http_status": response.status_code} if response.ok and result.get("code") == 0 else {"status": "failed", "http_status": response.status_code, "code": result.get("code")}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:250]}"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--shard-total", type=int, default=4); parser.add_argument("--notify", action="store_true"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: assert read_csv(Path("/missing.csv")).empty; print("FINAL_CHIP_SUMMARY_SELF_TEST_OK"); return 0
    if args.input_dir is None or args.output_dir is None: parser.error("--input-dir and --output-dir are required")
    statuses, frames, error_frames = [], [], []
    for status_path in args.input_dir.rglob("status.json"):
        try:
            statuses.append(json.loads(status_path.read_text(encoding="utf-8"))); frames.append(read_csv(status_path.parent / "records.csv")); error_frames.append(read_csv(status_path.parent / "errors.csv"))
        except Exception as exc: statuses.append({"state": "artifact_read_error", "error": f"{type(exc).__name__}:{str(exc)[:180]}"})
    completed = [item for item in statuses if item.get("state") in {"completed", "completed_zero_records"}]; args.output_dir.mkdir(parents=True, exist_ok=True)
    if not completed:
        result = {"schema_version": "final-chip-summary/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "skipped:no_completed_shards", "completed_shards": [], "candidate_count": 0, "notification": {"status": "skipped", "reason": "no_completed_shards"}, "disclosure": "无完成分片，不生成研究结论或通知。"}; (args.output_dir / "final_chip_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (args.output_dir / "final_chip_report.md").write_text("# FINAL Chip汇总\n\n无完成分片，未生成研究结论或通知。\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0
    records = pd.concat([item for item in frames if not item.empty], ignore_index=True) if any(not item.empty for item in frames) else pd.DataFrame(); errors = pd.concat([item for item in error_frames if not item.empty], ignore_index=True) if any(not item.empty for item in error_frames) else pd.DataFrame(columns=["code", "name", "error_type", "error_message"])
    if not records.empty:
        records["is_tradeable"] = records["is_tradeable"].astype(str).str.lower().eq("true") if "is_tradeable" in records else False; records["is_approaching"] = records["is_approaching"].astype(str).str.lower().eq("true") if "is_approaching" in records else False; records["total_score"] = pd.to_numeric(records["total_score"], errors="coerce").fillna(0) if "total_score" in records else 0.0; records = records.sort_values(["is_tradeable", "is_approaching", "total_score"], ascending=[False, False, False])
    candidates = records.head(100); preview = candidates.head(20); candidates.to_csv(args.output_dir / "final_chip_candidates.csv", index=False, encoding="utf-8-sig"); errors.to_csv(args.output_dir / "final_chip_errors.csv", index=False, encoding="utf-8-sig")
    lines = ["# FINAL Chip全市场汇总", "", f"- 完成分片：{len(completed)}/{args.shard_total}", f"- 有效记录：{len(records)}", f"- 错误台账：{len(errors)}", f"- artifact排序：{len(candidates)}", f"- 通知预览：{len(preview)}（最多20条）", "", "## 通知预览"]
    for number, (_, row) in enumerate(preview.iterrows(), 1): lines.append(f"{number:02d}. {row.get('code','')} {row.get('name','')}｜评分{row.get('total_score','')}｜峰距{row.get('dist_to_peak_pct','')}%｜集中度{row.get('conc90_pct','')}%｜{row.get('signal','')}")
    if preview.empty: lines.append("当日无可展示研究记录；详见artifact状态与错误台账。")
    lines.extend(["", "仅为FINAL Chip规则研究输出，不构成投资建议。"]); notification = notify(f"FINAL Chip | {len(completed)}/{args.shard_total}分片 | 展示{len(preview)}/排序{len(candidates)}", "\n".join(lines)) if args.notify else {"status": "not_requested"}
    result = {"schema_version": "final-chip-summary/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "ready" if len(completed) == args.shard_total else "partial", "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed), "record_count": len(records), "error_count": len(errors), "candidate_count": len(candidates), "notification_preview_count": len(preview), "notification": notification, "disclosure": "FINAL Chip专属汇总；通知失败不阻断artifact。"}
    (args.output_dir / "final_chip_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (args.output_dir / "final_chip_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
