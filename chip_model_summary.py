#!/usr/bin/env python3
"""筹码模型独立汇总器。仅在至少一个完成分片后尝试一次通知。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def read_shards(input_dir: Path, shard_total: int) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    candidates: list[pd.DataFrame] = []
    errors: list[pd.DataFrame] = []
    completed: list[int] = []
    for index in range(shard_total):
        folder = input_dir / f"chip-model-shard-{index}"
        candidate_file = folder / "chip_model_candidates.csv"
        error_file = folder / "chip_model_errors.csv"
        audit_file = folder / "chip_model_audit.json"
        if not audit_file.exists() or not candidate_file.exists() or not error_file.exists():
            continue
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        if audit.get("strategy_id") != "chip-model" or audit.get("schema_version") != "chip-model-scan/v1":
            continue
        completed.append(index)
        candidates.append(pd.read_csv(candidate_file))
        errors.append(pd.read_csv(error_file))
    return (pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(), pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(), completed, [index for index in range(shard_total) if index not in completed])


def send_notification(title: str, content: str) -> dict[str, Any]:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped:no_sendkey"}
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=30)
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text[:200]}
        if response.status_code == 200 and payload.get("code") == 0:
            return {"status": "sent", "http_status": response.status_code, "business_code": 0}
        return {"status": "failed:serverchan", "http_status": response.status_code, "business_code": payload.get("code"), "message": str(payload.get("message", ""))[:160]}
    except Exception as exc:  # noqa: BLE001
        return {"status": f"failed:{type(exc).__name__}", "message": str(exc)[:160]}


def main() -> int:
    parser = argparse.ArgumentParser(description="独立全市场筹码模型汇总")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--max-notification-items", type=int, default=20)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    candidates, errors, completed, missing = read_shards(Path(args.input_dir), args.shard_total)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not candidates.empty:
        candidates = candidates.sort_values(["concentration_10_pct", "winner_pct", "premium_pct"], ascending=[False, False, True]).drop_duplicates("code").head(args.max_records).reset_index(drop=True)
        candidates = candidates.drop(columns=["rank"], errors="ignore")
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
    status = "ready" if len(completed) == args.shard_total else ("partial" if completed else "unavailable")
    notification: dict[str, Any] = {"status": "skipped:not_requested"}
    if args.notify:
        if not completed:
            notification = {"status": "skipped:no_completed_shards"}
        else:
            preview = candidates.head(args.max_notification_items) if not candidates.empty else pd.DataFrame()
            lines = [f"完成分片：{len(completed)}/{args.shard_total}", f"有效记录：{len(candidates)}", f"数据错误：{len(errors)}", "", f"观察记录（最多{args.max_notification_items}条）："]
            if preview.empty:
                lines.append("当日无可展示研究记录；完整状态与错误台账见artifact。")
            else:
                for _, row in preview.iterrows():
                    lines.append(f"{int(row['rank'])}. {str(row['code']).zfill(6)} {row['name']}｜集中{row['concentration_10_pct']:.2f}%｜获利{row['winner_pct']:.2f}%｜溢价{row['premium_pct']:.2f}%")
            notification = send_notification(f"独立全市场筹码模型｜{len(completed)}/{args.shard_total}分片｜{len(candidates)}条排序", "\n".join(lines))
    snapshot = {"schema_version": "chip-model-summary/v1", "strategy_id": "chip-model", "status": status, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "expected_shards": args.shard_total, "completed_shards": completed, "missing_shards": missing, "candidate_count": len(candidates), "error_count": len(errors), "notification": notification, "research_disclaimer": "筹码分布为研究统计输出，不构成买卖建议或收益承诺。"}
    candidates.to_csv(output / "chip_model_candidates.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(output / "chip_model_errors.csv", index=False, encoding="utf-8-sig")
    (output / "chip_model_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 独立全市场筹码模型汇总", f"- 状态：`{status}`", f"- 完成分片：`{len(completed)}/{args.shard_total}`", f"- 排序记录：`{len(candidates)}`", f"- 数据错误：`{len(errors)}`", f"- 通知：`{notification['status']}`", "", "> 完整排序、错误台账与通知状态均保留在artifact；消息正文最多展示20条。"]
    if not candidates.empty:
        lines.extend(["", "## 前30条排序", candidates.head(30)[["rank", "code", "name", "concentration_10_pct", "winner_pct", "premium_pct", "daily_data_source"]].to_markdown(index=False)])
    (output / "chip_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"chip-model summary: status={status} completed={len(completed)} candidates={len(candidates)} notification={notification['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
