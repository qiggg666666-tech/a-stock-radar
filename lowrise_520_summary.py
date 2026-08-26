#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立520日低位首红汇总器；不读取DistilledQuant核心artifact。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PRIORITY_DISTANCE_PCT = 5.0


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def completed_shards(statuses: dict[str, dict[str, Any]]) -> list[str]:
    failed = {"", "missing", "unavailable", "failed", "error"}
    return [index for index, value in statuses.items() if str(value.get("status", "")).lower() not in failed]


def notification_body(summary: dict[str, Any], observations: pd.DataFrame) -> str:
    lines = [
        f"- 状态：`{summary['status']}`",
        f"- 已完成分片：`{len(summary['completed_shards'])}/{summary['shard_count']}`",
        f"- 观察：`{summary['observation_count']}`",
        f"- 错误：`{summary['error_count']}`",
    ]
    fields = [field for field in ["priority_rank", "distance_priority_group", "code", "name", "signal_date", "distance_to_520_low_pct", "volume_ratio_previous_5"] if field in observations.columns]
    if fields:
        lines.extend(["", "## 低位首红观察"])
        for _, row in observations[fields].head(20).iterrows():
            label = "优先＜5%" if row.get("distance_priority_group", "") == "priority_under_5pct" else "原版保留"
            lines.append(f"{row.get('priority_rank', '-')}. 【{label}】{row.get('code', '')} {row.get('name', '')} / 距520低点 {row.get('distance_to_520_low_pct', '')}")
    else:
        lines.extend(["", "当日没有符合当前低位首红观察条件的记录；完整状态和错误审计见artifact。"])
    lines.extend(["", "> 该输出为独立研究观察，不构成投资建议。"])
    return "\n".join(lines)


def notify_serverchan(summary: dict[str, Any], observations: pd.DataFrame) -> str:
    if not summary["completed_shards"]:
        return "skipped:no_completed_shards"
    sendkey = os.getenv("SENDKEY", "").strip()
    if not sendkey:
        return "skipped:no_sendkey"
    title = f"LowRise 520：{summary['status']} / 观察{summary['observation_count']}条"
    try:
        response = requests.post(
            f"https://sctapi.ftqq.com/{sendkey}.send",
            data={"title": title, "desp": notification_body(summary, observations)[:3800]},
            timeout=15,
        )
        payload = response.json()
        code = payload.get("code") if isinstance(payload, dict) else None
        return "sent" if response.ok and code == 0 else f"failed:http_{response.status_code}:serverchan_code_{code}"
    except (requests.RequestException, ValueError) as exc:
        return f"failed:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description="独立520日低位首红A-D分片汇总器")
    parser.add_argument("--input-root", type=Path, default=Path("collected"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("lowrise_summary"))
    parser.add_argument("--notify", action="store_true", help="仅在至少一个完成分片后尝试发送一条Server酱消息")
    args = parser.parse_args()
    raws: list[pd.DataFrame] = []
    errors_list: list[pd.DataFrame] = []
    statuses: dict[str, dict[str, Any]] = {}
    missing: list[int] = []
    for index in range(args.shard_count):
        matches = list(args.input_root.rglob(f"shard-{index}/status.json"))
        if not matches:
            missing.append(index)
            statuses[str(index)] = {"status": "missing"}
            continue
        folder = matches[0].parent
        statuses[str(index)] = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        raw = read(folder / "raw_records.csv")
        errors = read(folder / "errors.csv")
        if not raw.empty:
            raws.append(raw.assign(source_shard=index))
        if not errors.empty:
            errors_list.append(errors.assign(source_shard=index))
    raw = pd.concat(raws, ignore_index=True).drop_duplicates("code") if raws else pd.DataFrame()
    errors = pd.concat(errors_list, ignore_index=True) if errors_list else pd.DataFrame(columns=["code", "name", "stage", "error_type", "error_message", "attempts", "source_shard"])
    observations = raw.loc[raw["lowrise_observation"].fillna(False).astype(bool)].copy() if not raw.empty else pd.DataFrame()
    if not observations.empty:
        distances = pd.to_numeric(observations["distance_to_520_low_pct"], errors="coerce").fillna(float("inf"))
        observations["distance_priority_group"] = distances.lt(PRIORITY_DISTANCE_PCT).map({True: "priority_under_5pct", False: "standard_5pct_or_more"})
        observations["_priority_sort"] = distances.ge(PRIORITY_DISTANCE_PCT).astype(int)
        observations = observations.sort_values(["_priority_sort", "distance_to_520_low_pct", "volume_ratio_previous_5"], ascending=[True, True, False]).drop(columns=["_priority_sort"]).reset_index(drop=True)
        observations["priority_rank"] = range(1, len(observations) + 1)
    degraded = [index for index, value in statuses.items() if value.get("status") != "ready"]
    status = "partial" if missing else "degraded" if degraded or not errors.empty else "unavailable" if raw.empty else "ready"
    summary = {
        "schema_version": "lowrise-520-summary/v1",
        "status": status,
        "missing_shards": missing,
        "degraded_shards": degraded,
        "completed_shards": completed_shards(statuses),
        "shard_count": args.shard_count,
        "valid_record_count": len(raw),
        "observation_count": len(observations),
        "error_count": len(errors),
        "priority_distance_pct_exclusive": PRIORITY_DISTANCE_PCT,
        "priority_candidates_under_5pct": int((observations.get("distance_priority_group", pd.Series(dtype=str)) == "priority_under_5pct").sum()),
        "standard_candidates_5pct_or_more": int((observations.get("distance_priority_group", pd.Series(dtype=str)) == "standard_5pct_or_more").sum()),
        "distance_priority_is_filter": False,
        "shard_statuses": statuses,
        "disclosure": "独立520日低位首红观察任务；不与DistilledQuant核心任务共享输入或输出。",
    }
    summary["notification"] = notify_serverchan(summary, observations) if args.notify else "skipped:notify_not_requested"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "lowrise_520_observations.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(args.output_dir / "lowrise_520_errors.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "lowrise_520_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "lowrise_520_summary.md").write_text(
        "\n".join([
            "# 520低位首红",
            "",
            f"- 状态：`{status}`",
            f"- 观察：`{len(observations)}`",
            f"- 排序：距520日低点严格小于`{PRIORITY_DISTANCE_PCT:.1f}%`的候选置顶；其余原版候选保留",
            f"- 通知：`{summary['notification']}`",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
