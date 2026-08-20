#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原版520首红每日汇总器：宽松定义、严格当日口径。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def read_records(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [{str(key): str(value or "") for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]
    except Exception:
        return []


def candidate_line(item: dict[str, str]) -> str:
    return (
        f"- {item.get('code', '')} {item.get('name', '')}｜{item.get('signal_date', '')}"
        f"｜量比{item.get('volume_ratio', '')}｜距520低点{item.get('distance_to_520_low_pct', '')}%"
    )


def split_bodies(header: list[str], candidates: list[dict[str, str]], footer: list[str], max_chars: int = 3400) -> list[str]:
    lines = [candidate_line(item) for item in candidates]
    if not lines:
        return ["\n".join(header + ["- 当日无满足原版首红定义的检测结果。"] + footer)]
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and len("\n".join(header + current + [line] + footer)) > max_chars:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    return ["\n".join(header + [f"- 推送分段：{index}/{len(groups)}"] + group + footer) for index, group in enumerate(groups, start=1)]


def send_serverchan(title: str, body: str) -> str:
    key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
    if not key:
        return "skipped:no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body[:3800]}, timeout=15)
        payload = response.json()
        return "sent" if response.ok and payload.get("code") == 0 else f"failed:http_{response.status_code}:code_{payload.get('code', 'unknown')}"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def load_prepare_status(root: Path) -> dict[str, Any] | None:
    for path in root.rglob("a_share_universe_status.json"):
        payload = read_json(path)
        if payload:
            return payload
    return None


def consolidate(input_root: Path, output: Path, notify: bool) -> dict[str, Any]:
    states = [state for path in sorted(input_root.rglob("first_red_520_?.state.json")) if (state := read_json(path))]
    universe = max((int((item.get("universe") or {}).get("count") or 0) for item in states), default=0)
    processed = sum(int((item.get("stats") or {}).get("processed") or 0) for item in states)
    errors = sum(int((item.get("stats") or {}).get("source_error") or 0) for item in states)
    asof_dates = {str(item.get("max_data_last_date") or "") for item in states if item.get("max_data_last_date")}
    asof_date = max(asof_dates) if asof_dates else None
    stale = sorted(str(item.get("shard") or "") for item in states if asof_date and item.get("max_data_last_date") != asof_date)
    valid_shards = {str(item.get("shard") or "") for item in states if asof_date and item.get("max_data_last_date") == asof_date}
    records: list[dict[str, str]] = []
    for path in sorted(input_root.rglob("first_red_520_?.csv")):
        shard = path.stem.rsplit("_", 1)[-1]
        if shard in valid_shards:
            records.extend(read_records(path))
    candidates = [item for item in records if asof_date and item.get("signal_date") == asof_date and item.get("data_last_date") == asof_date]
    candidates.sort(key=lambda item: (-float(item.get("volume_ratio") or 0), float(item.get("distance_to_520_low_pct") or 999), item.get("code") or ""))
    fields = list(dict.fromkeys(key for item in candidates for key in item)) or ["code", "name", "signal_date", "data_last_date", "volume_ratio", "distance_to_520_low_pct"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "first_red_520_original_daily.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    write_json(output / "first_red_520_original_daily.json", candidates)
    coverage = round(processed * 100 / universe, 4) if universe else 0.0
    error_rate = round(errors * 100 / processed, 4) if processed else 100.0
    gate = {
        "expected_shards": 5,
        "visible_shards": len(states),
        "universe_count": universe,
        "processed": processed,
        "coverage_pct": coverage,
        "coverage_complete": bool(universe and processed == universe),
        "source_errors": errors,
        "source_error_rate_pct": error_rate,
        "max_source_error_rate_pct": 5.0,
        "asof_date": asof_date,
        "stale_shards": stale,
        "passed": bool(len(states) == 5 and universe and processed == universe and error_rate <= 5.0 and not stale and asof_date),
    }
    state = "completed" if gate["passed"] else "attention_required"
    summary = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "state": state, "quality_gate": gate, "candidates": len(candidates), "artifacts": {"csv": "first_red_520_original_daily.csv", "json": "first_red_520_original_daily.json"}, "prepare_status": load_prepare_status(input_root), "disclaimer": "自动化研究筛选结果，不构成投资建议。"}
    lines = ["# 原版520首红：全市场当日汇总", "", f"- 状态：`{state}`", f"- 统一数据与信号日期：`{asof_date or '无'}`", f"- 覆盖：{processed}/{universe}（{coverage}%）", f"- 数据源错误：{errors}（{error_rate}%）", f"- 当日观察候选：{len(candidates)}", f"- 质量闸门：`{'通过' if gate['passed'] else '未通过'}`", "", "> 宽松原版定义仅作为观察层；最终名单仍强制信号日等于标的最后可用日线，避免历史信号混入。"]
    if notify:
        bodies = split_bodies(lines[:-1], candidates, ["", "> 全部当日观察结果已分段完整发送；不是买卖建议。"])
        outcomes = []
        for index, body in enumerate(bodies, start=1):
            title = f"原版520首红：{state} | {len(candidates)}条"
            if len(bodies) > 1:
                title += f"（{index}/{len(bodies)}）"
            outcomes.append(send_serverchan(title, body))
        summary["notification"] = outcomes[0] if len(outcomes) == 1 else outcomes
    write_json(output / "first_red_520_original_daily_summary.json", summary)
    (output / "first_red_520_original_daily_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "collected"
        for index, shard in enumerate("abcde"):
            path = root / shard
            path.mkdir(parents=True)
            pd = "2026-08-19" if shard != "e" else "2026-08-18"
            (path / f"first_red_520_{shard}.state.json").write_text(json.dumps({"shard": shard, "universe": {"count": 5}, "max_data_last_date": pd, "stats": {"processed": 1, "source_error": 0}}), encoding="utf-8")
            (path / f"first_red_520_{shard}.csv").write_text("code,name,signal_date,data_last_date,volume_ratio,distance_to_520_low_pct\n000001,样本,2026-08-19,2026-08-19,1.2,1.0\n", encoding="utf-8")
        summary = consolidate(root, Path(temp_dir) / "output", False)
        assert summary["state"] == "attention_required" and summary["candidates"] == 4
    print("FIRST_RED_520_DAILY_SUMMARY_SELF_TEST_OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="原版520首红每日汇总器")
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(consolidate(Path(args.input_root), Path(args.output_dir), args.notify == "true"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
