#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""首红与资本流全市场生产运行的五分片artifact汇总器。"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def read_csv_records(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [{str(key): str(value or "") for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]
    except Exception:
        return []


def collect_first_red_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("first_red_520_[abcde].csv")):
        records.extend(read_csv_records(path))
    unique = {(item.get("code", ""), item.get("signal_date", "")): item for item in records if item.get("code")}
    return sorted(unique.values(), key=lambda item: (item.get("signal_date", ""), item.get("volume_ratio", ""), item.get("code", "")), reverse=True)


def first_red_line(item: dict[str, str]) -> str:
    return (
        f"- {item.get('code', '')} {item.get('name', '')}｜{item.get('signal_date', '')}"
        f"｜量比{item.get('volume_ratio', '')}｜距520低点{item.get('distance_to_520_low_pct', '')}%"
    )


def build_first_red_notification(header: list[str], records: list[dict[str, str]], footer: list[str], max_items: int = 20) -> str:
    """Send one mobile-readable preview while preserving every record in artifacts."""
    preview = records[:max_items]
    if not preview:
        return "\n".join(header + ["- 没有检测到520首红结果。"] + footer)
    lines = [f"- 推送展示：{len(preview)}/{len(records)}条（最多{max_items}条）"]
    lines.extend(first_red_line(item) for item in preview)
    return "\n".join(header + lines + footer)


def summarize_first_red(root: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for path in sorted(root.rglob("first_red_520_*.state.json")):
        payload = read_json(path)
        if payload:
            states.append(payload)
    if not states:
        return {"state": "missing", "shards": 0, "candidates": 0, "processed": 0, "detail": "缺少首红状态artifact"}
    completed = [item for item in states if item.get("state") == "completed"]
    stats = [item.get("stats") or {} for item in states]
    return {
        "state": "completed" if len(completed) == len(states) else "partial",
        "shards": len(states),
        "candidates": sum(int(item.get("candidate_count") or 0) for item in states),
        "processed": sum(int(item.get("processed") or 0) for item in stats),
        "timeouts": sum(int(item.get("timeout") or 0) for item in stats),
        "source_errors": sum(int(item.get("source_error") or 0) for item in stats),
        "detail": "所有可见分片完成" if len(completed) == len(states) else f"已完成{len(completed)}/{len(states)}片",
    }


def summarize_capital(root: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for path in sorted(root.rglob("capital_flow_market_status.json")):
        payload = read_json(path)
        if payload:
            states.append(payload)
    if not states:
        return {"state": "missing", "shards": 0, "candidates": 0, "processed": 0, "detail": "缺少资本流状态artifact"}
    ready = [item for item in states if str(item.get("status") or item.get("state")) in {"ready", "completed"}]
    return {
        "state": "completed" if len(ready) == len(states) else "partial",
        "shards": len(states),
        "candidates": sum(int(item.get("candidate_count") or 0) for item in states),
        "processed": sum(int(item.get("scanned") or item.get("processed") or item.get("processed_count") or 0) for item in states),
        "detail": "所有可见分片完成" if len(ready) == len(states) else f"已就绪{len(ready)}/{len(states)}片",
    }


def notify(title: str, content: str) -> str:
    key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
    if not key:
        return "skipped:no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content[:3800]}, timeout=15)
        payload = response.json()
        return "sent" if response.ok and payload.get("code") == 0 else f"failed:http_{response.status_code}"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description="全市场研究信号五分片汇总")
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    parser.add_argument("--prepare-status", default="", help="可选：共享股票池状态文件，用于说明上游失败原因")
    args = parser.parse_args()
    root = Path(args.input_root)
    first_red, capital = summarize_first_red(root), summarize_capital(root)
    first_red_records = collect_first_red_records(root)
    prepare_status = read_json(Path(args.prepare_status)) if args.prepare_status else None
    upstream_failed = bool(prepare_status and prepare_status.get("state") == "failed")
    state = "upstream_universe_failed" if upstream_failed else "completed" if first_red["state"] == "completed" and capital["state"] == "completed" else "attention_required"
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "state": state,
        "first_red_520": first_red,
        "capital_flow": capital,
        "first_red_all_detected": len(first_red_records),
        "prepare_status": prepare_status,
        "disclaimer": "全市场自动化筛选产物汇总，不构成投资建议。",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "research_signal_production_summary.json"
    md_path = output / "research_signal_production_summary.md"
    if first_red_records:
        with (output / "research_signal_first_red_all.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for item in first_red_records for key in item)))
            writer.writeheader()
            writer.writerows(first_red_records)
    else:
        with (output / "research_signal_first_red_all.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=["code", "name", "signal_date", "volume_ratio", "distance_to_520_low_pct"]).writeheader()
    (output / "research_signal_first_red_all.json").write_text(json.dumps(first_red_records, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 首红与资本流：全市场生产运行汇总", "",
        f"- 状态：`{state}`", f"- 生成时间：`{summary['generated_at']}`", "",
        f"- **520首红**：`{first_red['state']}`，分片{first_red['shards']}，已处理{first_red['processed']}，全部检测结果{len(first_red_records)}；{first_red['detail']}",
        f"- **资本流研究**：`{capital['state']}`，分片{capital['shards']}，已处理{capital['processed']}，候选{capital['candidates']}；{capital['detail']}",
        "", "> 首红使用信号后下一交易日开盘的无前视回测口径；资本流为公开数据快照代理，不识别真实账户资金。",
    ]
    if upstream_failed:
        diagnostics = prepare_status.get("diagnostics", []) if prepare_status else []
        reason = "; ".join(str(item.get("reason", "")) for item in diagnostics[-2:]) or str((prepare_status or {}).get("reason", "universe_unavailable"))
        lines.extend(["", f"> 上游共享股票池失败：`{reason}`。首红与资本流分片均被安全跳过，本次并非扫描到0条。"])
    if args.notify == "true":
        if upstream_failed:
            summary["notification"] = notify("研究信号全市场：上游股票池失败", "\n".join(lines))
        elif first_red["state"] == "completed":
            body = build_first_red_notification(lines[:-1], first_red_records, ["", "> 全部首红检测结果已保存至artifact；本条仅展示前20条。"])
            summary["notification"] = notify(f"研究信号全市场：{state} | 首红展示{min(len(first_red_records), 20)}/{len(first_red_records)}条", body)
        else:
            summary["notification"] = notify(f"研究信号全市场：{state}", "\n".join(lines))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
