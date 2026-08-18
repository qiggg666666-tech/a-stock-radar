#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""首红与资本流独立验收的artifact汇总器；只读artifact后发送至多一条Server酱状态消息。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def read_json_candidates(root: Path, filename: str) -> tuple[dict[str, Any] | None, str | None]:
    paths = sorted(root.rglob(filename))
    if not paths:
        return None, f"缺少{filename}"
    try:
        return json.loads(paths[-1].read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{filename}无法读取：{type(exc).__name__}"


def compact_status(label: str, payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"label": label, "state": "missing", "detail": error or "未知错误", "candidates": None}
    state = str(payload.get("state") or payload.get("status") or "unknown")
    candidates = payload.get("candidate_count")
    if candidates is None:
        candidates = (payload.get("stats") or {}).get("candidate_records")
    return {
        "label": label,
        "state": state,
        "detail": str(payload.get("reason") or payload.get("fund_flow_status") or ""),
        "candidates": candidates,
        "generated_at": payload.get("finished_at") or payload.get("generated_at"),
    }


def push_serverchan(title: str, content: str, key: str) -> str:
    if not key:
        return "skipped:no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content[:3800]}, timeout=15)
        body = response.json()
        return "sent" if response.ok and body.get("code") == 0 else f"failed:http_{response.status_code}"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description="研究脚本验收artifact汇总")
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    args = parser.parse_args()

    root, output = Path(args.input_root), Path(args.output_dir)
    first_raw, first_error = read_json_candidates(root, "first_red_520_acceptance.state.json")
    capital_raw, capital_error = read_json_candidates(root, "capital_flow_market_status.json")
    rows = [compact_status("520首红验收", first_raw, first_error), compact_status("资本流验收", capital_raw, capital_error)]
    acceptable = {"completed", "ready"}
    all_ready = all(row["state"] in acceptable for row in rows)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "state": "ready" if all_ready else "attention_required",
        "jobs": rows,
        "method": "独立手动验收；不属于all，不触发子脚本直推；只在本汇总器中最多发送一条通知。",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "research_signal_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 首红与资本流研究验收汇总", "", f"- 状态：`{summary['state']}`", f"- 生成时间：`{summary['generated_at']}`", ""]
    for row in rows:
        lines.append(f"- **{row['label']}**：`{row['state']}`；候选：`{row['candidates']}`；{row['detail'] or '无附加诊断'}")
    lines.extend(["", "> 独立研究任务的产物诊断，不构成投资建议。"])
    (output / "research_signal_acceptance_summary.md").write_text("\n".join(lines), encoding="utf-8")
    if args.notify == "true":
        key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
        summary["notification"] = push_serverchan("研究脚本验收：" + summary["state"], "\n".join(lines), key)
        (output / "research_signal_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
