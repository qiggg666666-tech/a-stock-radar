#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立全市场筹码扫描汇总与可选Server酱通知。

本文件只服务chip-full-market-scanner，不读取其他策略artifact。
SENDKEY仅从运行环境读取，绝不打印密钥值；通知失败不阻断研究artifact。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


def send_serverchan(sendkey: str, title: str, desp: str) -> dict[str, Any]:
    if not sendkey:
        return {"status": "disabled", "reason": "SENDKEY_not_configured"}
    if sendkey.startswith("sctp"):
        match = re.match(r"sctp(\d+)t", sendkey)
        if not match:
            return {"status": "failed", "reason": "invalid_sctp_key"}
        url = f"https://{match.group(1)}.push.ft07.com/send/{sendkey}.send"
    else:
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
    try:
        response = requests.post(
            url,
            json={"title": title[:100], "desp": desp},
            headers={"Content-Type": "application/json;charset=utf-8"},
            timeout=15,
        )
        payload = response.json()
        code = payload.get("code")
        return {
            "status": "sent" if response.ok and code in (0, "0", None) else "failed",
            "http_status": response.status_code,
            "business_code": code,
            "message": str(payload.get("message", ""))[:160],
        }
    except Exception as exc:  # 通知非核心产物，记录可审计原因但不抛出
        return {"status": "failed", "reason": type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="collected")
    parser.add_argument("--output-dir", default="full-market-summary")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_files: list[str] = []
    read_errors: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("**/*_data.json")):
        source_files.append(str(path))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows.extend(item for item in data if isinstance(item, dict))
            else:
                read_errors.append({"file": str(path), "error": "root_not_list"})
        except Exception as exc:
            read_errors.append({"file": str(path), "error": type(exc).__name__})

    ok = [row for row in rows if "error" not in row]
    errors = [row for row in rows if "error" in row]
    errors.extend({"error": item["error"], "file": item["file"]} for item in read_errors)
    tradeable = [row for row in ok if row.get("is_tradeable") is True]
    spikes = [row for row in ok if row.get("is_spike_tradeable") is True]
    top = sorted(tradeable, key=lambda row: float(row.get("total_score", 0) or 0), reverse=True)[:20]

    lines = [
        f"扫描文件 {len(source_files)} 个；成功 {len(ok)}；错误 {len(errors)}",
        f"可交易预警 {len(tradeable)}；尖峰关注 {len(spikes)}",
        "",
        "| 代码 | 板块 | 收盘 | 综合 | 信号 |",
        "|---|---|---:|---:|---|",
    ]
    for row in top:
        lines.append(
            f"| {row.get('symbol', '')} | {row.get('industry', '未知')} | {row.get('close', '')} | "
            f"{row.get('total_score', '')} | {row.get('signal', '')} |"
        )
    if errors:
        lines.extend(["", f"错误样本 {len(errors)} 条；完整错误记录见all_rows.json。"])
    body = "\n".join(lines)

    notification: dict[str, Any]
    if args.notify:
        notification = send_serverchan(
            os.environ.get("SENDKEY", "").strip(),
            f"独立筹码扫描汇总｜可交易{len(tradeable)}｜错误{len(errors)}",
            body,
        )
    else:
        notification = {"status": "disabled", "reason": "notify_flag_not_set"}

    summary = {
        "schema": "chip-full-market-summary/v2",
        "status": "ready" if ok and not read_errors else ("partial" if ok else "degraded"),
        "notification": notification,
        "shards_found": len(source_files),
        "scanned": len(rows),
        "success": len(ok),
        "errors": len(errors),
        "tradeable": len(tradeable),
        "spike_tradeable": len(spikes),
        "error_samples": errors[:20],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "all_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(body + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "shards_found", "scanned", "success", "errors", "tradeable", "notification")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
