#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FINAL Chip parameter comparison experiment.

This is an isolated experiment. It invokes only the sibling
``chip_param_base_scan.py`` copy, runs six explicit parameter profiles, and
writes one JSON/report bundle. It never reads a notification secret and never
sends a per-shard or per-profile message.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCAN_SCRIPT = SCRIPT_DIR / "chip_param_base_scan.py"
RESULTS_DIR = SCRIPT_DIR / "results"
LOGS_DIR = SCRIPT_DIR / "logs"

PROFILE_DEFAULTS = {
    "legacy": {"vol": 1.2, "conc": 0.30},
    "winrate": {"vol": 1.5, "conc": 0.20},
    "strict": {"vol": 2.0, "conc": 0.15},
    "balanced": {"vol": 1.3, "conc": 0.20},
}

CONFIGS = [
    {"name": "winrate_1.5", "env": {"CHIP_PROFILE": "winrate"}},
    {"name": "vol_1.3", "env": {"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "1.3"}},
    {"name": "vol_1.2", "env": {"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "1.2"}},
    {"name": "vol_1.1", "env": {"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "1.1"}},
    {"name": "vol_1.0", "env": {"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "1.0"}},
    {"name": "vol_0.9", "env": {"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "0.9"}},
]


def effective_thresholds(env: dict) -> dict:
    profile = env.get("CHIP_PROFILE", "winrate")
    base = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["winrate"])
    vol = float(env.get("CHIP_VOL_MULTIPLIER", base["vol"]))
    conc = float(env.get("CHIP_CONC_THRESHOLD", base["conc"]))
    return {"vol": vol, "conc": conc}


def build_funnel(rows: list[dict], thresholds: dict) -> dict:
    in_zone = [r for r in rows if r.get("is_approaching")]
    vol_ok = [r for r in in_zone if r.get("volume_ratio", 0) >= thresholds["vol"]]
    conc_ok = [r for r in in_zone if r.get("conc90_pct", 100) / 100.0 <= thresholds["conc"]]
    both = [
        r for r in in_zone
        if r.get("volume_ratio", 0) >= thresholds["vol"]
        and r.get("conc90_pct", 100) / 100.0 <= thresholds["conc"]
    ]
    return {
        "vol_threshold": thresholds["vol"],
        "conc_threshold": thresholds["conc"],
        "total": len(rows),
        "in_zone": len(in_zone),
        "vol_ok_in_zone": len(vol_ok),
        "conc_ok_in_zone": len(conc_ok),
        "vol_and_conc_ok": len(both),
        "trend_ok_among_vol_and_conc": sum(bool(r.get("trend")) for r in both),
        "final_tradeable": sum(bool(r.get("is_tradeable")) for r in rows),
    }


def run_one(config: dict, shard_index: int, shard_total: int, num_processes: int, timeout: int) -> dict:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(config["env"])
    env.update({
        "SHARD_INDEX": str(shard_index),
        "SHARD_TOTAL": str(shard_total),
        "NUM_PROCESSES": str(num_processes),
        "PUSH_ON_SHARD": "0",
        "FINAL_CHIP_PARAM_EXPERIMENT": "1",
    })
    log_path = LOGS_DIR / f"{config['name']}.log"
    result = {"name": config["name"], "env": config["env"], "status": "failed", "log_file": log_path.name}
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, str(SCAN_SCRIPT)],
                cwd=str(SCRIPT_DIR), env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        log_path.write_text(f"timeout after {timeout}s\n", encoding="utf-8")
        result["error"] = f"timeout_after_{timeout}s"
        return result
    data_path = RESULTS_DIR / f"shard_{shard_index}_data.json"
    if proc.returncode != 0 or not data_path.exists():
        result["error"] = "scanner_failed_or_no_output"
        return result
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    ok = [r for r in rows if "error" not in r]
    output_path = SCRIPT_DIR / f"results_{config['name']}.json"
    shutil.copy2(data_path, output_path)
    thresholds = effective_thresholds(config["env"])
    tradeable = [r for r in ok if r.get("is_tradeable")]
    result.update({
        "status": "completed",
        "scanned": len(rows),
        "ok": len(ok),
        "error": len(rows) - len(ok),
        "funnel": build_funnel(ok, thresholds),
        "tradeable": len(tradeable),
        "spike_tradeable": sum(bool(r.get("is_spike_tradeable")) for r in ok),
        "top_tradeable": sorted(tradeable, key=lambda r: r.get("total_score", 0), reverse=True)[:10],
        "output_file": output_path.name,
    })
    return result


def markdown_report(results: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# FINAL Chip 参数对比实验报告",
        "> 这是参数研究artifact，不是交易建议；本任务不发送通知。",
        f"> 分片 {args.shard_index}/{args.shard_total}；并行进程 {args.num_processes}；每组超时 {args.timeout}s",
        "",
        "| 配置 | 状态 | 扫描 | 成功 | 错误 | 可交易 | 尖峰关注 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['status']} | {r.get('scanned', 0)} | {r.get('ok', 0)} | "
            f"{r.get('error', 0)} | {r.get('tradeable', 0)} | {r.get('spike_tradeable', 0)} |"
        )
    lines.extend(["", "## 漏斗统计", ""])
    for r in results:
        f = r.get("funnel")
        if not f:
            lines.append(f"### {r['name']}：无有效输出")
            continue
        lines.extend([
            f"### {r['name']}",
            f"- 量比阈值：{f['vol_threshold']}；集中度阈值：{f['conc_threshold']}",
            f"- 接近区：{f['in_zone']}；量比达标：{f['vol_ok_in_zone']}；集中度达标：{f['conc_ok_in_zone']}",
            f"- 量比+集中度：{f['vol_and_conc_ok']}；其中趋势达标：{f['trend_ok_among_vol_and_conc']}",
            f"- 最终可交易字段：{f['final_tradeable']}",
            "",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="比较六组Chip VOL_MULTIPLIER参数并拆解信号漏斗")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=20)
    ap.add_argument("--num-processes", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("CHIP_PARAM_TIMEOUT", "900")))
    args = ap.parse_args()
    if not SCAN_SCRIPT.exists():
        print(f"missing private base scanner: {SCAN_SCRIPT}", file=sys.stderr)
        return 2
    results = [run_one(cfg, args.shard_index, args.shard_total, args.num_processes, args.timeout) for cfg in CONFIGS]
    summary = {
        "schema": "chip-param-comparison/v1",
        "status": "completed" if any(r["status"] == "completed" for r in results) else "failed",
        "notification": {"status": "disabled", "reason": "parameter_experiment_no_push"},
        "configs": results,
        "shard_index": args.shard_index,
        "shard_total": args.shard_total,
    }
    (SCRIPT_DIR / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (SCRIPT_DIR / "comparison_report.md").write_text(markdown_report(results, args), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "completed": sum(r["status"] == "completed" for r in results), "total": len(results)}, ensure_ascii=False))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
