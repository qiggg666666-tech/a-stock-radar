#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将四分片候选合并为每策略一条手机通知，不输出任何密钥。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    csv_pattern: str
    score_column: str
    type_column: str = ""


SPECS = {
    "bottom-accumulation": StrategySpec("bottom-accumulation", "底部吸筹快速版", "bottom_accumulation_fast_*.csv", "score", "阶段"),
    "pattern-breakout": StrategySpec("pattern-breakout", "形态突破安全版", "pattern_breakout_*.csv", "评分"),
    "smallcap-trend": StrategySpec("smallcap-trend", "小市值趋势扫描", "chi_diao_smallcap_*.csv", "信号评分", "信号类型"),
    "vcp-fast": StrategySpec("vcp-fast", "VCP快速精简版", "vcp_fast_????????.csv", "VCP_Score", "Breakout"),
    "bull-confirm": StrategySpec("bull-confirm", "牛市确认快速版", "bull_confirm_????????.csv", "多头得分", "综合倾向"),
    "yearline-limitup-v5": StrategySpec("yearline-limitup-v5", "年线涨停v5.2快速版", "年线涨停_评分_????????.csv", "综合评分"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分片策略单条汇总通知器")
    parser.add_argument("--strategy", choices=sorted(SPECS), required=True)
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--run-date", default="")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    parser.add_argument("--notify-zero", choices=["true", "false"], default="true")
    parser.add_argument("--report-path", default="output/strategy_notification_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_candidate_files(root: Path, spec: StrategySpec) -> list[Path]:
    return sorted(path for path in root.rglob(spec.csv_pattern) if not path.name.endswith(".checkpoint.csv"))


def read_candidates(paths: Iterable[Path], spec: StrategySpec) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str})
            frame["__artifact"] = str(path)
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{path.name}:{type(exc).__name__}")
    if not frames:
        return pd.DataFrame(), errors
    merged = pd.concat(frames, ignore_index=True)
    if "代码" in merged.columns:
        merged["代码"] = merged["代码"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    if spec.score_column in merged.columns:
        merged[spec.score_column] = pd.to_numeric(merged[spec.score_column], errors="coerce")
        merged = merged.sort_values(spec.score_column, ascending=False, na_position="last")
    if "代码" in merged.columns:
        merged = merged.drop_duplicates("代码", keep="first")
    return merged.reset_index(drop=True), errors


def shard_labels(paths: Iterable[Path]) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        name = str(path.parent).lower()
        for letter in "abcd":
            if name.endswith(f"-{letter}") or name.endswith(f"_{letter}") or f"results-{letter}" in name:
                labels.add(letter.upper())
    return sorted(labels)


def compact_row(row: dict[str, Any], spec: StrategySpec) -> str:
    code = str(row.get("代码", "------")).zfill(6)
    name = str(row.get("名称", "未知"))[:10]
    raw_score = row.get(spec.score_column, "-")
    try:
        score = f"{float(raw_score):.1f}"
    except (TypeError, ValueError):
        score = "-"
    suffix = f" {row.get(spec.type_column, '')}" if spec.type_column else ""
    return f"- {code} {name}{suffix}｜分{score}".rstrip()


def build_message(spec: StrategySpec, frame: pd.DataFrame, files: list[Path], read_errors: list[str], run_date: str, top: int) -> tuple[str, str]:
    shards = shard_labels(files)
    state = "/".join(shards) if shards else "无artifact"
    title = f"{spec.label} 汇总：{len(frame)}只｜分片{state}"
    lines = [
        f"# {spec.label} 分片汇总",
        f"- 运行日：{run_date or '未指定'}",
        f"- 已读取分片：{state}；候选去重后：{len(frame)}只。",
        f"- artifact候选文件：{len(files)} 份。",
    ]
    if read_errors:
        lines.append(f"- 读取异常文件：{len(read_errors)} 份（详见Actions日志）。")
    if frame.empty:
        lines.extend(["", "本次无候选；扫描与artifact仍可能正常完成。"])
    else:
        lines.extend(["", f"## Top {min(top, len(frame))}"])
        lines.extend(compact_row(row, spec) for row in frame.head(top).to_dict("records"))
    lines.extend(["", "仅为自动化筛选结果汇总，不构成投资建议。"])
    return title[:80], "\n".join(lines)[:3700]


def send_serverchan(sendkey: str, title: str, content: str) -> tuple[bool, str]:
    if not sendkey:
        return False, "SKIPPED: 未配置 SENDKEY"
    try:
        import requests
        response = requests.post(f"https://sctapi.ftqq.com/{sendkey}.send", data={"title": title, "desp": content}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) == 0:
            return True, "SUCCESS: Server酱业务返回 code=0"
        return False, f"FAILED: Server酱业务返回 code={payload.get('code')} message={payload.get('message', '未知错误')}"
    except Exception as exc:
        return False, f"FAILED: Server酱 {type(exc).__name__}"


def send_telegram(token: str, chat_id: str, title: str, content: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "SKIPPED: 未同时配置 TELEGRAM_BOT_TOKEN 与 TELEGRAM_CHAT_ID"
    try:
        import requests
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": f"{title}\n\n{content}"[:4096]}, timeout=15)
        payload = response.json()
        if response.ok and payload.get("ok") is True:
            return True, "SUCCESS: Telegram业务返回 ok=true"
        return False, f"FAILED: Telegram业务返回 {payload.get('description', response.status_code)}"
    except Exception as exc:
        return False, f"FAILED: Telegram {type(exc).__name__}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    spec = SPECS[args.strategy]
    files = find_candidate_files(Path(args.input_root), spec)
    frame, read_errors = read_candidates(files, spec)
    title, content = build_message(spec, frame, files, read_errors, args.run_date, args.top)
    print(title)
    print(content)
    channels: list[tuple[bool, str]] = []
    should_send = args.notify == "true" and (args.notify_zero == "true" or not frame.empty) and not args.dry_run
    if should_send:
        channels.append(send_serverchan(os.getenv("SENDKEY", ""), title, content))
        channels.append(send_telegram(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), title, content))
    else:
        channels.append((False, "SKIPPED: 通知开关关闭、dry-run或零候选提醒关闭"))
    for success, detail in channels:
        print(("INFO" if success else "WARN") + ": " + detail)
    write_report(Path(args.report_path), {
        "strategy": spec.key, "generated_at": datetime.now().isoformat(timespec="seconds"), "run_date": args.run_date,
        "candidate_files": [str(path) for path in files], "candidates_deduplicated": len(frame), "read_errors": read_errors,
        "notification_attempted": should_send, "channel_results": [{"success": ok, "detail": text} for ok, text in channels],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
