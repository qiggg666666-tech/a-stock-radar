#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总年线v7四分片产物，并向可选通知渠道发送一条安全摘要。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="年线v7分片汇总通知器")
    parser.add_argument("--input-root", default="collected", help="download-artifact后的根目录")
    parser.add_argument("--run-date", default="", help="运行日期，仅用于通知标题")
    parser.add_argument("--top", type=int, default=10, help="每层展示前N条")
    parser.add_argument("--notify", choices=["true", "false"], default="true", help="是否真正发送通知")
    parser.add_argument("--dry-run", action="store_true", help="仅打印通知内容，不发网络请求")
    return parser.parse_args()


def read_csvs(root: Path, pattern: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in root.rglob(pattern):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str})
            if not frame.empty:
                frame["__source"] = str(path)
                frames.append(frame)
        except Exception as exc:
            print(f"WARN: 无法读取 {path}: {exc}", file=sys.stderr)
    return frames


def deduplicate(frames: Iterable[pd.DataFrame], score_column: str) -> pd.DataFrame:
    usable = list(frames)
    if not usable:
        return pd.DataFrame()
    frame = pd.concat(usable, ignore_index=True)
    if score_column in frame.columns:
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
        frame = frame.sort_values(score_column, ascending=False, na_position="last")
    if "代码" in frame.columns:
        frame["代码"] = frame["代码"].astype(str).str.zfill(6)
        frame = frame.drop_duplicates("代码", keep="first")
    return frame.reset_index(drop=True)


def read_checkpoint_states(root: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for path in root.rglob("yearline_v7_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "checkpoint_key" in payload:
                states.append(payload)
        except Exception as exc:
            print(f"WARN: 无法读取checkpoint {path}: {exc}", file=sys.stderr)
    return states


def compact_row(row: Mapping[str, Any], score: str) -> str:
    code = str(row.get("代码", "------")).zfill(6)
    name = str(row.get("名称", "未知"))[:10]
    raw_score = row.get(score, "-")
    try:
        score_text = f"{float(raw_score):.1f}"
    except (TypeError, ValueError):
        score_text = "-"
    return f"{code} {name} 评分{score_text}"


def build_message(run_date: str, pre_alerts: pd.DataFrame, confirmations: pd.DataFrame, states: list[dict[str, Any]], top: int) -> tuple[str, str]:
    completed = sum(1 for state in states if state.get("status") == "completed")
    partial = sum(1 for state in states if state.get("status") in {"partial", "interrupted", "running"})
    title = f"年线v7 分片汇总 {run_date or '当前'}：预警{len(pre_alerts)} / 确认{len(confirmations)}"
    lines = [
        "年线涨停 v7 双层研究汇总",
        f"运行日：{run_date or '未指定'}",
        f"预警候选：{len(pre_alerts)} 只；涨停确认：{len(confirmations)} 只。",
        f"checkpoint：完成 {completed} 个；待恢复/部分完成 {partial} 个；已发现状态文件 {len(states)} 个。",
    ]
    if not pre_alerts.empty:
        lines.extend(["", f"T-1 预警 Top {min(top, len(pre_alerts))}："])
        lines.extend(compact_row(row, "pre_score") for row in pre_alerts.head(top).to_dict("records"))
    if not confirmations.empty:
        lines.extend(["", f"T 日确认 Top {min(top, len(confirmations))}："])
        lines.extend(compact_row(row, "confirm_score") for row in confirmations.head(top).to_dict("records"))
    if pre_alerts.empty and confirmations.empty:
        lines.extend(["", "本次未发现候选CSV；请查看四个分片job的artifact与checkpoint状态。"])
    lines.extend(["", "仅为技术形态研究汇总，不构成投资建议。"])
    return title[:80], "\n".join(lines)[:3500]


def send_serverchan(sendkey: str, title: str, content: str) -> tuple[bool, str]:
    if not sendkey:
        return False, "未配置 SENDKEY"
    try:
        import requests
        response = requests.post(
            f"https://sctapi.ftqq.com/{sendkey}.send",
            data={"title": title, "desp": content},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) == 0:
            return True, "Server酱已发送"
        return False, f"Server酱返回失败：{payload.get('message', '未知错误')}"
    except Exception as exc:
        return False, f"Server酱请求异常：{type(exc).__name__}"


def send_telegram(token: str, chat_id: str, title: str, content: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "未同时配置 TELEGRAM_BOT_TOKEN 与 TELEGRAM_CHAT_ID"
    try:
        import requests
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": f"{title}\n\n{content}"[:4096]},
            timeout=15,
        )
        payload = response.json()
        if response.ok and payload.get("ok") is True:
            return True, "Telegram已发送"
        return False, f"Telegram返回失败：{payload.get('description', response.status_code)}"
    except Exception as exc:
        return False, f"Telegram请求异常：{type(exc).__name__}"


def main() -> int:
    args = parse_args()
    root = Path(args.input_root)
    pre_alerts = deduplicate(read_csvs(root, "年线预警_*.csv"), "pre_score")
    confirmations = deduplicate(read_csvs(root, "年线涨停确认_*.csv"), "confirm_score")
    states = read_checkpoint_states(root)
    title, content = build_message(args.run_date, pre_alerts, confirmations, states, args.top)
    print(title)
    print(content)
    if args.dry_run or args.notify == "false":
        print("通知已禁用或处于dry-run；未发起网络请求。")
        return 0
    # 通知失败只记录状态，绝不让扫描汇总job失败，也绝不输出任何密钥内容。
    results = [
        send_serverchan(os.getenv("SENDKEY", ""), title, content),
        send_telegram(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), title, content),
    ]
    for ok, detail in results:
        print(("INFO" if ok else "WARN") + ": " + detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
