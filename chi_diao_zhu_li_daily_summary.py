#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吃掉主力研究：跨分片每日汇总与唯一通知器。

原始分片文件保留所有技术研究候选；最终名单只保留全体分片的同一最新信号日期。
仅本脚本发送Server酱通知，避免分片重复推送。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SHARDS = ("a", "b", "c", "d")
FINAL_COLUMNS = ["代码", "名称", "信号日期", "信号类型", "流通市值(亿)", "当日成交额(亿)", "最新价", "信号评分", "股牛股", "买进", "操纵", "趋势", "中线趋势", "相对强弱%", "数据源"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="吃掉主力研究跨分片单条汇总")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--universe-status", type=Path, default=Path("input/chi_diao_smallcap_universe_status.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--sendkey", default=os.getenv("SENDKEY", ""))
    parser.add_argument("--max-source-error-rate", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def send_serverchan(sendkey: str, title: str, content: str) -> str:
    if not sendkey:
        return "skipped_no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{sendkey}.send", data={"title": title, "desp": content}, timeout=15)
        response.raise_for_status()
        return "sent"
    except requests.RequestException as error:
        return f"failed:{type(error).__name__}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = load_json(args.universe_status)
    expected_universe = int(universe.get("universe_count") or 0)
    states: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for shard in SHARDS:
        state_path = args.input_dir / f"chi_diao_smallcap_{shard}.json"
        csv_path = args.input_dir / f"chi_diao_smallcap_{shard}.csv"
        state = load_json(state_path)
        if not state:
            missing.append(shard)
            continue
        states.append(state)
        if csv_path.is_file():
            try:
                frame = pd.read_csv(csv_path, dtype={"代码": str})
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                pass
    processed = sum(int(state.get("processed") or 0) for state in states)
    source_errors = sum(int((state.get("stats") or {}).get("source_error") or 0) for state in states)
    source_error_rate = round(source_errors * 100.0 / processed, 4) if processed else 100.0
    completed = all(
        str(state.get("stop_reason")) == "completed"
        and int(state.get("processed") or 0) == int(state.get("universe_size", -1))
        for state in states
    )
    coverage_pct = round(processed * 100.0 / expected_universe, 4) if expected_universe else 0.0
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FINAL_COLUMNS)
    latest_date = ""
    final = pd.DataFrame(columns=FINAL_COLUMNS)
    if not raw.empty and "信号日期" in raw.columns:
        raw["信号日期"] = pd.to_datetime(raw["信号日期"], errors="coerce")
        latest = raw["信号日期"].max()
        if pd.notna(latest):
            latest_date = pd.Timestamp(latest).strftime("%Y-%m-%d")
            final = raw.loc[raw["信号日期"] == latest].copy()
            final["信号日期"] = final["信号日期"].dt.strftime("%Y-%m-%d")
            final = final.sort_values(["买进", "信号评分"], ascending=[False, False]).drop_duplicates("代码", keep="first")
    quality_passed = bool(
        universe.get("state") == "completed" and not missing and completed and expected_universe > 0
        and processed == expected_universe and source_error_rate <= float(args.max_source_error_rate)
    )
    state = "completed" if quality_passed else "attention_required"
    final.to_csv(args.output_dir / "chi_diao_zhu_li_daily_global_latest.csv", index=False, encoding="utf-8-sig")
    selected = final.head(10)
    lines = [f"- {row['名称']}({row['代码']}) {row['信号类型']} 分{row['信号评分']}" for _, row in selected.iterrows()] or ["当日没有符合研究条件的最终候选。"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "state": state,
        "quality_gate": {
            "expected_shards": len(SHARDS), "visible_shards": len(states), "missing_shards": missing,
            "universe_count": expected_universe, "processed": processed, "coverage_pct": coverage_pct,
            "source_errors": source_errors, "source_error_rate_pct": source_error_rate,
            "max_source_error_rate_pct": float(args.max_source_error_rate), "passed": quality_passed,
        },
        "latest_signal_date": latest_date or None, "candidates": int(len(final)),
        "buy_points": int((final.get("买进", pd.Series(dtype=int)) == 1).sum()) if not final.empty else 0,
        "disclaimer": "价格、均线与相对强弱技术研究近似版，不代表可观测的主力账户行为，也不构成投资建议。",
    }
    notification = send_serverchan(
        args.sendkey,
        f"吃掉主力研究：{state} | {len(final)}条",
        "# 吃掉主力研究：每日全市场汇总\n\n"
        f"- 状态：`{state}`\n- 统一候选信号日：`{latest_date or '无候选'}`\n"
        f"- 覆盖：{processed}/{expected_universe}（{coverage_pct}%）\n"
        f"- 数据源错误：{source_errors}（{source_error_rate}%）\n"
        f"- 最终候选：{len(final)}；买点：{report['buy_points']}\n"
        f"- 质量闸门：`{'通过' if quality_passed else '需关注'}`\n\n"
        "> 指标仅为价格、均线和相对强弱技术研究近似版，不代表真实主力账户行为。\n\n## 候选\n\n" + "\n".join(lines),
    )
    report["notification"] = notification
    (args.output_dir / "chi_diao_zhu_li_daily_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "chi_diao_zhu_li_daily_summary.md").write_text(
        "# 吃掉主力研究：每日全市场汇总\n\n"
        f"- 状态：`{state}`\n- 统一候选信号日：`{latest_date or '无候选'}`\n"
        f"- 覆盖：{processed}/{expected_universe}（{coverage_pct}%）\n"
        f"- 数据源错误：{source_errors}（{source_error_rate}%）\n"
        f"- 最终候选：{len(final)}；买点：{report['buy_points']}\n"
        f"- 质量闸门：`{'通过' if quality_passed else '需关注'}`\n\n"
        "> 本指标是技术研究近似版，不代表可观测的主力账户行为，也不构成投资建议。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
