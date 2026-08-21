#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吃掉主力研究：跨分片每日汇总与唯一通知器。

原始分片文件保留所有技术研究候选；最终名单只保留全体分片的同一最新信号日期。
仅本脚本发送Server酱通知，避免分片重复推送。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SHARDS = ("a", "b", "c", "d")
FINAL_COLUMNS = ["代码", "名称", "信号日期", "信号类型", "流通市值(亿)", "当日成交额(亿)", "最新价", "信号评分", "股牛股", "首个粉红柱", "预备首粉", "粉红准备度", "预备首粉缺失条件", "买进", "操纵", "趋势", "中线趋势", "相对强弱%", "数据源"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="吃掉主力研究跨分片单条汇总")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--universe-status", type=Path, default=Path("input/chi_diao_smallcap_universe_status.json"))
    parser.add_argument("--universe-file", type=Path, default=Path("input/chi_diao_smallcap_universe.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--sendkey", default=os.getenv("SENDKEY", ""))
    parser.add_argument("--max-source-error-rate", type=float, default=5.0)
    parser.add_argument("--run-label", default="收盘后确认", help="午盘预警、收盘后确认或手动研究")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_universe_codes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        frame = pd.read_csv(path, dtype={"代码": str})
        return set(frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False).dropna())
    except Exception:
        return set()


def signal_rank(value: Any) -> int:
    """已触发首粉优先，其次买点、预备首粉和趋势观察。"""
    return {"首个粉红柱": 0, "买点": 1, "预备首粉": 2, "趋势观察": 3}.get(str(value or ""), 9)


def send_serverchan(sendkey: str, title: str, content: str) -> str:
    if not sendkey:
        return "skipped_no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{sendkey}.send", data={"title": title, "desp": content}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        code = payload.get("code")
        return "sent" if code == 0 else f"failed:serverchan_code_{code}"
    except (requests.RequestException, ValueError) as error:
        return f"failed:{type(error).__name__}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = load_json(args.universe_status)
    expected_universe = int(universe.get("universe_count") or 0)
    universe_codes = load_universe_codes(args.universe_file)
    expected_run_id = str(universe.get("universe_run_id") or "")
    expected_hash = str(universe.get("universe_sha256") or "")
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
    processed_lists = [list(state.get("processed_codes") or []) for state in states]
    processed_codes = [str(code) for codes in processed_lists for code in codes]
    code_counts = Counter(processed_codes)
    unique_processed_codes = set(processed_codes)
    duplicate_processed_codes = sorted(code for code, count in code_counts.items() if count > 1)
    out_of_universe_codes = sorted(unique_processed_codes.difference(universe_codes))
    missing_universe_codes = sorted(universe_codes.difference(unique_processed_codes))
    covered_codes = unique_processed_codes.intersection(universe_codes)
    ledger_available = bool(states) and all("source_error_ledger" in state for state in states)
    error_ledger: list[dict[str, Any]] = []
    for state in states:
        for row in list(state.get("source_error_ledger") or []):
            error_ledger.append({"分片": state.get("shard", ""), **row})
    source_errors = sum(1 for row in error_ledger if row.get("最终状态", row.get("final_status", "unresolved")) == "unresolved") if ledger_available else sum(int((state.get("stats") or {}).get("source_error") or 0) for state in states)
    source_recovered = sum(1 for row in error_ledger if row.get("最终状态", row.get("final_status")) == "recovered") if ledger_available else sum(int((state.get("stats") or {}).get("source_recovered") or 0) for state in states)
    source_error_rate = round(source_errors * 100.0 / len(covered_codes), 4) if covered_codes else 100.0
    completed = all(
        str(state.get("stop_reason")) == "completed"
        and int(state.get("processed") or 0) == int(state.get("universe_size", -1))
        for state in states
    )
    contracts_match = bool(expected_run_id and expected_hash) and all(
        (state.get("universe_contract") or {}).get("universe_run_id") == expected_run_id
        and (state.get("universe_contract") or {}).get("universe_sha256") == expected_hash
        and int((state.get("universe_contract") or {}).get("universe_count") or 0) == expected_universe
        for state in states
    )
    coverage_pct = round(len(covered_codes) * 100.0 / expected_universe, 4) if expected_universe else 0.0
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
            if "首个粉红柱" not in final.columns:
                final["首个粉红柱"] = 0
            if "预备首粉" not in final.columns:
                final["预备首粉"] = 0
            final["信号优先级"] = final["信号类型"].map(signal_rank)
            final = final.sort_values(["信号优先级", "信号评分"], ascending=[True, False]).drop_duplicates("代码", keep="first").drop(columns=["信号优先级"])
    error_ledger_frame = pd.DataFrame(error_ledger)
    error_ledger_frame.to_csv(args.output_dir / "chi_diao_zhu_li_daily_source_error_ledger.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "chi_diao_zhu_li_daily_source_error_ledger.json").write_text(json.dumps(error_ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    quality_passed = bool(
        universe.get("state") == "completed" and not missing and completed and expected_universe > 0
        and expected_universe == len(universe_codes) and len(covered_codes) == expected_universe
        and not duplicate_processed_codes and not out_of_universe_codes and not missing_universe_codes
        and contracts_match and ledger_available and source_error_rate <= float(args.max_source_error_rate)
    )
    state = "completed" if quality_passed else "attention_required"
    observations = final.copy()
    first_pink_bars = observations.loc[observations.get("首个粉红柱", pd.Series(index=observations.index, dtype=int)) == 1].copy() if not observations.empty else observations.copy()
    pre_pink_bars = observations.loc[observations.get("预备首粉", pd.Series(index=observations.index, dtype=int)) == 1].copy() if not observations.empty else observations.copy()
    buy_points = observations.loc[observations.get("买进", pd.Series(index=observations.index, dtype=int)) == 1].copy() if not observations.empty else observations.copy()
    priority_events = pd.concat([first_pink_bars, buy_points, pre_pink_bars], ignore_index=True).drop_duplicates("代码", keep="first") if not observations.empty else pd.DataFrame(columns=FINAL_COLUMNS)
    publishable = priority_events.copy() if quality_passed else pd.DataFrame(columns=FINAL_COLUMNS)
    observations.to_csv(args.output_dir / "chi_diao_zhu_li_daily_observation_latest.csv", index=False, encoding="utf-8-sig")
    buy_points.to_csv(args.output_dir / "chi_diao_zhu_li_daily_buy_points_latest.csv", index=False, encoding="utf-8-sig")
    publishable.to_csv(args.output_dir / "chi_diao_zhu_li_daily_global_latest.csv", index=False, encoding="utf-8-sig")
    selected = publishable.head(10)
    lines = [f"- {row['名称']}({row['代码']}) {row['信号类型']} 分{row['信号评分']}" for _, row in selected.iterrows()] or ["质量闸门未通过或当日没有可发布研究候选。"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "run_label": args.run_label, "state": state,
        "quality_gate": {
            "expected_shards": len(SHARDS), "visible_shards": len(states), "missing_shards": missing,
            "universe_run_id": expected_run_id or None, "universe_count": expected_universe,
            "processed_events": processed, "unique_covered_codes": len(covered_codes), "coverage_pct": coverage_pct,
            "duplicate_processed_codes": duplicate_processed_codes, "out_of_universe_codes": out_of_universe_codes,
            "missing_universe_codes": missing_universe_codes, "contracts_match": contracts_match,
            "error_ledger_available": ledger_available, "source_errors": source_errors, "source_recovered": source_recovered,
            "source_error_rate_pct": source_error_rate, "max_source_error_rate_pct": float(args.max_source_error_rate), "passed": quality_passed,
        },
        "latest_signal_date": latest_date or None,
        "observation_candidates": int(len(observations)),
        "first_pink_bars": int(len(first_pink_bars)),
        "pre_pink_bars": int(len(pre_pink_bars)),
        "buy_points": int(len(buy_points)),
        "publishable_candidates": int(len(publishable)),
        "publication_blocked": not quality_passed,
        "disclaimer": "价格、均线与相对强弱技术研究近似版，不代表可观测的主力账户行为，也不构成投资建议。",
    }
    notification = send_serverchan(
        args.sendkey,
        f"吃掉主力·{args.run_label}：{state} | 首粉{len(first_pink_bars)} | 预备{len(pre_pink_bars)} | 可发布{len(publishable)}条",
        "# 吃掉主力研究：每日全市场汇总\n\n"
        f"- 运行口径：`{args.run_label}`\n- 状态：`{state}`\n- 统一候选信号日：`{latest_date or '无候选'}`\n"
        f"- 共同股票池运行标识：`{expected_run_id or '缺失'}`\n"
        f"- 覆盖（唯一代码）：{len(covered_codes)}/{expected_universe}（{coverage_pct}%）；处理事件：{processed}\n"
        f"- 数据源未恢复错误：{source_errors}（{source_error_rate}%）；定向恢复：{source_recovered}\n"
        f"- 观察研究记录：{len(observations)}；买点研究记录：{report['buy_points']}\n"
        f"- 可发布研究候选：{len(publishable)}\n"
        f"- 质量闸门：`{'通过' if quality_passed else '需关注'}`\n\n"
        "> 指标仅为价格、均线和相对强弱技术研究近似版，不代表真实主力账户行为。\n\n## 候选\n\n" + "\n".join(lines),
    )
    report["notification"] = notification
    (args.output_dir / "chi_diao_zhu_li_daily_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "chi_diao_zhu_li_daily_summary.md").write_text(
        "# 吃掉主力研究：每日全市场汇总\n\n"
        f"- 运行口径：`{args.run_label}`\n- 状态：`{state}`\n- 统一候选信号日：`{latest_date or '无候选'}`\n"
        f"- 共同股票池运行标识：`{expected_run_id or '缺失'}`\n"
        f"- 覆盖（唯一代码）：{len(covered_codes)}/{expected_universe}（{coverage_pct}%）；处理事件：{processed}\n"
        f"- 数据源未恢复错误：{source_errors}（{source_error_rate}%）；定向恢复：{source_recovered}\n"
        f"- 观察研究记录：{len(observations)}；买点研究记录：{report['buy_points']}\n"
        f"- 可发布研究候选：{len(publishable)}\n"
        f"- 质量闸门：`{'通过' if quality_passed else '需关注'}`\n\n"
        "> 本指标是技术研究近似版，不代表可观测的主力账户行为，也不构成投资建议。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
