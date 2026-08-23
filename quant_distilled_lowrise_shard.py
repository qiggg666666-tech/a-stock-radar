#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-D分片扫描：真实DistilledQuant因子 + 520日低位首红观察层。

此脚本不执行网格回测、不会以未来收益更新参数、不会生成合成数据。520日首红
仅为截至信号日的观察条件；DistilledQuant分数为全截面汇总前的原始因子记录。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import quant_distilled_core_research as core


def load_universe(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("universe")
    if not isinstance(items, list) or not items:
        raise ValueError("universe_file_has_no_valid_universe")
    rows = []
    for item in items:
        rows.append({"code": core.normalize_code(item["code"]), "name": str(item.get("name", ""))})
    return sorted(rows, key=lambda row: row["code"])


def slice_universe(universe: list[dict[str, str]], shard_index: int, shard_count: int) -> list[dict[str, str]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid_shard_index_or_count")
    return universe[shard_index::shard_count]


def lowrise_observation(frame: pd.DataFrame, signal_date: pd.Timestamp, max_stale_days: int, max_rise_pct: float) -> dict[str, Any]:
    visible = frame.loc[frame.index <= signal_date].copy()
    if len(visible) < 521:
        raise ValueError(f"insufficient_history_for_520_low:{len(visible)}_rows")
    last_date = visible.index.max().normalize()
    staleness_days = int((signal_date - last_date).days)
    if staleness_days > max_stale_days:
        raise ValueError(f"stale_daily_data:{staleness_days}_days")
    low_520 = float(visible["low"].tail(520).min())
    close = float(visible["close"].iloc[-1])
    distance_pct = (close / low_520 - 1.0) * 100.0
    previous = visible.iloc[-2]
    current = visible.iloc[-1]
    volume_mean_previous_5 = float(visible["volume"].iloc[-6:-1].mean())
    volume_ratio = float(current["volume"] / (volume_mean_previous_5 + 1e-12))
    ma5 = float(visible["close"].tail(5).mean())
    first_red = bool(current["close"] > current["open"] and previous["close"] <= previous["open"])
    observation = bool(first_red and distance_pct <= max_rise_pct and current["close"] > ma5)
    return {
        "lowrise_data_last_date": last_date.strftime("%Y-%m-%d"),
        "lowrise_staleness_days": staleness_days,
        "low_520": low_520,
        "distance_to_520_low_pct": distance_pct,
        "volume_ratio_previous_5": volume_ratio,
        "first_red": first_red,
        "above_ma5": bool(current["close"] > ma5),
        "lowrise_observation": observation,
    }


def scan_shard(args: argparse.Namespace) -> int:
    signal_date = core.parse_signal_date(args.signal_date)
    universe = load_universe(args.universe_file)
    shard = slice_universe(universe, args.shard_index, args.shard_count)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    source_counts = {"akshare": 0, "baostock": 0}
    for item in shard:
        code, name = item["code"], item["name"]
        fetched = core.fetch_real_daily(code, args.start_date, signal_date.strftime("%Y-%m-%d"), args.timeout_seconds, args.retries)
        attempts = json.dumps([core.asdict(attempt) for attempt in fetched.attempts], ensure_ascii=False)
        if fetched.frame is None or fetched.source is None:
            errors.append({"code": code, "name": name, "stage": "daily_fetch", "error_type": "DataSourceUnavailable", "error_message": "both_public_sources_failed_or_returned_no_usable_rows", "attempts": attempts})
            continue
        try:
            raw = core.calculate_raw_factors(fetched.frame, signal_date, args.max_stale_days)
            observation = lowrise_observation(fetched.frame, signal_date, args.max_stale_days, args.max_rise_pct)
            records.append({"code": code, "name": name, "daily_data_source": fetched.source, **raw, **observation})
            source_counts[fetched.source] = source_counts.get(fetched.source, 0) + 1
        except Exception as exc:  # Explicitly retained in audit output.
            errors.append({"code": code, "name": name, "stage": "factor_or_lowrise", "error_type": type(exc).__name__, "error_message": str(exc), "attempts": attempts})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "raw_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["code", "name", "stage", "error_type", "error_message", "attempts"]).to_csv(output / "errors.csv", index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": "distilled-quant-lowrise-shard/v1",
        "status": "ready" if not errors else "degraded",
        "signal_date_requested": args.signal_date,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "scan_queue_count": len(shard),
        "valid_record_count": len(records),
        "error_count": len(errors),
        "daily_data_source_counts": source_counts,
        "max_rise_pct": args.max_rise_pct,
        "disclosure": "真实日线研究记录；520低位首红为观察条件，非未来收益预测或交易建议。",
    }
    (output / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_self_test() -> None:
    universe = [{"code": f"{i:06d}", "name": str(i)} for i in range(1, 17)]
    slices = [slice_universe(universe, index, 4) for index in range(4)]
    flattened = [item["code"] for shard in slices for item in shard]
    assert len(flattened) == len(set(flattened)) == 16
    index = pd.bdate_range("2022-01-03", periods=600)
    step = np.arange(len(index), dtype=float)
    close = 10.0 + step * 0.01 + np.sin(step / 11.0) * 0.04
    frame = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + step * 500,
        },
        index=index,
    )
    signal = frame.index[-1]
    output = lowrise_observation(frame, signal, 0, 5.0)
    assert "lowrise_observation" in output and "distance_to_520_low_pct" in output
    print("SELF_TEST_OK: shard_union, lowrise_observation")


def main() -> int:
    parser = argparse.ArgumentParser(description="DistilledQuant低位首红协同A-D分片扫描")
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--signal-date")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("joint_shard_output"))
    parser.add_argument("--timeout-seconds", type=int, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-stale-days", type=int, default=3)
    parser.add_argument("--max-rise-pct", type=float, default=5.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.universe_file or not args.signal_date:
        parser.error("--universe-file and --signal-date are required unless --self-test")
    if args.timeout_seconds <= 0 or args.retries <= 0 or args.max_rise_pct < 0:
        parser.error("invalid timeout/retries/max-rise-pct")
    return scan_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
