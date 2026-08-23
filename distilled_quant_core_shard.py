#!/usr/bin/env python3
"""独立DistilledQuant核心A-D扫描器；仅消费其专属共同股票池。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import quant_distilled_core_research as core


def load_universe(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    universe = payload.get("universe")
    if not isinstance(universe, list) or not universe:
        raise ValueError("core_universe_invalid_or_empty")
    return sorted(
        [{"code": core.normalize_code(item["code"]), "name": str(item.get("name", ""))} for item in universe],
        key=lambda item: item["code"],
    )


def shard_slice(universe: list[dict[str, str]], index: int, count: int) -> list[dict[str, str]]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid_shard_index_or_count")
    return universe[index::count]


def main() -> int:
    parser = argparse.ArgumentParser(description="独立DistilledQuant核心A-D扫描")
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--signal-date")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--timeout-seconds", type=int, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-stale-days", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("distilled_core_shard_output"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = [{"code": f"{i:06d}", "name": str(i)} for i in range(1, 17)]
        all_codes = [item["code"] for shard in [shard_slice(sample, n, 4) for n in range(4)] for item in shard]
        assert len(all_codes) == len(set(all_codes)) == 16
        print("SELF_TEST_OK: core_shard_partition")
        return 0
    if not args.universe_file or not args.signal_date:
        parser.error("--universe-file and --signal-date are required")
    signal_date = core.parse_signal_date(args.signal_date)
    queue = shard_slice(load_universe(args.universe_file), args.shard_index, args.shard_count)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    sources = {"akshare": 0, "baostock": 0}
    for item in queue:
        fetched = core.fetch_real_daily(item["code"], args.start_date, signal_date.strftime("%Y-%m-%d"), args.timeout_seconds, args.retries)
        attempts = json.dumps([core.asdict(attempt) for attempt in fetched.attempts], ensure_ascii=False)
        if fetched.frame is None or fetched.source is None:
            errors.append({"code": item["code"], "name": item["name"], "stage": "daily_fetch", "error_type": "DataSourceUnavailable", "error_message": "both_sources_failed", "attempts": attempts})
            continue
        try:
            factors = core.calculate_raw_factors(fetched.frame, signal_date, args.max_stale_days)
            records.append({"code": item["code"], "name": item["name"], "daily_data_source": fetched.source, **factors})
            sources[fetched.source] += 1
        except Exception as exc:  # no silent skip
            errors.append({"code": item["code"], "name": item["name"], "stage": "factor_calculation", "error_type": type(exc).__name__, "error_message": str(exc), "attempts": attempts})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "raw_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["code", "name", "stage", "error_type", "error_message", "attempts"]).to_csv(args.output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    status = {"schema_version": "distilled-quant-core-shard/v1", "status": "ready" if not errors else "degraded", "signal_date_requested": args.signal_date, "shard_index": args.shard_index, "shard_count": args.shard_count, "scan_queue_count": len(queue), "valid_record_count": len(records), "error_count": len(errors), "daily_data_source_counts": sources}
    (args.output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
