#!/usr/bin/env python3
"""FINAL Chip专属A–D分片扫描器；只输出研究记录和显式错误。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from final_chip_research import analyze, fetch_intraday_ma5_turnup, fetch_ohlcv, self_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--intraday-timeout-seconds", type=float, default=20, help="首红候选二次确认(盘中MA5拐头)的单次超时")
    parser.add_argument("--intraday-retries", type=int, default=1, help="首红候选二次确认的重试次数")
    parser.add_argument("--skip-intraday-confirm", action="store_true", help="跳过首红候选的盘中二次确认，只保留日线 is_first_red_daily")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); assert [0, 1, 2, 3][1::2] == [1, 3]; print("FINAL_CHIP_SHARD_SELF_TEST_OK"); return 0
    if None in (args.universe, args.shard_index, args.output_dir): parser.error("--universe, --shard-index and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    state = universe.get("state")
    if state not in {"ready", "degraded_cache"}:
        status = {"schema_version": "final-chip-shard/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "skipped:universe_unavailable", "shard_index": args.shard_index, "shard_total": args.shard_total, "record_count": 0, "error_count": 0, "universe_state": state}
        pd.DataFrame().to_csv(args.output_dir / "records.csv", index=False); pd.DataFrame(columns=["code", "name", "error_type", "error_message"]).to_csv(args.output_dir / "errors.csv", index=False); (args.output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(status, ensure_ascii=False)); return 0
    selected = universe["universe"][args.shard_index::args.shard_total]; records: list[dict[str, object]] = []; errors: list[dict[str, str]] = []; source_counts: dict[str, int] = {}; intraday_confirm_count = 0
    for position, item in enumerate(selected, 1):
        code, name = str(item["code"]), str(item.get("name", ""))
        try:
            frame, source, prior_errors = fetch_ohlcv(code, args.timeout_seconds, args.retries); row = analyze(code, name, frame); row["data_source"] = source; row["source_errors"] = " | ".join(prior_errors)
            # 首红候选(is_first_red_daily=True)才额外拉盘中5分钟数据做MA5拐头二次确认；其余股票不调用，控制接口调用量。
            if row.get("is_first_red_daily") and not args.skip_intraday_confirm:
                is_turn_up, slope, note, intraday_errors = fetch_intraday_ma5_turnup(code, args.intraday_timeout_seconds, args.intraday_retries)
                row["intraday_ma5_turn_up"] = is_turn_up; row["intraday_ma5_slope"] = round(slope, 5); row["intraday_ma5_note"] = note; row["intraday_errors"] = " | ".join(intraday_errors)
                row["is_first_red_confirmed"] = bool(is_turn_up)
                intraday_confirm_count += 1
            else:
                row["intraday_ma5_turn_up"] = False; row["intraday_ma5_slope"] = 0.0; row["intraday_ma5_note"] = "跳过" if args.skip_intraday_confirm else "非首红候选未调用"; row["intraday_errors"] = ""
                row["is_first_red_confirmed"] = False
            records.append(row); source_counts[source] = source_counts.get(source, 0) + 1
        except Exception as exc:
            errors.append({"code": code, "name": name, "error_type": type(exc).__name__, "error_message": str(exc)[:500]})
        if position % 50 == 0: print(json.dumps({"shard": args.shard_index, "processed": position, "total": len(selected), "records": len(records), "errors": len(errors), "intraday_confirm_calls": intraday_confirm_count}, ensure_ascii=False))
    pd.DataFrame(records).to_csv(args.output_dir / "records.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(errors, columns=["code", "name", "error_type", "error_message"]).to_csv(args.output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    status = {"schema_version": "final-chip-shard/v2", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "completed" if records else "completed_zero_records", "shard_index": args.shard_index, "shard_total": args.shard_total, "universe_state": state, "universe_count": len(selected), "record_count": len(records), "error_count": len(errors), "data_source_counts": source_counts, "intraday_confirm_calls": intraday_confirm_count, "disclosure": "FINAL Chip专属分片；没有合成记录或分片推送；仅对is_first_red_daily候选调用盘中二次确认。"}
    (args.output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(status, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
