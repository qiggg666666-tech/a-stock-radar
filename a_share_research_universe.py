#!/usr/bin/env python3
"""A股全市场透明研究的专属共同股票池准备器。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _worker(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send(("ok", function()))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}:{str(exc)[:300]}"))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, function: Callable[[], Any]) -> Any:
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    if context is None:
        return function()
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, function), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=3)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        state, payload = parent.recv()
        process.join(timeout=3)
        if state != "ok":
            raise RuntimeError(f"provider_error:{label}:{payload}")
        return payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        parent.close()


def normalize_universe(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame is None or frame.empty or not {"code", "name"}.issubset(frame.columns):
        return []
    clean = frame[["code", "name"]].copy()
    clean["code"] = clean["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    clean["name"] = clean["name"].astype(str)
    clean = clean[clean["code"].str.fullmatch(r"(?:00|30|60|68)\d{4}", na=False)]
    clean = clean[~clean["name"].str.contains(r"ST|\*ST|退", regex=True, case=False, na=False)]
    clean = clean.drop_duplicates("code").sort_values("code")
    return [{"code": row.code, "name": row.name} for row in clean.itertuples(index=False)]


def akshare_universe() -> pd.DataFrame:
    import akshare as ak

    return ak.stock_info_a_code_name().rename(columns={"代码": "code", "名称": "name"})


def baostock_universe() -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_stock_basic()
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
        if "status" in raw.columns:
            raw = raw[raw["status"].astype(str) == "1"]
        return pd.DataFrame({"code": raw.get("code", pd.Series(dtype=str)).astype(str).str.split(".").str[-1], "name": raw.get("code_name", pd.Series(dtype=str)).astype(str)})
    finally:
        bs.logout()


def fetch_universe(timeout_seconds: float, retries: int) -> tuple[pd.DataFrame, str, list[str]]:
    errors: list[str] = []
    for source, function in (("akshare", akshare_universe), ("baostock", baostock_universe)):
        for attempt in range(1, max(1, retries) + 1):
            try:
                return provider_call(f"{source}_universe", timeout_seconds, function), source, errors
            except Exception as exc:
                errors.append(f"{source}:{attempt}:{type(exc).__name__}:{str(exc)[:240]}")
    raise RuntimeError("universe_unavailable:" + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="A股全市场透明研究共同股票池")
    parser.add_argument("--signal-date", default="")
    parser.add_argument("--output", type=Path, default=Path("a_share_research_universe.json"))
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame({"code": ["000001", "302132", "830001"], "name": ["甲", "乙", "退市样本"]})
        assert [row["code"] for row in normalize_universe(sample)] == ["000001", "302132"]
        print("SELF_TEST_OK")
        return 0
    raw, source, source_errors = fetch_universe(args.timeout_seconds, args.retries)
    universe = normalize_universe(raw)
    if not universe:
        raise RuntimeError("empty_universe_after_normalization")
    payload = {"schema_version": "a-share-transparent-research-universe/v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "signal_date_requested": args.signal_date, "universe_snapshot_state": "current_universe_not_historical_point_in_time", "count": len(universe), "data_source": source, "source_errors": source_errors, "universe": universe, "disclosure": "仅供A股全市场透明研究任务使用的运行时股票池快照，不与任何其他任务共享。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(universe), "source": source, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
