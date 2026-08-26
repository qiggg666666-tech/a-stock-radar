#!/usr/bin/env python3
"""筹码模型独立全市场共同股票池。仅服务chip-model任务，绝不与其他策略复用。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "chip-model-universe/v1"


def _child(conn: Any, func: Callable[[], Any]) -> None:
    try:
        conn.send(("ok", func()))
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", f"{type(exc).__name__}:{str(exc)[:240]}"))
    finally:
        conn.close()


def isolated_call(label: str, timeout_seconds: int, func: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return func()
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, func), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=3)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds}s")
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


def akshare_universe() -> list[dict[str, str]]:
    import akshare as ak

    frame = ak.stock_info_a_code_name()
    records: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        code = str(row.get("code", "")).zfill(6)
        name = str(row.get("name", "")).strip()
        if code.startswith(("60", "68", "00", "30")) and name:
            records.append({"code": code, "name": name, "source": "akshare"})
    return records


def baostock_universe() -> list[dict[str, str]]:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_stock_basic()
        records: list[dict[str, str]] = []
        while result.error_code == "0" and result.next():
            row = result.get_row_data()
            code, code_name, stock_type, status = row[0], row[1], row[2], row[5]
            if stock_type != "1" or status != "1" or not code.startswith(("sh.6", "sz.0", "sz.3")):
                continue
            records.append({"code": code.split(".", 1)[1], "name": code_name.strip(), "source": "baostock"})
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        return records
    finally:
        bs.logout()


def normalize(records: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for record in records:
        code = str(record.get("code", "")).zfill(6)
        name = str(record.get("name", "")).strip()
        if code.startswith(("60", "68", "00", "30")) and name and code not in deduped:
            deduped[code] = {"code": code, "name": name, "universe_source": str(record.get("source", "unknown"))}
    return [deduped[code] for code in sorted(deduped)]


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="筹码模型独立全市场共同股票池")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-output", default="")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    errors: list[str] = []
    records: list[dict[str, str]] = []
    provider = ""
    for name, loader in (("akshare", akshare_universe), ("baostock", baostock_universe)):
        for attempt in range(1, args.retries + 1):
            try:
                records = normalize(isolated_call(name, args.timeout_seconds, loader))
                if records:
                    provider = name
                    break
                errors.append(f"{name}:attempt{attempt}:empty")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}:attempt{attempt}:{type(exc).__name__}:{str(exc)[:180]}")
            time.sleep(min(attempt * 2, 6))
        if records:
            break

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = "ready" if records else "unavailable"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": "chip-model",
        "status": status,
        "generated_at": generated_at,
        "provider": provider or None,
        "stock_count": len(records),
        "stocks": records,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.status_output:
        write_status(Path(args.status_output), {key: payload[key] for key in ["schema_version", "strategy_id", "status", "generated_at", "provider", "stock_count", "errors"]})
    print(f"chip-model universe: status={status} count={len(records)} provider={provider or 'none'}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
