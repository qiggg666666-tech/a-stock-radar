#!/usr/bin/env python3
"""FINAL Chip专属A股共同股票池：双源、ST剔除、缓存降级和显式状态。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


def _call(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send((True, function()))
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}:{str(exc)[:400]}"))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, function: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return function()
    parent, child = mp.get_context("fork").Pipe(duplex=False)
    process = mp.get_context("fork").Process(target=_call, args=(child, function), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=2)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        ok, value = parent.recv()
        process.join(timeout=2)
        if not ok:
            raise RuntimeError(f"provider_error:{label}:{value}")
        return value
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        parent.close()


def valid_code(value: object) -> bool:
    code = str(value).replace("sh.", "").replace("sz.", "").strip().zfill(6)
    return len(code) == 6 and code.isdigit() and code.startswith(("00", "30", "60", "68"))


def is_st_name(value: object) -> bool:
    """与用户新版脚本一致：剔除名称中含ST的风险警示普通股。"""
    return "ST" in str(value or "").upper().replace(" ", "")


def baostock_universe() -> list[dict[str, str]]:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_stock_basic()
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        fields = {name: index for index, name in enumerate(result.fields)}
        required = {"code", "code_name", "type", "status"}
        if not required.issubset(fields):
            raise ValueError(f"baostock_schema_missing:{sorted(required - set(fields))}")
        rows: list[dict[str, str]] = []
        while result.next():
            row = result.get_row_data()
            code = row[fields["code"]].replace("sh.", "").replace("sz.", "").zfill(6)
            out_date = row[fields["outDate"]] if "outDate" in fields else ""
            name = row[fields["code_name"]].strip()
            if (
                valid_code(code)
                and row[fields["type"]] == "1"
                and row[fields["status"]] == "1"
                and out_date in ("", "None")
                and not is_st_name(name)
            ):
                rows.append({"code": code, "name": name})
        if not rows:
            raise ValueError("baostock_empty_universe")
        return rows
    finally:
        bs.logout()


def akshare_universe() -> list[dict[str, str]]:
    import akshare as ak

    table = ak.stock_info_a_code_name()
    if table is None or table.empty:
        raise ValueError("akshare_empty_universe")
    rows: list[dict[str, str]] = []
    for _, item in table.iterrows():
        code = str(item.iloc[0]).strip().zfill(6)
        name = str(item.iloc[1]).strip() if len(item) > 1 else ""
        if valid_code(code) and not is_st_name(name):
            rows.append({"code": code, "name": name})
    if not rows:
        raise ValueError("akshare_no_normal_a_share")
    return rows


def fresh_cache(path: Path, max_age_hours: float) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(str(data["generated_at"]).replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - created.astimezone(timezone.utc)
        if age <= timedelta(hours=max_age_hours) and data.get("universe"):
            return data
    except Exception:
        return None
    return None


def provider_sources() -> tuple[tuple[str, Callable[[], list[dict[str, str]]]], ...]:
    """使用用户新版脚本的BaoStock优先顺序，但仍由provider_call实施硬超时。"""
    return (("baostock", baostock_universe), ("akshare", akshare_universe))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=Path(".final-chip-cache/universe.json"))
    parser.add_argument("--cache-max-age-hours", type=float, default=72)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    errors: list[str] = []
    for label, source in provider_sources():
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                universe = provider_call(label, args.timeout_seconds, source)
                payload = {"schema_version": "final-chip-universe/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "ready", "source": label, "universe": sorted(universe, key=lambda item: item["code"]), "errors": errors}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.cache_path.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                args.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"state": "ready", "source": label, "count": len(universe)}, ensure_ascii=False))
                return 0
            except Exception as exc:
                errors.append(f"{label}:{attempt}:{type(exc).__name__}:{str(exc)[:300]}")
    cached = fresh_cache(args.cache_path, args.cache_max_age_hours)
    if cached:
        payload = {"schema_version": "final-chip-universe/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "degraded_cache", "source": "final_chip_cache", "universe": cached["universe"], "cache_generated_at": cached["generated_at"], "errors": errors}
    else:
        payload = {"schema_version": "final-chip-universe/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "unavailable", "source": "none", "universe": [], "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": payload["state"], "errors": errors}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
