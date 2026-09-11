#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定海神针独立共同股票池准备器；不读取其他策略的股票池或artifact。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _worker(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send(("ok", function()))
    except Exception as exc:  # noqa: BLE001
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
            # 修复：SIGTERM(terminate)可能被卡在阻塞C调用里的子进程忽略/来不及响应，
            # 子进程变僵尸继续占用连接，后续fork()在容器里可能变慢甚至卡住，多次累积后
            # 总耗时会远超"单次超时×重试次数"的理论上限。改用SIGKILL(kill)保证一定能
            # 杀死，跟final_chip_universe.py里同一个真实事故修复口径一致。
            process.kill()
            process.join(timeout=3)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        state, payload = parent.recv()
        process.join(timeout=3)
        if state != "ok":
            raise RuntimeError(f"provider_error:{label}:{payload}")
        return payload
    finally:
        if process.is_alive():
            process.kill()
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
        if "type" in raw.columns:
            raw = raw[raw["type"].astype(str).eq("1")]
        if "status" in raw.columns:
            raw = raw[raw["status"].astype(str).eq("1")]
        return pd.DataFrame({"code": raw.get("code", pd.Series(dtype=str)).astype(str).str.split(".").str[-1], "name": raw.get("code_name", pd.Series(dtype=str)).astype(str)})
    finally:
        bs.logout()


def fetch_universe(timeout_seconds: float, retries: int, deadline: float) -> tuple[pd.DataFrame, str, list[str]]:
    """deadline是time.monotonic()口径的绝对截止时间。到点即使还有重试次数剩余也立即
    放弃，转由调用方走缓存兜底，避免像final_chip那次一样一路卡到被GitHub Actions
    作业级超时强杀、什么产物都留不下。"""
    errors: list[str] = []
    for source, function in (("akshare", akshare_universe), ("baostock", baostock_universe)):
        for attempt in range(1, max(1, retries) + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"budget_exhausted:before:{source}:{attempt}")
                raise RuntimeError("universe_unavailable:" + " | ".join(errors))
            per_attempt_timeout = min(timeout_seconds, remaining)
            try:
                return provider_call(f"{source}_universe", per_attempt_timeout, function), source, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}:{attempt}:{type(exc).__name__}:{str(exc)[:240]}")
    raise RuntimeError("universe_unavailable:" + " | ".join(errors))


def fresh_cache(path: Path, max_age_hours: float) -> dict[str, Any] | None:
    """跟final_chip_universe.py同一套缓存新鲜度检查：磁盘缓存存在、在有效期内、且
    universe非空，才认为可用。"""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="定海神针独立共同股票池")
    parser.add_argument("--signal-date", default="")
    parser.add_argument("--output", type=Path, default=Path("dinghai_universe.json"))
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=Path(".dinghai-cache/universe.json"))
    parser.add_argument("--cache-max-age-hours", type=float, default=72.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    # 整个重试循环的总耗时硬上限。理论上限是 timeout-seconds×retries×2源=180秒左右；
    # 这里留出余量，不管内部单次超时有没有按预期生效，到点就必须退出转缓存兜底。
    parser.add_argument("--max-total-seconds", type=float, default=240.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame({"code": ["000001", "302132", "688001", "830001"], "name": ["甲", "乙", "丙", "退市样本"]})
        assert [item["code"] for item in normalize_universe(sample)] == ["000001", "302132", "688001"]
        print("SELF_TEST_OK")
        return 0

    deadline = time.monotonic() + args.max_total_seconds
    try:
        raw, source, source_errors = fetch_universe(args.timeout_seconds, args.retries, deadline)
        universe = normalize_universe(raw)
        if not universe:
            raise RuntimeError("empty_universe_after_normalization")
        payload = {
            "schema_version": "dinghai-universe/v1",
            "status": "ready",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "signal_date_requested": args.signal_date,
            "universe_snapshot_state": "current_universe_not_historical_point_in_time",
            "count": len(universe),
            "data_source": source,
            "source_errors": source_errors,
            "universe": universe,
            "disclosure": "定海神针独立运行时A股股票池快照，不与DistilledQuant核心或520低位首红共享输入。",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        args.cache_path.parent.mkdir(parents=True, exist_ok=True)
        args.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.status_output:
            args.status_output.parent.mkdir(parents=True, exist_ok=True)
            args.status_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"count": len(universe), "source": source, "output": str(args.output)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        live_error = f"{type(exc).__name__}:{str(exc)[:1200]}"

    # 实时源(含规范化)全部失败：先试磁盘缓存兜底，跟final_chip_universe.py同一套逻辑，
    # 让当天的scan/summary还能用一份稍旧但真实的股票池继续跑，而不是直接断供。
    cached = fresh_cache(args.cache_path, args.cache_max_age_hours)
    if cached:
        payload = {
            "schema_version": "dinghai-universe/v1",
            "status": "degraded_cache",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "signal_date_requested": args.signal_date,
            "universe_snapshot_state": "degraded_cache_not_current",
            "count": len(cached["universe"]),
            "data_source": "dinghai_cache",
            "cache_generated_at": cached.get("generated_at"),
            "source_errors": [live_error],
            "universe": cached["universe"],
            "disclosure": "定海神针独立运行时A股股票池快照（实时源失败，回退到磁盘缓存）；不与DistilledQuant核心或520低位首红共享输入。",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.status_output:
            args.status_output.parent.mkdir(parents=True, exist_ok=True)
            args.status_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"count": len(cached["universe"]), "source": "dinghai_cache", "output": str(args.output)}, ensure_ascii=False))
        return 0

    # 缓存也没有：保留原有行为——写失败artifact供审计，然后让这一步失败退出。
    # 这样dinghai_shard.py不用改（它遇到空universe仍然会明确报错，而不是悄悄假装有数据）。
    failure = {
        "schema_version": "dinghai-universe/v1",
        "status": "universe_unavailable",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_date_requested": args.signal_date,
        "error": live_error,
        "count": 0,
        "universe": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.status_output:
        args.status_output.parent.mkdir(parents=True, exist_ok=True)
        args.status_output.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
    raise RuntimeError(live_error)


if __name__ == "__main__":
    raise SystemExit(main())
