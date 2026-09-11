#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立520日低位首红任务的共同股票池准备器。

该股票池是运行时快照，不应被描述为历史时点成分池。AkShare不可用时使用
BaoStock ``query_stock_basic()`` 的当前兼容调用；两源都失败时尝试东方财富
公开接口作为第三备用源。所有实时源和磁盘缓存都失败时明确退出，不伪造池。
它仅由LowRise 520工作流消费，不与DistilledQuant核心共享输入或输出。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
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
    clean = clean[clean["code"].str.fullmatch(r"0\d{5}|3\d{5}|6\d{5}", na=False)]
    clean = clean[~clean["name"].str.contains(r"ST|\*ST|退", regex=True, case=False, na=False)]
    clean = clean.drop_duplicates("code").sort_values("code")
    return [{"code": row.code, "name": row.name} for row in clean.itertuples(index=False)]


def akshare_universe() -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_info_a_code_name()
    return raw.rename(columns={"代码": "code", "名称": "name"})


def baostock_universe() -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        # 当前发布版仅接受code/code_name；无参数调用能避免旧接口参数TypeError。
        result = bs.query_stock_basic()
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
        return pd.DataFrame(
            {
                "code": raw.get("code", pd.Series(dtype=str)).astype(str).str.split(".").str[-1],
                "name": raw.get("code_name", pd.Series(dtype=str)).astype(str),
            }
        )
    finally:
        bs.logout()


def eastmoney_universe() -> pd.DataFrame:
    """直接请求东方财富公开接口获取 A 股列表（第三备用源）"""
    import requests

    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "6000",
        "po": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",  # 沪深主板+创业板+科创板
        "fields": "f12,f14",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    diff = data.get("data", {}).get("diff") or []
    if not diff:
        raise RuntimeError("eastmoney_empty_diff")
    rows = [{"code": str(item.get("f12", "")), "name": str(item.get("f14", ""))} for item in diff]
    return pd.DataFrame(rows)


def _backoff_sleep(attempt: int) -> None:
    """指数退避 + 抖动，在同一数据源的重试之间等待。

    抖动是为了避免和同一时刻触发的其他独立管线（dinghai/distilled_quant等，
    经常在相近的cron窗口跑）撞到同一秒重试、对同一上游造成新一轮突发流量。
    单次最长等待封顶在20秒左右，避免在--timeout-seconds预算里占用过多时间。
    """
    backoff = min(20.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
    time.sleep(backoff)


def fetch_universe(timeout_seconds: float, retries: int, deadline: float) -> tuple[pd.DataFrame, str, list[str]]:
    """deadline是time.monotonic()口径的绝对截止时间。到点即使还有重试次数或源剩余
    也立即放弃，转由调用方走缓存兜底，避免像final_chip那次一样一路卡到被GitHub
    Actions作业级超时强杀、什么产物都留不下。"""
    errors: list[str] = []
    # 优先级：akshare → baostock → eastmoney（新增备用）
    providers = (
        ("akshare", akshare_universe),
        ("baostock", baostock_universe),
        ("eastmoney", eastmoney_universe),
    )
    for source, function in providers:
        for attempt in range(1, retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"budget_exhausted:before:{source}:{attempt}")
                raise RuntimeError("universe_unavailable:" + " | ".join(errors))
            per_attempt_timeout = min(timeout_seconds, remaining)
            try:
                return provider_call(f"{source}_universe", per_attempt_timeout, function), source, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}:{attempt}:{type(exc).__name__}:{str(exc)[:240]}")
                if attempt < retries and deadline - time.monotonic() > 0:
                    _backoff_sleep(attempt)
        # 换到下一个源前留出短暂间隔，降低同一秒内连续打三个不同上游的概率。
        if deadline - time.monotonic() > 0:
            time.sleep(1.5)
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
    parser = argparse.ArgumentParser(description="独立520低位首红共同股票池")
    parser.add_argument("--signal-date", default="")
    parser.add_argument("--output", type=Path, default=Path("lowrise_520_universe.json"))
    parser.add_argument("--cache-path", type=Path, default=Path(".lowrise-520-cache/universe.json"))
    parser.add_argument("--cache-max-age-hours", type=float, default=72.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    # 整个重试循环的总耗时硬上限。3个源、每源最多2次尝试再加上退避等待，理论上限
    # 比dinghai(2源)更高；这里留了更大的余量，到点就必须退出转缓存兜底。
    parser.add_argument("--max-total-seconds", type=float, default=360.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame(
            {"code": ["000001", "300001", "600001", "830001"], "name": ["甲", "乙", "丙", "退市样本"]}
        )
        assert [row["code"] for row in normalize_universe(sample)] == ["000001", "300001", "600001"]
        print("SELF_TEST_OK")
        return 0

    deadline = time.monotonic() + args.max_total_seconds
    try:
        raw, source, source_errors = fetch_universe(args.timeout_seconds, max(1, args.retries), deadline)
        universe = normalize_universe(raw)
        if not universe:
            raise RuntimeError("empty_universe_after_normalization")
        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {
            "schema_version": "lowrise-520-universe/v1",
            "status": "ready",
            "generated_at": generated_at,
            "signal_date_requested": args.signal_date,
            "universe_snapshot_state": "current_universe_not_historical_point_in_time",
            "count": len(universe),
            "data_source": source,
            "source_errors": source_errors,
            "universe": universe,
            "disclosure": "独立520日低位首红运行时A股股票池快照；排除ST/退市名称和非沪深创业板前缀，不与DistilledQuant核心共享，不代表历史时点成分池。",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        args.cache_path.parent.mkdir(parents=True, exist_ok=True)
        args.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"count": len(universe), "source": source, "output": str(args.output)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        live_error = f"{type(exc).__name__}:{str(exc)[:1200]}"

    # 实时源(含规范化)全部失败：先试磁盘缓存兜底，跟final_chip_universe.py同一套逻辑，
    # 让当天的分片扫描还能用一份稍旧但真实的股票池继续跑，而不是直接断供。这个脚本
    # 原来没有这一层——任何失败都是直接崩溃、连失败原因都没有落盘。
    cached = fresh_cache(args.cache_path, args.cache_max_age_hours)
    if cached:
        payload = {
            "schema_version": "lowrise-520-universe/v1",
            "status": "degraded_cache",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "signal_date_requested": args.signal_date,
            "universe_snapshot_state": "degraded_cache_not_current",
            "count": len(cached["universe"]),
            "data_source": "lowrise_520_cache",
            "cache_generated_at": cached.get("generated_at"),
            "source_errors": [live_error],
            "universe": cached["universe"],
            "disclosure": "独立520日低位首红运行时A股股票池快照（实时源失败，回退到磁盘缓存）；不与DistilledQuant核心共享，不代表历史时点成分池。",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"count": len(cached["universe"]), "source": "lowrise_520_cache", "output": str(args.output)}, ensure_ascii=False))
        return 0

    # 缓存也没有：写一份失败artifact落盘供审计，再让这一步失败退出——总比之前
    # 什么都不留、直接崩溃要好排查。
    failure = {
        "schema_version": "lowrise-520-universe/v1",
        "status": "universe_unavailable",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_date_requested": args.signal_date,
        "error": live_error,
        "count": 0,
        "universe": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
    raise RuntimeError(live_error)


if __name__ == "__main__":
    raise SystemExit(main())
