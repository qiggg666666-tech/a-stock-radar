#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DistilledQuant + 520低位首红协同任务的共同股票池准备器。

该股票池是运行时快照，不应被描述为历史时点成分池。AkShare不可用时使用
BaoStock ``query_stock_basic()`` 的当前兼容调用；两源都失败时尝试东方财富
公开接口作为第三备用源。所有源都失败时明确退出，不伪造池。
"""
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
    clean = clean[clean["code"].str.fullmatch(r"0\d{5}|3\d{5}|6\d{5}", na=False)]
    clean = clean[\~clean["name"].str.contains(r"ST|\*ST|退", regex=True, case=False, na=False)]
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


def fetch_universe(timeout_seconds: float, retries: int) -> tuple[pd.DataFrame, str, list[str]]:
    errors: list[str] = []
    # 优先级：akshare → baostock → eastmoney（新增备用）
    for source, function in (
        ("akshare", akshare_universe),
        ("baostock", baostock_universe),
        ("eastmoney", eastmoney_universe),
    ):
        for attempt in range(1, retries + 1):
            try:
                return provider_call(f"{source}_universe", timeout_seconds, function), source, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}:{attempt}:{type(exc).__name__}:{str(exc)[:240]}")
    raise RuntimeError("universe_unavailable:" + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="DistilledQuant低位首红协同共同股票池")
    parser.add_argument("--signal-date", default="")
    parser.add_argument("--output", type=Path, default=Path("joint_universe.json"))
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame(
            {"code": ["000001", "300001", "600001", "830001"], "name": ["甲", "乙", "丙", "退市样本"]}
        )
        assert [row["code"] for row in normalize_universe(sample)] == ["000001", "300001", "600001"]
        print("SELF_TEST_OK")
        return 0
    raw, source, source_errors = fetch_universe(args.timeout_seconds, max(1, args.retries))
    universe = normalize_universe(raw)
    if not universe:
        raise RuntimeError("empty_universe_after_normalization")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": "distilled-quant-lowrise-universe/v1",
        "generated_at": generated_at,
        "signal_date_requested": args.signal_date,
        "universe_snapshot_state": "current_universe_not_historical_point_in_time",
        "count": len(universe),
        "data_source": source,
        "source_errors": source_errors,
        "universe": universe,
        "disclosure": "运行时A股股票池快照；排除ST/退市名称和非沪深创业板前缀，不代表历史时点成分池。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(universe), "source": source, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
