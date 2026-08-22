#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为自适应行业轮动全市场扫描生成共同股票池快照。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import pandas as pd


def _worker(connection: Any, func: Callable[[], Any]) -> None:
    try:
        connection.send(("ok", func()))
    except Exception as exc:  # noqa: BLE001
        connection.send(("error", f"{type(exc).__name__}:{str(exc)[:160]}"))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, func: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return func()
    parent, child = mp.get_context("fork").Pipe(duplex=False)
    process = mp.get_context("fork").Process(target=_worker, args=(child, func), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=3)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        status, payload = parent.recv()
        process.join(timeout=3)
        if status != "ok":
            raise RuntimeError(f"provider_error:{label}:{payload}")
        return payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        parent.close()


def normalize_universe(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame is None or frame.empty or "code" not in frame.columns or "name" not in frame.columns:
        return []
    cleaned = frame[["code", "name"]].copy()
    cleaned["code"] = cleaned["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    cleaned["name"] = cleaned["name"].astype(str)
    cleaned = cleaned[cleaned["code"].str.fullmatch(r"0\d{5}|3\d{5}|6\d{5}", na=False)]
    cleaned = cleaned[~cleaned["name"].str.contains(r"ST|\*ST|退", case=False, regex=True, na=False)]
    cleaned = cleaned.drop_duplicates("code").sort_values("code")
    return [{"code": row.code, "name": row.name, "industry": "全市场待映射"} for row in cleaned.itertuples(index=False)]


def main() -> int:
    parser = argparse.ArgumentParser(description="自适应行业轮动共同股票池准备器")
    parser.add_argument("--signal-date", default="", help="运行期望信号日；仅记录审计，不将当前股票清单伪装成历史时点池")
    parser.add_argument("--output", type=Path, default=Path("adaptive_industry_rotation_universe.json"))
    parser.add_argument("--provider-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame({"code": ["000001", "300001", "600001", "830001", "000002"], "name": ["样本甲", "样本乙", "样本丙", "北交所样本", "ST样本"]})
        rows = normalize_universe(sample)
        assert [row["code"] for row in rows] == ["000001", "300001", "600001"]
        print("SELF_TEST_OK")
        return 0
    raw = provider_call("stock_info_a_code_name", args.provider_timeout_seconds, ak.stock_info_a_code_name)
    raw = raw.rename(columns={"code": "code", "名称": "name", "name": "name", "代码": "code"})
    universe = normalize_universe(raw)
    if not universe:
        raise RuntimeError("empty_universe_after_normalization")
    generated_at = datetime.now().astimezone()
    signal_date = args.signal_date or generated_at.strftime("%Y%m%d")
    payload = {
        "schema_version": "a-share-adaptive-industry-rotation-universe/v1",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "signal_date_requested": signal_date,
        "universe_snapshot_state": "current_universe_not_historical_point_in_time" if signal_date != generated_at.strftime("%Y%m%d") else "current_universe",
        "count": len(universe),
        "universe": universe,
        "disclosure": "该共同股票池由运行时A股代码清单生成，排除ST/退市名称及非沪深创业板前缀。历史signal_date不代表可回放的历史成分池。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(universe), "output": str(args.output), "signal_date": signal_date}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
