#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为自适应行业轮动全市场扫描生成共同股票池快照。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
from datetime import datetime, timezone
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


def fetch_akshare_universe(timeout_seconds: float, retries: int) -> pd.DataFrame:
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            raw = provider_call("akshare_stock_info_a_code_name", timeout_seconds, ak.stock_info_a_code_name)
            return raw.rename(columns={"名称": "name", "代码": "code"})
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"akshare_universe_unavailable:{last_error}")


def _baostock_universe() -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        # BaoStock当前发布版仅接受code与code_name；不要传旧版不存在的ipo_date等参数。
        result = bs.query_stock_basic()
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
        if frame.empty:
            return pd.DataFrame(columns=["code", "name"])
        return pd.DataFrame(
            {
                "code": frame["code"].astype(str).str.split(".").str[-1],
                "name": frame["code_name"].astype(str),
            }
        )
    finally:
        bs.logout()


def fetch_baostock_universe(timeout_seconds: float, retries: int) -> pd.DataFrame:
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return provider_call("baostock_query_stock_basic", timeout_seconds, _baostock_universe)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"baostock_universe_unavailable:{last_error}")


def fetch_universe(source: str, timeout_seconds: float, retries: int) -> tuple[pd.DataFrame, str, list[str]]:
    errors: list[str] = []
    if source in {"auto", "akshare"}:
        try:
            return fetch_akshare_universe(timeout_seconds, retries), "akshare", errors
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:240])
            if source == "akshare":
                raise
    try:
        return fetch_baostock_universe(timeout_seconds, retries), "baostock", errors
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc)[:240])
        raise RuntimeError("universe_unavailable:" + " | ".join(errors)) from exc


def load_recent_cache(path: Path, max_age_hours: float) -> tuple[list[dict[str, str]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    origin = str(payload.get("cache_origin_generated_at") or payload.get("generated_at") or "")
    if not origin:
        raise ValueError("cache_missing_generated_at")
    generated_at = datetime.fromisoformat(origin.replace("Z", "+00:00"))
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise ValueError(f"cache_stale_hours:{age_hours:.2f}")
    rows = normalize_universe(pd.DataFrame(payload.get("universe", [])))
    if not rows:
        raise ValueError("cache_empty_after_normalization")
    return rows, origin


def write_unavailable_status(output: Path, generated_at: datetime, errors: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path = output.parent / "adaptive_industry_rotation_universe_status.json"
    status_path.write_text(json.dumps({
        "schema_version": "a-share-adaptive-industry-rotation-universe-status/v1",
        "status": "universe_unavailable",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "source_errors": errors,
        "disclosure": "双源与有效期内专属缓存均不可用；未生成空股票池或合成股票池。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="自适应行业轮动共同股票池准备器")
    parser.add_argument("--signal-date", default="", help="运行期望信号日；仅记录审计，不将当前股票清单伪装成历史时点池")
    parser.add_argument("--output", type=Path, default=Path("adaptive_industry_rotation_universe.json"))
    parser.add_argument("--provider-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--source", choices=["auto", "akshare", "baostock"], default="auto")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache-file", type=Path, default=None, help="仅在双源失败时读取的本任务专属共同池缓存")
    parser.add_argument("--cache-max-age-hours", type=float, default=72.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = pd.DataFrame({"code": ["000001", "300001", "600001", "830001", "000002"], "name": ["样本甲", "样本乙", "样本丙", "北交所样本", "ST样本"]})
        rows = normalize_universe(sample)
        assert [row["code"] for row in rows] == ["000001", "300001", "600001"]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "universe.json"
            cache_path.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "universe": [{"code": "000001", "name": "样本甲"}]}), encoding="utf-8")
            cached_rows, _ = load_recent_cache(cache_path, 1.0)
            assert cached_rows[0]["code"] == "000001"
        print("SELF_TEST_OK")
        return 0
    generated_at = datetime.now().astimezone()
    cache_origin_generated_at: str | None = None
    try:
        raw, data_source, source_errors = fetch_universe(args.source, args.provider_timeout_seconds, args.retries)
        raw = raw.rename(columns={"code": "code", "名称": "name", "name": "name", "代码": "code"})
        universe = normalize_universe(raw)
        if not universe:
            raise RuntimeError("empty_universe_after_normalization")
    except Exception as exc:  # noqa: BLE001
        source_errors = [str(exc)[:500]]
        try:
            if args.source != "auto" or args.cache_file is None:
                raise ValueError("cache_not_available_for_requested_source")
            universe, cache_origin_generated_at = load_recent_cache(args.cache_file, args.cache_max_age_hours)
            data_source = "degraded_cache"
            source_errors.append(f"cache_fallback:{args.cache_file}")
        except Exception as cache_exc:  # noqa: BLE001
            source_errors.append(f"cache:{type(cache_exc).__name__}:{str(cache_exc)[:300]}")
            write_unavailable_status(args.output, generated_at, source_errors)
            print(json.dumps({"status": "universe_unavailable", "status_file": str(args.output.parent / "adaptive_industry_rotation_universe_status.json")}, ensure_ascii=False))
            return 1
    signal_date = args.signal_date or generated_at.strftime("%Y%m%d")
    payload = {
        "schema_version": "a-share-adaptive-industry-rotation-universe/v1",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "signal_date_requested": signal_date,
        "universe_snapshot_state": "current_universe_not_historical_point_in_time" if signal_date != generated_at.strftime("%Y%m%d") else "current_universe",
        "count": len(universe),
        "universe": universe,
        "data_source": data_source,
        "source_errors": source_errors,
        "cache_origin_generated_at": cache_origin_generated_at,
        "disclosure": "该共同股票池由运行时A股代码清单生成，排除ST/退市名称及非沪深创业板前缀。历史signal_date不代表可回放的历史成分池。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(universe), "output": str(args.output), "signal_date": signal_date, "data_source": data_source}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
