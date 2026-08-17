#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为A股分片策略一次性准备、校验并缓存股票池。

设计目标：
1. 每次workflow只请求一次股票池，A/B/C/D四片下载同一份artifact；
2. AkShare主用、BaoStock回退；单源超时在独立进程中强制终止；
3. 两源暂不可用时可使用不超过限定天数的本地缓存；
4. 所有诊断均脱敏，且两源均不可用时以退出码2结束，禁止伪装为成功。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import akshare as ak
import baostock as bs
import pandas as pd


SCHEMA_VERSION = "a-share-universe/v1"
CODE_PATTERN = re.compile(r"(\d{6})")
SENSITIVE_PATTERN = re.compile(r"(?i)(sendkey|serverchan|token|secret|password|apikey|api[_-]?key)=?[^\s,;]+")


class UniverseError(RuntimeError):
    """股票池不可用；调用方必须把它视为业务失败。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(text: object, limit: int = 220) -> str:
    """将外部异常压缩为artifact可安全保存的诊断文本。"""
    compact = " ".join(str(text).replace("\n", " ").split())
    compact = SENSITIVE_PATTERN.sub("<redacted>", compact)
    return compact[:limit] or "未提供详情"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_universe(raw: pd.DataFrame, source: str) -> list[dict[str, str]]:
    """统一两源字段并只保留沪深主板/创业板的非ST、非退市代码。"""
    if raw is None or raw.empty:
        raise UniverseError(f"{source}返回空股票池")
    if source == "akshare":
        frame = raw.rename(columns={"code": "代码", "name": "名称"})
    else:
        frame = raw.rename(columns={"code": "代码", "code_name": "名称"})
    if not {"代码", "名称"}.issubset(frame.columns):
        raise UniverseError(f"{source}字段不完整")
    frame = frame[["代码", "名称"]].copy()
    frame["代码"] = frame["代码"].astype(str).str.extract(CODE_PATTERN, expand=False)
    frame["名称"] = frame["名称"].astype(str).str.strip()
    selected = frame[
        frame["代码"].notna()
        & frame["代码"].str.startswith(("0", "3", "6"))
        & ~frame["名称"].str.contains("ST|退", regex=True, na=False)
    ].drop_duplicates("代码").sort_values("代码")
    if len(selected) < 1000:
        raise UniverseError(f"{source}清洗后代码过少:{len(selected)}")
    return [{"代码": str(row.代码), "名称": str(row.名称)} for row in selected.itertuples(index=False)]


def fetch_akshare() -> list[dict[str, str]]:
    return clean_universe(ak.stock_info_a_code_name(), "akshare")


def fetch_baostock() -> list[dict[str, str]]:
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        raise UniverseError(f"BaoStock登录失败:{redact(getattr(login, 'error_msg', ''))}")
    try:
        response = bs.query_stock_basic()
        if getattr(response, "error_code", "1") != "0":
            raise UniverseError(f"BaoStock股票池查询失败:{redact(getattr(response, 'error_msg', ''))}")
        return clean_universe(response.get_data(), "baostock")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def child_fetch(result_queue: Any, source: str) -> None:
    try:
        universe = fetch_akshare() if source == "akshare" else fetch_baostock()
        result_queue.put({"ok": True, "universe": universe})
    except Exception as error:
        result_queue.put({"ok": False, "reason": redact(f"{type(error).__name__}: {error}")})


def fetch_with_hard_timeout(source: str, timeout_seconds: int) -> list[dict[str, str]]:
    """使用子进程确保连接卡住时能在规定秒数后真实结束。"""
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=child_fetch, args=(result_queue, source))
    process.start()
    try:
        payload = result_queue.get(timeout=max(1, timeout_seconds))
    except queue.Empty as error:
        if process.is_alive():
            process.terminate()
        process.join(5)
        result_queue.close()
        raise UniverseError(f"{source}超过{timeout_seconds}秒无响应") from error
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    result_queue.close()
    if not payload.get("ok"):
        raise UniverseError(str(payload.get("reason", f"{source}未知错误")))
    return list(payload["universe"])


def load_fresh_cache(path: Path, max_age_days: int) -> tuple[list[dict[str, str]], str] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)
        if age < timedelta(seconds=0) or age > timedelta(days=max(0, max_age_days)):
            return None
        universe = payload.get("universe")
        if not isinstance(universe, list) or len(universe) < 1000:
            return None
        normalized = [{"代码": str(item["代码"]), "名称": str(item["名称"])} for item in universe]
        return normalized, generated_at.isoformat(timespec="seconds")
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次性准备A股股票池")
    parser.add_argument("--output", default="output/a_share_universe.json")
    parser.add_argument("--status-output", default="output/a_share_universe_status.json")
    parser.add_argument("--cache-file", default=".a_share_universe_cache/a_share_universe.json")
    parser.add_argument("--cache-max-age-days", type=int, default=3)
    parser.add_argument("--ak-timeout", type=int, default=45)
    parser.add_argument("--bao-timeout", type=int, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=4.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> int:
    codes = [f"6{i:05d}" for i in range(1001)] + ["sh.000001", "300750", "002888"]
    names = ["样本股"] * 1001 + ["平安银行", "宁德时代", "*ST样本"]
    sample = pd.DataFrame({"code": codes, "name": names})
    cleaned = clean_universe(sample, "akshare")
    normalized_codes = {item["代码"] for item in cleaned}
    assert "000001" in normalized_codes and "300750" in normalized_codes
    assert "002888" not in normalized_codes
    assert len(cleaned) == 1003
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    output, status_output, cache_file = Path(args.output), Path(args.status_output), Path(args.cache_file)
    diagnostics: list[dict[str, Any]] = []
    attempts = max(1, args.retries)
    for attempt in range(1, attempts + 1):
        for source, timeout in (("akshare", args.ak_timeout), ("baostock", args.bao_timeout)):
            started = time.monotonic()
            try:
                universe = fetch_with_hard_timeout(source, timeout)
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "generated_at": now_iso(),
                    "source": source,
                    "degraded": False,
                    "universe": universe,
                }
                write_json(output, payload)
                write_json(cache_file, payload)
                write_json(status_output, {
                    "schema_version": "1.0",
                    "state": "ready",
                    "source": source,
                    "universe_size": len(universe),
                    "generated_at": payload["generated_at"],
                    "diagnostics": diagnostics,
                })
                print(f"UNIVERSE_READY source={source} size={len(universe)}")
                return 0
            except UniverseError as error:
                diagnostics.append({"attempt": attempt, "source": source, "elapsed_seconds": round(time.monotonic() - started, 1), "reason": redact(error)})
        if attempt < attempts:
            time.sleep(max(0, args.retry_wait_seconds))

    cached = load_fresh_cache(cache_file, args.cache_max_age_days)
    if cached is not None:
        universe, cached_at = cached
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_iso(),
            "source": "cache",
            "degraded": True,
            "cache_generated_at": cached_at,
            "universe": universe,
        }
        write_json(output, payload)
        write_json(status_output, {
            "schema_version": "1.0",
            "state": "degraded_cache",
            "source": "cache",
            "universe_size": len(universe),
            "generated_at": payload["generated_at"],
            "diagnostics": diagnostics,
        })
        print(f"UNIVERSE_READY source=cache size={len(universe)}")
        return 0

    write_json(status_output, {
        "schema_version": "1.0",
        "state": "failed",
        "reason": "universe_unavailable",
        "generated_at": now_iso(),
        "diagnostics": diagnostics,
    })
    print("UNIVERSE_FAILED reason=universe_unavailable", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
