#!/usr/bin/env python3
"""
first_red_520.py
================
520天周期低点后的首根红K扫描器（生产替代版）。

设计要点：
1. detect_first_red_to_520_low() 始终只返回一个 signals DataFrame，实盘与回测共用。
2. 信号只使用信号日及以前的数据；回测从下一交易日开盘进入，避免前视偏差。
3. BaoStock不调用 get_data()，避免pandas 2.x/3.x DataFrame.append兼容问题。
4. BaoStock日线优先、AkShare日线回退；单源连续故障只熔断该源，不提前结束全市场扫描。
5. 支持 offset/limit 分片、checkpoint续跑、CSV/JSON/Markdown/状态JSON产物和Server酱单条汇总。

使用示例：
  python first_red_520.py --output-dir output --processes 3
  python first_red_520.py --shard a --offset 0 --limit 1300 --resume
  python first_red_520.py --backtest --scan-limit 200 --hold-days 20 --no-push
  python first_red_520.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import akshare as ak
import baostock as bs
import pandas as pd
import requests
from tqdm import tqdm


LOG = logging.getLogger("first_red_520")
VERSION = "2026.08.17-production"
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
WORKER_BS_READY = False
WORKER_BS_DISABLED = False
WORKER_BS_FAILURES = 0
WORKER_CONFIG: dict[str, Any] = {}


class SourceError(RuntimeError):
    """数据源可恢复错误。"""


class SourceTimeout(SourceError):
    """数据源在指定秒数内未返回。"""


@dataclass(frozen=True)
class Config:
    low_window: int = 520
    volume_window: int = 20
    near_low_tolerance: float = 1.02
    first_red_lookahead: int = 5
    min_price: float = 5.0
    min_red_body_pct: float = 0.0
    min_volume_ratio: float = 0.0
    processes: int = 3
    query_timeout: int = 12
    max_bs_worker_failures: int = 20
    checkpoint_every: int = 50
    universe_cache_days: int = 3
    max_runtime_seconds: int = 19200


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        low_window=args.low_window,
        volume_window=args.volume_window,
        near_low_tolerance=args.near_low_tolerance,
        first_red_lookahead=args.first_red_lookahead,
        min_price=args.min_price,
        min_red_body_pct=args.min_red_body_pct,
        min_volume_ratio=args.min_volume_ratio,
        processes=args.processes,
        query_timeout=args.query_timeout,
        max_bs_worker_failures=args.max_bs_worker_failures,
        checkpoint_every=args.checkpoint_every,
        universe_cache_days=args.universe_cache_days,
        max_runtime_seconds=args.max_runtime_seconds,
    )


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def six_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def market_code(code: str) -> str:
    code = six_code(code)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    if code.startswith(("4", "8")):
        return f"bj.{code}"
    return f"sz.{code}"


def redact(message: Any) -> str:
    text = str(message or "")
    for marker in ("SENDKEY", "SERVERCHAN", "TOKEN", "SECRET", "KEY"):
        text = text.replace(marker, "[REDACTED]")
    return text[:240]


class AlarmTimeout:
    """Linux runner中的主线程真实闹钟超时；不支持时退化为无闹钟。"""

    def __init__(self, seconds: int):
        self.seconds = max(1, int(seconds))
        self.enabled = False
        self.previous: Any = None

    def _raise_timeout(self, *_: Any) -> None:
        raise SourceTimeout(f"查询超过{self.seconds}秒")

    def __enter__(self) -> "AlarmTimeout":
        try:
            self.previous = signal.signal(signal.SIGALRM, self._raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self.enabled = True
        except (AttributeError, ValueError):
            self.enabled = False
        return self

    def __exit__(self, *_: Any) -> None:
        if self.enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)


def response_to_frame(response: Any, label: str) -> pd.DataFrame:
    """逐行读取BaoStock游标，绝不使用response.get_data()。"""
    if getattr(response, "error_code", "1") != "0":
        raise SourceError(f"{label}失败:{redact(getattr(response, 'error_msg', ''))}")
    fields = list(getattr(response, "fields", []) or [])
    if not fields:
        raise SourceError(f"{label}字段为空")
    rows: list[list[str]] = []
    while response.next():
        row = list(response.get_row_data())
        if len(row) != len(fields):
            raise SourceError(f"{label}行字段异常:{len(row)}/{len(fields)}")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields)


def normalize_history(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {
        "日期": "date", "date": "date",
        "开盘": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "收盘": "close", "close": "close",
        "成交量": "volume", "volume": "volume",
    }
    out = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise SourceError(f"{source}日线缺字段:{','.join(sorted(missing))}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "low", "close", "volume"])
    out = out[(out["close"] > 0) & (out["low"] > 0) & (out["volume"] >= 0)]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise SourceError(f"{source}日线清洗后为空")
    return out[["date", "open", "high", "low", "close", "volume"]]


def clean_universe(frame: pd.DataFrame, source: str) -> list[dict[str, str]]:
    code_col = next((c for c in ("code", "代码") if c in frame.columns), None)
    name_col = next((c for c in ("code_name", "name", "名称") if c in frame.columns), None)
    if not code_col or not name_col:
        raise SourceError(f"{source}股票池缺代码或名称字段")
    rows: list[dict[str, str]] = []
    for _, item in frame[[code_col, name_col]].iterrows():
        code = six_code(item[code_col])
        name = str(item[name_col] or "").strip()
        if not code or not name or "ST" in name.upper() or "退" in name:
            continue
        # A股主板、创业板、科创板和北交所；保留全部A股可映射代码。
        if not code.startswith(("0", "2", "3", "4", "6", "8", "9")):
            continue
        rows.append({"code": code, "name": name, "bs_code": market_code(code)})
    dedup = {row["code"]: row for row in rows}
    values = [dedup[code] for code in sorted(dedup)]
    if len(values) < 1000:
        raise SourceError(f"{source}有效股票池过小:{len(values)}")
    return values


def fetch_akshare_universe(timeout: int) -> list[dict[str, str]]:
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            frame = pool.submit(ak.stock_info_a_code_name).result(timeout=timeout)
    except FutureTimeoutError as exc:
        raise SourceTimeout(f"AkShare股票池超时{timeout}秒") from exc
    except Exception as exc:
        raise SourceError(f"AkShare股票池失败:{redact(exc)}") from exc
    return clean_universe(frame, "AkShare")


def fetch_baostock_universe(timeout: int) -> list[dict[str, str]]:
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        raise SourceError(f"BaoStock登录失败:{redact(getattr(login, 'error_msg', ''))}")
    try:
        with AlarmTimeout(timeout):
            frame = response_to_frame(bs.query_stock_basic(), "BaoStock股票池")
        return clean_universe(frame, "BaoStock")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def load_or_fetch_universe(cache_path: Path, config: Config) -> tuple[list[dict[str, str]], dict[str, Any]]:
    now = datetime.now()
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(payload["saved_at"])
            records = payload.get("records", [])
            if now - saved_at <= timedelta(days=config.universe_cache_days) and len(records) >= 1000:
                return records, {"source": "cache", "saved_at": payload["saved_at"], "count": len(records), "diagnostics": []}
        except Exception:
            pass

    diagnostics: list[str] = []
    for source, getter in (("AkShare", fetch_akshare_universe), ("BaoStock", fetch_baostock_universe)):
        for attempt in range(1, 4):
            try:
                records = getter(config.query_timeout)
                payload = {"saved_at": now.isoformat(timespec="seconds"), "source": source, "records": records}
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return records, {"source": source, "saved_at": payload["saved_at"], "count": len(records), "diagnostics": diagnostics}
            except Exception as exc:
                diagnostics.append(f"{source}第{attempt}次:{redact(exc)}")
                time.sleep(min(2 * attempt, 4))
    raise SourceError("股票池不可用；" + " | ".join(diagnostics[-6:]))


def load_shared_universe(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """读取prepare_a_share_universe.py的共享产物，拒绝不完整或过小的股票池。"""
    if not path.is_file():
        raise SourceError(f"共享股票池文件不存在:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceError(f"共享股票池无法读取:{type(exc).__name__}") from exc
    raw = payload.get("universe") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise SourceError("共享股票池缺少universe列表")
    records = clean_universe(pd.DataFrame(raw), "共享股票池")
    meta = {
        "source": "shared_universe",
        "path": str(path),
        "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
        "count": len(records),
        "diagnostics": [],
    }
    return records, meta


def worker_init(config_dict: dict[str, Any]) -> None:
    global WORKER_BS_READY, WORKER_BS_DISABLED, WORKER_BS_FAILURES, WORKER_CONFIG
    WORKER_CONFIG = config_dict
    WORKER_BS_READY = False
    WORKER_BS_DISABLED = False
    WORKER_BS_FAILURES = 0
    try:
        login = bs.login()
        WORKER_BS_READY = getattr(login, "error_code", "1") == "0"
    except Exception:
        WORKER_BS_READY = False


def fetch_baostock_history(bs_code: str, start_date: str, timeout: int) -> pd.DataFrame:
    if not WORKER_BS_READY or WORKER_BS_DISABLED:
        raise SourceError("BaoStock当前worker不可用")
    with AlarmTimeout(timeout):
        response = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        raw = response_to_frame(response, "BaoStock日线")
    return normalize_history(raw, "BaoStock")


def fetch_akshare_history(code: str, start_date: str, timeout: int) -> pd.DataFrame:
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            frame = pool.submit(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            ).result(timeout=timeout)
    except FutureTimeoutError as exc:
        raise SourceTimeout(f"AkShare日线超时{timeout}秒") from exc
    except Exception as exc:
        raise SourceError(f"AkShare日线失败:{redact(exc)}") from exc
    return normalize_history(frame, "AkShare")


def detect_first_red_to_520_low(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """返回所有有效信号；只使用每个信号日及此前的OHLCV，不读取未来bar。"""
    required = {"date", "open", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"日线缺字段:{','.join(sorted(required - set(df.columns)))}")
    frame = normalize_history(df, "信号输入")
    if len(frame) <= config.low_window:
        return pd.DataFrame(columns=["signal_index", "date", "close", "volume_ratio", "distance_to_520_low_pct", "prior_520_low", "five_day_low"])

    frame["prior_520_low"] = frame["low"].rolling(config.low_window, min_periods=config.low_window).min().shift(1)
    frame["five_day_low"] = frame["low"].rolling(5, min_periods=1).min()
    frame["avg_volume"] = frame["volume"].rolling(config.volume_window, min_periods=config.volume_window).mean()
    frame["red_body_pct"] = (frame["close"] - frame["open"]) / frame["open"].replace(0, pd.NA) * 100
    frame["near_520_low"] = frame["low"] <= frame["prior_520_low"] * config.near_low_tolerance

    signals: list[dict[str, Any]] = []
    low_zone_age: int | None = None
    for index in range(config.low_window, len(frame)):
        row = frame.iloc[index]
        if pd.isna(row["prior_520_low"]):
            continue
        if bool(row["near_520_low"]):
            low_zone_age = 0
        elif low_zone_age is not None:
            low_zone_age += 1
        if low_zone_age is None or low_zone_age > config.first_red_lookahead:
            continue

        volume_ratio = float(row["volume"] / row["avg_volume"]) if pd.notna(row["avg_volume"]) and row["avg_volume"] > 0 else 0.0
        is_first_red = row["close"] > row["open"] and float(row["red_body_pct"] or 0.0) >= config.min_red_body_pct
        is_valid = is_first_red and row["close"] >= config.min_price and volume_ratio >= config.min_volume_ratio
        if not is_valid:
            continue
        distance = (float(row["close"]) - float(row["prior_520_low"])) / float(row["prior_520_low"]) * 100
        signals.append({
            "signal_index": int(index),
            "date": pd.Timestamp(row["date"]),
            "close": round(float(row["close"]), 4),
            "volume_ratio": round(volume_ratio, 4),
            "distance_to_520_low_pct": round(distance, 4),
            "prior_520_low": round(float(row["prior_520_low"]), 4),
            "five_day_low": round(float(row["five_day_low"]), 4),
            "red_body_pct": round(float(row["red_body_pct"]), 4),
            "low_zone_age": int(low_zone_age),
        })
        low_zone_age = None
    return pd.DataFrame(signals)


def _fetch_history_for_task(task: dict[str, Any], config: Config) -> tuple[pd.DataFrame | None, str, str | None, bool]:
    global WORKER_BS_DISABLED, WORKER_BS_FAILURES
    start_date = (datetime.now() - timedelta(days=int(config.low_window * 1.75))).strftime("%Y-%m-%d")
    diagnostics: list[str] = []
    timed_out = False
    if not WORKER_BS_DISABLED:
        try:
            data = fetch_baostock_history(task["bs_code"], start_date, config.query_timeout)
            WORKER_BS_FAILURES = 0
            if len(data) >= config.low_window:
                return data, "baostock", None, False
            diagnostics.append(f"BaoStock数据不足:{len(data)}")
        except SourceTimeout as exc:
            timed_out = True
            WORKER_BS_FAILURES += 1
            diagnostics.append(redact(exc))
        except Exception as exc:
            WORKER_BS_FAILURES += 1
            diagnostics.append(redact(exc))
        if WORKER_BS_FAILURES >= config.max_bs_worker_failures:
            WORKER_BS_DISABLED = True
            diagnostics.append("BaoStock连续失败，当前worker已熔断，仅使用AkShare")
    try:
        data = fetch_akshare_history(task["code"], start_date, config.query_timeout)
        if len(data) >= config.low_window:
            return data, "akshare", " | ".join(diagnostics) or None, timed_out
        diagnostics.append(f"AkShare数据不足:{len(data)}")
    except SourceTimeout as exc:
        timed_out = True
        diagnostics.append(redact(exc))
    except Exception as exc:
        diagnostics.append(redact(exc))
    return None, "none", " | ".join(diagnostics[-4:]), timed_out


def process_task(payload: tuple[dict[str, Any], bool, int]) -> dict[str, Any]:
    task, backtest, hold_days = payload
    config = Config(**WORKER_CONFIG)
    started = time.monotonic()
    history, source, diagnostic, timed_out = _fetch_history_for_task(task, config)
    base = {
        "code": task["code"], "name": task["name"], "source": source,
        "elapsed_sec": round(time.monotonic() - started, 3), "timed_out": timed_out,
    }
    if history is None:
        return {**base, "status": "source_error", "diagnostic": diagnostic or "双源无可用日线", "signals": []}
    try:
        signals = detect_first_red_to_520_low(history, config)
    except Exception as exc:
        return {**base, "status": "logic_error", "diagnostic": redact(exc), "signals": []}

    if signals.empty:
        return {**base, "status": "no_signal", "diagnostic": diagnostic, "signals": []}
    records: list[dict[str, Any]] = []
    for _, row in signals.iterrows():
        record = {**base, "status": "candidate", "diagnostic": diagnostic, "signal_date": row["date"].strftime("%Y-%m-%d"), "close": float(row["close"]), "volume_ratio": float(row["volume_ratio"]), "distance_to_520_low_pct": float(row["distance_to_520_low_pct"]), "prior_520_low": float(row["prior_520_low"]), "five_day_low": float(row["five_day_low"]), "red_body_pct": float(row["red_body_pct"]), "low_zone_age": int(row["low_zone_age"])}
        if backtest:
            signal_index = int(row["signal_index"])
            entry_index = signal_index + 1  # 信号日收盘后才可知，下一交易日才允许进入。
            exit_index = entry_index + hold_days
            if exit_index < len(history):
                entry_price = float(history.iloc[entry_index]["open"])
                exit_price = float(history.iloc[exit_index]["close"])
                record.update({"entry_date": history.iloc[entry_index]["date"].strftime("%Y-%m-%d"), "exit_date": history.iloc[exit_index]["date"].strftime("%Y-%m-%d"), "entry_price": round(entry_price, 4), "exit_price": round(exit_price, 4), "hold_days": hold_days, "return_pct": round((exit_price / entry_price - 1) * 100, 4) if entry_price > 0 else None})
            else:
                record.update({"hold_days": hold_days, "return_pct": None, "backtest_status": "insufficient_future_bars"})
        records.append(record)
    return {**base, "status": "candidate", "diagnostic": diagnostic, "signals": records}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("completed_codes", []))
    except Exception:
        return set()


def save_checkpoint(path: Path, completed: set[str], stats: dict[str, int], shard: str) -> None:
    write_json(path, {"version": VERSION, "saved_at": now_text(), "shard": shard, "completed_codes": sorted(completed), "completed_count": len(completed), "stats": stats})


def select_tasks(universe: list[dict[str, str]], offset: int, limit: int, scan_limit: int) -> list[dict[str, str]]:
    selected = universe[max(0, offset):]
    if limit > 0:
        selected = selected[:limit]
    if scan_limit > 0:
        selected = selected[:scan_limit]
    return selected


def scan_market(args: argparse.Namespace, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard = args.shard.lower()
    universe_cache = output / "first_red_520_universe_cache.json"
    if args.universe_json:
        universe, universe_meta = load_shared_universe(Path(args.universe_json))
    else:
        universe, universe_meta = load_or_fetch_universe(universe_cache, config)
    tasks = select_tasks(universe, args.offset, args.limit, args.scan_limit)
    checkpoint_path = output / f"first_red_520_{shard}.checkpoint.json"
    completed = load_checkpoint(checkpoint_path) if args.resume else set()
    pending = [task for task in tasks if task["code"] not in completed]
    stats = {"universe": len(universe), "slice": len(tasks), "resumed_completed": len(completed.intersection({t['code'] for t in tasks})), "processed": 0, "candidate_records": 0, "source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0}
    records: list[dict[str, Any]] = []
    payloads = [(task, bool(args.backtest), int(args.hold_days)) for task in pending]
    LOG.info("开始扫描：股票池=%s，当前分片=%s，待处理=%s，进程=%s", len(universe), len(tasks), len(pending), config.processes)
    worker_config = asdict(config)
    stop_reason = "completed"
    with mp.Pool(processes=max(1, config.processes), initializer=worker_init, initargs=(worker_config,)) as pool:
        for item in tqdm(pool.imap_unordered(process_task, payloads), total=len(payloads), desc=f"520首红-{shard}", unit="只"):
            code = item["code"]
            completed.add(code)
            stats["processed"] += 1
            if item.get("timed_out"):
                stats["timeout"] += 1
            status = item["status"]
            if status == "source_error":
                stats["source_error"] += 1
            elif status == "logic_error":
                stats["logic_error"] += 1
            elif status == "no_signal":
                stats["no_signal"] += 1
            elif status == "candidate":
                signals = item.get("signals", [])
                records.extend(signals)
                stats["candidate_records"] += len(signals)
            if stats["processed"] % max(1, config.checkpoint_every) == 0:
                save_checkpoint(checkpoint_path, completed, stats, shard)
            if time.monotonic() - started >= config.max_runtime_seconds:
                stop_reason = "runtime_limit"
                LOG.warning("分片%s触发主动运行时限：%s秒；已处理%s/%s", shard, config.max_runtime_seconds, stats["processed"], len(tasks))
                break
    save_checkpoint(checkpoint_path, completed, stats, shard)
    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates = candidates.sort_values(["volume_ratio", "distance_to_520_low_pct"], ascending=[False, True]).drop_duplicates(["code", "signal_date"], keep="first").reset_index(drop=True)
    state = {"version": VERSION, "state": "completed" if stop_reason == "completed" else stop_reason, "finished_at": now_text(), "shard": shard, "offset": args.offset, "limit": args.limit, "backtest": bool(args.backtest), "hold_days": args.hold_days if args.backtest else None, "universe": universe_meta, "stats": stats, "candidate_count": int(len(candidates)), "coverage_complete": len(completed.intersection({t['code'] for t in tasks})) == len(tasks), "config": asdict(config), "stop_reason": stop_reason, "elapsed_seconds": round(time.monotonic() - started, 2)}
    return candidates, state


def backtest_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_pct"), errors="coerce").dropna() if not frame.empty else pd.Series(dtype=float)
    if values.empty:
        return {"evaluated_signals": 0, "note": "没有具有完整持有期的信号；不输出收益结论。"}
    return {"evaluated_signals": int(len(values)), "win_rate_pct": round(float((values > 0).mean() * 100), 4), "mean_return_pct": round(float(values.mean()), 4), "median_return_pct": round(float(values.median()), 4), "best_return_pct": round(float(values.max()), 4), "worst_return_pct": round(float(values.min()), 4)}


def write_outputs(output: Path, shard: str, candidates: pd.DataFrame, state: dict[str, Any]) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"first_red_520_{shard}"
    csv_path = output / f"{prefix}.csv"
    json_path = output / f"{prefix}.json"
    state_path = output / f"{prefix}.state.json"
    markdown_path = output / f"{prefix}.md"
    empty_columns = ["code", "name", "signal_date", "close", "volume_ratio", "distance_to_520_low_pct", "source", "return_pct"]
    (candidates if not candidates.empty else pd.DataFrame(columns=empty_columns)).to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_json(json_path, candidates.to_dict("records") if not candidates.empty else [])
    write_json(state_path, state)
    lines = [f"# 520天首红扫描：分片 {shard.upper()}", "", f"- 完成时间：{state['finished_at']}", f"- 股票池：{state['universe']['count']}只（{state['universe']['source']}）", f"- 分片处理：{state['stats']['processed']} / {state['stats']['slice']}", f"- 候选记录：{state['candidate_count']}", f"- 双源失败：{state['stats']['source_error']}；超时：{state['stats']['timeout']}；逻辑错误：{state['stats']['logic_error']}", "", "> 自动化筛选记录，不构成投资建议。", ""]
    if state.get("backtest"):
        lines.extend(["## 回测口径", "信号仅使用信号日及此前数据；入场为下一交易日开盘，退出为入场后指定持有期的收盘。", "", "```json", json.dumps(backtest_summary(candidates), ensure_ascii=False, indent=2), "```"])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "state": state_path, "markdown": markdown_path}


def send_serverchan(title: str, body: str, key: str) -> bool:
    if not key:
        return False
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body[:3800]}, timeout=15)
        payload = response.json()
        return response.ok and payload.get("code") == 0
    except Exception as exc:
        LOG.warning("Server酱发送失败：%s", redact(exc))
        return False


def run_self_test() -> None:
    dates = pd.bdate_range("2021-01-01", periods=540)
    frame = pd.DataFrame({"date": dates, "open": [10.0] * 540, "high": [10.2] * 540, "low": [9.9] * 540, "close": [10.0] * 540, "volume": [1000.0] * 540})
    frame.loc[520, ["open", "high", "low", "close", "volume"]] = [9.4, 9.5, 9.0, 9.2, 1200]
    frame.loc[521, ["open", "high", "low", "close", "volume"]] = [9.2, 9.6, 9.2, 9.5, 1800]
    config = Config()
    signals = detect_first_red_to_520_low(frame, config)
    assert len(signals) == 1, f"期望1个信号，得到{len(signals)}"
    assert signals.iloc[0]["signal_index"] == 521
    assert int(signals.iloc[0]["low_zone_age"]) == 1
    fake_response = type("R", (), {"error_code": "0", "fields": ["code", "name"], "_rows": [["sh.600000", "样本"], ["sz.000001", "样本2"]], "_i": 0, "next": lambda self: setattr(self, "_i", self._i + 1) or self._i <= len(self._rows), "get_row_data": lambda self: self._rows[self._i - 1]})()
    assert list(response_to_frame(fake_response, "自检").columns) == ["code", "name"]
    print("FIRST_RED_520_SELF_TEST_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="520天周期低点首红扫描器（生产版）")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))
    parser.add_argument("--universe-json", default=os.environ.get("UNIVERSE_JSON", ""), help="可选：prepare_a_share_universe.py生成的共享股票池JSON")
    parser.add_argument("--shard", default=os.environ.get("SHARD", "full"))
    parser.add_argument("--offset", type=int, default=env_int("OFFSET", 0))
    parser.add_argument("--limit", type=int, default=env_int("LIMIT", 0), help="0表示扫描offset后全部股票")
    parser.add_argument("--scan-limit", type=int, default=env_int("SCAN_LIMIT", 0), help="测试限额，0表示不额外限制")
    parser.add_argument("--processes", type=int, default=env_int("NUM_PROCESSES", 3))
    parser.add_argument("--query-timeout", type=int, default=env_int("QUERY_TIMEOUT_SEC", 12))
    parser.add_argument("--low-window", type=int, default=env_int("LOW_WINDOW", 520))
    parser.add_argument("--volume-window", type=int, default=env_int("VOL_WINDOW", 20))
    parser.add_argument("--near-low-tolerance", type=float, default=env_float("NEW_LOW_TOLERANCE", 1.02))
    parser.add_argument("--first-red-lookahead", type=int, default=env_int("FIRST_RED_LOOKAHEAD", 5))
    parser.add_argument("--min-price", type=float, default=env_float("MIN_PRICE", 5.0))
    parser.add_argument("--min-red-body-pct", type=float, default=env_float("MIN_RED_BODY_PCT", 0.0))
    parser.add_argument("--min-volume-ratio", type=float, default=env_float("MIN_VOLUME_RATIO", 0.0))
    parser.add_argument("--max-bs-worker-failures", type=int, default=env_int("MAX_BS_WORKER_FAILURES", 20))
    parser.add_argument("--checkpoint-every", type=int, default=env_int("CHECKPOINT_EVERY", 50))
    parser.add_argument("--max-runtime-seconds", type=int, default=env_int("MAX_RUNTIME_SECONDS", 19200), help="主动停止并写出产物的最长运行秒数")
    parser.add_argument("--universe-cache-days", type=int, default=env_int("UNIVERSE_CACHE_DAYS", 3))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backtest", action="store_true", help="以同一信号规则进行无前视偏差持有期回测")
    parser.add_argument("--hold-days", type=int, default=env_int("BACKTEST_HOLD_DAYS", 20))
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.low_window < 60 or args.volume_window < 2 or args.hold_days < 1:
        raise SystemExit("LOW_WINDOW至少60、VOL_WINDOW至少2、HOLD_DAYS至少1")
    config = build_config(args)
    candidates, state = scan_market(args, config)
    output_paths = write_outputs(Path(args.output_dir), args.shard.lower(), candidates, state)
    LOG.info("产物已写入：%s", ", ".join(str(path) for path in output_paths.values()))
    if args.backtest:
        LOG.info("回测统计：%s", json.dumps(backtest_summary(candidates), ensure_ascii=False))
    if not args.no_push and not args.backtest:
        key = os.environ.get("SENDKEY") or os.environ.get("SERVERCHAN_KEY", "")
        title = f"520天首红：{len(candidates)}条 | {args.shard.upper()}片"
        body = "\n".join([
            f"时间：{state['finished_at']}",
            f"股票池：{state['universe']['count']}只（{state['universe']['source']}）",
            f"分片处理：{state['stats']['processed']}/{state['stats']['slice']}，双源失败：{state['stats']['source_error']}，超时：{state['stats']['timeout']}",
            "",
            *[f"- {row['code']} {row['name']} | {row['signal_date']} | 量比{row['volume_ratio']:.2f} | 距520低点{row['distance_to_520_low_pct']:.2f}%" for _, row in candidates.head(20).iterrows()],
            "",
            "自动化筛选记录，不构成投资建议。",
        ])
        if key:
            LOG.info("Server酱推送：%s", "成功" if send_serverchan(title, body, key) else "失败")
        else:
            LOG.info("未配置SENDKEY，跳过推送")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        LOG.exception("脚本异常终止：%s", redact(exc))
        raise SystemExit(1)
