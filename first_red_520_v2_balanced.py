#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""520首红 V2 均衡版：全覆盖分片扫描与跨分片统一交易日汇总。

生产原则：
1. 只使用信号日及此前OHLCV；回测入场固定为下一交易日开盘。
2. AkShare为主数据源，BaoStock仅作备用；两者均有硬超时。
3. 单标的实盘只保留其最新日线上的首红；最终汇总再统一到全市场同一交易日。
4. 输出观察/确认/强确认三层，并由质量闸门决定是否发送完成通知。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import baostock as bs
import pandas as pd
import requests
from tqdm import tqdm


LOG = logging.getLogger("first_red_520_v2")
VERSION = "2026.08.19-v2-balanced"
WORKER_CONFIG: dict[str, Any] = {}
WORKER_BAO_READY = False
WORKER_BAO_DISABLED = False
WORKER_BAO_FAILURES = 0
TIER_RANK = {"observation": 1, "confirmation": 2, "strong_confirmation": 3}


class SourceError(RuntimeError):
    """可恢复的数据源错误。"""


class SourceTimeout(SourceError):
    """数据源硬超时。"""


@dataclass(frozen=True)
class Config:
    low_window: int = 520
    volume_window: int = 20
    near_low_tolerance: float = 1.015
    first_red_lookahead: int = 3
    min_price: float = 5.0
    observation_min_body_pct: float = 0.5
    observation_min_volume_ratio: float = 0.8
    observation_min_close_position: float = 0.50
    confirmation_min_body_pct: float = 1.0
    confirmation_min_volume_ratio: float = 1.1
    confirmation_min_close_position: float = 0.65
    strong_min_volume_ratio: float = 1.5
    max_distance_to_low_pct: float = 6.0
    tier_min: str = "confirmation"
    processes: int = 2
    query_timeout: int = 12
    max_bao_worker_failures: int = 20
    checkpoint_every: int = 25
    max_runtime_seconds: int = 19200
    max_source_error_rate_pct: float = 5.0


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


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def redact(value: Any) -> str:
    text = str(value or "")
    for marker in ("SENDKEY", "SERVERCHAN", "TOKEN", "SECRET", "KEY"):
        text = text.replace(marker, "[REDACTED]")
    return text[:240]


class AlarmTimeout:
    """仅在Linux worker主线程启用的真实硬超时。"""

    def __init__(self, seconds: int):
        self.seconds = max(1, int(seconds))
        self.enabled = False
        self.previous: Any = None

    def _raise(self, *_: Any) -> None:
        raise SourceTimeout(f"查询超过{self.seconds}秒")

    def __enter__(self) -> "AlarmTimeout":
        try:
            self.previous = signal.signal(signal.SIGALRM, self._raise)
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
    """逐行读取BaoStock游标，绝不调用get_data。"""
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
    return pd.DataFrame(rows, columns=fields)


def normalize_history(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {"日期": "date", "date": "date", "开盘": "open", "open": "open", "最高": "high", "high": "high", "最低": "low", "low": "low", "收盘": "close", "close": "close", "成交量": "volume", "volume": "volume"}
    out = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise SourceError(f"{source}日线缺字段:{','.join(sorted(missing))}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0) & (out["volume"] >= 0)]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise SourceError(f"{source}日线清洗后为空")
    return out[["date", "open", "high", "low", "close", "volume"]]


def load_shared_universe(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        raise SourceError(f"共享股票池不存在:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceError(f"共享股票池无法读取:{type(exc).__name__}") from exc
    raw = payload.get("universe") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise SourceError("共享股票池缺少universe列表")
    dedup: dict[str, dict[str, str]] = {}
    for item in raw:
        code = six_code((item.get("code") or item.get("代码")) if isinstance(item, dict) else "")
        name = str((item.get("name") or item.get("code_name") or item.get("名称") if isinstance(item, dict) else "") or "").strip()
        if not code or not name or "ST" in name.upper() or "退" in name:
            continue
        dedup[code] = {"code": code, "name": name, "bs_code": str(item.get("bs_code") or market_code(code))}
    values = [dedup[code] for code in sorted(dedup)]
    if len(values) < 1000:
        raise SourceError(f"共享股票池有效数量异常:{len(values)}")
    return values, {"source": "shared_universe", "path": str(path), "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None, "count": len(values)}


def worker_init(config_dict: dict[str, Any]) -> None:
    global WORKER_CONFIG, WORKER_BAO_READY, WORKER_BAO_DISABLED, WORKER_BAO_FAILURES
    WORKER_CONFIG = config_dict
    WORKER_BAO_READY, WORKER_BAO_DISABLED, WORKER_BAO_FAILURES = False, False, 0
    try:
        login = bs.login()
        WORKER_BAO_READY = getattr(login, "error_code", "1") == "0"
    except Exception:
        WORKER_BAO_READY = False


def fetch_akshare_history(code: str, start_date: str, timeout: int) -> pd.DataFrame:
    try:
        # ThreadPoolExecutor在future超时后会在退出上下文时等待底层线程，形成伪超时。
        # Pool worker的主线程可直接接收SIGALRM，因此用AlarmTimeout保证到期立即中断请求。
        with AlarmTimeout(timeout):
            frame = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.replace("-", ""), end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
    except SourceTimeout:
        raise
    except Exception as exc:
        raise SourceError(f"AkShare日线失败:{redact(exc)}") from exc
    return normalize_history(frame, "AkShare")


def fetch_baostock_history(bs_code: str, start_date: str, timeout: int) -> pd.DataFrame:
    if not WORKER_BAO_READY or WORKER_BAO_DISABLED:
        raise SourceError("BaoStock当前worker不可用")
    with AlarmTimeout(timeout):
        response = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume", start_date=start_date, end_date=datetime.now().strftime("%Y-%m-%d"), frequency="d", adjustflag="2")
        return normalize_history(response_to_frame(response, "BaoStock日线"), "BaoStock")


def fetch_history(task: dict[str, str], config: Config) -> tuple[pd.DataFrame | None, str, str | None, bool]:
    """AkShare主路径；BaoStock仅在主路径失败/不足时备用。"""
    global WORKER_BAO_DISABLED, WORKER_BAO_FAILURES
    start_date = (datetime.now() - timedelta(days=int(config.low_window * 1.8))).strftime("%Y-%m-%d")
    diagnostics: list[str] = []
    timed_out = False
    try:
        history = fetch_akshare_history(task["code"], start_date, config.query_timeout)
        if len(history) >= config.low_window:
            return history, "akshare", None, False
        diagnostics.append(f"AkShare数据不足:{len(history)}")
    except SourceTimeout as exc:
        timed_out = True
        diagnostics.append(redact(exc))
    except Exception as exc:
        diagnostics.append(redact(exc))
    if not WORKER_BAO_DISABLED:
        try:
            history = fetch_baostock_history(task["bs_code"], start_date, config.query_timeout)
            WORKER_BAO_FAILURES = 0
            if len(history) >= config.low_window:
                return history, "baostock", " | ".join(diagnostics) or None, timed_out
            diagnostics.append(f"BaoStock数据不足:{len(history)}")
        except SourceTimeout as exc:
            timed_out = True
            WORKER_BAO_FAILURES += 1
            diagnostics.append(redact(exc))
        except Exception as exc:
            WORKER_BAO_FAILURES += 1
            diagnostics.append(redact(exc))
        if WORKER_BAO_FAILURES >= config.max_bao_worker_failures:
            WORKER_BAO_DISABLED = True
            diagnostics.append("BaoStock连续失败，当前worker熔断")
    return None, "none", " | ".join(diagnostics[-4:]), timed_out


def detect_v2(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """返回全部历史信号；实盘worker只会保留最新日线上的信号。"""
    history = normalize_history(frame, "V2信号输入")
    if len(history) <= config.low_window:
        return pd.DataFrame()
    history["prior_520_low"] = history["low"].rolling(config.low_window, min_periods=config.low_window).min().shift(1)
    history["five_day_low"] = history["low"].rolling(5, min_periods=1).min()
    history["avg_volume"] = history["volume"].rolling(config.volume_window, min_periods=config.volume_window).mean()
    history["ma20"] = history["close"].rolling(20, min_periods=20).mean()
    history["ma20_slope_pct"] = (history["ma20"] / history["ma20"].shift(5) - 1) * 100
    history["red_body_pct"] = (history["close"] - history["open"]) / history["open"] * 100
    history["near_520_low"] = history["low"] <= history["prior_520_low"] * config.near_low_tolerance
    signals: list[dict[str, Any]] = []
    low_zone_age: int | None = None
    for index in range(config.low_window, len(history)):
        row = history.iloc[index]
        if pd.isna(row["prior_520_low"]):
            continue
        if bool(row["near_520_low"]):
            low_zone_age = 0
        elif low_zone_age is not None:
            low_zone_age += 1
        if low_zone_age is None or low_zone_age > config.first_red_lookahead:
            continue
        volume_ratio = float(row["volume"] / row["avg_volume"]) if pd.notna(row["avg_volume"]) and row["avg_volume"] > 0 else 0.0
        body_pct = float(row["red_body_pct"]) if pd.notna(row["red_body_pct"]) else 0.0
        intraday_range = float(row["high"] - row["low"])
        close_position = float((row["close"] - row["low"]) / intraday_range) if intraday_range > 0 else 0.5
        distance = (float(row["close"]) - float(row["prior_520_low"])) / float(row["prior_520_low"]) * 100
        observation = row["close"] > row["open"] and row["close"] >= config.min_price and body_pct >= config.observation_min_body_pct and volume_ratio >= config.observation_min_volume_ratio and close_position >= config.observation_min_close_position and distance <= config.max_distance_to_low_pct
        if not observation:
            continue
        confirmation = body_pct >= config.confirmation_min_body_pct and volume_ratio >= config.confirmation_min_volume_ratio and close_position >= config.confirmation_min_close_position
        ma20_slope = float(row["ma20_slope_pct"]) if pd.notna(row["ma20_slope_pct"]) else -99.0
        strong = confirmation and volume_ratio >= config.strong_min_volume_ratio and ma20_slope >= -1.0
        tier = "strong_confirmation" if strong else "confirmation" if confirmation else "observation"
        score = min(100.0, 40.0 + min(20.0, body_pct * 5.0) + min(20.0, max(0.0, volume_ratio - 0.8) * 20.0) + close_position * 12.0 + max(0.0, 8.0 - distance) * 1.0 + (8.0 if strong else 3.0 if confirmation else 0.0))
        signals.append({"signal_index": int(index), "date": pd.Timestamp(row["date"]), "close": round(float(row["close"]), 4), "volume_ratio": round(volume_ratio, 4), "distance_to_520_low_pct": round(distance, 4), "prior_520_low": round(float(row["prior_520_low"]), 4), "five_day_low": round(float(row["five_day_low"]), 4), "red_body_pct": round(body_pct, 4), "close_position": round(close_position, 4), "ma20_slope_pct": round(ma20_slope, 4), "low_zone_age": int(low_zone_age), "tier": tier, "signal_score": round(score, 2)})
        low_zone_age = None
    return pd.DataFrame(signals)


def process_task(task: dict[str, str]) -> dict[str, Any]:
    config = Config(**WORKER_CONFIG)
    started = time.monotonic()
    history, source, diagnostic, timed_out = fetch_history(task, config)
    base = {"code": task["code"], "name": task["name"], "source": source, "elapsed_sec": round(time.monotonic() - started, 3), "timed_out": timed_out}
    if history is None:
        return {**base, "status": "source_error", "diagnostic": diagnostic or "双源无可用日线", "records": [], "data_last_date": None}
    data_last_date = history.iloc[-1]["date"].strftime("%Y-%m-%d")
    try:
        signals = detect_v2(history, config)
    except Exception as exc:
        return {**base, "status": "logic_error", "diagnostic": redact(exc), "records": [], "data_last_date": data_last_date}
    if signals.empty:
        return {**base, "status": "no_signal", "diagnostic": diagnostic, "records": [], "data_last_date": data_last_date}
    # 实盘仅允许信号日等于该标的最后一根可用日线；历史信号仅供回测模式使用。
    signals = signals.loc[signals["date"].dt.strftime("%Y-%m-%d") == data_last_date]
    signals = signals.loc[signals["tier"].map(TIER_RANK).fillna(0) >= TIER_RANK[config.tier_min]]
    if signals.empty:
        return {**base, "status": "no_signal", "diagnostic": diagnostic, "records": [], "data_last_date": data_last_date}
    records = [{**base, "status": "candidate", "diagnostic": diagnostic, "data_last_date": data_last_date, "signal_date": row["date"].strftime("%Y-%m-%d"), **{key: row[key] for key in ("close", "volume_ratio", "distance_to_520_low_pct", "prior_520_low", "five_day_low", "red_body_pct", "close_position", "ma20_slope_pct", "low_zone_age", "tier", "signal_score")}} for _, row in signals.iterrows()]
    return {**base, "status": "candidate", "diagnostic": diagnostic, "records": records, "data_last_date": data_last_date}


def select_tasks(universe: list[dict[str, str]], offset: int, limit: int) -> list[dict[str, str]]:
    selected = universe[max(0, offset):]
    return selected if limit <= 0 else selected[:limit]


def read_checkpoint(path: Path) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    if not path.is_file():
        return set(), [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("completed_codes", [])), list(payload.get("records", [])), dict(payload.get("stats", {}))
    except Exception:
        return set(), [], {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_checkpoint(path: Path, completed: set[str], records: list[dict[str, Any]], stats: dict[str, int], shard: str) -> None:
    write_json(path, {"version": VERSION, "saved_at": now_text(), "shard": shard, "completed_codes": sorted(completed), "records": records, "stats": stats})


def scan_market(args: argparse.Namespace, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.monotonic()
    output = Path(args.output_dir)
    shard = args.shard.lower()
    universe, universe_meta = load_shared_universe(Path(args.universe_json))
    tasks = select_tasks(universe, args.offset, args.limit)
    checkpoint_path = output / f"first_red_520_v2_{shard}.checkpoint.json"
    completed, records, recovered_stats = read_checkpoint(checkpoint_path) if args.resume else (set(), [], {})
    stats = {"universe": len(universe), "slice": len(tasks), "processed": 0, "candidate_records": 0, "source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "stale_data": 0}
    stats.update({key: int(value) for key, value in recovered_stats.items() if key in stats})
    pending = [task for task in tasks if task["code"] not in completed]
    data_dates: Counter[str] = Counter()
    stop_reason = "completed"
    with mp.Pool(processes=max(1, config.processes), initializer=worker_init, initargs=(asdict(config),)) as pool:
        for item in tqdm(pool.imap_unordered(process_task, pending), total=len(pending), desc=f"首红V2-{shard}", unit="只"):
            completed.add(item["code"])
            stats["processed"] += 1
            if item.get("timed_out"):
                stats["timeout"] += 1
            if item.get("data_last_date"):
                data_dates[str(item["data_last_date"])] += 1
            status = item["status"]
            if status == "source_error":
                stats["source_error"] += 1
            elif status == "logic_error":
                stats["logic_error"] += 1
            elif status == "no_signal":
                stats["no_signal"] += 1
            elif status == "candidate":
                records.extend(item["records"])
                stats["candidate_records"] += len(item["records"])
            if stats["processed"] % max(1, config.checkpoint_every) == 0:
                save_checkpoint(checkpoint_path, completed, records, stats, shard)
            if time.monotonic() - started >= config.max_runtime_seconds:
                stop_reason = "runtime_limit"
                break
    save_checkpoint(checkpoint_path, completed, records, stats, shard)
    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates = candidates.sort_values(["signal_score", "volume_ratio"], ascending=[False, False]).drop_duplicates(["code", "signal_date"], keep="first").reset_index(drop=True)
    max_data_last_date = max(data_dates) if data_dates else None
    state = {"version": VERSION, "state": "completed" if stop_reason == "completed" else stop_reason, "finished_at": now_text(), "shard": shard, "offset": args.offset, "limit": args.limit, "universe": universe_meta, "assigned_count": len(tasks), "coverage_complete": len(completed.intersection({task['code'] for task in tasks})) == len(tasks), "max_data_last_date": max_data_last_date, "data_last_date_counts": dict(sorted(data_dates.items())), "stats": stats, "candidate_count": int(len(candidates)), "config": asdict(config), "stop_reason": stop_reason, "elapsed_seconds": round(time.monotonic() - started, 2)}
    return candidates, state


def write_shard_outputs(output: Path, shard: str, candidates: pd.DataFrame, state: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"first_red_520_v2_{shard}"
    empty = pd.DataFrame(columns=["code", "name", "signal_date", "data_last_date", "tier", "signal_score", "close", "volume_ratio"])
    (candidates if not candidates.empty else empty).to_csv(output / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    write_json(output / f"{prefix}.json", candidates.to_dict("records") if not candidates.empty else [])
    write_json(output / f"{prefix}.state.json", state)
    lines = [f"# 520首红V2均衡版：分片 {shard.upper()}", "", f"- 已处理：{state['stats']['processed']} / {state['assigned_count']}", f"- 候选：{state['candidate_count']}", f"- 本片最大数据日期：{state.get('max_data_last_date') or '无'}", f"- 数据源错误：{state['stats']['source_error']}；超时：{state['stats']['timeout']}", "", "> 分片CSV仅用于汇总诊断；最终当日名单以全市场汇总CSV为准。"]
    (output / f"{prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def read_csv_records(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [{str(key): str(value or "") for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]
    except Exception:
        return []


def consolidate(input_root: Path, output: Path, notify_enabled: bool) -> dict[str, Any]:
    states = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(input_root.rglob("first_red_520_v2_*.state.json"))]
    records: list[dict[str, str]] = []
    for path in sorted(input_root.rglob("first_red_520_v2_?.csv")):
        records.extend(read_csv_records(path))
    universe_count = max((int(item.get("universe", {}).get("count") or 0) for item in states), default=0)
    processed = sum(int((item.get("stats") or {}).get("processed") or 0) for item in states)
    source_errors = sum(int((item.get("stats") or {}).get("source_error") or 0) for item in states)
    max_dates = {str(item.get("max_data_last_date") or "") for item in states if item.get("max_data_last_date")}
    asof_date = max(max_dates) if max_dates else None
    stale_shards = sorted(str(item.get("shard") or "") for item in states if asof_date and item.get("max_data_last_date") != asof_date)
    final_candidates = [item for item in records if asof_date and item.get("signal_date") == asof_date and item.get("data_last_date") == asof_date]
    final_candidates.sort(key=lambda item: (-float(item.get("signal_score") or 0), str(item.get("code") or "")))
    fieldnames = list(dict.fromkeys(key for item in final_candidates for key in item)) or ["code", "name", "signal_date", "tier", "signal_score"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "first_red_520_v2_global_latest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_candidates)
    write_json(output / "first_red_520_v2_global_latest.json", final_candidates)
    coverage_pct = round(processed * 100 / universe_count, 4) if universe_count else 0.0
    error_rate = round(source_errors * 100 / processed, 4) if processed else 100.0
    threshold = max((float((item.get("config") or {}).get("max_source_error_rate_pct") or 5.0) for item in states), default=5.0)
    quality_gate = {"expected_shards": 5, "visible_shards": len(states), "universe_count": universe_count, "processed": processed, "coverage_pct": coverage_pct, "coverage_complete": bool(universe_count and processed == universe_count), "source_errors": source_errors, "source_error_rate_pct": error_rate, "max_source_error_rate_pct": threshold, "asof_date": asof_date, "stale_shards": stale_shards, "passed": bool(len(states) == 5 and universe_count and processed == universe_count and error_rate <= threshold and not stale_shards and asof_date)}
    state = "completed" if quality_gate["passed"] else "attention_required"
    tiers = Counter(item.get("tier") or "unknown" for item in final_candidates)
    summary = {"generated_at": now_text(), "state": state, "quality_gate": quality_gate, "candidates": len(final_candidates), "tier_counts": dict(tiers), "artifacts": {"csv": "first_red_520_v2_global_latest.csv", "json": "first_red_520_v2_global_latest.json"}, "disclaimer": "自动化研究筛选结果，不构成投资建议。"}
    write_json(output / "first_red_520_v2_summary.json", summary)
    lines = ["# 520首红V2均衡版：全市场汇总", "", f"- 状态：`{state}`", f"- 统一数据与信号日期：`{asof_date or '无'}`", f"- 覆盖：{processed}/{universe_count}（{coverage_pct}%）", f"- 数据源错误：{source_errors}（{error_rate}%）", f"- 最终候选：{len(final_candidates)}；强确认{tiers.get('strong_confirmation', 0)}，确认{tiers.get('confirmation', 0)}", f"- 质量闸门：`{'通过' if quality_gate['passed'] else '未通过'}`", "", "> 只有全市场统一日期、100%分片覆盖和质量闸门通过时，才将最终CSV视为当日研究名单。"]
    if notify_enabled:
        key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
        if key:
            try:
                response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": f"首红V2：{state} | {len(final_candidates)}条", "desp": "\n".join(lines)[:3800]}, timeout=15)
                payload = response.json()
                summary["notification"] = "sent" if response.ok and payload.get("code") == 0 else f"failed:http_{response.status_code}"
            except Exception as exc:
                summary["notification"] = f"failed:{type(exc).__name__}"
            write_json(output / "first_red_520_v2_summary.json", summary)
    (output / "first_red_520_v2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_self_test() -> None:
    coverage_universe = [{"code": f"{index:06d}", "name": "测试", "bs_code": "sz.000001"} for index in range(5007)]
    coverage_parts = [select_tasks(coverage_universe, offset, limit) for offset, limit in ((0, 1000), (1000, 1000), (2000, 1000), (3000, 1000), (4000, 0))]
    coverage_codes = [item["code"] for part in coverage_parts for item in part]
    assert len(coverage_codes) == 5007 and len(set(coverage_codes)) == 5007
    dates = pd.bdate_range("2023-01-02", periods=540)
    frame = pd.DataFrame({"date": dates, "open": [10.0] * 540, "high": [10.3] * 540, "low": [10.0] * 540, "close": [10.0] * 540, "volume": [1000.0] * 540})
    frame.loc[538, ["open", "high", "low", "close", "volume"]] = [9.4, 9.5, 9.0, 9.2, 900]
    frame.loc[539, ["open", "high", "low", "close", "volume"]] = [9.2, 9.6, 9.2, 9.5, 1800]
    signals = detect_v2(frame, Config())
    assert len(signals) == 1
    assert signals.iloc[0]["tier"] in {"confirmation", "strong_confirmation"}
    assert signals.iloc[0]["date"].strftime("%Y-%m-%d") == dates[-1].strftime("%Y-%m-%d")
    original_akshare = ak.stock_zh_a_hist
    try:
        def slow_akshare(**_: Any) -> pd.DataFrame:
            time.sleep(2)
            return pd.DataFrame()
        ak.stock_zh_a_hist = slow_akshare
        timeout_started = time.monotonic()
        try:
            fetch_akshare_history("000001", "2024-01-01", 1)
            raise AssertionError("AkShare硬超时未触发")
        except SourceTimeout:
            assert time.monotonic() - timeout_started < 1.5
    finally:
        ak.stock_zh_a_hist = original_akshare
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "collected"
        for index, shard in enumerate(("a", "b", "c", "d", "e")):
            shard_dir = root / f"first-red-v2-{shard}"
            shard_dir.mkdir(parents=True)
            signal_date = "2026-08-17" if shard in {"d", "e"} else "2026-08-16"
            pd.DataFrame([{"code": f"00000{index + 1}", "name": f"样本{shard}", "signal_date": signal_date, "data_last_date": signal_date, "tier": "confirmation", "signal_score": 70 + index}]).to_csv(shard_dir / f"first_red_520_v2_{shard}.csv", index=False, encoding="utf-8-sig")
            write_json(shard_dir / f"first_red_520_v2_{shard}.state.json", {"shard": shard, "universe": {"count": 5}, "max_data_last_date": "2026-08-17", "stats": {"processed": 1, "source_error": 0}, "config": {"max_source_error_rate_pct": 5.0}})
        summary = consolidate(root, Path(temp_dir) / "output", False)
        assert summary["state"] == "completed" and summary["candidates"] == 2 and summary["quality_gate"]["coverage_complete"]
    print("FIRST_RED_520_V2_SELF_TEST_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="520首红V2均衡版生产筛选器")
    parser.add_argument("--mode", choices=["scan", "consolidate"], default="scan")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--universe-json", default=os.environ.get("UNIVERSE_JSON", ""))
    parser.add_argument("--shard", default=os.environ.get("SHARD", "full"))
    parser.add_argument("--offset", type=int, default=env_int("OFFSET", 0))
    parser.add_argument("--limit", type=int, default=env_int("LIMIT", 0), help="0表示扫描offset后的全部剩余股票")
    parser.add_argument("--processes", type=int, default=env_int("NUM_PROCESSES", 2))
    parser.add_argument("--query-timeout", type=int, default=env_int("QUERY_TIMEOUT_SEC", 12))
    parser.add_argument("--max-runtime-seconds", type=int, default=env_int("MAX_RUNTIME_SECONDS", 19200))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.mode == "consolidate":
        summary = consolidate(Path(args.input_root), Path(args.output_dir), args.notify == "true")
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if not args.universe_json:
        raise SystemExit("scan模式必须提供--universe-json")
    config = Config(processes=args.processes, query_timeout=args.query_timeout, max_runtime_seconds=args.max_runtime_seconds)
    candidates, state = scan_market(args, config)
    write_shard_outputs(Path(args.output_dir), args.shard.lower(), candidates, state)
    LOG.info("分片%s完成：处理%s，候选%s，本片最大数据日期=%s", args.shard, state["stats"]["processed"], state["candidate_count"], state.get("max_data_last_date"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        LOG.exception("脚本异常终止：%s", redact(exc))
        raise SystemExit(1)
