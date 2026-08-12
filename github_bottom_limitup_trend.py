#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 单文件版：A 股底部涨停趋势筛选器 (矩阵接入版)
================================================
特性：
- 只使用 Python 标准库；无需 pip install；
- 使用东方财富公开行情接口，先批量获取资金流及实时行情，再对少量候选拉取日线；
- 融合：肯纳特通道、520 日布林带、120 周布林带周线首红、近期底部、涨停趋势；
- 输出 CSV / JSON / Markdown；支持 GitHub Actions、Codespaces、本地 Python 3.10+；
- 【矩阵接入】支持环境变量传参，自动接入 ServerChan 微信推送。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

VERSION = "1.1.0-matrix"
CN_TZ = timezone(timedelta(hours=8), name="CST")
LOGGER = logging.getLogger("bottom_limitup_trend")
USER_AGENT = "Mozilla/5.0 (compatible; GitHub-Ashare-Screener/1.0; +https://github.com/)"

SPOT_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
FUND_RANK_URL = "https://push2.eastmoney.com/api/qt/clist/get"
DAILY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

CSV_FIELDS = (
    "code", "name", "market", "last_price", "spot_pct", "market_cap_yi", "turnover",
    "main_inflow", "super_inflow", "score_fund", "signal_count", "trend_score",
    "kc_signal", "kc_position", "bb520_signal", "bb520_position", "weekly_first_red",
    "weekly_bar_complete", "bb120w_signal", "bb120w_position", "bottom_signal",
    "bottom_distance_pct", "limitup_trend_signal", "limitup_count", "history_status",
)

@dataclass(frozen=True)
class Config:
    min_main_inflow: float = 1_000_000.0
    min_super_inflow: float = 0.0
    min_market_cap: float = 1_000_000_000.0
    max_market_cap: float = 500_000_000_000.0
    top_n: int = 30
    candidate_cap: int = 150
    min_signals: int = 5
    include_bj: bool = False
    include_st: bool = False
    workers: int = 4
    retries: int = 3
    timeout_sec: float = 20.0
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("output")
    cache_ttl_sec: int = 900
    force_refresh: bool = False
    history_calendar_days: int = 1_000
    kc_ema_period: int = 20
    kc_atr_period: int = 10
    kc_multiplier: float = 2.0
    kc_slope_lookback: int = 3
    kc_position_max: float = 1.05
    bb_daily_period: int = 520
    bb_daily_std_multiplier: float = 2.0
    bb_daily_position_max: float = 0.35
    bb_weekly_period: int = 120
    bb_weekly_std_multiplier: float = 2.0
    bb_weekly_position_max: float = 0.60
    only_closed_weeks: bool = True
    bottom_lookback_days: int = 120
    bottom_max_distance: float = 0.35
    limitup_lookback_days: int = 60
    limitup_min_count: int = 1
    trend_ema_period: int = 20
    trend_slope_lookback: int = 3

class DataSourceError(RuntimeError):
    pass

def cn_now() -> datetime:
    return datetime.now(CN_TZ)

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def clean_code(value: Any) -> str:
    if value is None: return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text): text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6) if digits and len(digits) <= 6 else ""

def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or isinstance(value, bool): return default if value is None else float(value)
    if isinstance(value, (int, float)): return float(value) if math.isfinite(float(value)) else default
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "--", "-"}: return default
    matched = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
    if not matched: return default
    try: number = float(matched.group())
    except ValueError: return default
    if "万亿" in text: return number * 1_000_000_000_000
    if "亿" in text: return number * 100_000_000
    if "万" in text: return number * 10_000
    return number

def finite_or_none(value: Any) -> Optional[float]:
    number = to_float(value)
    return number if math.isfinite(number) else None

def infer_market(code: str) -> str:
    if code.startswith("6"): return "sh"
    if code.startswith(("0", "3")): return "sz"
    if code.startswith(("4", "8")): return "bj"
    return ""

def is_regular_a_share(code: str, include_bj: bool) -> bool:
    if not re.fullmatch(r"\d{6}", code): return False
    if code.startswith(("0", "3", "6")): return True
    return include_bj and code.startswith(("4", "8"))

def limitup_threshold(code: str) -> float:
    if code.startswith(("300", "301", "688", "689")): return 19.8
    if code.startswith(("4", "8")): return 29.8
    return 9.8

def format_money(value: Any) -> str:
    number = to_float(value, default=0.0)
    if abs(number) >= 100_000_000: return f"{number / 100_000_000:.2f}亿"
    return f"{number / 10_000:.0f}万"

def format_percent(value: Any) -> str:
    number = finite_or_none(value)
    return "" if number is None else f"{number:.2f}%"

def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (datetime, date)): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

class JsonCache:
    def __init__(self, directory: Path, ttl_sec: int, force_refresh: bool) -> None:
        self.directory = directory
        self.ttl_sec = ttl_sec
        self.force_refresh = force_refresh
        self.lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        if self.force_refresh: return None
        try:
            with self._path(key).open("r", encoding="utf-8") as file:
                payload = json.load(file)
            created_at = float(payload["created_at_epoch"])
            if time.time() - created_at > self.ttl_sec: return None
            return payload["data"]
        except Exception: return None

    def set(self, key: str, data: Any) -> None:
        payload = {"created_at_epoch": time.time(), "data": json_safe(data)}
        destination = self._path(key)
        with self.lock:
            descriptor, temporary = tempfile.mkstemp(prefix=".cache-", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, destination)
            except OSError as exc:
                LOGGER.debug("缓存写入失败：%s", exc)
                try: os.unlink(temporary)
                except OSError: pass

def http_json(url: str, params: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    encoded = urlencode({key: str(value) for key, value in params.items()})
    request = Request(f"{url}?{encoded}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception as exc: raise DataSourceError(str(exc)) from exc
    if not isinstance(data, dict): raise DataSourceError("数据源返回非对象 JSON")
    return data

def retry_call(action: Callable[[], Any], label: str, cfg: Config) -> Any:
    error: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try: return action()
        except Exception as exc:
            error = exc
            if attempt == cfg.retries: break
            delay = min(2 ** (attempt - 1), 8)
            LOGGER.warning("%s 失败（%s/%s）：%s；%s 秒后重试", label, attempt, cfg.retries, exc, delay)
            time.sleep(delay)
    raise DataSourceError(f"{label} 在 {cfg.retries} 次尝试后仍失败：{error}")

def extract_diff(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    data = payload.get("data")
    if not isinstance(data, dict): raise DataSourceError("数据源没有 data 对象")
    diff = data.get("diff")
    if isinstance(diff, list): rows = [row for row in diff if isinstance(row, dict)]
    elif isinstance(diff, dict): rows = [row for row in diff.values() if isinstance(row, dict)]
    else: raise DataSourceError("数据源没有可用 diff 字段")
    total = int(to_float(data.get("total"), default=len(rows)))
    return rows, max(total, len(rows))

def fetch_paginated(url: str, base_params: dict[str, Any], label: str, cfg: Config, page_size: int = 200) -> list[dict[str, Any]]:
    params = dict(base_params)
    params.update({"pn": 1, "pz": page_size, "np": 1})
    first = retry_call(lambda: http_json(url, params, cfg.timeout_sec), f"{label}（第1页）", cfg)
    rows, total = extract_diff(first)
    pages = max(1, math.ceil(total / page_size))
    if pages > 1: LOGGER.info("%s：预计 %s 页，共 %s 行", label, pages, total)
    for page in range(2, pages + 1):
        page_params = dict(params)
        page_params["pn"] = page
        payload = retry_call(lambda page_params=page_params: http_json(url, page_params, cfg.timeout_sec), f"{label}（第{page}页）", cfg)
        page_rows, _ = extract_diff(payload)
        rows.extend(page_rows)
    return rows

def get_spot(cfg: Config, cache: JsonCache) -> dict[str, dict[str, Any]]:
    key = "spot:all_a:v1"
    cached = cache.get(key)
    if isinstance(cached, list): raw_rows = cached
    else:
        raw_rows = fetch_paginated(SPOT_URL, {
            "po": 1, "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": 2, "invt": 2, "fid": "f12",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22",
        }, "获取 A 股实时行情", cfg)
        cache.set(key, raw_rows)

    result: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        code = clean_code(row.get("f12"))
        if not code: continue
        result[code] = {
            "code": code, "name": str(row.get("f14") or "").strip(),
            "last_price": finite_or_none(row.get("f2")), "spot_pct": finite_or_none(row.get("f3")),
            "amount": finite_or_none(row.get("f6")), "turnover": finite_or_none(row.get("f8")),
            "market_cap": finite_or_none(row.get("f21")),
        }
    if not result: raise DataSourceError("实时行情清洗后为空")
    return result

def get_fund_rank(cfg: Config, cache: JsonCache) -> list[dict[str, Any]]:
    key = "fund_rank:today:v1"
    cached = cache.get(key)
    if isinstance(cached, list): raw_rows = cached
    else:
        raw_rows = fetch_paginated(FUND_RANK_URL, {
            "po": 1, "ut": "b2884a393a59ad64002292a3e90d46a5", "fltt": 2, "invt": 2, "fid": "f62",
            "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
            "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f124",
        }, "获取今日个股资金流排行", cfg)
        cache.set(key, raw_rows)

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        code = clean_code(row.get("f12"))
        if not code or code in seen: continue
        seen.add(code)
        result.append({
            "code": code, "name": str(row.get("f14") or "").strip(),
            "last_price": finite_or_none(row.get("f2")), "spot_pct": finite_or_none(row.get("f3")),
            "main_inflow": finite_or_none(row.get("f62")), "main_inflow_pct": finite_or_none(row.get("f184")),
            "super_inflow": finite_or_none(row.get("f66")), "super_inflow_pct": finite_or_none(row.get("f69")),
            "large_inflow": finite_or_none(row.get("f72")), "large_inflow_pct": finite_or_none(row.get("f75")),
            "mid_inflow": finite_or_none(row.get("f78")), "small_inflow": finite_or_none(row.get("f84")),
        })
    if not result: raise DataSourceError("资金流排行清洗后为空")
    return result

def base_score(record: dict[str, Any]) -> float:
    main = max(to_float(record.get("main_inflow"), 0.0), 0.0)
    super_flow = max(to_float(record.get("super_inflow"), 0.0), 0.0)
    main_pct = min(max(to_float(record.get("main_inflow_pct"), 0.0), 0.0), 20.0)
    change = min(max(to_float(record.get("spot_pct"), 0.0), 0.0), 10.0)
    turnover = min(max(to_float(record.get("turnover"), 0.0), 0.0), 15.0)
    main_component = min(math.log1p(main) / math.log(100_000_000), 1.0) * 40
    super_component = min(super_flow / max(main, 1.0), 1.0) * 20
    return round(main_component + super_component + main_pct / 20 * 15 + change / 10 * 15 + turnover / 15 * 10, 2)

def build_candidates(spot: dict[str, dict[str, Any]], funds: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for fund in funds:
        code = str(fund["code"])
        snapshot = spot.get(code)
        if snapshot is None or not is_regular_a_share(code, cfg.include_bj): continue
        name = fund["name"] or snapshot["name"]
        if not cfg.include_st and "ST" in name.upper(): continue
        market_cap = finite_or_none(snapshot.get("market_cap"))
        main_inflow = finite_or_none(fund.get("main_inflow"))
        super_inflow = finite_or_none(fund.get("super_inflow"))
        if market_cap is None or main_inflow is None: continue
        if not (cfg.min_market_cap <= market_cap <= cfg.max_market_cap): continue
        if main_inflow < cfg.min_main_inflow or (super_inflow or 0.0) < cfg.min_super_inflow: continue
        last_price = finite_or_none(fund.get("last_price")) or finite_or_none(snapshot.get("last_price"))
        spot_pct = finite_or_none(fund.get("spot_pct"))
        if spot_pct is None: spot_pct = finite_or_none(snapshot.get("spot_pct"))
        record = {
            **fund, "code": code, "name": name, "market": infer_market(code),
            "last_price": last_price, "spot_pct": spot_pct,
            "turnover": finite_or_none(snapshot.get("turnover")),
            "market_cap": market_cap, "market_cap_yi": round(market_cap / 100_000_000, 2),
        }
        record["score_fund"] = base_score(record)
        candidates.append(record)
    return sorted(candidates, key=lambda item: (to_float(item["score_fund"], 0.0), to_float(item["main_inflow"], 0.0)), reverse=True)

def parse_ymd(value: str) -> Optional[date]:
    try: return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except Exception: return None

def fetch_daily_history(code: str, cfg: Config, cache: JsonCache, as_of: date) -> list[dict[str, Any]]:
    start = as_of - timedelta(days=cfg.history_calendar_days)
    key = f"daily:{code}:{start.isoformat()}:{as_of.isoformat()}:v1"
    cached = cache.get(key)
    if isinstance(cached, list): return cached

    market_id = 1 if code.startswith("6") else 0
    payload = retry_call(
        lambda: http_json(DAILY_URL, {
            "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989", "klt": 101, "fqt": 0,
            "secid": f"{market_id}.{code}", "beg": start.strftime("%Y%m%d"), "end": as_of.strftime("%Y%m%d"),
        }, cfg.timeout_sec), f"获取 {code} 长周期日线", cfg)
    
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("klines"), list): raise DataSourceError("日线接口没有 kline 数据")

    rows: list[dict[str, Any]] = []
    for line in data["klines"]:
        values = str(line).split(",")
        if len(values) < 11: continue
        trading_day = parse_ymd(values[0])
        open_price = finite_or_none(values[1])
        close_price = finite_or_none(values[2])
        high = finite_or_none(values[3])
        low = finite_or_none(values[4])
        pct = finite_or_none(values[8])
        if trading_day is None or None in {open_price, close_price, high, low}: continue
        rows.append({"date": trading_day, "open": open_price, "close": close_price, "high": high, "low": low, "pct": pct})
    rows.sort(key=lambda item: item["date"])
    if not rows: raise DataSourceError("日线数据清洗后为空")
    cache.set(key, rows)
    return rows

def ema(values: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period: return result
    current = fmean(values[:period])
    result[period - 1] = current
    multiplier = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        current = (values[index] - current) * multiplier + current
        result[index] = current
    return result

def wilder_atr(rows: list[dict[str, Any]], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(rows)
    if period <= 0 or len(rows) < period: return result
    true_ranges: list[float] = []
    for index, row in enumerate(rows):
        high = float(row["high"]); low = float(row["low"])
        if index == 0: true_ranges.append(high - low)
        else:
            previous = float(rows[index - 1]["close"])
            true_ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
    current = fmean(true_ranges[:period])
    result[period - 1] = current
    for index in range(period, len(rows)):
        current = (current * (period - 1) + true_ranges[index]) / period
        result[index] = current
    return result

def bollinger(values: list[float], period: int, multiplier: float) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    middle: list[Optional[float]] = [None] * len(values)
    upper: list[Optional[float]] = [None] * len(values)
    lower: list[Optional[float]] = [None] * len(values)
    position: list[Optional[float]] = [None] * len(values)
    if period <= 1 or len(values) < period: return middle, upper, lower, position
    rolling_sum = sum(values[:period])
    rolling_sq = sum(value * value for value in values[:period])
    for index in range(period - 1, len(values)):
        if index > period - 1:
            outgoing = values[index - period]
            incoming = values[index]
            rolling_sum += incoming - outgoing
            rolling_sq += incoming * incoming - outgoing * outgoing
        average = rolling_sum / period
        variance = max(rolling_sq / period - average * average, 0.0)
        deviation = math.sqrt(variance)
        high = average + multiplier * deviation
        low = average - multiplier * deviation
        middle[index], upper[index], lower[index] = average, high, low
        if high > low: position[index] = (values[index] - low) / (high - low)
    return middle, upper, lower, position

def weekly_ohlc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        day = row["date"]
        friday = day + timedelta(days=4 - day.weekday())
        grouped[friday].append(row)
    weeks: list[dict[str, Any]] = []
    for friday in sorted(grouped):
        bars = sorted(grouped[friday], key=lambda item: item["date"])
        weeks.append({
            "date": friday, "open": float(bars[0]["open"]),
            "high": max(float(item["high"]) for item in bars),
            "low": min(float(item["low"]) for item in bars),
            "close": float(bars[-1]["close"]),
        })
    return weeks

def week_is_complete(friday: date, now: datetime) -> bool:
    local_now = now.astimezone(CN_TZ)
    if friday < local_now.date(): return True
    return friday == local_now.date() and local_now.weekday() == 4 and local_now.time() >= clock_time(15, 0)

def technical_features(code: str, rows: list[dict[str, Any]], cfg: Config, now: datetime) -> dict[str, Any]:
    required = max(cfg.bb_daily_period, cfg.kc_ema_period + cfg.kc_slope_lookback, cfg.bottom_lookback_days, cfg.limitup_lookback_days + 1, cfg.trend_ema_period + cfg.trend_slope_lookback)
    if len(rows) < required: return {"history_status": f"insufficient_daily_bars:{len(rows)}/{required}"}

    closes = [float(row["close"]) for row in rows]
    kc_middle = ema(closes, cfg.kc_ema_period)
    atr = wilder_atr(rows, cfg.kc_atr_period)
    last = len(rows) - 1
    kc_position: Optional[float] = None
    kc_signal: Optional[bool] = None
    slope_index = last - cfg.kc_slope_lookback
    if kc_middle[last] is not None and atr[last] is not None and kc_middle[slope_index] is not None:
        kc_upper = float(kc_middle[last]) + cfg.kc_multiplier * float(atr[last])
        kc_lower = float(kc_middle[last]) - cfg.kc_multiplier * float(atr[last])
        if kc_upper > kc_lower:
            kc_position = (closes[last] - kc_lower) / (kc_upper - kc_lower)
            kc_signal = bool(closes[last] >= float(kc_middle[last]) and float(kc_middle[last]) >= float(kc_middle[slope_index]) and kc_position <= cfg.kc_position_max)

    _, _, _, bb520_position_series = bollinger(closes, cfg.bb_daily_period, cfg.bb_daily_std_multiplier)
    bb520_position = bb520_position_series[last]
    bb520_signal = None if bb520_position is None else bool(bb520_position <= cfg.bb_daily_position_max)

    period_lows = [float(row["low"]) for row in rows[-cfg.bottom_lookback_days :]]
    low_price = min(period_lows)
    bottom_distance = (closes[last] / low_price - 1) * 100 if low_price > 0 else None
    bottom_signal = None if bottom_distance is None else bool(bottom_distance <= cfg.bottom_max_distance * 100)

    recent_pcts: list[float] = []
    for index in range(max(1, len(rows) - cfg.limitup_lookback_days), len(rows)):
        explicit_pct = finite_or_none(rows[index].get("pct"))
        calculated_pct = (closes[index] / closes[index - 1] - 1) * 100 if closes[index - 1] else 0.0
        recent_pcts.append(explicit_pct if explicit_pct is not None else calculated_pct)
    limit_count = sum(1 for pct in recent_pcts if pct >= limitup_threshold(code))
    trend_ema = ema(closes, cfg.trend_ema_period)
    trend_ref = last - cfg.trend_slope_lookback
    limitup_trend: Optional[bool] = None
    if trend_ema[last] is not None and trend_ema[trend_ref] is not None:
        limitup_trend = bool(limit_count >= cfg.limitup_min_count and closes[last] >= float(trend_ema[last]) and float(trend_ema[last]) >= float(trend_ema[trend_ref]))

    weekly = weekly_ohlc(rows)
    weekly_complete: Optional[bool] = None
    if weekly:
        weekly_complete = week_is_complete(weekly[-1]["date"], now)
        if cfg.only_closed_weeks and not weekly_complete: weekly = weekly[:-1]
    weekly_first_red: Optional[bool] = None
    bb120w_position: Optional[float] = None
    bb120w_signal: Optional[bool] = None
    if len(weekly) >= cfg.bb_weekly_period:
        current, previous = weekly[-1], weekly[-2]
        weekly_red = bool(current["close"] > current["open"])
        weekly_first_red = bool(weekly_red and previous["close"] <= previous["open"])
        weekly_closes = [float(item["close"]) for item in weekly]
        _, _, _, weekly_position_series = bollinger(weekly_closes, cfg.bb_weekly_period, cfg.bb_weekly_std_multiplier)
        bb120w_position = weekly_position_series[-1]
        bb120w_signal = None if bb120w_position is None else bool(bb120w_position <= cfg.bb_weekly_position_max)

    return {
        "history_status": "ok", "history_bars": len(rows),
        "kc_signal": kc_signal, "kc_position": kc_position,
        "bb520_signal": bb520_signal, "bb520_position": bb520_position,
        "weekly_first_red": weekly_first_red, "weekly_bar_complete": weekly_complete,
        "bb120w_signal": bb120w_signal, "bb120w_position": bb120w_position,
        "bottom_signal": bottom_signal, "bottom_distance_pct": bottom_distance,
        "limitup_trend_signal": limitup_trend, "limitup_count": limit_count,
    }

def enrich_one(record: dict[str, Any], cfg: Config, cache: JsonCache, now: datetime) -> dict[str, Any]:
    code = str(record["code"])
    try:
        rows = fetch_daily_history(code, cfg, cache, now.date())
        return {**record, **technical_features(code, rows, cfg, now)}
    except DataSourceError as exc: return {**record, "history_status": f"unavailable:{exc}"}
    except Exception as exc:
        LOGGER.exception("%s 技术计算异常：%s", code, exc)
        return {**record, "history_status": f"error:{type(exc).__name__}"}

def enrich_all(candidates: list[dict[str, Any]], cfg: Config, cache: JsonCache, now: datetime) -> tuple[list[dict[str, Any]], Counter[str]]:
    results: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    if not candidates:
        issues["no_base_candidates"] += 1
        return results, issues
    with ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="history") as executor:
        futures = [executor.submit(enrich_one, record, cfg, cache, now) for record in candidates]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = str(result.get("history_status", "unknown"))
            if status != "ok": issues[status.split(":", 1)[0]] += 1
            results.append(result)
            if completed % 20 == 0 or completed == len(futures):
                LOGGER.info("长周期日线与技术指标：%s/%s", completed, len(futures))
    return results, issues

def add_ranking(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        weekly_confirm = record.get("weekly_first_red") is True and record.get("bb120w_signal") is True
        signal_count = sum([
            record.get("kc_signal") is True, record.get("bb520_signal") is True,
            weekly_confirm, record.get("bottom_signal") is True, record.get("limitup_trend_signal") is True,
        ])
        technical_score = signal_count / 5 * 100
        record["weekly_confirmation"] = weekly_confirm
        record["signal_count"] = signal_count
        record["technical_score"] = round(technical_score, 2)
        record["trend_score"] = round(to_float(record.get("score_fund"), 0.0) * 0.5 + technical_score * 0.5, 2)
    return sorted(records, key=lambda item: (to_float(item.get("trend_score"), 0.0), int(item.get("signal_count") or 0), to_float(item.get("main_inflow"), 0.0)), reverse=True)

def send_serverchan(title: str, content: str, sendkey: str) -> bool:
    if not sendkey: return False
    LIMIT = 3800
    chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur: chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur: chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        try:
            data = urlencode({"title": t, "desp": ch}).encode("utf-8")
            req = Request(f"https://sctapi.ftqq.com/{sendkey}.send", data=data, method="POST")
            with urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                if res.get("code") != 0: ok = False
        except Exception as e:
            LOGGER.warning("推送失败: %s", e); ok = False
        if i < len(chunks) - 1: time.sleep(1)
    return ok

def write_reports(selected: list[dict[str, Any]], cfg: Config, counts: dict[str, int], issues: Counter[str], started: datetime) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.output_dir / f"bottom_limitup_trend_{stamp}.csv"
    json_path = cfg.output_dir / f"bottom_limitup_trend_{stamp}.json"
    markdown_path = cfg.output_dir / f"bottom_limitup_trend_{stamp}.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for record in selected:
            row = {field: record.get(field) for field in CSV_FIELDS}
            writer.writerow(json_safe(row))

    payload = {
        "generated_at": started.isoformat(timespec="seconds"), "version": VERSION,
        "config": json_safe(asdict(cfg)), "counts": counts, "data_quality": dict(issues), "selected": json_safe(selected),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)

    lines = [
        "# 🚀 A 股底部涨停趋势筛选报告", "", f"生成时间：{started.strftime('%Y-%m-%d %H:%M')}", "",
        "## 摘要", "",
        f"资金流基础条件通过 **{counts['base_filtered']}** 只；取其中前 **{counts['technical_candidates']}** 只计算长周期技术指标；最终满足至少 **{cfg.min_signals}/5** 项技术维度的有 **{len(selected)}** 只。", "",
        "## TOP 20", "",
    ]
    if not selected:
        lines.append("本次没有股票满足所选条件。可在研究场景下将 `--min-signals` 调整为 `4` 建立观察池。")
    else:
        headers = ["代码", "名称", "主力净流入", "涨跌幅", "信号数", "趋势评分", "KC", "BB520", "周线", "底部", "涨停"]
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        for record in selected[:20]:
            values = [
                record.get("code", ""), record.get("name", ""), format_money(record.get("main_inflow")),
                format_percent(record.get("spot_pct")), str(record.get("signal_count", "")),
                f"{to_float(record.get('trend_score'), 0.0):.1f}",
                "✅" if record.get("kc_signal") else "❌", "✅" if record.get("bb520_signal") else "❌",
                "✅" if record.get("weekly_confirmation") else "❌", "✅" if record.get("bottom_signal") else "❌",
                "✅" if record.get("limitup_trend_signal") else "❌",
            ]
            lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend(["", "*本报告仅用于研究与数据分析，不构成个性化投资建议。*", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="零依赖单文件：资金流 + 肯纳特 + 520日布林 + 120周首红 + 底部涨停趋势筛选器")
    basic = parser.add_argument_group("资金流基础条件")
    basic.add_argument("--min-main", type=float, default=float(os.environ.get('MIN_MAIN', 1_000_000)))
    basic.add_argument("--min-super", type=float, default=float(os.environ.get('MIN_SUPER', 0)))
    basic.add_argument("--min-cap", type=float, default=float(os.environ.get('MIN_CAP', 10)))
    basic.add_argument("--max-cap", type=float, default=float(os.environ.get('MAX_CAP', 5000)))
    basic.add_argument("--candidate-cap", type=int, default=int(os.environ.get('CANDIDATE_CAP', 150)))
    basic.add_argument("--top", type=int, default=int(os.environ.get('TOP_N', 30)))
    basic.add_argument("--include-bj", action="store_true")
    basic.add_argument("--include-st", action="store_true")

    technical = parser.add_argument_group("技术条件")
    technical.add_argument("--min-signals", type=int, default=int(os.environ.get('MIN_SIGNALS', 5)))
    technical.add_argument("--kc-ema", type=int, default=20)
    technical.add_argument("--kc-atr", type=int, default=10)
    technical.add_argument("--kc-multiplier", type=float, default=2.0)
    technical.add_argument("--kc-position-max", type=float, default=1.05)
    technical.add_argument("--bb520-position-max", type=float, default=0.35)
    technical.add_argument("--bb120w-position-max", type=float, default=0.60)
    technical.add_argument("--bottom-days", type=int, default=120)
    technical.add_argument("--bottom-max-distance", type=float, default=0.35)
    technical.add_argument("--limitup-days", type=int, default=60)
    technical.add_argument("--limitup-min-count", type=int, default=1)
    technical.add_argument("--include-current-week", action="store_true")

    runtime = parser.add_argument_group("运行控制")
    runtime.add_argument("--workers", type=int, default=4)
    runtime.add_argument("--retries", type=int, default=3)
    runtime.add_argument("--timeout", type=float, default=20)
    runtime.add_argument("--history-calendar-days", type=int, default=1000)
    runtime.add_argument("--cache-dir", type=Path, default=Path("cache"))
    runtime.add_argument("--cache-ttl", type=int, default=900)
    runtime.add_argument("--force-refresh", action="store_true")
    runtime.add_argument("--output-dir", type=Path, default=Path("output"))
    runtime.add_argument("--log-level", default="INFO")
    runtime.add_argument("--self-test", action="store_true")
    return parser.parse_args()

def build_config(args: argparse.Namespace) -> Config:
    return Config(
        min_main_inflow=args.min_main, min_super_inflow=args.min_super,
        min_market_cap=args.min_cap * 100_000_000, max_market_cap=args.max_cap * 100_000_000,
        top_n=args.top, candidate_cap=args.candidate_cap, min_signals=args.min_signals,
        include_bj=args.include_bj, include_st=args.include_st,
        workers=args.workers, retries=args.retries, timeout_sec=args.timeout,
        cache_dir=args.cache_dir, output_dir=args.output_dir, cache_ttl_sec=args.cache_ttl,
        force_refresh=args.force_refresh, history_calendar_days=args.history_calendar_days,
        kc_ema_period=args.kc_ema, kc_atr_period=args.kc_atr, kc_multiplier=args.kc_multiplier,
        kc_position_max=args.kc_position_max, bb_daily_position_max=args.bb520_position_max,
        bb_weekly_position_max=args.bb120w_position_max, only_closed_weeks=not args.include_current_week,
        bottom_lookback_days=args.bottom_days, bottom_max_distance=args.bottom_max_distance,
        limitup_lookback_days=args.limitup_days, limitup_min_count=args.limitup_min_count,
    )

def run_self_test() -> int:
    assert clean_code("600519.0") == "600519"
    LOGGER.info("离线自检通过。")
    return 0

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    if args.self_test: return run_self_test()
    
    cfg = build_config(args)
    started = cn_now()
    cache = JsonCache(cfg.cache_dir, cfg.cache_ttl_sec, cfg.force_refresh)
    
    LOGGER.info("开始：资金流 + 肯纳特 + 520日布林 + 120周首红 + 底部涨停趋势筛选")
    spot = get_spot(cfg, cache)
    funds = get_fund_rank(cfg, cache)
    base_filtered = build_candidates(spot, funds, cfg)
    technical_candidates = base_filtered[: cfg.candidate_cap]
    LOGGER.info("行情 %s 只；资金流 %s 只；基础条件通过 %s 只；技术计算 %s 只", len(spot), len(funds), len(base_filtered), len(technical_candidates))
    
    enriched, issues = enrich_all(technical_candidates, cfg, cache, started)
    ranked = add_ranking(enriched)
    selected = [record for record in ranked if int(record.get("signal_count") or 0) >= cfg.min_signals][: cfg.top_n]

    counts = {
        "spot_rows": len(spot), "fund_rank_rows": len(funds), "base_filtered": len(base_filtered),
        "technical_candidates": len(technical_candidates),
        "technical_available": sum(1 for record in enriched if record.get("history_status") == "ok"),
        "selected": len(selected),
    }
    paths = write_reports(selected, cfg, counts, issues, started)
    
    print("\n" + "=" * 76)
    print(f"资金流基础通过：{len(base_filtered)} | 技术计算：{len(technical_candidates)} | 满足 {cfg.min_signals}/5：{len(selected)}")
    if selected:
        for record in selected[:10]:
            print(f"{record['code']} {record['name'][:6]:<6} 主力:{format_money(record.get('main_inflow')):>8} 信号:{record['signal_count']} 评分:{record['trend_score']:.1f}")
    for name, path in paths.items(): print(f"{name.upper():<9}: {path}")

    sendkey = os.environ.get("SERVERCHAN_KEY") or os.environ.get("SENDKEY", "")
    if sendkey and selected:
        push_title = f"🚀 底部涨停趋势 命中{len(selected)}只 (满足{cfg.min_signals}/5维)"
        md_content = paths["markdown"].read_text(encoding="utf-8")
        send_serverchan(push_title, md_content, sendkey)
        LOGGER.info("微信推送完成")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
