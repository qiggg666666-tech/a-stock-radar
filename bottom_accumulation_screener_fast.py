#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""底部坑底/蓄势全市场筛选器（快速容错四分片版）。

设计原则：
1. 保留旧版的四档信号语义：坑底止跌、坑底探底、低位横盘、低位止跌；
2. AkShare 为主，BaoStock 仅在单标的 AkShare 失败后回退；
3. 每只股票在独立子进程中完成，因此父进程可在超时后真正终止卡住的调用；
4. 父进程最多并行 NUM_WORKERS 个子进程；每 25 只保存可上传的 checkpoint；
5. 到达 MAX_RUNTIME_SECONDS 或 MAX_FAILURES 时安全结束并保留部分结果。

GitHub Actions 推荐四片：
  A: SCAN_OFFSET=0    SCAN_LIMIT=1500
  B: SCAN_OFFSET=1500 SCAN_LIMIT=1500
  C: SCAN_OFFSET=3000 SCAN_LIMIT=1500
  D: SCAN_OFFSET=4500 SCAN_LIMIT=1500
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd
import requests


LOGGER = logging.getLogger("bottom_accumulation_fast")


@dataclass(frozen=True)
class Config:
    # 策略口径：与旧版四档语义保持一致。
    bottom_zone_pct: float = 0.20
    drawdown_min: float = 0.30
    low_lookback: int = 250
    pit_pct: float = 0.05
    rsi_low: float = 35.0
    ma_spread_max: float = 0.02
    squeeze_lookback: int = 10
    squeeze_max: float = 0.06
    bb_narrow_max: float = 0.12
    strict_non_pit: bool = True
    require_pit_macd: bool = False
    require_duck: bool = True
    duck_score_min: int = 4
    turnover_min: float = 0.30
    min_data_len: int = 450
    history_calendar_days: int = 1100
    min_price: float = 3.0
    pre_amount_min: float = 30_000_000.0
    pre_turnover_min: float = 0.20

    # 运行保护：所有值均可由 workflow 环境变量覆盖。
    scan_offset: int = 0
    scan_limit: int = 1500
    shard_name: str = "a"
    num_workers: int = 2
    ak_timeout_seconds: int = 12
    bao_timeout_seconds: int = 8
    per_symbol_timeout_seconds: int = 36
    max_runtime_seconds: int = 19_200
    max_failures: int = 200
    checkpoint_every: int = 25
    resume: bool = True
    output_dir: Path = Path("output")
    enable_notify: bool = False
    serverchan_key: str = ""


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    return Config(
        bottom_zone_pct=env_float("BOTTOM_ZONE_PCT", 0.20),
        drawdown_min=env_float("DRAWDOWN_MIN", 0.30),
        low_lookback=env_int("LOW_LOOKBACK", 250),
        pit_pct=env_float("PIT_PCT", 0.05),
        rsi_low=env_float("RSI_LOW", 35.0),
        ma_spread_max=env_float("MA_SPREAD_MAX", 0.02),
        squeeze_lookback=env_int("SQUEEZE_LOOKBACK", 10),
        squeeze_max=env_float("SQUEEZE_MAX", 0.06),
        bb_narrow_max=env_float("BB_NARROW_MAX", 0.12),
        strict_non_pit=env_bool("STRICT_NON_PIT", True),
        require_pit_macd=env_bool("REQUIRE_PIT_MACD", False),
        require_duck=env_bool("REQUIRE_DUCK", True),
        duck_score_min=env_int("DUCK_SCORE_MIN", 4),
        turnover_min=env_float("TURNOVER_MIN", 0.30),
        min_data_len=env_int("MIN_DATA_LEN", 450),
        history_calendar_days=env_int("HISTORY_CALENDAR_DAYS", 1100),
        min_price=env_float("MIN_PRICE", 3.0),
        pre_amount_min=env_float("PRE_AMOUNT_MIN", 30_000_000.0),
        pre_turnover_min=env_float("PRE_TURNOVER_MIN", 0.20),
        scan_offset=max(0, env_int("SCAN_OFFSET", 0)),
        scan_limit=max(0, env_int("SCAN_LIMIT", 1500)),
        shard_name=os.getenv("SCAN_SHARD", "a").strip().lower() or "a",
        num_workers=max(1, min(4, env_int("NUM_WORKERS", 2))),
        ak_timeout_seconds=max(1, env_int("AK_TIMEOUT", 12)),
        bao_timeout_seconds=max(1, env_int("BAO_TIMEOUT", 8)),
        per_symbol_timeout_seconds=max(5, env_int("PER_SYMBOL_TIMEOUT", 36)),
        max_runtime_seconds=max(60, env_int("MAX_RUNTIME_SECONDS", 19_200)),
        max_failures=max(1, env_int("MAX_FAILURES", 200)),
        checkpoint_every=max(1, env_int("CHECKPOINT_EVERY", 25)),
        resume=env_bool("RESUME", True),
        output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        enable_notify=env_bool("ENABLE_NOTIFY", False),
        serverchan_key=os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY", ""),
    )


STAGE_ORDER = {"🔴坑底止跌": 0, "⚠️坑底探底": 1, "🟡低位横盘": 2, "🟠低位止跌": 3}
RESULT_COLUMNS = [
    "代码", "名称", "最新价", "信号日期", "阶段", "距低点%", "距高点回撤%",
    "均线极差%", "近10日振幅%", "RSI", "换手%", "MACD状态", "布林", "鸭头分",
    "score", "数据源",
]


class DataSourceError(RuntimeError):
    """单标的数据源失败；父进程会把它记入跳过统计。"""


def normalize_frame(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    columns = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "换手率": "turn"}
    frame = raw.rename(columns=columns).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataSourceError(f"缺少字段:{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "turn"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "turn" not in frame.columns:
        frame["turn"] = np.nan
    frame = frame.dropna(subset=required)
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0) & (frame["volume"] > 0)]
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if len(frame) < cfg.min_data_len:
        raise DataSourceError(f"历史不足:{len(frame)}/{cfg.min_data_len}")
    return frame[["date", "open", "high", "low", "close", "volume", "turn"]]


def fetch_ak_history(symbol: str, cfg: Config) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=cfg.history_calendar_days)
    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if raw is None or raw.empty:
        raise DataSourceError("AkShare返回空表")
    return normalize_frame(raw, cfg)


def fetch_bao_history(symbol: str, cfg: Config) -> pd.DataFrame:
    exchange = "sh" if symbol.startswith(("6", "9")) else "sz"
    end = date.today()
    start = end - timedelta(days=cfg.history_calendar_days)
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        raise DataSourceError(f"BaoStock登录失败:{getattr(login, 'error_msg', '')}")
    try:
        response = bs.query_history_k_data_plus(
            f"{exchange}.{symbol}",
            "date,open,high,low,close,volume,turn,tradestatus,isST",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag="2",
        )
        if getattr(response, "error_code", "1") != "0":
            raise DataSourceError(f"BaoStock日线失败:{getattr(response, 'error_msg', '')}")
        rows: list[list[str]] = []
        while response.next():
            rows.append(response.get_row_data())
        raw = pd.DataFrame(rows, columns=response.fields)
        if raw.empty:
            raise DataSourceError("BaoStock返回空表")
        if "tradestatus" in raw.columns:
            raw = raw[raw["tradestatus"].astype(str) == "1"]
        if "isST" in raw.columns:
            raw = raw[raw["isST"].astype(str) != "1"]
        return normalize_frame(raw, cfg)
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def source_fetch_worker(result_queue: Any, source: str, symbol: str, cfg_dict: dict[str, Any]) -> None:
    """数据源最小隔离单元：结果通过队列返回，超时由调用方直接终止该进程。"""
    cfg = Config(**{**cfg_dict, "output_dir": Path(cfg_dict["output_dir"])})
    try:
        frame = fetch_ak_history(symbol, cfg) if source == "akshare" else fetch_bao_history(symbol, cfg)
        result_queue.put({"ok": True, "frame": frame})
    except Exception as error:
        result_queue.put({"ok": False, "reason": f"{type(error).__name__}:{str(error)[:160]}"})


def fetch_source_hard(source: str, symbol: str, cfg: Config) -> pd.DataFrame:
    """真正墙钟超时，不依赖线程池取消语义。"""
    timeout = cfg.ak_timeout_seconds if source == "akshare" else cfg.bao_timeout_seconds
    ctx = context()
    result_queue = ctx.Queue(maxsize=1)
    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    process = ctx.Process(target=source_fetch_worker, args=(result_queue, source, symbol, cfg_dict))
    process.start()
    try:
        # 必须先消费队列：日线DataFrame可能大于管道缓冲，先join会阻塞子进程退出。
        payload = result_queue.get(timeout=timeout)
    except queue.Empty as error:
        process.terminate()
        process.join(5)
        result_queue.close()
        raise DataSourceError(f"{source}超过{timeout}秒或子进程无返回，已强制终止") from error
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    result_queue.close()
    if not payload.get("ok"):
        raise DataSourceError(str(payload.get("reason", f"{source}未知错误")))
    return payload["frame"]


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def duck_score(frame: pd.DataFrame) -> int:
    """保留旧版周线老鸭头的核心评分，但不作额外网络调用。"""
    weekly = frame.set_index("date").resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(weekly) < 80:
        return 0
    weekly["ma5"] = weekly["close"].rolling(5).mean()
    weekly["ma10"] = weekly["close"].rolling(10).mean()
    weekly["ma60"] = weekly["close"].rolling(60).mean()
    if pd.isna(weekly["ma60"].iloc[-1]) or pd.isna(weekly["ma60"].iloc[-20]):
        return 0
    score = 0
    if (weekly["ma60"].iloc[-1] - weekly["ma60"].iloc[-20]) / 20 > 0:
        score += 1
    cross5 = (weekly["ma5"].shift(1) < weekly["ma60"].shift(1)) & (weekly["ma5"] > weekly["ma60"])
    cross10 = (weekly["ma10"].shift(1) < weekly["ma60"].shift(1)) & (weekly["ma10"] > weekly["ma60"])
    if (cross5 | cross10).tail(30).any():
        score += 1
    recent = weekly.tail(25)
    peak, trough = float(recent["high"].max()), float(recent["low"].min())
    if peak > 0 and 0.08 <= (peak - trough) / peak <= 0.35:
        score += 1
    if weekly["ma5"].iloc[-1] > weekly["ma10"].iloc[-1] and weekly["ma10"].iloc[-1] > weekly["ma60"].iloc[-1] * 0.98:
        score += 1
    dif, dea, _ = macd(weekly["close"])
    golden = (dif.shift(1) < dea.shift(1)) & (dif > dea)
    if golden.tail(6).any():
        score += 2
    elif dif.iloc[-1] > dea.iloc[-1]:
        score += 1
    return score


def evaluate(symbol: str, name: str, frame: pd.DataFrame, source: str, cfg: Config) -> dict[str, Any] | None:
    """保留旧版四档判定的价格、均线、MACD、布林和周线过滤口径。"""
    close, high, low, open_, volume = (frame[column].astype(float) for column in ("close", "high", "low", "open", "volume"))
    last = len(frame) - 1
    ma5, ma10, ma20 = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean()
    dif, dea, histogram = macd(close)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rsi = 100 - 100 / (1 + gain / loss)
    required = (ma20.iloc[last], dif.iloc[last], dea.iloc[last], histogram.iloc[last], rsi.iloc[last])
    if any(pd.isna(value) for value in required):
        return None

    low_ref = float(low.rolling(cfg.low_lookback, min_periods=120).min().iloc[last])
    high_ref = float(high.rolling(cfg.low_lookback, min_periods=120).max().iloc[last])
    price = float(close.iloc[last])
    if low_ref <= 0 or high_ref <= 0 or price <= 0:
        return None
    above_low = (price - low_ref) / low_ref
    drawdown = (high_ref - price) / high_ref
    if not (above_low <= cfg.bottom_zone_pct and drawdown >= cfg.drawdown_min):
        return None

    open_price, high_price, low_price = float(open_.iloc[last]), float(high.iloc[last]), float(low.iloc[last])
    body, amplitude = abs(price - open_price), high_price - low_price
    lower_shadow = min(open_price, price) - low_price
    yang = price > open_price
    long_lower = amplitude > 0 and lower_shadow > max(body * 1.2, amplitude * 0.5)
    rsi_low = float(rsi.iloc[last]) < cfg.rsi_low
    no_new_low = bool(low.iloc[-5:].min() >= low.iloc[-15:].min() * 0.99)
    macd_turn = bool(histogram.iloc[last] > histogram.iloc[last - 1])
    macd_golden = bool(dif.iloc[last] > dea.iloc[last] and dif.iloc[last - 1] <= dea.iloc[last - 1])
    macd_strong = macd_turn or macd_golden
    above_ma5 = price > float(ma5.iloc[last])
    stop = yang or long_lower or rsi_low or no_new_low or macd_turn or above_ma5
    ma_values = [float(ma5.iloc[last]), float(ma10.iloc[last]), float(ma20.iloc[last])]
    ma_spread = (max(ma_values) - min(ma_values)) / price
    squeeze = (float(high.iloc[-cfg.squeeze_lookback:].max()) - float(low.iloc[-cfg.squeeze_lookback:].min())) / price
    sideways = ma_spread < cfg.ma_spread_max and squeeze < cfg.squeeze_max
    bb_narrow = bool(pd.notna(bb_width.iloc[last]) and float(bb_width.iloc[last]) < cfg.bb_narrow_max)
    bb_low = bool(pd.notna(bb_position.iloc[last]) and float(bb_position.iloc[last]) < 0.30)
    is_pit = above_low <= cfg.pit_pct
    duck: int | None = None

    if is_pit:
        if stop:
            if cfg.require_pit_macd and not macd_strong:
                return None
            if cfg.require_duck:
                duck = duck_score(frame)
                if duck < cfg.duck_score_min:
                    return None
            stage = "🔴坑底止跌"
        else:
            stage = "⚠️坑底探底"
    elif sideways:
        if cfg.strict_non_pit and not (bb_narrow and macd_strong):
            return None
        stage = "🟡低位横盘"
    elif stop:
        if cfg.strict_non_pit and not macd_strong:
            return None
        stage = "🟠低位止跌"
    else:
        return None

    turn_value = pd.to_numeric(frame["turn"], errors="coerce").iloc[last]
    if cfg.turnover_min > 0 and (pd.isna(turn_value) or float(turn_value) < cfg.turnover_min):
        return None
    score = round(
        (1 - min(above_low, cfg.bottom_zone_pct) / cfg.bottom_zone_pct) * 40
        + (25 if is_pit else 0) + (15 if stop else 0) + (10 if sideways else 0)
        + (10 if macd_strong else 0) + (10 if bb_narrow else 0) + (10 if ma_spread < cfg.ma_spread_max else 0)
        + ((duck or 0) * 2),
        1,
    )
    return {
        "代码": symbol, "名称": name, "最新价": round(price, 2),
        "信号日期": frame["date"].iloc[last].strftime("%Y-%m-%d"), "阶段": stage,
        "距低点%": round(above_low * 100, 1), "距高点回撤%": round(drawdown * 100, 1),
        "均线极差%": round(ma_spread * 100, 2), "近10日振幅%": round(squeeze * 100, 2),
        "RSI": round(float(rsi.iloc[last]), 1), "换手%": round(float(turn_value), 2),
        "MACD状态": "金叉" if macd_golden else ("转强" if macd_strong else "未转"),
        "布林": "收窄" if bb_narrow else ("下轨" if bb_low else "—"),
        "鸭头分": duck, "score": score, "数据源": source,
    }


def child_evaluate(result_queue: Any, symbol: str, name: str, cfg_dict: dict[str, Any]) -> None:
    """每只股票独立子进程。父进程可直接 terminate 它，确保超时真实生效。"""
    cfg = Config(**{**cfg_dict, "output_dir": Path(cfg_dict["output_dir"])})
    try:
        try:
            frame = fetch_source_hard("akshare", symbol, cfg)
            source = "akshare"
        except Exception as ak_error:
            try:
                frame = fetch_source_hard("baostock", symbol, cfg)
                source = "baostock"
            except Exception as bao_error:
                result_queue.put({"kind": "source_error", "symbol": symbol, "reason": f"AkShare:{str(ak_error)[:120]} | BaoStock:{str(bao_error)[:120]}"})
                return
        result = evaluate(symbol, name, frame, source, cfg)
        result_queue.put({"kind": "candidate" if result else "no_signal", "symbol": symbol, "result": result})
    except Exception as error:
        result_queue.put({"kind": "logic_error", "symbol": symbol, "reason": f"{type(error).__name__}:{str(error)[:160]}"})


def child_universe(result_queue: Any, source: str, cfg_dict: dict[str, Any]) -> None:
    """仅取代码名单，避免全市场实时快照在交易时段频繁超时。"""
    _ = Config(**{**cfg_dict, "output_dir": Path(cfg_dict["output_dir"])})
    try:
        if source == "akshare":
            data = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
        else:
            login = bs.login()
            if getattr(login, "error_code", "1") != "0":
                raise DataSourceError(f"BaoStock股票池登录失败:{getattr(login, 'error_msg', '')}")
            try:
                raw = bs.query_stock_basic().get_data()
            finally:
                try:
                    bs.logout()
                except Exception:
                    pass
            if raw is None or raw.empty:
                raise DataSourceError("BaoStock股票池为空")
            data = raw.rename(columns={"code": "代码", "code_name": "名称"})
        if data is None or data.empty or not {"代码", "名称"}.issubset(data.columns):
            result_queue.put({"kind": "error", "reason": f"{source}股票池为空或字段不完整"})
            return
        data = data[["代码", "名称"]].copy()
        data["代码"] = data["代码"].astype(str).str.extract(r"(\d{6})", expand=False)
        data["名称"] = data["名称"].astype(str)
        selected = data[
            data["代码"].notna()
            & data["代码"].str.startswith(("0", "3", "6"))
            & ~data["名称"].str.contains("ST|退", regex=True, na=False)
        ].drop_duplicates("代码").sort_values("代码")
        if selected.empty:
            result_queue.put({"kind": "error", "reason": f"{source}清洗后无可用A股代码"})
            return
        result_queue.put({"kind": "ok", "universe": selected.to_dict(orient="records")})
    except Exception as error:
        result_queue.put({"kind": "error", "reason": f"{type(error).__name__}:{str(error)[:160]}"})


def context() -> Any:
    # Actions Ubuntu 下 fork 启动快且父进程可可靠终止单标的子进程。
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def isolated_universe(cfg: Config) -> list[dict[str, str]]:
    ctx = context()
    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    reasons: list[str] = []
    for source, timeout in (("akshare", cfg.ak_timeout_seconds), ("baostock", cfg.bao_timeout_seconds)):
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=child_universe, args=(result_queue, source, cfg_dict))
        process.start()
        try:
            payload = result_queue.get(timeout=timeout)
        except queue.Empty:
            if process.is_alive():
                process.terminate()
            process.join(5)
            result_queue.close()
            reasons.append(f"{source}股票池超过{timeout}秒")
            continue
        process.join(5)
        if process.is_alive():
            process.terminate(); process.join(5)
        result_queue.close()
        if payload.get("kind") == "ok":
            return payload["universe"]
        reasons.append(f"{source}:{payload.get('reason', '未知错误')}")
    raise DataSourceError("；".join(reasons))


def checkpoint_paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    prefix = f"bottom_accumulation_fast_{cfg.shard_name}"
    return (
        cfg.output_dir / f"{prefix}.checkpoint.json",
        cfg.output_dir / f"{prefix}.csv",
        cfg.output_dir / f"{prefix}.json",
        cfg.output_dir / f"{prefix}.md",
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """不使用DataFrame.to_markdown，避免新增tabulate依赖。"""
    if frame.empty:
        return "无候选。"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_checkpoint(path: Path, cfg: Config) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    if not cfg.resume or not path.exists():
        return set(), [], {"source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        processed = set(str(item) for item in payload.get("processed", []))
        results = list(payload.get("results", []))
        stats = {str(key): int(value) for key, value in payload.get("stats", {}).items()}
        return processed, results, {**{"source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}, **stats}
    except Exception as error:
        LOGGER.warning("checkpoint读取失败，将从头开始：%s", error)
        return set(), [], {"source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}


def save_outputs(
    cfg: Config,
    universe_size: int,
    processed: set[str],
    results: list[dict[str, Any]],
    stats: dict[str, int],
    reason: str,
    started_at: float,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, csv_path, json_path, md_path = checkpoint_paths(cfg)
    elapsed = round(time.monotonic() - started_at, 1)
    ordered = sorted(results, key=lambda row: (STAGE_ORDER.get(row.get("阶段", ""), 9), -float(row.get("score", 0))))
    frame = pd.DataFrame(ordered, columns=RESULT_COLUMNS)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = {
        "version": "fast-v1", "shard": cfg.shard_name, "scan_offset": cfg.scan_offset,
        "scan_limit": cfg.scan_limit, "universe_size": universe_size, "processed": len(processed),
        "candidates": len(ordered), "stats": stats, "stop_reason": reason,
        "elapsed_seconds": elapsed, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "results": ordered,
    }
    atomic_json(json_path, metadata)
    atomic_json(checkpoint_path, {"processed": sorted(processed), "results": ordered, "stats": stats, "metadata": metadata})
    markdown = [
        f"# 底部吸筹快速扫描（分片 {cfg.shard_name.upper()}）", "",
        f"- 股票池：{universe_size} 只", f"- 已处理：{len(processed)} 只", f"- 候选：{len(ordered)} 只",
        f"- 停止原因：`{reason}`", f"- 耗时：{elapsed:.1f} 秒", f"- 异常统计：`{json.dumps(stats, ensure_ascii=False)}`", "",
    ]
    if frame.empty:
        markdown.append("本分片暂无符合四档底部条件的标的，或在数据源保护机制下安全结束。")
    else:
        markdown.extend(["## 候选", "", markdown_table(frame)])
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def notify(cfg: Config, results: list[dict[str, Any]], reason: str) -> None:
    if not cfg.enable_notify or not cfg.serverchan_key:
        return
    top = sorted(results, key=lambda row: (STAGE_ORDER.get(row.get("阶段", ""), 9), -float(row.get("score", 0))))[:20]
    lines = [f"- {row['阶段']} {row['名称']}({row['代码']})｜分{row['score']}｜MACD{row['MACD状态']}" for row in top]
    body = "\n".join(lines) if lines else "本分片无候选。"
    try:
        response = requests.post(
            f"https://sctapi.ftqq.com/{cfg.serverchan_key}.send",
            data={"title": f"底部吸筹快速版[{cfg.shard_name.upper()}]：{len(results)}只", "desp": f"停止原因：{reason}\n\n{body}"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        LOGGER.warning("通知失败，不影响筛选结果：%s", error)


def run_scan(cfg: Config) -> int:
    started_at = time.monotonic()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, _, _, _ = checkpoint_paths(cfg)
    try:
        full_universe = isolated_universe(cfg)
    except DataSourceError as error:
        LOGGER.error("股票池获取失败：%s", error)
        save_outputs(cfg, 0, set(), [], {"source_error": 1, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}, "universe_unavailable", started_at)
        return 0

    shard = full_universe[cfg.scan_offset: cfg.scan_offset + cfg.scan_limit] if cfg.scan_limit else full_universe[cfg.scan_offset:]
    processed, results, stats = load_checkpoint(checkpoint_path, cfg)
    tasks = [item for item in shard if item["代码"] not in processed]
    LOGGER.info("分片%s：预筛后全池%s只，本片%s只，恢复后待处理%s只，worker=%s", cfg.shard_name.upper(), len(full_universe), len(shard), len(tasks), cfg.num_workers)
    if not tasks:
        save_outputs(cfg, len(shard), processed, results, stats, "already_complete", started_at)
        notify(cfg, results, "already_complete")
        return 0

    ctx = context()
    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    active: dict[str, tuple[Any, Any, float]] = {}
    task_index = 0
    last_checkpoint_count = len(processed)
    stop_reason = "completed"

    def start_one(stock: dict[str, str]) -> None:
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=child_evaluate, args=(result_queue, stock["代码"], stock["名称"], cfg_dict))
        process.start()
        active[stock["代码"]] = (process, result_queue, time.monotonic())

    while task_index < len(tasks) or active:
        elapsed = time.monotonic() - started_at
        if elapsed >= cfg.max_runtime_seconds and stop_reason == "completed":
            stop_reason = "runtime_budget_reached"
        if stats["source_error"] >= cfg.max_failures and stop_reason == "completed":
            stop_reason = "circuit_breaker_open"
        while stop_reason == "completed" and len(active) < cfg.num_workers and task_index < len(tasks):
            start_one(tasks[task_index]); task_index += 1

        for symbol, (process, result_queue, launched_at) in list(active.items()):
            payload: dict[str, Any] | None = None
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                pass
            if payload is not None:
                process.join(2)
                result_queue.close()
                active.pop(symbol, None)
                processed.add(symbol)
                kind = payload.get("kind", "worker_crash")
                if kind == "candidate" and payload.get("result"):
                    results.append(payload["result"])
                elif kind in stats:
                    stats[kind] += 1
                else:
                    stats["worker_crash"] += 1
            elif time.monotonic() - launched_at >= cfg.per_symbol_timeout_seconds:
                if process.is_alive():
                    process.terminate()
                process.join(5)
                result_queue.close()
                active.pop(symbol, None)
                processed.add(symbol)
                stats["timeout"] += 1
                LOGGER.warning("%s 超过%s秒，已由父进程强制终止", symbol, cfg.per_symbol_timeout_seconds)
            elif process.exitcode is not None:
                result_queue.close()
                active.pop(symbol, None)
                processed.add(symbol)
                stats["worker_crash"] += 1

        if len(processed) - last_checkpoint_count >= cfg.checkpoint_every:
            save_outputs(cfg, len(shard), processed, results, stats, "in_progress" if stop_reason == "completed" else stop_reason, started_at)
            last_checkpoint_count = len(processed)
            LOGGER.info("分片%s进度：%s/%s，候选%s，统计%s", cfg.shard_name.upper(), len(processed), len(shard), len(results), stats)
        if stop_reason != "completed":
            for process, result_queue, _ in active.values():
                if process.is_alive():
                    process.terminate()
                process.join(5)
                result_queue.close()
            active.clear()
            break
        time.sleep(0.10)

    save_outputs(cfg, len(shard), processed, results, stats, stop_reason, started_at)
    notify(cfg, results, stop_reason)
    LOGGER.info("分片%s结束：原因=%s，已处理%s/%s，候选%s，统计=%s", cfg.shard_name.upper(), stop_reason, len(processed), len(shard), len(results), stats)
    return 0


def self_test() -> int:
    cfg = Config()
    assert Config(scan_offset=500, scan_limit=500).scan_offset == 500
    assert STAGE_ORDER["🔴坑底止跌"] < STAGE_ORDER["🟠低位止跌"]
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    base = np.linspace(20, 12, 500)
    frame = pd.DataFrame({"date": dates, "open": base, "high": base * 1.01, "low": base * 0.99, "close": base, "volume": 1_000_000, "turn": 1.0})
    assert evaluate("600000", "测试", frame, "test", cfg) is None or isinstance(evaluate("600000", "测试", frame, "test", cfg), dict)
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    if "--self-test" in sys.argv:
        return self_test()
    return run_scan(load_config())


if __name__ == "__main__":
    raise SystemExit(main())
