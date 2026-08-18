#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股资本流研究筛选器：价格结构、量价资金代理与公开资金流快照的可解释组合。

本脚本不读取券商账户、席位或所谓“主力”真实持仓，不能据此断言主力行为。它将：
1) 同类“吃掉主力”价格结构（中线趋势、操纵/趋势上穿、相对强弱）；
2) 日线可复算的量价资金代理（CLV × 成交额）；
3) AkShare公开的当日主力净流入排行快照（仅作为实盘时点确认）；
整合为可解释研究评分。公式经用户截图验证仅为近似实现，不是同花顺加密指标复刻。

安全与研究口径：
* AkShare主路径、BaoStock游标回退；每一次外部请求均使用可终止子进程硬超时。
* 全市场先以公开资金流/市值快照预筛，再按分片逐股下载日线，避免无边界串行扫描。
* 数据不足、失败比例过高、资金流字段缺失均写入状态JSON；不吞掉异常。
* 实盘评分可使用“当日主力净流入”快照；回测严格只使用日线可复算的价格/量价代理，
  信号日后下一交易日开盘入场，因此不会把今天的资金流快照泄漏到过去。

示例：
  python capital_flow_research_screener.py --self-test
  python capital_flow_research_screener.py --code 300650 --backtest --hold-days 20
  python capital_flow_research_screener.py --max-circ-mv 80 --preselect-top 80 --shard-index 0 --shard-count 5
  python capital_flow_research_screener.py --max-circ-mv 200 --require-live-fund-flow --push
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger("capital-flow-research")
EPS = 1e-12
CN_TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class Params:
    control_ema: int = 13
    trend_ema: int = 3
    trend_weight: float = 0.618
    cooldown_days: int = 5
    min_history_days: int = 300
    min_score: int = 60


@dataclass(frozen=True)
class FetchConfig:
    history_days: int
    ak_timeout: int
    bao_timeout: int
    index_code: str


class DataError(RuntimeError):
    """外部数据、字段或数据质量不满足研究脚本要求。"""


def context() -> Any:
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def clean_code(value: object) -> str:
    hit = re.search(r"(\d{6})", str(value))
    return hit.group(1) if hit else ""


def ema(values: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("EMA周期必须为正数")
    return pd.to_numeric(values, errors="coerce").astype(float).ewm(span=period, adjust=False).mean()


def chinese_sma(values: pd.Series, period: int, weight: int = 1) -> pd.Series:
    if period < 1 or not 0 < weight <= period:
        raise ValueError("中国式SMA参数非法")
    return pd.to_numeric(values, errors="coerce").astype(float).ewm(alpha=weight / period, adjust=False).mean()


def cooldown(events: pd.Series, days: int) -> pd.Series:
    accepted = np.zeros(len(events), dtype=int)
    last = -days - 1
    for pos, value in enumerate(events.fillna(False).astype(bool).to_numpy()):
        if value and pos - last > days:
            accepted[pos] = 1
            last = pos
    return pd.Series(accepted, index=events.index)


def worker(result_queue: Any, kind: str, payload: dict[str, Any]) -> None:
    """所有网络库只在子进程导入，父进程可可靠终止阻塞请求。"""
    try:
        import akshare as ak

        if kind == "stock_ak":
            frame = ak.stock_zh_a_hist(
                symbol=payload["code"], period="daily", start_date=payload["start"], end_date=payload["end"], adjust="qfq"
            )
        elif kind == "index_ak":
            frame = ak.stock_zh_index_daily(symbol=payload["code"])
            dates = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.loc[(dates >= pd.Timestamp(payload["start"])) & (dates <= pd.Timestamp(payload["end"]))]
        elif kind == "spot_ak":
            frame = ak.stock_zh_a_spot_em()
        elif kind == "fund_rank_ak":
            frame = ak.stock_individual_fund_flow_rank(indicator="今日")
        elif kind == "stock_bao":
            import baostock as bs

            code = payload["code"]
            market = "sh" if code.startswith(("6", "9")) else "sz"
            login = bs.login()
            if getattr(login, "error_code", "1") != "0":
                raise DataError(f"BaoStock登录失败：{getattr(login, 'error_msg', '')}")
            try:
                query = bs.query_history_k_data_plus(
                    f"{market}.{code}", "date,open,high,low,close,volume,tradestatus",
                    start_date=payload["start_dash"], end_date=payload["end_dash"], frequency="d", adjustflag="2",
                )
                if getattr(query, "error_code", "1") != "0":
                    raise DataError(f"BaoStock日线失败：{getattr(query, 'error_msg', '')}")
                rows: list[list[str]] = []
                while query.next():
                    rows.append(query.get_row_data())
                frame = pd.DataFrame(rows, columns=query.fields)
                if "tradestatus" in frame.columns:
                    frame = frame.loc[frame["tradestatus"].astype(str) == "1"]
            finally:
                try:
                    bs.logout()
                except Exception:
                    pass
        else:
            raise DataError(f"未知数据请求：{kind}")
        if frame is None or frame.empty:
            raise DataError(f"{kind}返回空数据")
        result_queue.put({"ok": True, "data": frame})
    except Exception as exc:
        result_queue.put({"ok": False, "reason": f"{type(exc).__name__}: {str(exc)[:240]}"})


def hard_fetch(kind: str, payload: dict[str, Any], timeout_seconds: int) -> pd.DataFrame:
    result_queue = context().Queue(maxsize=1)
    process = context().Process(target=worker, args=(result_queue, kind, payload))
    process.start()
    try:
        reply = result_queue.get(timeout=max(1, timeout_seconds))
    except queue.Empty as exc:
        if process.is_alive():
            process.terminate()
        process.join(5)
        result_queue.close()
        raise DataError(f"{kind}超过{timeout_seconds}秒") from exc
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    result_queue.close()
    if not reply.get("ok"):
        raise DataError(str(reply.get("reason", f"{kind}失败")))
    return reply["data"]


def normalize_ohlc(raw: pd.DataFrame, source: str, min_rows: int = 160) -> pd.DataFrame:
    aliases = {
        "日期": "date", "date": "date", "开盘": "open", "open": "open", "最高": "high", "high": "high",
        "最低": "low", "low": "low", "收盘": "close", "close": "close", "成交量": "volume", "volume": "volume",
        "成交额": "amount", "amount": "amount",
    }
    frame = raw.rename(columns=aliases).copy()
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataError(f"{source}缺少字段：{','.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).drop_duplicates("date", keep="last").sort_values("date")
    valid = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
    keep = [column for column in ["date", "open", "high", "low", "close", "volume", "amount"] if column in frame.columns]
    frame = frame.loc[valid, keep].set_index("date").sort_index()
    if len(frame) < min_rows:
        raise DataError(f"{source}有效日线不足{min_rows}根：{len(frame)}")
    if "volume" not in frame.columns:
        frame["volume"] = np.nan
    if "amount" not in frame.columns:
        frame["amount"] = frame["close"] * frame["volume"]
    return frame


def fetch_ohlc(code: str, config: FetchConfig) -> tuple[pd.DataFrame, str, Optional[str]]:
    end = date.today()
    start = end - timedelta(days=config.history_days)
    payload = {
        "code": code, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "start_dash": start.isoformat(), "end_dash": end.isoformat(),
    }
    try:
        return normalize_ohlc(hard_fetch("stock_ak", payload, config.ak_timeout), "AkShare个股"), "akshare", None
    except DataError as first_error:
        LOG.warning("%s AkShare失败，尝试BaoStock：%s", code, first_error)
        frame = normalize_ohlc(hard_fetch("stock_bao", payload, config.bao_timeout), "BaoStock个股")
        return frame, "baostock", str(first_error)


def fetch_index(config: FetchConfig) -> tuple[Optional[pd.DataFrame], str]:
    end = date.today()
    start = end - timedelta(days=config.history_days)
    try:
        raw = hard_fetch(
            "index_ak", {"code": config.index_code, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d")}, config.ak_timeout
        )
        return normalize_ohlc(raw, "AkShare指数"), "akshare"
    except DataError as exc:
        LOG.warning("指数不可用，本次不计相对强弱：%s", exc)
        return None, f"unavailable:{exc}"


def compute_features(stock: pd.DataFrame, benchmark: Optional[pd.DataFrame], params: Params) -> pd.DataFrame:
    """只使用每个交易日及此前OHLCV计算特征，保留预热期NaN。"""
    frame = stock.copy()
    if benchmark is not None:
        common = frame.index.intersection(benchmark.index)
        if len(common) >= 160:
            frame = frame.loc[common].copy()
            benchmark_close = benchmark.loc[common, "close"].astype(float)
        else:
            benchmark_close = None
    else:
        benchmark_close = None
    if len(frame) < 160:
        raise DataError("共同有效日线不足160根")

    close, high, low, volume, amount = (frame[column].astype(float) for column in ["close", "high", "low", "volume", "amount"])
    ma5, ma10, ma20 = (close.rolling(days, min_periods=days).mean() for days in (5, 10, 20))
    low125, high125 = low.rolling(125, min_periods=125).min(), high.rolling(125, min_periods=125).max()
    position125 = (close - low125) / (high125 - low125).replace(0, np.nan) * 100.0
    middle_fast = chinese_sma(position125, 72)
    middle_slow = chinese_sma(middle_fast, 34)
    medium_trend = 3.0 * middle_fast - 2.0 * middle_slow

    typical = (2.0 * close + high + low) / 4.0
    low27, high27 = low.rolling(27, min_periods=27).min(), high.rolling(27, min_periods=27).max()
    control = ema((typical - low27) / (high27 - low27).replace(0, np.nan) * 100.0 - 50.0, params.control_ema)
    trend_input = params.trend_weight * control.shift(1) + (1.0 - params.trend_weight) * control
    trend = ema(trend_input, params.trend_ema)

    ma5_up = ma5.gt(ma5.shift(1))
    ma10_up = ma10.gt(ma10.shift(1))
    medium_up = medium_trend.gt(medium_trend.shift(1))
    cross_up = control.gt(trend) & control.shift(1).le(trend.shift(1))
    negative_zone = trend.lt(0)

    # CLV为收盘在日内区间的位置；它与成交额的乘积只是量价资金代理，不是账户资金流。
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_proxy = clv * amount
    proxy_5 = money_proxy.rolling(5, min_periods=5).sum()
    proxy_mean = proxy_5.rolling(20, min_periods=20).mean()
    proxy_std = proxy_5.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    proxy_z = (proxy_5 - proxy_mean) / proxy_std
    volume_ratio = volume / volume.rolling(5, min_periods=5).mean().replace(0, np.nan)
    ret5 = close / close.shift(5) - 1.0
    above_ma = close.gt(ma5) & close.gt(ma10) & close.gt(ma20)

    if benchmark_close is not None:
        stock_curve = (1.0 + close.pct_change(fill_method=None).fillna(0.0)).cumprod()
        index_curve = (1.0 + benchmark_close.pct_change(fill_method=None).fillna(0.0)).cumprod()
        relative_strength = stock_curve / index_curve - 1.0
        relative_leader = relative_strength.ge(relative_strength.rolling(20, min_periods=20).mean())
        rs_available = pd.Series(True, index=frame.index)
    else:
        relative_strength = pd.Series(np.nan, index=frame.index)
        relative_leader = pd.Series(False, index=frame.index)
        rs_available = pd.Series(False, index=frame.index)

    bull_state = ma5_up & ma10_up & medium_up & (relative_leader | ~rs_available)
    raw_event = cross_up & ma5_up & ma10_up & medium_up & negative_zone & (relative_leader | ~rs_available)
    event = cooldown(raw_event, params.cooldown_days).astype(bool)
    not_overheated = ret5.lt(0.18)
    price_score = (
        event.astype(int) * 28
        + bull_state.astype(int) * 20
        + ma5_up.astype(int) * 8
        + ma10_up.astype(int) * 8
        + medium_up.astype(int) * 10
        + cross_up.astype(int) * 8
        + negative_zone.astype(int) * 5
        + above_ma.astype(int) * 5
        + not_overheated.astype(int) * 4
        + (relative_leader & rs_available).astype(int) * 4
    )
    proxy_score = (proxy_z.ge(0.5).astype(int) * 12 + volume_ratio.ge(1.15).astype(int) * 8).astype(int)

    frame["MA5"] = ma5
    frame["MA10"] = ma10
    frame["MA20"] = ma20
    frame["股牛股"] = bull_state.astype(int) * 4
    frame["操纵"] = control
    frame["趋势"] = trend
    frame["操纵上穿"] = cross_up.astype(int)
    frame["价格结构事件"] = event.astype(int)
    frame["中线趋势"] = medium_trend
    frame["相对强弱%"] = relative_strength * 100.0
    frame["量比"] = volume_ratio
    frame["近5日涨幅%"] = ret5 * 100.0
    frame["站上三均线"] = above_ma.astype(int)
    frame["资金进出代理Z"] = proxy_z
    frame["价格结构分"] = price_score.astype(int)
    frame["量价代理分"] = proxy_score.astype(int)
    frame["基础研究分"] = (price_score + proxy_score).astype(int)
    frame["数据预热完成"] = (np.arange(len(frame)) >= 159).astype(int)
    return frame


def fund_snapshot(timeout_seconds: int) -> tuple[pd.DataFrame, str]:
    try:
        raw = hard_fetch("fund_rank_ak", {}, timeout_seconds)
    except DataError as exc:
        return pd.DataFrame(), f"unavailable:{exc}"
    rename: dict[Any, str] = {}
    for column in raw.columns:
        text = str(column).replace(" ", "")
        if "代码" in text:
            rename[column] = "code"
        elif "名称" in text:
            rename[column] = "name"
        elif "主力净流入" in text and "净额" in text:
            rename[column] = "main_net_amount"
        elif "主力净流入" in text and "净占比" in text:
            rename[column] = "main_net_ratio"
        elif "最新价" in text:
            rename[column] = "last_price"
        elif "涨跌幅" in text:
            rename[column] = "change_pct"
    flow = raw.rename(columns=rename).copy()
    if not {"code", "main_net_amount"}.issubset(flow.columns):
        return pd.DataFrame(), f"schema_missing:{','.join(map(str, raw.columns.tolist()[:16]))}"
    flow["code"] = flow["code"].map(clean_code)
    for column in ["main_net_amount", "main_net_ratio", "last_price", "change_pct"]:
        if column in flow.columns:
            flow[column] = pd.to_numeric(flow[column], errors="coerce")
    flow = flow.loc[flow["code"].str.len() == 6].drop_duplicates("code", keep="first")
    return flow, "akshare_today_snapshot"


def stock_universe(timeout_seconds: int, min_mv: float, max_mv: float) -> tuple[pd.DataFrame, str]:
    raw = hard_fetch("spot_ak", {}, timeout_seconds)
    rename: dict[Any, str] = {}
    for column in raw.columns:
        text = str(column).replace(" ", "")
        if "代码" in text:
            rename[column] = "code"
        elif "名称" in text:
            rename[column] = "name"
        elif "流通市值" in text:
            rename[column] = "circ_mv"
        elif "成交额" in text:
            rename[column] = "turnover"
    frame = raw.rename(columns=rename).copy()
    if not {"code", "name", "circ_mv"}.issubset(frame.columns):
        raise DataError(f"股票池字段缺失，实际字段：{','.join(map(str, raw.columns.tolist()[:16]))}")
    frame["code"] = frame["code"].map(clean_code)
    frame["name"] = frame["name"].astype(str)
    frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce") / 1e8
    if "turnover" in frame.columns:
        frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce")
    else:
        frame["turnover"] = np.nan
    prefixes = ("00", "30", "60", "68")
    excluded_name = frame["name"].str.contains(r"ST|退|\*ST", regex=True, na=False) | frame["name"].str.startswith(("N", "C"), na=False)
    valid = frame["code"].str.startswith(prefixes) & ~excluded_name
    valid &= frame["circ_mv"].between(float(min_mv), float(max_mv), inclusive="both")
    result = frame.loc[valid, ["code", "name", "circ_mv", "turnover"]].dropna(subset=["circ_mv"]).drop_duplicates("code")
    return result.sort_values("code").reset_index(drop=True), "akshare_snapshot"


def load_universe_json(path: Path, min_mv: float, max_mv: float, allow_missing_mv: bool) -> tuple[pd.DataFrame, str]:
    """读取上游共享股票池；若没有市值字段，默认拒绝悄然绕过小市值约束。"""
    if not path.is_file():
        raise DataError(f"共享股票池文件不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("universe", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("顶层不是股票列表")
        frame = pd.DataFrame(rows)
    except Exception as exc:
        raise DataError(f"共享股票池读取失败：{type(exc).__name__}:{str(exc)[:160]}") from exc
    rename = {"代码": "code", "名称": "name", "流通市值_亿": "circ_mv", "circ_mv_yi": "circ_mv", "turnover": "turnover", "成交额": "turnover"}
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    if not {"code", "name"}.issubset(frame.columns):
        raise DataError("共享股票池至少必须含代码/名称字段")
    frame["code"] = frame["code"].map(clean_code)
    frame["name"] = frame["name"].astype(str)
    if "circ_mv" not in frame.columns:
        if not allow_missing_mv:
            raise DataError("共享股票池不含流通市值；为避免静默绕过市值过滤，请补充市值或显式传入--allow-universe-without-mv")
        frame["circ_mv"] = np.nan
    else:
        frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce")
        frame = frame.loc[frame["circ_mv"].between(float(min_mv), float(max_mv), inclusive="both")]
    if "turnover" not in frame.columns:
        frame["turnover"] = np.nan
    else:
        frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce")
    frame = frame.loc[frame["code"].str.len() == 6].drop_duplicates("code")
    if frame.empty:
        raise DataError("共享股票池过滤后为空")
    return frame[["code", "name", "circ_mv", "turnover"]].sort_values("code").reset_index(drop=True), f"shared_json:{path.name}"


def candidate_from_features(code: str, name: str, circ_mv: float, features: pd.DataFrame, live_flow: Optional[pd.Series], min_live_flow_wan: float, flow_status: str) -> dict[str, Any]:
    latest = features.iloc[-1]
    base = int(latest["基础研究分"])
    live_amount = math.nan
    live_ratio = math.nan
    live_score = 0
    if live_flow is not None:
        live_amount = float(live_flow.get("main_net_amount", math.nan))
        live_ratio = float(live_flow.get("main_net_ratio", math.nan))
        if np.isfinite(live_amount) and live_amount >= min_live_flow_wan * 10_000:
            live_score += 15
        if np.isfinite(live_ratio) and live_ratio > 0:
            live_score += 5
    total = min(100, base + live_score)
    if total >= 75 and int(latest["价格结构事件"]) == 1 and live_score >= 15:
        tier = "确认候选"
    elif total >= 60:
        tier = "研究观察"
    else:
        tier = "不入选"
    return {
        "code": code, "name": name, "signal_date": pd.Timestamp(features.index[-1]).strftime("%Y-%m-%d"),
        "tier": tier, "total_score": int(total), "price_structure_score": base, "live_fund_score": int(live_score),
        "circ_mv_yi": round(float(circ_mv), 2) if np.isfinite(circ_mv) else None, "close": round(float(latest["close"]), 2),
        "bull_state": int(latest["股牛股"]), "control": round(float(latest["操纵"]), 3), "trend": round(float(latest["趋势"]), 3),
        "price_event": int(latest["价格结构事件"]), "volume_ratio": round(float(latest["量比"]), 3) if np.isfinite(latest["量比"]) else None,
        "return_5d_pct": round(float(latest["近5日涨幅%"]), 3) if np.isfinite(latest["近5日涨幅%"]) else None,
        "flow_proxy_z": round(float(latest["资金进出代理Z"]), 3) if np.isfinite(latest["资金进出代理Z"]) else None,
        "live_main_net_inflow_wan": round(live_amount / 10_000, 2) if np.isfinite(live_amount) else None,
        "live_main_net_ratio_pct": round(live_ratio, 3) if np.isfinite(live_ratio) else None,
        "live_flow_status": flow_status,
        "interpretation": "价格结构与量价代理的研究评分；公开主力资金流仅为当日快照确认，非账户级资金证据。",
    }


def run_backtest(features: pd.DataFrame, hold_days: int, cost_bps: float, min_score: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """回测不使用实时资金流快照，仅使用当日及此前可复算的价格/量价特征。"""
    if hold_days < 1 or cost_bps < 0:
        raise ValueError("hold_days必须为正，cost_bps不能为负")
    signals = (features["价格结构事件"].eq(1) & features["基础研究分"].ge(min_score) & features["数据预热完成"].eq(1))
    positions = np.flatnonzero(signals.to_numpy())
    rows: list[dict[str, Any]] = []
    incomplete = 0
    cost = cost_bps / 10_000.0
    for signal_pos in positions:
        entry_pos = int(signal_pos) + 1
        exit_pos = entry_pos + int(hold_days) - 1
        if exit_pos >= len(features):
            incomplete += 1
            continue
        entry = float(features.iloc[entry_pos]["open"])
        exit_price = float(features.iloc[exit_pos]["close"])
        if not (np.isfinite(entry) and np.isfinite(exit_price) and entry > 0 and exit_price > 0):
            incomplete += 1
            continue
        gross = exit_price / entry - 1.0
        net = (1.0 + gross) * (1.0 - cost) ** 2 - 1.0
        rows.append({
            "signal_date": pd.Timestamp(features.index[signal_pos]).strftime("%Y-%m-%d"),
            "entry_date": pd.Timestamp(features.index[entry_pos]).strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(features.index[exit_pos]).strftime("%Y-%m-%d"),
            "entry_open": round(entry, 4), "exit_close": round(exit_price, 4), "hold_days": int(hold_days),
            "base_score": int(features.iloc[signal_pos]["基础研究分"]), "gross_return_pct": round(gross * 100, 4), "net_return_pct": round(net * 100, 4),
        })
    trades = pd.DataFrame(rows)
    returns = trades["net_return_pct"] if not trades.empty else pd.Series(dtype=float)
    summary = {
        "method": "price_volume_proxy_only__signal_day_information_only__next_trading_day_open_entry__holding_period_close_exit",
        "live_fund_flow_used": False, "hold_days": int(hold_days), "one_way_cost_bps": float(cost_bps),
        "min_score": int(min_score), "signal_count": int(len(positions)), "completed_trade_count": int(len(trades)), "insufficient_future_bars": int(incomplete),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 4) if len(returns) else None,
        "mean_net_return_pct": round(float(returns.mean()), 4) if len(returns) else None,
        "median_net_return_pct": round(float(returns.median()), 4) if len(returns) else None,
        "warning": "单标的或用户选定样本不是样本外组合验证；历史市值筛选在回测中未使用，以避免实时市值时点泄漏。",
    }
    return trades, summary


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无记录。"
    headers = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        parts = []
        for value in row:
            if isinstance(value, float):
                text = f"{value:.3f}" if np.isfinite(value) else ""
            else:
                text = str(value)
            parts.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def write_bundle(output_dir: Path, records: pd.DataFrame, status: dict[str, Any], prefix: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz=CN_TZ).strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{stamp}"
    csv_path, json_path, md_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.json", output_dir / f"{stem}.md"
    records.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    json_path.write_text(json.dumps({"status": status, "records": records.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = (
        "# A股资本流研究筛选报告\n\n"
        f"- 生成时间：`{status['generated_at']}`\n- 状态：`{status['status']}`\n- 已扫描：`{status.get('scanned', 0)}`\n"
        f"- 成功：`{status.get('succeeded', 0)}`\n- 失败：`{status.get('failed', 0)}`\n- 当日公开资金流：`{status.get('fund_flow_status')}`\n\n"
        "> 研究评分由价格结构、量价资金代理及（如可用）当日公开主力资金流快照构成。它不代表账户级资金事实，也不构成投资建议。\n\n"
        f"## 候选\n\n{markdown_table(records)}\n"
    )
    md_path.write_text(report, encoding="utf-8")
    status_path = output_dir / f"{prefix}_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path, "status": status_path}


def serverchan_push(title: str, content: str) -> bool:
    key = (os.getenv("SERVERCHAN_SENDKEY") or os.getenv("SENDKEY") or "").strip()
    if not key:
        LOG.warning("未配置SERVERCHAN_SENDKEY/SENDKEY，跳过推送")
        return False
    try:
        import requests
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15)
        return bool(response.ok and response.json().get("code") == 0)
    except Exception as exc:
        LOG.warning("Server酱推送失败：%s", str(exc)[:180])
        return False


def run_single(args: argparse.Namespace, config: FetchConfig, params: Params) -> int:
    code = clean_code(args.code)
    stock, source, fallback = fetch_ohlc(code, config)
    index, index_status = fetch_index(config)
    features = compute_features(stock, index, params)
    fake_record = candidate_from_features(code, "单标的", math.nan, features, None, args.min_live_flow_wan, "not_requested_for_single")
    records = pd.DataFrame([fake_record])
    status: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz=CN_TZ).isoformat(), "status": "ready", "mode": "single",
        "stock_source": source, "fallback_reason": fallback, "index_status": index_status, "fund_flow_status": "not_used_for_backtest_or_single_default",
        "scanned": 1, "succeeded": 1, "failed": 0, "parameters": asdict(params),
    }
    paths = write_bundle(Path(args.output_dir), records, status, f"capital_flow_single_{code}")
    if args.backtest:
        trades, summary = run_backtest(features, args.hold_days, args.cost_bps, params.min_score)
        trades_path = Path(args.output_dir) / f"capital_flow_backtest_{code}.csv"
        summary_path = Path(args.output_dir) / f"capital_flow_backtest_{code}.json"
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.update({"backtest_csv": trades_path, "backtest_json": summary_path})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n===== 单标的资本流研究状态 =====")
    print(records.to_string(index=False))
    for name, path in paths.items():
        print(f"{name.upper()}: {path}")
    return 0


def run_scan(args: argparse.Namespace, config: FetchConfig, params: Params) -> int:
    out = Path(args.output_dir)
    started = pd.Timestamp.now(tz=CN_TZ)
    status: dict[str, Any] = {
        "generated_at": started.isoformat(), "status": "running", "mode": "market_scan", "scanned": 0, "succeeded": 0, "failed": 0,
        "parameters": asdict(params), "fund_flow_status": "not_requested", "failures": [],
    }
    try:
        if args.universe_json:
            universe, universe_status = load_universe_json(Path(args.universe_json), args.min_circ_mv, args.max_circ_mv, args.allow_universe_without_mv)
        else:
            universe, universe_status = stock_universe(args.ak_timeout, args.min_circ_mv, args.max_circ_mv)
        status["universe_status"] = universe_status
    except DataError as exc:
        status.update({"status": "universe_unavailable", "reason": str(exc)})
        write_bundle(out, pd.DataFrame(), status, "capital_flow_market")
        return 2
    try:
        flow, flow_status = fund_snapshot(args.ak_timeout)
    except Exception as exc:
        flow, flow_status = pd.DataFrame(), f"unavailable:{type(exc).__name__}"
    status["fund_flow_status"] = flow_status
    if args.require_live_fund_flow and flow.empty:
        status.update({"status": "live_fund_flow_unavailable", "reason": flow_status})
        write_bundle(out, pd.DataFrame(), status, "capital_flow_market")
        return 2
    if not flow.empty:
        universe = universe.merge(flow[[column for column in ["code", "main_net_amount", "main_net_ratio"] if column in flow.columns]], on="code", how="left")
        universe["main_net_amount"] = universe["main_net_amount"].fillna(-np.inf)
        universe = universe.sort_values(["main_net_amount", "turnover"], ascending=[False, False])
    else:
        universe = universe.sort_values(["turnover", "circ_mv"], ascending=[False, True])
    universe = universe.iloc[args.shard_index::args.shard_count].head(args.preselect_top).reset_index(drop=True)
    status["preselected"] = int(len(universe))
    index, index_status = fetch_index(config)
    status["index_status"] = index_status
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for position, row in universe.iterrows():
        code, name, circ_mv = str(row["code"]), str(row["name"]), float(row["circ_mv"])
        status["scanned"] += 1
        try:
            stock, source, fallback = fetch_ohlc(code, config)
            listing_days = (pd.Timestamp(stock.index[-1]) - pd.Timestamp(stock.index[0])).days
            if listing_days < params.min_history_days:
                continue
            features = compute_features(stock, index, params)
            live = flow.loc[flow["code"] == code].iloc[0] if not flow.empty and (flow["code"] == code).any() else None
            record = candidate_from_features(code, name, circ_mv, features, live, args.min_live_flow_wan, flow_status)
            record["stock_source"] = source
            record["fallback_reason"] = fallback
            if record["total_score"] >= params.min_score:
                records.append(record)
            status["succeeded"] += 1
        except Exception as exc:
            failures.append(f"{code}:{type(exc).__name__}:{str(exc)[:140]}")
            status["failed"] += 1
        if status["scanned"] >= 20 and status["failed"] / max(1, status["scanned"]) > args.max_failure_rate:
            status["status"] = "circuit_breaker_open"
            break
        if (position + 1) % max(1, args.checkpoint_every) == 0:
            status["failures"] = failures[-50:]
            write_bundle(out, pd.DataFrame(records), status, "capital_flow_market_checkpoint")
    output = pd.DataFrame(records)
    if not output.empty:
        output = output.sort_values(["total_score", "live_main_net_inflow_wan", "price_structure_score"], ascending=[False, False, False]).head(args.top)
    if status["status"] == "running":
        status["status"] = "ready"
    status["failures"] = failures[-50:]
    status["candidate_count"] = int(len(output))
    paths = write_bundle(out, output, status, "capital_flow_market")
    print("\n===== A股资本流研究候选 =====")
    print(output.to_string(index=False) if not output.empty else "无满足研究阈值的候选。")
    for name, path in paths.items():
        print(f"{name.upper()}: {path}")
    if args.push:
        lines = [f"**资本流研究候选：{len(output)}只**", f"状态：{status['status']}；资金流：{flow_status}", ""]
        for _, item in output.head(10).iterrows():
            lines.append(f"- {item['code']} {item['name']}｜总分{item['total_score']}｜资金流{item.get('live_main_net_inflow_wan')}")
        serverchan_push(f"资本流研究 {started:%m-%d %H:%M}", "\n".join(lines))
    return 0 if status["status"] == "ready" else 2


def self_test() -> int:
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    close = pd.Series(np.linspace(10, 22, len(dates)) + np.sin(np.linspace(0, 12, len(dates))), index=dates)
    stock = pd.DataFrame({"open": close * .995, "high": close * 1.02, "low": close * .98, "close": close, "volume": np.linspace(1e6, 2e6, len(dates)), "amount": close * np.linspace(1e6, 2e6, len(dates))}, index=dates)
    benchmark = pd.DataFrame({"open": np.linspace(3000, 3200, len(dates)), "high": np.linspace(3010, 3210, len(dates)), "low": np.linspace(2990, 3190, len(dates)), "close": np.linspace(3000, 3200, len(dates)), "volume": 1, "amount": 1}, index=dates)
    output = compute_features(stock, benchmark, Params())
    required = {"基础研究分", "资金进出代理Z", "价格结构事件", "操纵", "趋势", "股牛股"}
    assert required.issubset(output.columns)
    assert output.index.equals(stock.index)
    events = np.flatnonzero(output["价格结构事件"].to_numpy() == 1)
    assert all(np.diff(events) > Params().cooldown_days)
    test = pd.DataFrame({"open": [10, 10.1, 10.3, 10.4, 10.6], "close": [10, 10.2, 10.4, 10.5, 10.8], "价格结构事件": [0, 1, 0, 0, 0], "基础研究分": [0, 70, 0, 0, 0], "数据预热完成": [1, 1, 1, 1, 1]}, index=pd.date_range("2024-01-02", periods=5, freq="B"))
    trades, summary = run_backtest(test, 3, 12, 60)
    assert len(trades) == 1 and summary["completed_trade_count"] == 1
    assert pd.Timestamp(trades.iloc[0]["entry_date"]) > pd.Timestamp(trades.iloc[0]["signal_date"])
    print("SELF_TEST_OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股资本流研究筛选器（非账户级资金识别）")
    parser.add_argument("--code", help="六位A股代码；设置后仅运行单标的模式")
    parser.add_argument("--history-days", type=int, default=1100, help="日线下载自然日数，建议不少于900")
    parser.add_argument("--index", default="sh000001", help="AkShare基准指数代码")
    parser.add_argument("--min-circ-mv", type=float, default=5.0, help="实盘股票池最低流通市值（亿元）")
    parser.add_argument("--max-circ-mv", type=float, default=200.0, help="实盘股票池最高流通市值（亿元）")
    parser.add_argument("--universe-json", help="可选：上游共享股票池JSON；默认仍请求实时市值股票池")
    parser.add_argument("--allow-universe-without-mv", action="store_true", help="仅在共享股票池不含市值且已明确接受放弃市值过滤时使用")
    parser.add_argument("--preselect-top", type=int, default=80, help="按当日资金流/成交额预筛后的最大日线扫描数")
    parser.add_argument("--shard-index", type=int, default=0, help="分片编号，从0开始")
    parser.add_argument("--shard-count", type=int, default=1, help="分片总数")
    parser.add_argument("--top", type=int, default=20, help="输出前N个候选")
    parser.add_argument("--min-live-flow-wan", type=float, default=1000.0, help="公开主力净流入确认阈值（万元）")
    parser.add_argument("--require-live-fund-flow", action="store_true", help="当日资金流快照不可用时失败退出")
    parser.add_argument("--min-score", type=int, default=60, help="研究候选最低总分")
    parser.add_argument("--hold-days", type=int, default=20, help="回测持有交易日；信号后下一日开盘入场")
    parser.add_argument("--cost-bps", type=float, default=12.0, help="回测单边成本（基点）")
    parser.add_argument("--backtest", action="store_true", help="单标的模式启用无前视回测")
    parser.add_argument("--ak-timeout", type=int, default=12, help="AkShare单请求硬超时秒数")
    parser.add_argument("--bao-timeout", type=int, default=8, help="BaoStock单请求硬超时秒数")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="市场扫描每N只写一次checkpoint")
    parser.add_argument("--max-failure-rate", type=float, default=0.65, help="扫描至少20只后触发断路器的失败率")
    parser.add_argument("--output-dir", default="output", help="CSV/JSON/Markdown/状态文件目录")
    parser.add_argument("--push", action="store_true", help="读取SERVERCHAN_SENDKEY/SENDKEY推送单条汇总")
    parser.add_argument("--self-test", action="store_true", help="执行不联网的指标与回测契约自检")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("分片参数非法：要求0 <= shard-index < shard-count")
    if not (0 <= args.max_failure_rate <= 1):
        raise SystemExit("max-failure-rate必须在0至1之间")
    params = Params(min_score=max(1, min(100, args.min_score)))
    config = FetchConfig(history_days=max(800, args.history_days), ak_timeout=max(1, args.ak_timeout), bao_timeout=max(1, args.bao_timeout), index_code=args.index)
    try:
        return run_single(args, config, params) if args.code else run_scan(args, config, params)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOG.exception("脚本异常终止：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
