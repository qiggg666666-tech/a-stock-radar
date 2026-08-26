#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股自适应行业轮动研究模块（独立运行）。

本模块使用截至 ``signal_date`` 的前复权日线和运行时行业快照，构造市场状态、
均值回归位置、区间稳定性、左尾稳定性、趋势及波动五个统计研究代理。市场状态
只改变预先固定的因子权重乘数；不以未来收益拟合IC，不把横截面离散度称为预测
能力，也不宣称Hawkes过程、幂律尾部、Gaussian Copula、因果效应或主力资金识别。

行业快照接口为当期截面。若 ``signal_date`` 早于运行日，审计文件会标记
``historical_industry_snapshot_not_point_in_time``，此结果不可用作无前视回测结论。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("adaptive_industry_rotation")
TRADING_DAYS = 252
FACTOR_COLUMNS = ["reversion", "range_stability", "tail_stability", "trend", "volatility"]


class MarketRegime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEUTRAL = "neutral"


def _worker(connection: Any, func: Callable[[], Any]) -> None:
    try:
        connection.send(("ok", func()))
    except Exception as exc:  # noqa: BLE001
        connection.send(("error", f"{type(exc).__name__}:{str(exc)[:160]}"))
    finally:
        connection.close()


@dataclass(frozen=True)
class Config:
    start_date: str = "20220101"
    signal_date: str = ""
    index_code: str = "000001"
    universe_file: str = ""
    shard_index: int = 0
    shard_count: int = 1
    top_industries: int = 8
    max_per_industry: int = 18
    max_total: int = 0
    min_history_days: int = 140
    min_average_turnover_wan: float = 3000.0
    request_pause_seconds: float = 0.50
    retries: int = 1
    retry_wait_seconds: float = 1.0
    provider_timeout_seconds: float = 35.0
    regime_window: int = 60
    factor_window: int = 120
    risk_top_n: int = 8
    risk_bootstrap_samples: int = 3000
    risk_seed: int = 20260822

    def validate(self) -> None:
        if self.top_industries <= 0 or self.max_per_industry <= 0 or self.max_total < 0:
            raise ValueError("扫描数量无效")
        if self.shard_count < 1 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index必须位于[0, shard_count)")
        if self.min_history_days < 100 or self.factor_window < 80 or self.regime_window < 40:
            raise ValueError("历史与计算窗口过短")
        if self.min_average_turnover_wan < 0 or self.provider_timeout_seconds <= 0:
            raise ValueError("流动性门槛或超时时间无效")
        if self.retries < 0:
            raise ValueError("retries不能为负数")


def baostock_symbol(code: str) -> str:
    if code.startswith("6"):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    raise ValueError(f"unsupported_a_share_code:{code}")


def baostock_daily(code: str, start_date: str, signal_date: str) -> pd.DataFrame:
    """在独立子进程内登录、查询并登出，避免跨分片共享BaoStock会话。"""
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            baostock_symbol(code),
            "date,open,high,low,close,volume,amount",
            start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        if result.error_code != "0":
            raise RuntimeError(f"baostock_daily:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()


def normalize_daily(raw: pd.DataFrame, source: str) -> tuple[pd.DataFrame | None, str | None]:
    mappings = {
        "akshare": {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount_yuan",
        },
        "baostock": {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount_yuan",
        },
    }
    mapping = mappings[source]
    missing = [column for column in mapping if column not in raw.columns]
    if missing:
        return None, f"{source}_daily_schema_missing:{','.join(missing)}"
    frame = raw.rename(columns=mapping)[["date", "open", "high", "low", "close", "volume", "amount_yuan"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount_yuan"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, None


def split_frozen_universe(universe: list[dict[str, str]], shard_index: int, shard_count: int, max_total: int) -> list[dict[str, str]]:
    shard = universe[shard_index::shard_count]
    return shard[:max_total] if max_total else shard


class AkShareFetcher:
    """同一次运行内缓存成功结果，并把卡住的供应商调用隔离到子进程。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache: dict[str, pd.DataFrame] = {}
        self.cache_sources: dict[str, str] = {}

    def _call(self, label: str, func: Callable[[], Any]) -> Any:
        if "fork" not in mp.get_all_start_methods():
            return func()
        context = mp.get_context("fork")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(child, func), daemon=True)
        process.start()
        child.close()
        try:
            if not parent.poll(self.cfg.provider_timeout_seconds):
                process.terminate()
                process.join(timeout=3)
                raise TimeoutError(f"provider_timeout:{label}:{self.cfg.provider_timeout_seconds:.0f}s")
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

    @staticmethod
    def _numeric(value: pd.Series) -> pd.Series:
        return pd.to_numeric(value, errors="coerce")

    def _retry(self, label: str, func: Callable[[], Any]) -> tuple[Any | None, str | None]:
        last_error = "unknown"
        for attempt in range(self.cfg.retries + 1):
            try:
                return self._call(label, func), None
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.retry_wait_seconds * (attempt + 1))
        return None, f"{label}_failed:{last_error}"

    def daily(self, code: str, start_date: str, signal_date: str) -> tuple[pd.DataFrame | None, str | None, str | None]:
        key = f"daily:{code}:{start_date}:{signal_date}"
        if key in self.cache:
            return self.cache[key].copy(), None, self.cache_sources.get(key)
        source_errors: list[str] = []
        raw, error = self._retry(
            f"daily:{code}",
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=signal_date, adjust="qfq"),
        )
        if error is None:
            frame, schema_error = normalize_daily(raw, "akshare")
            if schema_error is None and frame is not None:
                source = "akshare"
            else:
                source_errors.append(schema_error or "akshare_daily_normalization_failed")
                frame = None
                source = ""
        else:
            source_errors.append(error)
            frame = None
            source = ""
        if frame is None:
            raw, fallback_error = self._retry(
                f"baostock_daily:{code}", lambda: baostock_daily(code, start_date, signal_date)
            )
            if fallback_error is not None:
                source_errors.append(fallback_error)
                return None, "daily_unavailable:" + " | ".join(source_errors), None
            frame, schema_error = normalize_daily(raw, "baostock")
            if schema_error is not None or frame is None:
                source_errors.append(schema_error or "baostock_daily_normalization_failed")
                return None, "daily_unavailable:" + " | ".join(source_errors), None
            source = "baostock"
        frame = frame.dropna(subset=["date", "open", "high", "low", "close", "amount_yuan"])
        frame = frame[(frame["close"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["amount_yuan"] >= 0)]
        frame = frame[frame["date"] <= pd.Timestamp(signal_date)].sort_values("date").drop_duplicates("date")
        if frame.empty:
            return None, f"empty_daily_history_before_signal_date:{source}", source
        self.cache[key] = frame
        self.cache_sources[key] = source
        return frame.copy(), None, source

    def index_daily(self, index_code: str, start_date: str, signal_date: str) -> tuple[pd.DataFrame | None, str | None]:
        key = f"index:{index_code}:{start_date}:{signal_date}"
        if key in self.cache:
            return self.cache[key].copy(), None
        raw, error = self._retry(
            f"index:{index_code}",
            lambda: ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date, end_date=signal_date),
        )
        if error:
            return None, error
        required = ["日期", "开盘", "最高", "最低", "收盘"]
        missing = [column for column in required if column not in raw.columns]
        if missing:
            return None, f"index_schema_missing:{','.join(missing)}"
        frame = raw.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"})[
            ["date", "open", "high", "low", "close"]
        ].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["open", "high", "low", "close"]:
            frame[column] = self._numeric(frame[column])
        frame = frame.dropna().query("close > 0 and high > 0 and low > 0")
        frame = frame[frame["date"] <= pd.Timestamp(signal_date)].sort_values("date").drop_duplicates("date")
        if frame.empty:
            return None, "empty_index_history_before_signal_date"
        self.cache[key] = frame
        return frame.copy(), None

    def industry_snapshot(self) -> tuple[pd.DataFrame | None, str | None]:
        raw, error = self._retry("industry_snapshot", ak.stock_board_industry_name_em)
        if error:
            return None, error
        required = ["板块名称", "涨跌幅", "上涨家数", "下跌家数", "换手率"]
        missing = [column for column in required if column not in raw.columns]
        if missing:
            return None, f"industry_snapshot_schema_missing:{','.join(missing)}"
        frame = raw[required].copy()
        for column in required[1:]:
            frame[column] = self._numeric(frame[column])
        frame = frame.dropna()
        denominator = (frame["上涨家数"] + frame["下跌家数"]).replace(0, np.nan)
        frame["up_ratio"] = (frame["上涨家数"] / denominator).fillna(0.5)
        frame["industry_strength"] = (
            0.50 * frame["涨跌幅"].rank(pct=True, method="average")
            + 0.30 * frame["up_ratio"].rank(pct=True, method="average")
            + 0.20 * frame["换手率"].rank(pct=True, method="average")
        )
        return frame.sort_values("industry_strength", ascending=False).reset_index(drop=True), None

    def industry_components(self, industry: str) -> tuple[pd.DataFrame | None, str | None]:
        raw, error = self._retry(f"industry_components:{industry}", lambda: ak.stock_board_industry_cons_em(symbol=industry))
        if error:
            return None, error
        if "代码" not in raw.columns or "名称" not in raw.columns:
            return None, "industry_components_schema_missing:代码/名称"
        frame = raw.copy()
        frame["code"] = frame["代码"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        frame["name"] = frame["名称"].astype(str)
        frame = frame[~frame["name"].str.contains(r"ST|\*ST|退", regex=True, na=False)]
        if "成交额" in frame.columns:
            frame["component_amount_yuan"] = self._numeric(frame["成交额"]).fillna(-1.0)
            frame = frame.sort_values(["component_amount_yuan", "code"], ascending=[False, True])
            frame["component_selection_basis"] = "current_component_amount"
        else:
            frame = frame.sort_values("code")
            frame["component_amount_yuan"] = np.nan
            frame["component_selection_basis"] = "code_order_fallback"
        return frame[["code", "name", "component_amount_yuan", "component_selection_basis"]].drop_duplicates("code"), None


class RegimeDetector:
    """环境标签是观察性分类，仅选择预设权重乘数，不构成市场预测。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def detect(self, index_frame: pd.DataFrame | None) -> tuple[MarketRegime, dict[str, float | str]]:
        neutral = {"regime_input_state": "unavailable", "annualized_volatility": np.nan, "volatility_percentile": np.nan, "return_20d": np.nan, "return_60d": np.nan, "trend_strength": np.nan}
        if index_frame is None or len(index_frame) < self.cfg.regime_window + 60:
            return MarketRegime.NEUTRAL, neutral
        close = index_frame["close"].astype(float)
        returns = close.pct_change().dropna()
        recent_vol = float(returns.tail(self.cfg.regime_window).std(ddof=1) * math.sqrt(TRADING_DAYS))
        historical_vol = returns.rolling(self.cfg.regime_window).std(ddof=1).dropna() * math.sqrt(TRADING_DAYS)
        vol_percentile = float((historical_vol <= recent_vol).mean()) if not historical_vol.empty else np.nan
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        ret60 = float(close.iloc[-1] / close.iloc[-61] - 1.0)
        ma20 = close.tail(20).mean()
        ma60 = close.tail(60).mean()
        trend_strength = float(abs(ma20 / ma60 - 1.0))
        metrics = {"regime_input_state": "ready", "annualized_volatility": recent_vol, "volatility_percentile": vol_percentile, "return_20d": ret20, "return_60d": ret60, "trend_strength": trend_strength}
        if vol_percentile >= 0.80:
            return MarketRegime.HIGH_VOLATILITY, metrics
        if vol_percentile <= 0.20:
            return MarketRegime.LOW_VOLATILITY, metrics
        if abs(ret20) <= 0.025 and trend_strength <= 0.012:
            return MarketRegime.MEAN_REVERTING, metrics
        if abs(ret60) >= 0.06 and trend_strength >= 0.012:
            return MarketRegime.TRENDING, metrics
        return MarketRegime.NEUTRAL, metrics


class FactorEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0

    def reversion(self, close: pd.Series, regime: MarketRegime) -> tuple[float, dict[str, float]]:
        window = 100 if regime == MarketRegime.HIGH_VOLATILITY else self.cfg.factor_window
        values = np.log(close.tail(window).dropna())
        if len(values) < 80:
            return 0.0, {"reversion_z": np.nan, "reversion_half_life_days": np.nan}
        x, y = values.iloc[:-1].to_numpy(), values.iloc[1:].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        if not 0.0 < slope < 1.0:
            return 0.0, {"reversion_z": np.nan, "reversion_half_life_days": np.nan}
        half_life = math.log(2.0) / -math.log(slope)
        residual = y - (intercept + slope * x)
        stationary_sd = float(np.std(residual, ddof=1)) / math.sqrt(max(1 - slope**2, 1e-12))
        mean_level = intercept / (1 - slope)
        z = (float(values.iloc[-1]) - mean_level) / stationary_sd if stationary_sd > 0 else np.nan
        target_half_life = 14.0 if regime == MarketRegime.MEAN_REVERTING else 21.0
        half_life_fit = 1.0 - min(abs(half_life - target_half_life) / 35.0, 1.0)
        score = self._clip01((-z - 0.25) / 2.4) * half_life_fit
        return self._clip01(score), {"reversion_z": z, "reversion_half_life_days": half_life}

    def range_stability(self, frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
        high, low, close = frame["high"], frame["low"], frame["close"]
        prior = close.shift(1)
        true_range = pd.concat([(high - low), (high - prior).abs(), (low - prior).abs()], axis=1).max(axis=1)
        normalized = true_range / close.replace(0, np.nan)
        intensity = normalized.fillna(0).ewm(span=10, adjust=False).mean().tail(60)
        if len(intensity) < 40:
            return 0.0, {"range_intensity_percentile": np.nan}
        percentile = float((intensity <= intensity.iloc[-1]).mean())
        return self._clip01((0.92 - percentile) / 0.70), {"range_intensity_percentile": percentile}

    def tail_stability(self, returns: pd.Series) -> tuple[float, dict[str, float]]:
        values = returns.dropna().tail(self.cfg.factor_window)
        if len(values) < 80:
            return 0.0, {"left_tail_z": np.nan, "left_tail_5pct": np.nan}
        volatility = float(values.std(ddof=1))
        q05 = float(values.quantile(0.05))
        left_tail_z = abs(q05) / volatility if volatility > 0 else np.nan
        return self._clip01((2.80 - left_tail_z) / 1.45), {"left_tail_z": left_tail_z, "left_tail_5pct": q05}

    def trend(self, close: pd.Series, regime: MarketRegime) -> tuple[float, dict[str, float]]:
        if len(close) < 61:
            return 0.0, {"return_20d_pct": np.nan, "above_ma20": np.nan}
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        above_ma20 = float(close.iloc[-1] > close.tail(20).mean())
        ceiling = 0.16 if regime == MarketRegime.HIGH_VOLATILITY else 0.20
        raw = self._clip01((ret20 + 0.05) / ceiling)
        return raw * (0.65 + 0.35 * above_ma20), {"return_20d_pct": ret20, "above_ma20": above_ma20}

    def volatility(self, returns: pd.Series) -> tuple[float, dict[str, float]]:
        values = returns.dropna().tail(20)
        if len(values) < 15:
            return 0.0, {"annualized_volatility": np.nan}
        annualized = float(values.std(ddof=1) * math.sqrt(TRADING_DAYS))
        return self._clip01((0.70 - annualized) / 0.52), {"annualized_volatility": annualized}

    def score(self, code: str, name: str, industry: str, frame: pd.DataFrame, regime: MarketRegime) -> tuple[dict[str, Any] | None, str | None]:
        if len(frame) < self.cfg.min_history_days:
            return None, "insufficient_history"
        amount20 = float(frame["amount_yuan"].tail(20).mean())
        if amount20 < self.cfg.min_average_turnover_wan * 10000:
            return None, "below_min_average_turnover"
        close = frame["close"].astype(float)
        returns = close.pct_change()
        reversion, reversion_meta = self.reversion(close, regime)
        range_score, range_meta = self.range_stability(frame)
        tail_score, tail_meta = self.tail_stability(returns)
        trend_score, trend_meta = self.trend(close, regime)
        vol_score, vol_meta = self.volatility(returns)
        return {
            "code": code,
            "name": name,
            "industry": industry,
            "signal_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
            "close": float(close.iloc[-1]),
            "average_turnover_20d_yuan": amount20,
            "average_turnover_20d_wan": amount20 / 10000,
            "reversion": reversion,
            "range_stability": range_score,
            "tail_stability": tail_score,
            "trend": trend_score,
            "volatility": vol_score,
            **reversion_meta,
            **range_meta,
            **tail_meta,
            **trend_meta,
            **vol_meta,
            "factor_disclosure": "五项均为截至信号日的统计研究代理；不表示因果、主力资金或收益预测",
        }, None


class AdaptiveWeights:
    """权重由预设环境乘数与当期横截面可分性限定调整，不使用未来收益或伪IC。"""

    BASE = {"reversion": 0.24, "range_stability": 0.18, "tail_stability": 0.20, "trend": 0.24, "volatility": 0.14}
    MULTIPLIERS = {
        MarketRegime.TRENDING: {"reversion": 0.75, "range_stability": 1.00, "tail_stability": 1.00, "trend": 1.45, "volatility": 0.85},
        MarketRegime.MEAN_REVERTING: {"reversion": 1.45, "range_stability": 1.00, "tail_stability": 1.05, "trend": 0.65, "volatility": 1.10},
        MarketRegime.HIGH_VOLATILITY: {"reversion": 0.80, "range_stability": 0.85, "tail_stability": 1.35, "trend": 0.80, "volatility": 1.35},
        MarketRegime.LOW_VOLATILITY: {"reversion": 1.10, "range_stability": 1.20, "tail_stability": 0.90, "trend": 1.10, "volatility": 0.80},
        MarketRegime.NEUTRAL: {name: 1.0 for name in FACTOR_COLUMNS},
    }

    def allocate(self, candidates: pd.DataFrame, regime: MarketRegime) -> tuple[dict[str, float], dict[str, float]]:
        multipliers = self.MULTIPLIERS[regime]
        separability: dict[str, float] = {}
        raw_weights: dict[str, float] = {}
        for factor in FACTOR_COLUMNS:
            series = candidates[factor].dropna() if factor in candidates else pd.Series(dtype=float)
            mad = float((series - series.median()).abs().median()) if len(series) >= 20 else np.nan
            # Limited 0.9–1.1 adjustment: dispersion is only a ranking-separation proxy, not IC.
            adjustment = float(np.clip(0.9 + 2.0 * mad, 0.9, 1.1)) if np.isfinite(mad) else 1.0
            separability[factor] = adjustment
            raw_weights[factor] = self.BASE[factor] * multipliers[factor] * adjustment
        total = sum(raw_weights.values())
        return ({factor: raw_weights[factor] / total for factor in FACTOR_COLUMNS}, separability)

    @staticmethod
    def correlations(candidates: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        if len(candidates) < 3:
            return {}
        matrix = candidates[FACTOR_COLUMNS].corr()
        return {
            row: {column: (None if pd.isna(matrix.loc[row, column]) else float(matrix.loc[row, column])) for column in FACTOR_COLUMNS}
            for row in FACTOR_COLUMNS
        }


def load_universe(path: Path) -> list[dict[str, str]]:
    """读取预先冻结的共同股票池；行业字段缺失时明确保留“待映射”标签。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = (payload.get("universe") or payload.get("stocks") or payload.get("codes") or payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ValueError("共同股票池必须是股票对象列表")
    normalized: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("代码") or "").replace(".0", "").zfill(6)
        name = str(item.get("name") or item.get("名称") or code)
        industry = str(item.get("industry") or item.get("行业") or "全市场待映射")
        if len(code) == 6 and code.isdigit() and "ST" not in name.upper() and "退" not in name:
            normalized[code] = {"code": code, "name": name, "industry": industry}
    return [normalized[code] for code in sorted(normalized)]


class AdaptiveIndustryRotationResearch:
    def __init__(self, cfg: Config) -> None:
        cfg.validate()
        self.cfg = cfg
        self.fetcher = AkShareFetcher(cfg)
        self.factors = FactorEngine(cfg)
        self.errors: list[dict[str, str]] = []
        self.returns: dict[str, pd.Series] = {}
        self.daily_source_counts: dict[str, int] = {}

    def _error(self, code: str, name: str, industry: str, stage: str, reason: str) -> None:
        self.errors.append({"code": code, "name": name, "industry": industry, "stage": stage, "reason": reason})

    def _risk_summary(self, top_codes: list[str]) -> dict[str, Any]:
        columns = [self.returns[code] for code in top_codes if code in self.returns]
        if len(columns) < 4:
            return {"state": "skipped", "reason": "fewer_than_four_candidates"}
        matrix = pd.concat(columns, axis=1).dropna()
        if len(matrix) < 60:
            return {"state": "skipped", "reason": "fewer_than_60_aligned_rows"}
        portfolio = matrix.mean(axis=1).to_numpy()
        rng = np.random.default_rng(self.cfg.risk_seed)
        sampled = rng.choice(portfolio, size=self.cfg.risk_bootstrap_samples, replace=True)
        var = float(np.quantile(sampled, 0.05))
        cvar = float(sampled[sampled <= var].mean())
        return {
            "state": "ready",
            "method": "deterministic_empirical_bootstrap_not_copula_forecast",
            "candidate_count": len(columns),
            "aligned_rows": len(matrix),
            "one_day_var_5pct": var,
            "one_day_cvar_5pct": cvar,
            "random_seed": self.cfg.risk_seed,
        }

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        run_at = datetime.now().astimezone()
        signal_date = self.cfg.signal_date or run_at.strftime("%Y%m%d")
        try:
            signal_timestamp = pd.Timestamp(signal_date)
        except ValueError as exc:
            raise ValueError("signal_date必须是YYYYMMDD") from exc
        index_frame, index_error = self.fetcher.index_daily(self.cfg.index_code, self.cfg.start_date, signal_date)
        regime, regime_metrics = RegimeDetector(self.cfg).detect(index_frame)
        snapshot, snapshot_error = self.fetcher.industry_snapshot()
        selected_industries = snapshot.head(self.cfg.top_industries).copy() if snapshot is not None else pd.DataFrame()
        snapshot_state = "current_snapshot"
        if snapshot is None:
            snapshot_state = "unavailable"
            self._error("", "", "", "industry_snapshot", snapshot_error or "unknown")
        elif signal_timestamp.normalize() < pd.Timestamp(run_at.date()):
            snapshot_state = "historical_industry_snapshot_not_point_in_time"
        queue: list[tuple[str, str, str, str]] = []
        universe_count: int | None = None
        if self.cfg.universe_file:
            universe = load_universe(Path(self.cfg.universe_file))
            universe_count = len(universe)
            shard = split_frozen_universe(universe, self.cfg.shard_index, self.cfg.shard_count, self.cfg.max_total)
            queue = [(item["code"], item["name"], item["industry"], "frozen_universe_shard") for item in shard]
        else:
            seen: set[str] = set()
            for _, industry_row in selected_industries.iterrows():
                industry = str(industry_row["板块名称"])
                components, component_error = self.fetcher.industry_components(industry)
                if components is None:
                    self._error("", "", industry, "industry_components", component_error or "unknown")
                    continue
                for _, component in components.head(self.cfg.max_per_industry).iterrows():
                    code = str(component["code"])
                    if code not in seen:
                        seen.add(code)
                        queue.append((code, str(component["name"]), industry, str(component["component_selection_basis"])))
            if self.cfg.max_total:
                queue = queue[: self.cfg.max_total]
        rows: list[dict[str, Any]] = []
        for position, (code, name, industry, selection_basis) in enumerate(queue, start=1):
            frame, daily_error, daily_source = self.fetcher.daily(code, self.cfg.start_date, signal_date)
            if frame is None:
                self._error(code, name, industry, "daily", daily_error or "unknown")
            else:
                source_name = daily_source or "unknown"
                self.daily_source_counts[source_name] = self.daily_source_counts.get(source_name, 0) + 1
                row, score_error = self.factors.score(code, name, industry, frame, regime)
                if row is None:
                    self._error(code, name, industry, "eligibility", score_error or "unknown")
                else:
                    row["component_selection_basis"] = selection_basis
                    row["daily_data_source"] = source_name
                    rows.append(row)
                    self.returns[code] = frame.set_index("date")["close"].pct_change().rename(code)
            if position < len(queue):
                time.sleep(self.cfg.request_pause_seconds)
        candidates = pd.DataFrame(rows)
        weights: dict[str, float] = {}
        separability: dict[str, float] = {}
        correlations: dict[str, dict[str, float | None]] = {}
        if not candidates.empty:
            if self.cfg.universe_file:
                # A-D分片只产生原始因子。全部分片合并后才按完整截面计算权重和排序。
                candidates = candidates.sort_values(["trend", "average_turnover_20d_yuan"], ascending=False).reset_index(drop=True)
            else:
                weights, separability = AdaptiveWeights().allocate(candidates, regime)
                correlations = AdaptiveWeights.correlations(candidates)
                candidates["research_score"] = 100.0 * sum(candidates[factor] * weights[factor] for factor in FACTOR_COLUMNS)
                candidates = candidates.sort_values(["research_score", "trend", "average_turnover_20d_yuan"], ascending=False).reset_index(drop=True)
                candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
                candidates["rank_in_industry"] = candidates.groupby("industry")["research_score"].rank(method="min", ascending=False).astype(int)
        status = "ready" if snapshot is not None else "degraded"
        audit = self._base_audit(run_at, signal_date, regime, regime_metrics, status)
        audit.update(
            {
                "industry_snapshot_state": snapshot_state,
                "industry_selection": selected_industries[["板块名称", "涨跌幅", "上涨家数", "下跌家数", "换手率", "up_ratio", "industry_strength"]].to_dict("records") if not selected_industries.empty else [],
                "industry_snapshot_error": snapshot_error,
                "scan_mode": "frozen_universe_shard" if self.cfg.universe_file else "top_industry_components",
                "universe_count": universe_count,
                "shard_index": self.cfg.shard_index,
                "shard_count": self.cfg.shard_count,
                "scan_queue_count": len(queue),
                "daily_data_source_counts": self.daily_source_counts,
                "candidate_count": int(len(candidates)),
                "adaptive_weights": weights,
                "cross_sectional_separability_adjustment": separability,
                "factor_cross_sectional_correlation": correlations,
                "ranking_state": "pending_global_summary" if self.cfg.universe_file else "ranked_in_this_run",
                "risk_summary": self._risk_summary(candidates.head(self.cfg.risk_top_n)["code"].tolist() if not candidates.empty else []),
            }
        )
        return candidates, audit

    def _base_audit(self, run_at: datetime, signal_date: str, regime: MarketRegime, metrics: dict[str, float | str], status: str) -> dict[str, Any]:
        return {
            "schema_version": "a-share-adaptive-industry-rotation-research/v1",
            "status": status,
            "generated_at": run_at.isoformat(timespec="seconds"),
            "signal_date_requested": signal_date,
            "market_regime": regime.value,
            "market_regime_metrics": metrics,
            "error_count": len(self.errors),
            "config": asdict(self.cfg),
            "research_disclaimer": "市场状态与五个因子均为观察性统计研究代理。权重只由预设环境规则和当期横截面可分性有限调整，不以未来收益估计IC；行业快照为运行时截面。结果不表示因果、主力资金、Hawkes校准、幂律检验、Copula预测或收益承诺，不构成投资建议。",
        }


def write_outputs(output_dir: Path, candidates: pd.DataFrame, errors: list[dict[str, str]], audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "adaptive_industry_rotation_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["code", "name", "industry", "stage", "reason"]).to_csv(
        output_dir / "adaptive_industry_rotation_errors.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "adaptive_industry_rotation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# A股自适应行业轮动研究输出",
        f"- 状态：`{audit['status']}`",
        f"- 信号日：`{audit['signal_date_requested']}`",
        f"- 环境代理：`{audit['market_regime']}`",
        f"- 行业快照状态：`{audit.get('industry_snapshot_state', 'not_recorded')}`",
        f"- 研究候选：`{audit.get('candidate_count', 0)}`",
        f"- 错误/跳过：`{audit['error_count']}`",
        "",
        "> 因子和环境标签只使用截至信号日的数据；行业快照是运行时截面。该输出只用于研究，不构成投资建议。",
    ]
    if not candidates.empty:
        global_display = ["rank", "code", "name", "industry", "research_score", "close", "return_20d_pct", "average_turnover_20d_wan"]
        if set(global_display).issubset(candidates.columns):
            lines.extend(["", "## 前30名", candidates.head(30)[global_display].to_markdown(index=False, floatfmt=".4f")])
        else:
            # A-D共同股票池分片只输出原始因子；全局权重、评分和rank只能在汇总阶段生成。
            shard_display = [
                column
                for column in ["code", "name", "industry", "trend", "close", "return_20d_pct", "average_turnover_20d_wan", "daily_data_source"]
                if column in candidates.columns
            ]
            lines.extend(
                [
                    "",
                    "## 分片原始候选（待全局汇总排序）",
                    "> 本分片不生成全局权重、研究评分或rank；这些字段只在A-D全部合并后由汇总器统一计算。",
                    candidates.head(30)[shard_display].to_markdown(index=False, floatfmt=".4f"),
                ]
            )
    (output_dir / "adaptive_industry_rotation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    cfg = Config(min_average_turnover_wan=0.0, factor_window=100, min_history_days=120, regime_window=40, provider_timeout_seconds=0.01)
    cfg.validate()
    dates = pd.bdate_range("2025-01-02", periods=160)
    # Deterministic mathematical series used only to test code paths, not to infer a financial outcome.
    cycle = np.sin(np.linspace(0, 8 * math.pi, len(dates))) * 0.012
    close = 10.0 * np.exp(np.cumsum(0.0005 + cycle))
    frame = pd.DataFrame({"date": dates, "open": close * 0.997, "high": close * 1.012, "low": close * 0.988, "close": close, "volume": 1_000_000.0, "amount_yuan": close * 1_000_000.0})
    engine = FactorEngine(cfg)
    row, error = engine.score("000001", "测试样本", "测试行业", frame, MarketRegime.NEUTRAL)
    assert error is None and row is not None
    assert all(0.0 <= float(row[factor]) <= 1.0 for factor in FACTOR_COLUMNS)
    weights, adjustment = AdaptiveWeights().allocate(pd.DataFrame([row] * 20), MarketRegime.NEUTRAL)
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    assert set(adjustment) == set(FACTOR_COLUMNS)
    fetcher = AkShareFetcher(cfg)
    try:
        fetcher._call("self_test_timeout", lambda: time.sleep(0.05))
        raise AssertionError("provider timeout did not fire")
    except TimeoutError:
        pass
    LOGGER.info("self-test passed")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="A股自适应行业轮动研究模块（非投资建议）")
    command.add_argument("--self-test", action="store_true", help="离线自检，不访问数据源")
    command.add_argument("--start-date", default="20220101")
    command.add_argument("--signal-date", default="", help="截至日期YYYYMMDD；留空为运行当日")
    command.add_argument("--index-code", default="000001")
    command.add_argument("--universe-file", default="", help="共同股票池JSON；全市场生产扫描必须提供")
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--shard-count", type=int, default=1)
    command.add_argument("--top-industries", type=int, default=8)
    command.add_argument("--max-per-industry", type=int, default=18)
    command.add_argument("--max-total", type=int, default=0, help="每片最多扫描数；0表示共同股票池该片全部代码")
    command.add_argument("--min-average-turnover-wan", type=float, default=3000.0)
    command.add_argument("--request-pause-seconds", type=float, default=0.20)
    command.add_argument("--retries", type=int, default=1)
    command.add_argument("--provider-timeout-seconds", type=float, default=35.0)
    command.add_argument("--output-dir", default="output_adaptive_industry_rotation")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    cfg = Config(
        start_date=args.start_date,
        signal_date=args.signal_date,
        index_code=args.index_code,
        universe_file=args.universe_file,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        top_industries=args.top_industries,
        max_per_industry=args.max_per_industry,
        max_total=args.max_total,
        min_average_turnover_wan=args.min_average_turnover_wan,
        request_pause_seconds=args.request_pause_seconds,
        retries=args.retries,
        provider_timeout_seconds=args.provider_timeout_seconds,
    )
    try:
        app = AdaptiveIndustryRotationResearch(cfg)
        candidates, audit = app.run()
        write_outputs(Path(args.output_dir), candidates, app.errors, audit)
        LOGGER.info("完成：状态=%s，候选=%s，错误=%s，输出=%s", audit["status"], len(candidates), len(app.errors), args.output_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    raise SystemExit(main())
