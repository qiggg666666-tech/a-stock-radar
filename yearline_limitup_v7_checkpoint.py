#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年线涨停双层预警系统 v7.0（baostock）

层 A：提前预警（T-1 收盘后）
    仅使用决策日及更早的数据，输出下一交易日值得观察的年线附近强势候选。

层 B：涨停确认（T 日收盘后）
    确认“涨停 + 年线突破 + 前序放量 + 年线向上”的已发生信号。

回测模块：
    对历史每个 T 日的预警信号，只在 T+1 至 T+N 的区间计算结果；未来数据只用于
    评估标签，不会参与 T 日预警特征或评分计算，避免未来函数。

依赖：
    pip install baostock pandas numpy tqdm requests

常用命令：
    # 单股票验证：同时输出预警和确认
    python yearline_limitup_v7_dual_layer.py --symbols 600000 --as-of 2025-12-31 --processes 1

    # 全市场双层筛选
    python yearline_limitup_v7_dual_layer.py --all --processes 2 --min-pre-score 65

    # 在指定股票池执行无未来函数的 3 日窗口历史评估
    python yearline_limitup_v7_dual_layer.py --symbols 600000,000001 --backtest --backtest-days 180 --backtest-horizon 3

说明：
    本程序用于量化研究与技术形态筛选，不构成投资建议，也不保证任何股票会涨停。
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ================================ 配置 ======================================
@dataclass
class Config:
    # 数据与运行
    LOOKBACK_CALENDAR_DAYS: int = 760
    MIN_LIST_DAYS: int = 250
    NUM_PROCESSES: int = 2
    REQUEST_DELAY_SEC: float = 0.20
    MAX_RETRIES: int = 3
    QUERY_TIMEOUT_SEC: int = 25
    LOGIN_STAGGER_SEC: float = 1.20
    ADJUST_FLAG: str = "2"  # 1 后复权，2 前复权，3 不复权
    AS_OF_DATE: Optional[str] = None
    OUTPUT_DIR: str = "./选股结果"
    VERBOSE: bool = False
    SCAN_OFFSET: int = 0
    SCAN_LIMIT: int = 0  # 0 表示从 offset 扫描到当前股票池末尾
    SHARD_ID: str = "single"
    CHECKPOINT_EVERY: int = 25
    RESUME: bool = True

    # 通用基础条件
    YEAR_LINE_PERIOD: int = 250
    YEAR_LINE_SLOPE_DAYS: int = 5
    VOLUME_BASE_DAYS: int = 5
    MIN_PRICE: float = 3.0
    MAX_PRICE: float = 300.0
    MIN_AMOUNT_WAN: float = 5000.0
    MIN_MARKET_CAP_YI: float = 15.0
    REQUIRE_MARKET_CAP: bool = True
    EXCLUDE_ST: bool = True
    EXCLUDE_KCB: bool = False
    EXCLUDE_CHUANGYE: bool = False
    EXCLUDE_BJ: bool = True

    # 层 A：提前预警（只可使用 T 日及更早数据）
    PRE_NEAR_YEARLINE_PCT: float = 0.03      # 收盘允许低于 MA250 的最大比例
    PRE_MAX_ABOVE_YEARLINE_PCT: float = 0.12 # 避免远离年线后的过度追高
    PRE_RESISTANCE_DAYS: int = 20
    PRE_NEAR_RESISTANCE_PCT: float = 0.02
    PRE_MIN_VOLUME_RATIO: float = 1.20
    PRE_MIN_CLOSE_POSITION: float = 0.55
    PRE_MAX_DAILY_RISE_PCT: float = 7.0
    PRE_REQUIRE_SHORT_TREND: bool = True
    PRE_COOLDOWN_DAYS: int = 5
    PRE_MIN_SCORE: float = 65.0

    # 收盘后行业差异化预警（科技 / 周期 / 其他）；均仅使用 T 日及之前的数据
    ENABLE_SECTOR_PROFILES: bool = True
    HIGH_VOL_ATR20_PCT: float = 0.04
    LOW_VOL_ATR20_PCT: float = 0.02

    # 层 B：涨停确认
    LIMIT_UP_PCT_MAIN: float = 9.85
    LIMIT_UP_PCT_20: float = 19.80
    LIMIT_UP_PCT_BJ: float = 29.70
    YEAR_LINE_BREAK_WINDOW: int = 5
    CONFIRM_VOLUME_RATIO: float = 1.60
    USE_MACD_CONFIRM_FILTER: bool = True

    # 回测
    BACKTEST_DAYS: int = 180
    BACKTEST_HORIZON: int = 3

    # 评分权重，总和为 100
    WEIGHT_YEARLINE: float = 22.0
    WEIGHT_RESISTANCE: float = 22.0
    WEIGHT_VOLUME: float = 18.0
    WEIGHT_CANDLE: float = 14.0
    WEIGHT_SHORT_TREND: float = 14.0
    WEIGHT_MOMENTUM: float = 10.0


cfg = Config()


@dataclass(frozen=True)
class SectorProfile:
    """收盘后预警档位；单只股票计算时传入，绝不修改全局 cfg。"""
    label: str
    near_yearline: float
    max_above_yearline: float
    near_resistance: float
    min_volume_ratio: float
    min_close_position: float
    max_daily_rise_pct: float
    min_score: float
    min_market_cap_yi: float
    require_macd: bool


TECH_PROFILE = SectorProfile(
    label="科技", near_yearline=0.025, max_above_yearline=0.090, near_resistance=0.025,
    min_volume_ratio=1.60, min_close_position=0.65, max_daily_rise_pct=6.50,
    min_score=72.0, min_market_cap_yi=30.0, require_macd=True,
)
CYCLE_PROFILE = SectorProfile(
    label="周期", near_yearline=0.015, max_above_yearline=0.050, near_resistance=0.010,
    min_volume_ratio=1.80, min_close_position=0.70, max_daily_rise_pct=4.50,
    min_score=78.0, min_market_cap_yi=50.0, require_macd=True,
)


def classify_sector(industry: str) -> str:
    """将数据源行业名称归并为收盘后规则档位；未命中则使用统一档位。"""
    text = str(industry or "未知")
    tech_words = ("半导体", "芯片", "电子", "软件", "IT", "通信", "计算机", "元器件", "互联网", "新能源", "电气设备")
    cycle_words = ("化工", "有色", "煤炭", "钢铁", "石油", "采掘", "矿", "建材", "机械", "造纸", "橡胶", "航运")
    if any(word in text for word in tech_words):
        return "科技"
    if any(word in text for word in cycle_words):
        return "周期"
    return "统一"


def resolve_sector_profile(industry: str, atr20_pct: float) -> SectorProfile:
    """返回行业基础档位，并按截至 T 日的 ATR20 有界修正。"""
    if not cfg.ENABLE_SECTOR_PROFILES:
        base = SectorProfile(
            label="统一", near_yearline=cfg.PRE_NEAR_YEARLINE_PCT,
            max_above_yearline=cfg.PRE_MAX_ABOVE_YEARLINE_PCT,
            near_resistance=cfg.PRE_NEAR_RESISTANCE_PCT,
            min_volume_ratio=cfg.PRE_MIN_VOLUME_RATIO,
            min_close_position=cfg.PRE_MIN_CLOSE_POSITION,
            max_daily_rise_pct=cfg.PRE_MAX_DAILY_RISE_PCT,
            min_score=cfg.PRE_MIN_SCORE, min_market_cap_yi=cfg.MIN_MARKET_CAP_YI,
            require_macd=False,
        )
    else:
        group = classify_sector(industry)
        base = TECH_PROFILE if group == "科技" else CYCLE_PROFILE if group == "周期" else SectorProfile(
            label="统一", near_yearline=cfg.PRE_NEAR_YEARLINE_PCT,
            max_above_yearline=cfg.PRE_MAX_ABOVE_YEARLINE_PCT,
            near_resistance=cfg.PRE_NEAR_RESISTANCE_PCT,
            min_volume_ratio=cfg.PRE_MIN_VOLUME_RATIO,
            min_close_position=cfg.PRE_MIN_CLOSE_POSITION,
            max_daily_rise_pct=cfg.PRE_MAX_DAILY_RISE_PCT,
            min_score=cfg.PRE_MIN_SCORE, min_market_cap_yi=cfg.MIN_MARKET_CAP_YI,
            require_macd=False,
        )
    if not np.isfinite(atr20_pct):
        return base
    if atr20_pct > cfg.HIGH_VOL_ATR20_PCT:
        return replace(
            base,
            near_yearline=min(base.near_yearline * 1.20, 0.03),
            max_above_yearline=min(base.max_above_yearline * 1.20, 0.12),
            near_resistance=min(base.near_resistance * 1.20, 0.03),
            min_volume_ratio=min(base.min_volume_ratio + 0.15, 2.20),
            min_close_position=min(base.min_close_position + 0.05, 0.85),
            min_score=min(base.min_score + 3.0, 90.0),
        )
    if atr20_pct <= cfg.LOW_VOL_ATR20_PCT:
        return replace(
            base,
            near_yearline=max(base.near_yearline * 0.80, 0.005),
            max_above_yearline=max(base.max_above_yearline * 0.80, 0.02),
            near_resistance=max(base.near_resistance * 0.80, 0.005),
        )
    return base


# ================================ 日志 ======================================
def configure_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(processName)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger("yearline_v7")


log = configure_logging()


# ================================ 运行状态 ==================================
_bs: Any = None
_bs_session_ready = False
_industry_map: Dict[str, str] = {}


class QueryTimeoutError(TimeoutError):
    """baostock 调用超过配置时限。"""


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise QueryTimeoutError(f"查询超过 {cfg.QUERY_TIMEOUT_SEC} 秒")


def call_with_timeout(function: Callable[..., Any], *args: Any, timeout_sec: Optional[int] = None, **kwargs: Any) -> Any:
    """在 POSIX 主线程中用 SIGALRM 中断阻塞调用，避免失控线程残留。"""
    timeout = timeout_sec if timeout_sec is not None else cfg.QUERY_TIMEOUT_SEC
    if os.name != "posix" or not hasattr(signal, "setitimer"):
        return function(*args, **kwargs)
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return function(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _logout_quietly() -> None:
    global _bs_session_ready
    was_ready = _bs_session_ready
    _bs_session_ready = False
    try:
        if was_ready and _bs is not None:
            _bs.logout()
    except Exception:
        pass


def _ensure_session() -> bool:
    global _bs, _bs_session_ready
    if _bs is None:
        try:
            import baostock as bs
            _bs = bs
        except ImportError:
            return False
    if _bs_session_ready:
        return True
    try:
        login_result = call_with_timeout(_bs.login)
        _bs_session_ready = login_result.error_code == "0"
        if not _bs_session_ready:
            log.debug("baostock 登录失败：%s", login_result.error_msg)
        return _bs_session_ready
    except Exception as exc:
        _bs_session_ready = False
        log.debug("baostock 登录异常：%s", exc)
        return False


def _reset_session() -> None:
    _logout_quietly()
    time.sleep(random.uniform(0.15, 0.45))


def retry_call(function: Callable[..., Any], *args: Any, timeout_sec: Optional[int] = None, **kwargs: Any) -> Any:
    """带会话重建和指数退避的查询封装。"""
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            if not _ensure_session():
                raise ConnectionError("baostock 未登录")
            return call_with_timeout(function, *args, timeout_sec=timeout_sec, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < cfg.MAX_RETRIES:
                _reset_session()
                delay = min(8.0, 0.8 * (1.8 ** (attempt - 1))) + random.uniform(0.0, 0.3)
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def _worker_init(config_values: Mapping[str, Any], industry_map: Mapping[str, str]) -> None:
    global cfg, _industry_map
    cfg = Config(**dict(config_values))
    _industry_map = dict(industry_map)
    configure_logging(cfg.VERBOSE)
    time.sleep(random.uniform(0.0, cfg.LOGIN_STAGGER_SEC * max(1, cfg.NUM_PROCESSES)))
    _ensure_session()
    atexit.register(_logout_quietly)


# ================================ 数据层 ===================================
HIST_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
NUMERIC_COLUMNS = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")


def parse_as_of_date(value: Optional[str]) -> date:
    if not value:
        return datetime.now().date()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("--as-of 必须使用 YYYY-MM-DD 格式，例如 2025-12-31") from exc


def normalize_code(value: Any) -> str:
    raw = str(value).strip()
    plain = raw.split(".")[-1]
    return plain.zfill(6) if plain.isdigit() and len(plain) <= 6 else plain


def to_bs_code(code: str) -> Optional[str]:
    plain = normalize_code(code)
    if not plain.isdigit() or len(plain) != 6:
        return None
    if plain.startswith("6"):
        return f"sh.{plain}"
    if plain.startswith(("0", "3")):
        return f"sz.{plain}"
    if plain.startswith(("4", "8", "9")):
        return f"bj.{plain}"
    return None


def get_limit_rule(code: str) -> Tuple[float, str]:
    plain = normalize_code(code)
    if plain.startswith(("4", "8", "9")):
        return cfg.LIMIT_UP_PCT_BJ, "北交所涨停"
    if plain.startswith(("300", "301")):
        return cfg.LIMIT_UP_PCT_20, "创业板涨停"
    if plain.startswith(("688", "689")):
        return cfg.LIMIT_UP_PCT_20, "科创板涨停"
    return cfg.LIMIT_UP_PCT_MAIN, "主板涨停"


def is_limit_up(pct_change: Any, code: str) -> Tuple[bool, str]:
    if pd.isna(pct_change):
        return False, ""
    threshold, label = get_limit_rule(code)
    return (float(pct_change) >= threshold, label if float(pct_change) >= threshold else "")


def fetch_history(code: str) -> Optional[pd.DataFrame]:
    bs_code = to_bs_code(code)
    if bs_code is None:
        return None
    as_of = parse_as_of_date(cfg.AS_OF_DATE)
    start = as_of - timedelta(days=cfg.LOOKBACK_CALENDAR_DAYS)

    def query() -> Tuple[List[List[str]], List[str]]:
        assert _bs is not None
        result = _bs.query_history_k_data_plus(
            bs_code,
            HIST_FIELDS,
            start_date=start.isoformat(),
            end_date=as_of.isoformat(),
            frequency="d",
            adjustflag=cfg.ADJUST_FLAG,
        )
        if result.error_code != "0":
            raise RuntimeError(f"query_history_k_data_plus: {result.error_msg}")
        rows: List[List[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        return rows, list(result.fields)

    rows, fields = retry_call(query)
    time.sleep(cfg.REQUEST_DELAY_SEC)
    if not rows:
        return None

    frame = pd.DataFrame(rows, columns=fields)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["isST"] = pd.to_numeric(frame.get("isST", 0), errors="coerce").fillna(0).astype("int8")
    frame = frame.loc[frame["tradestatus"].astype(str) == "1"].copy()
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.loc[
        (frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0)
        & (frame["close"] > 0) & (frame["volume"] > 0)
    ].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame if len(frame) >= cfg.MIN_LIST_DAYS else None


def get_stock_pool(specified_codes: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    """获取扫描股票池。

    指定代码模式不再请求数千只股票的全市场列表，避免公共接口拥塞；名称暂以代码回填，
    不影响技术计算。全市场模式则为慢接口使用较长时限和重试。
    """
    if specified_codes:
        pool: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for item in specified_codes:
            plain = normalize_code(item)
            if plain in seen:
                continue
            if to_bs_code(plain) is None:
                log.warning("跳过无效股票代码：%s", item)
                continue
            seen.add(plain)
            pool.append((plain, plain))
        return pool

    global _bs
    try:
        import baostock as bs
        _bs = bs
    except ImportError as exc:
        raise RuntimeError("缺少 baostock，请先执行 pip install baostock") from exc

    if not _ensure_session():
        raise ConnectionError("股票池登录失败")
    try:
        # query_stock_basic 返回全市场列表，公共服务器高峰时响应可明显超过单股票日线查询。
        result = retry_call(_bs.query_stock_basic, timeout_sec=max(cfg.QUERY_TIMEOUT_SEC, 120))
        if result.error_code != "0":
            raise RuntimeError(f"query_stock_basic 失败：{result.error_msg}")
        rows: List[List[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
    finally:
        _logout_quietly()

    required = {"code", "code_name", "type", "status"}
    if not required.issubset(frame.columns):
        raise ValueError(f"股票池字段缺失：{', '.join(sorted(required.difference(frame.columns)))}")
    frame = frame.loc[(frame["type"].astype(str) == "1") & (frame["status"].astype(str) == "1")].copy()
    frame["plain_code"] = frame["code"].map(normalize_code)
    frame = frame.loc[frame["plain_code"].str.match(r"^(00|30|60|68|4|8|9)", na=False)].copy()
    if cfg.EXCLUDE_KCB:
        frame = frame.loc[~frame["plain_code"].str.startswith(("688", "689"))]
    if cfg.EXCLUDE_CHUANGYE:
        frame = frame.loc[~frame["plain_code"].str.startswith(("300", "301"))]
    if cfg.EXCLUDE_BJ:
        frame = frame.loc[~frame["plain_code"].str.startswith(("4", "8", "9"))]

    if specified_codes:
        wanted = {normalize_code(item) for item in specified_codes}
        frame = frame.loc[frame["plain_code"].isin(wanted)].copy()
        missing = sorted(wanted.difference(set(frame["plain_code"])))
        if missing:
            log.warning("以下指定代码不在正常上市股票池中：%s", ", ".join(missing))

    frame = frame.sort_values("plain_code").drop_duplicates("plain_code", keep="last")
    return list(frame[["plain_code", "code_name"]].itertuples(index=False, name=None))


def get_industry_map() -> Dict[str, str]:
    """行业映射失败不影响主流程，输出中回退为“未知”。"""
    global _bs
    try:
        import baostock as bs
        _bs = bs
        if not _ensure_session():
            return {}
        result = retry_call(_bs.query_stock_industry, timeout_sec=max(cfg.QUERY_TIMEOUT_SEC, 120))
        if result.error_code != "0":
            return {}
        rows: List[List[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
        if not {"code", "industry"}.issubset(frame.columns):
            return {}
        return {normalize_code(row.code): str(row.industry or "未知") for row in frame[["code", "industry"]].itertuples(index=False)}
    except Exception as exc:
        log.warning("行业映射获取失败：%s", exc)
        return {}
    finally:
        _logout_quietly()


# ================================ 指标层 ===================================
def calc_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """所有特征均由截至当前行的历史数据计算，避免对预警层引入未来信息。"""
    data = frame.copy()
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)

    data["MA250"] = close.rolling(cfg.YEAR_LINE_PERIOD, min_periods=cfg.YEAR_LINE_PERIOD).mean()
    data["MA5"] = close.rolling(5, min_periods=5).mean()
    data["MA10"] = close.rolling(10, min_periods=10).mean()
    data["PREV_AVG_VOL5"] = volume.shift(1).rolling(cfg.VOLUME_BASE_DAYS, min_periods=cfg.VOLUME_BASE_DAYS).mean()
    data["PREV_HIGH20"] = data["high"].shift(1).rolling(cfg.PRE_RESISTANCE_DAYS, min_periods=cfg.PRE_RESISTANCE_DAYS).max()
    previous_close = close.shift(1)
    true_range = pd.concat([
        data["high"] - data["low"],
        (data["high"] - previous_close).abs(),
        (data["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data["ATR20_PCT"] = true_range.rolling(20, min_periods=20).mean() / close

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    data["DIF"] = ema_fast - ema_slow
    data["DEA"] = data["DIF"].ewm(span=9, adjust=False).mean()
    data["MACD"] = (data["DIF"] - data["DEA"]) * 2.0
    return data


def estimate_market_cap_yi(amount_yuan: float, turnover_pct: float) -> Optional[float]:
    if not np.isfinite(amount_yuan) or not np.isfinite(turnover_pct) or amount_yuan <= 0 or turnover_pct <= 0:
        return None
    return round(amount_yuan / (turnover_pct / 100.0) / 1e8, 2)


def basic_filters(data: pd.DataFrame, index: int, min_market_cap_yi: Optional[float] = None) -> Optional[Dict[str, float]]:
    """每个层共用的流动性、价格、ST 与市值前置检查。"""
    row = data.iloc[index]
    if cfg.EXCLUDE_ST and int(row.get("isST", 0) or 0) == 1:
        return None
    close = float(row["close"])
    amount_yuan = float(row.get("amount", 0.0) or 0.0)
    turn = float(row.get("turn", 0.0) or 0.0)
    if not cfg.MIN_PRICE <= close <= cfg.MAX_PRICE:
        return None
    if amount_yuan < cfg.MIN_AMOUNT_WAN * 10_000.0:
        return None
    market_cap = estimate_market_cap_yi(amount_yuan, turn)
    required_market_cap = cfg.MIN_MARKET_CAP_YI if min_market_cap_yi is None else min_market_cap_yi
    if required_market_cap > 0:
        if market_cap is None and cfg.REQUIRE_MARKET_CAP:
            return None
        if market_cap is not None and market_cap < required_market_cap:
            return None
    return {"amount_wan": amount_yuan / 10_000.0, "turn": turn, "market_cap_yi": market_cap}


def _close_position(row: pd.Series) -> float:
    spread = float(row["high"]) - float(row["low"])
    if spread <= 0:
        return 0.5
    return float(np.clip((float(row["close"]) - float(row["low"])) / spread, 0.0, 1.0))


def calc_pre_score(data: pd.DataFrame, index: int, industry: str = "未知") -> Optional[Dict[str, Any]]:
    """层 A：在 index 当日收盘后计算“下一交易日预警”分数。

    本函数不会访问 index 之后的任何行。回测时使用相同函数，确保预警与历史评估口径一致。
    """
    min_index = max(cfg.YEAR_LINE_PERIOD, cfg.PRE_RESISTANCE_DAYS, cfg.YEAR_LINE_SLOPE_DAYS, 10)
    if index < min_index or index >= len(data):
        return None
    row = data.iloc[index]
    atr20_pct = float(row.get("ATR20_PCT", np.nan))
    profile = resolve_sector_profile(industry, atr20_pct)
    base = basic_filters(data, index, profile.min_market_cap_yi)
    if base is None:
        return None

    ma250 = row.get("MA250")
    old_ma250 = data.iloc[index - cfg.YEAR_LINE_SLOPE_DAYS].get("MA250")
    prev_high20 = row.get("PREV_HIGH20")
    prev_avg_vol = row.get("PREV_AVG_VOL5")
    ma5 = row.get("MA5")
    ma10 = row.get("MA10")
    if any(pd.isna(value) for value in (ma250, old_ma250, prev_high20, prev_avg_vol, ma5, ma10)):
        return None
    if float(ma250) <= 0 or float(old_ma250) <= 0 or float(prev_high20) <= 0 or float(prev_avg_vol) <= 0:
        return None

    close = float(row["close"])
    open_price = float(row["open"])
    volume_ratio = float(row["volume"]) / float(prev_avg_vol)
    yearline_distance_pct = (close / float(ma250) - 1.0) * 100.0
    yearline_slope_pct = (float(ma250) / float(old_ma250) - 1.0) * 100.0
    resistance_distance_pct = (float(prev_high20) - close) / float(prev_high20) * 100.0
    close_position = _close_position(row)
    pct_change = float(row.get("pctChg", 0.0) or 0.0)
    macd_rising = float(row["MACD"]) > float(data.iloc[index - 1]["MACD"])
    short_trend = float(ma5) >= float(ma10)

    # 这些是预警层的“可观测状态”，不包含“已经涨停”的未来结果或条件。
    if yearline_distance_pct < -profile.near_yearline * 100.0:
        return None
    if yearline_distance_pct > profile.max_above_yearline * 100.0:
        return None
    if yearline_slope_pct < 0:
        return None
    if resistance_distance_pct > profile.near_resistance * 100.0:
        return None
    if volume_ratio < profile.min_volume_ratio:
        return None
    if close_position < profile.min_close_position:
        return None
    if close <= open_price:
        return None
    if pct_change > profile.max_daily_rise_pct:
        return None
    if cfg.PRE_REQUIRE_SHORT_TREND and not short_trend:
        return None
    if profile.require_macd and not (macd_rising and float(row["DIF"]) >= float(row["DEA"])):
        return None

    # 组件均标准化为 0–100，之后按透明权重相加。
    yearline_score = float(np.clip(100.0 - abs(yearline_distance_pct) / max(profile.max_above_yearline * 100.0, 0.1) * 100.0, 0.0, 100.0))
    resistance_score = float(np.clip(100.0 - resistance_distance_pct / max(profile.near_resistance * 100.0, 0.1) * 100.0, 0.0, 100.0))
    volume_score = float(np.clip(volume_ratio / 2.0 * 100.0, 0.0, 100.0))
    candle_score = close_position * 100.0
    short_trend_score = 100.0 if short_trend else 35.0
    momentum_score = 100.0 if (macd_rising and float(row["DIF"]) >= float(row["DEA"])) else (65.0 if macd_rising else 35.0)
    pre_score = (
        yearline_score * cfg.WEIGHT_YEARLINE
        + resistance_score * cfg.WEIGHT_RESISTANCE
        + volume_score * cfg.WEIGHT_VOLUME
        + candle_score * cfg.WEIGHT_CANDLE
        + short_trend_score * cfg.WEIGHT_SHORT_TREND
        + momentum_score * cfg.WEIGHT_MOMENTUM
    ) / 100.0

    return {
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "price": round(close, 2),
        "pre_score": round(float(pre_score), 2),
        "yearline": round(float(ma250), 2),
        "yearline_distance_pct": round(yearline_distance_pct, 2),
        "yearline_slope_pct": round(yearline_slope_pct, 4),
        "resistance_price": round(float(prev_high20), 2),
        "resistance_distance_pct": round(resistance_distance_pct, 2),
        "volume_ratio": round(volume_ratio, 2),
        "close_position_pct": round(close_position * 100.0, 1),
        "pct_change": round(pct_change, 2),
        "macd_rising": bool(macd_rising),
        "short_trend": bool(short_trend),
        "策略档位": profile.label,
        "atr20_pct": round(atr20_pct * 100.0, 2) if np.isfinite(atr20_pct) else None,
        "profile_min_score": round(profile.min_score, 2),
        "profile_min_volume_ratio": round(profile.min_volume_ratio, 2),
        "profile_min_market_cap_yi": round(profile.min_market_cap_yi, 2),
        "score_yearline": round(yearline_score, 1),
        "score_resistance": round(resistance_score, 1),
        "score_volume": round(volume_score, 1),
        "score_candle": round(candle_score, 1),
        "score_short_trend": round(short_trend_score, 1),
        "score_momentum": round(momentum_score, 1),
        **base,
    }


def check_confirmation(data: pd.DataFrame, index: int, code: str) -> Optional[Dict[str, Any]]:
    """层 B：确认当日已经形成的涨停年线突破；不可用于提前预测。"""
    if index < cfg.YEAR_LINE_PERIOD or index >= len(data):
        return None
    base = basic_filters(data, index)
    if base is None:
        return None
    row = data.iloc[index]
    is_lu, limit_type = is_limit_up(row.get("pctChg"), code)
    if not is_lu:
        return None

    ma250 = row.get("MA250")
    old_ma250 = data.iloc[index - cfg.YEAR_LINE_SLOPE_DAYS].get("MA250")
    avg_volume = row.get("PREV_AVG_VOL5")
    if any(pd.isna(value) for value in (ma250, old_ma250, avg_volume)):
        return None
    if float(ma250) <= 0 or float(old_ma250) <= 0 or float(avg_volume) <= 0:
        return None
    if float(row["close"]) < float(ma250):
        return None

    prior_window = data.iloc[max(0, index - cfg.YEAR_LINE_BREAK_WINDOW):index]
    if prior_window.empty or not (prior_window["close"] < prior_window["MA250"]).fillna(False).any():
        return None
    volume_ratio = float(row["volume"]) / float(avg_volume)
    if volume_ratio < cfg.CONFIRM_VOLUME_RATIO:
        return None
    if float(ma250) <= float(old_ma250):
        return None

    macd_ok = float(row["DIF"]) > float(row["DEA"]) and float(row["MACD"]) > 0
    if cfg.USE_MACD_CONFIRM_FILTER and not macd_ok:
        return None

    break_pct = (float(row["close"]) / float(ma250) - 1.0) * 100.0
    score = min(max(break_pct, 0.0), 15.0) * 2.0 + min(volume_ratio, 5.0) * 8.0 + (5.0 if macd_ok else 0.0)
    return {
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "price": round(float(row["close"]), 2),
        "pct_change": round(float(row["pctChg"]), 2),
        "limit_type": limit_type,
        "yearline": round(float(ma250), 2),
        "break_pct": round(break_pct, 2),
        "yearline_slope_pct": round((float(ma250) / float(old_ma250) - 1.0) * 100.0, 4),
        "volume_ratio": round(volume_ratio, 2),
        "macd_signal": "零轴上红柱" if macd_ok else "未通过",
        "dif": round(float(row["DIF"]), 4),
        "dea": round(float(row["DEA"]), 4),
        "macd": round(float(row["MACD"]), 4),
        "confirm_score": round(score, 2),
        **base,
    }


# ================================ 回测层 ===================================
def backtest_pre_alerts(data: pd.DataFrame, code: str, name: str, industry: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """无未来函数预警回测。

    信号使用 T 日行调用 calc_pre_score；T+1 至 T+H 的价格与涨停标签仅在信号产生后
    才读取，用来评价该历史信号，而不反向影响 T 日评分。
    """
    horizon = cfg.BACKTEST_HORIZON
    start = max(cfg.YEAR_LINE_PERIOD, len(data) - cfg.BACKTEST_DAYS - horizon)
    end = len(data) - horizon
    rows: List[Dict[str, Any]] = []
    last_signal_index = -10_000

    for index in range(start, end):
        if index - last_signal_index < cfg.PRE_COOLDOWN_DAYS:
            continue
        signal_row = calc_pre_score(data, index, industry)
        if signal_row is None or float(signal_row["pre_score"]) < float(signal_row["profile_min_score"]):
            continue
        last_signal_index = index
        decision_close = float(data.iloc[index]["close"])
        future = data.iloc[index + 1:index + horizon + 1]
        day1 = future.iloc[0]
        one_day_return = (float(day1["close"]) / decision_close - 1.0) * 100.0
        max_return = (float(future["high"].max()) / decision_close - 1.0) * 100.0
        min_return = (float(future["low"].min()) / decision_close - 1.0) * 100.0
        hit_limit_1d = bool(is_limit_up(day1.get("pctChg"), code)[0])
        hit_limit_horizon = any(is_limit_up(value, code)[0] for value in future["pctChg"])
        confirmed_horizon = any(check_confirmation(data, future_index, code) is not None for future_index in range(index + 1, index + horizon + 1))

        rows.append({
            "代码": normalize_code(code),
            "名称": name,
            "板块": industry,
            "决策日期": signal_row["date"],
            "预警评分": signal_row["pre_score"],
            "决策收盘价": round(decision_close, 2),
            "未来窗口_交易日": horizon,
            "次日收益_pct": round(one_day_return, 2),
            "窗口最大涨幅_pct": round(max_return, 2),
            "窗口最大回撤_pct": round(min_return, 2),
            "次日涨停": hit_limit_1d,
            f"{horizon}日内涨停": bool(hit_limit_horizon),
            f"{horizon}日内确认": bool(confirmed_horizon),
        })

    if not rows:
        return [], {
            "代码": normalize_code(code), "名称": name, "板块": industry, "预警信号数": 0,
            "次日涨停数": 0, f"{horizon}日内涨停数": 0, f"{horizon}日内确认数": 0,
            "次日涨停率_pct": None, f"{horizon}日内涨停率_pct": None,
            "平均次日收益_pct": None, f"平均{horizon}日最大涨幅_pct": None,
            f"平均{horizon}日最大回撤_pct": None,
        }

    detail = pd.DataFrame(rows)
    summary = {
        "代码": normalize_code(code),
        "名称": name,
        "板块": industry,
        "预警信号数": len(detail),
        "次日涨停数": int(detail["次日涨停"].sum()),
        f"{horizon}日内涨停数": int(detail[f"{horizon}日内涨停"].sum()),
        f"{horizon}日内确认数": int(detail[f"{horizon}日内确认"].sum()),
        "次日涨停率_pct": round(float(detail["次日涨停"].mean() * 100.0), 2),
        f"{horizon}日内涨停率_pct": round(float(detail[f"{horizon}日内涨停"].mean() * 100.0), 2),
        "平均次日收益_pct": round(float(detail["次日收益_pct"].mean()), 2),
        f"平均{horizon}日最大涨幅_pct": round(float(detail["窗口最大涨幅_pct"].mean()), 2),
        f"平均{horizon}日最大回撤_pct": round(float(detail["窗口最大回撤_pct"].mean()), 2),
    }
    return rows, summary


# ================================ 扫描层 ===================================
def process_stock(task: Tuple[str, str, bool]) -> Dict[str, Any]:
    code, name, run_backtest = task
    try:
        raw = fetch_history(code)
        if raw is None:
            return {"kind": "no_data", "code": code}
        data = calc_indicators(raw)
        latest_index = len(data) - 1
        latest_data_date = pd.Timestamp(data.iloc[latest_index]["date"]).strftime("%Y-%m-%d")
        industry = _industry_map.get(normalize_code(code), "未知")
        pre_alert = calc_pre_score(data, latest_index, industry)
        confirmation = check_confirmation(data, latest_index, code)

        if pre_alert is not None and float(pre_alert["pre_score"]) < float(pre_alert["profile_min_score"]):
            pre_alert = None
        if pre_alert is not None:
            pre_alert = {"代码": normalize_code(code), "名称": name, "板块": industry, **pre_alert}
        if confirmation is not None:
            confirmation = {"代码": normalize_code(code), "名称": name, "板块": industry, **confirmation}

        backtest_rows: List[Dict[str, Any]] = []
        backtest_summary: Optional[Dict[str, Any]] = None
        if run_backtest:
            backtest_rows, backtest_summary = backtest_pre_alerts(data, code, name, industry)
        return {
            "kind": "ok",
            "code": code,
            "latest_data_date": latest_data_date,
            "pre_alert": pre_alert,
            "confirmation": confirmation,
            "backtest_rows": backtest_rows,
            "backtest_summary": backtest_summary,
        }
    except Exception as exc:
        return {
            "kind": "error",
            "code": code,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc() if cfg.VERBOSE else "",
        }


def scan_pool(stock_pool: Sequence[Tuple[str, str]], industry_map: Mapping[str, str], run_backtest: bool) -> Iterator[Dict[str, Any]]:
    global _industry_map
    tasks = [(code, name, run_backtest) for code, name in stock_pool]
    if cfg.NUM_PROCESSES == 1:
        _industry_map = dict(industry_map)
        for task in tasks:
            yield process_stock(task)
        return

    context = mp.get_context("spawn")
    workers = min(cfg.NUM_PROCESSES, len(tasks))
    with context.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(asdict(cfg), dict(industry_map)),
        maxtasksperchild=600,
    ) as pool:
        for result in pool.imap_unordered(process_stock, tasks, chunksize=1):
            yield result


# ================================ 输出层 ===================================
PRE_COLUMNS = [
    "代码", "名称", "板块", "date", "price", "pre_score", "yearline", "yearline_distance_pct", "yearline_slope_pct",
    "resistance_price", "resistance_distance_pct", "volume_ratio", "close_position_pct", "pct_change", "macd_rising",
    "short_trend", "策略档位", "atr20_pct", "profile_min_score", "profile_min_volume_ratio", "profile_min_market_cap_yi",
    "amount_wan", "turn", "market_cap_yi", "score_yearline", "score_resistance", "score_volume",
    "score_candle", "score_short_trend", "score_momentum",
]
CONFIRM_COLUMNS = [
    "代码", "名称", "板块", "date", "price", "pct_change", "limit_type", "yearline", "break_pct", "yearline_slope_pct",
    "volume_ratio", "macd_signal", "dif", "dea", "macd", "confirm_score", "amount_wan", "turn", "market_cap_yi",
]
BACKTEST_COLUMNS = [
    "代码", "名称", "板块", "决策日期", "预警评分", "决策收盘价", "未来窗口_交易日", "次日收益_pct", "窗口最大涨幅_pct",
    "窗口最大回撤_pct", "次日涨停", "3日内涨停", "3日内确认",
]


def save_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str], sort_by: Sequence[str], ascending: Sequence[bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reindex(columns=columns)
    if not output.empty and sort_by:
        effective_sort = [column for column in sort_by if column in output.columns]
        if effective_sort:
            output = output.sort_values(effective_sort, ascending=list(ascending)[:len(effective_sort)])
    temporary_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temporary_path, index=False, encoding="utf-8-sig", na_rep="")
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str}).where(pd.notna, None).to_dict("records")
    except Exception as exc:
        log.warning("读取阶段性结果失败，忽略该文件：%s", exc)
        return []


def checkpoint_key(stock_pool: Sequence[Tuple[str, str]], as_of: str) -> str:
    codes = ",".join(code for code, _name in stock_pool)
    fingerprint = hashlib.sha256(codes.encode("utf-8")).hexdigest()[:12]
    safe_shard = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in cfg.SHARD_ID)
    return f"yearline_v7_{as_of}_{safe_shard}_o{cfg.SCAN_OFFSET}_n{cfg.SCAN_LIMIT or len(stock_pool)}_{fingerprint}"


def checkpoint_paths(output_dir: Path, key: str) -> Dict[str, Path]:
    root = output_dir / "checkpoints"
    return {
        "state": root / f"{key}.json",
        "pre_alerts": root / f"{key}_pre_alerts.csv",
        "confirmations": root / f"{key}_confirmations.csv",
        "backtest_rows": root / f"{key}_backtest_rows.csv",
        "summaries": root / f"{key}_stock_summaries.csv",
    }


def write_checkpoint(
    paths: Mapping[str, Path],
    *,
    key: str,
    status: str,
    processed_codes: Set[str],
    planned_codes: Sequence[str],
    pre_alerts: Sequence[Mapping[str, Any]],
    confirmations: Sequence[Mapping[str, Any]],
    backtest_rows: Sequence[Mapping[str, Any]],
    stock_summaries: Sequence[Mapping[str, Any]],
    stats: Mapping[str, int],
    errors: Sequence[str],
    latest_data_dates: Mapping[str, int],
    horizon: int,
) -> None:
    save_csv(pd.DataFrame(list(pre_alerts)), paths["pre_alerts"], PRE_COLUMNS, ["pre_score", "代码"], [False, True])
    save_csv(pd.DataFrame(list(confirmations)), paths["confirmations"], CONFIRM_COLUMNS, ["confirm_score", "代码"], [False, True])
    backtest_columns = [
        "代码", "名称", "板块", "决策日期", "预警评分", "决策收盘价", "未来窗口_交易日", "次日收益_pct",
        "窗口最大涨幅_pct", "窗口最大回撤_pct", "次日涨停", f"{horizon}日内涨停", f"{horizon}日内确认",
    ]
    save_csv(pd.DataFrame(list(backtest_rows)), paths["backtest_rows"], backtest_columns, ["决策日期", "预警评分"], [False, False])
    summary_columns = [
        "代码", "名称", "板块", "预警信号数", "次日涨停数", f"{horizon}日内涨停数", f"{horizon}日内确认数",
        "次日涨停率_pct", f"{horizon}日内涨停率_pct", "平均次日收益_pct",
        f"平均{horizon}日最大涨幅_pct", f"平均{horizon}日最大回撤_pct",
    ]
    save_csv(pd.DataFrame(list(stock_summaries)), paths["summaries"], summary_columns, [f"{horizon}日内涨停率_pct", "预警信号数"], [False, False])
    atomic_write_json(paths["state"], {
        "checkpoint_key": key,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "shard_id": cfg.SHARD_ID,
        "scan_offset": cfg.SCAN_OFFSET,
        "scan_limit": cfg.SCAN_LIMIT,
        "planned_codes": list(planned_codes),
        "processed_codes": sorted(processed_codes),
        "statistics": dict(stats),
        "errors": list(errors)[-200:],
        "latest_data_date_counts": dict(latest_data_dates),
        "partial_files": {name: str(path) for name, path in paths.items() if name != "state"},
    })


def load_checkpoint(paths: Mapping[str, Path], key: str) -> Optional[Dict[str, Any]]:
    state_path = paths["state"]
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("检查点 JSON 损坏，将从头扫描当前分片：%s", exc)
        return None
    if state.get("checkpoint_key") != key:
        log.warning("检查点与当前分片不匹配，忽略旧检查点")
        return None
    return state


def aggregate_backtest(rows: Sequence[Mapping[str, Any]], horizon: int) -> Dict[str, Any]:
    if not rows:
        return {"预警信号总数": 0}
    frame = pd.DataFrame(list(rows))
    hit_column = f"{horizon}日内涨停"
    confirm_column = f"{horizon}日内确认"
    return {
        "预警信号总数": len(frame),
        "次日涨停数": int(frame["次日涨停"].sum()),
        f"{horizon}日内涨停数": int(frame[hit_column].sum()),
        f"{horizon}日内确认数": int(frame[confirm_column].sum()),
        "次日涨停率_pct": round(float(frame["次日涨停"].mean() * 100.0), 2),
        f"{horizon}日内涨停率_pct": round(float(frame[hit_column].mean() * 100.0), 2),
        "平均次日收益_pct": round(float(frame["次日收益_pct"].mean()), 2),
        f"平均{horizon}日最大涨幅_pct": round(float(frame["窗口最大涨幅_pct"].mean()), 2),
        f"平均{horizon}日最大回撤_pct": round(float(frame["窗口最大回撤_pct"].mean()), 2),
    }


def build_push_content(pre_alerts: Sequence[Mapping[str, Any]], confirmations: Sequence[Mapping[str, Any]], top: int) -> str:
    lines = [f"### 年线 v7 双层系统", "", f"**提前预警：{len(pre_alerts)} 只；涨停确认：{len(confirmations)} 只。**", ""]
    if pre_alerts:
        lines.extend(["#### T-1 收盘预警候选（非涨停预测）", "", "| 代码 | 名称 | 预警评分 | 距年线 | 距20日压力 | 量比 |", "|---|---|---:|---:|---:|---:|"])
        for row in list(pre_alerts)[:top]:
            lines.append(f"| {row['代码']} | {row['名称']} | {row['pre_score']:.2f} | {row['yearline_distance_pct']:.2f}% | {row['resistance_distance_pct']:.2f}% | {row['volume_ratio']:.2f} |")
    if confirmations:
        lines.extend(["", "#### T 日涨停确认", "", "| 代码 | 名称 | 涨跌幅 | 突破年线 | 量比 | 评分 |", "|---|---|---:|---:|---:|---:|"])
        for row in list(confirmations)[:top]:
            lines.append(f"| {row['代码']} | {row['名称']} | {row['pct_change']:.2f}% | {row['break_pct']:.2f}% | {row['volume_ratio']:.2f} | {row['confirm_score']:.2f} |")
    return "\n".join(lines)


def send_serverchan(title: str, content: str) -> None:
    send_key = os.getenv("SENDKEY")
    if not send_key:
        log.info("未设置 SENDKEY，跳过推送")
        return
    try:
        import requests
    except ImportError:
        log.warning("未安装 requests，无法推送")
        return

    lines, chunks, current, length = content.splitlines(), [], [], 0
    for line in lines:
        if current and length + len(line) + 1 > 3600:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    for index, chunk in enumerate(chunks or [""], start=1):
        page_title = title if len(chunks) <= 1 else f"{title}（{index}/{len(chunks)}）"
        try:
            response = requests.post(f"https://sctapi.ftqq.com/{send_key}.send", data={"title": page_title, "desp": chunk}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("code", -1)) != 0:
                log.warning("推送失败：%s", payload.get("message", payload))
        except Exception as exc:
            log.warning("推送异常：%s", exc)


# ================================ 命令行 ===================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="年线涨停双层预警系统 v7.0（baostock）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="扫描正常上市 A 股")
    group.add_argument("--symbols", help="指定股票代码，多个代码以英文逗号分隔")
    parser.add_argument("--as-of", help="数据截至日期 YYYY-MM-DD；默认今天")
    parser.add_argument("--processes", type=int, default=cfg.NUM_PROCESSES, help="并发进程数；公共数据源建议 1-3")
    parser.add_argument("--output-dir", default=cfg.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--min-pre-score", type=float, default=cfg.PRE_MIN_SCORE, help="预警层最低评分")
    parser.add_argument("--no-macd-confirm", action="store_true", help="关闭确认层的 MACD 零轴上红柱过滤")
    parser.add_argument("--min-cap", type=float, default=cfg.MIN_MARKET_CAP_YI, help="最小估算流通市值（亿元）；0 表示关闭")
    parser.add_argument("--allow-missing-cap", action="store_true", help="市值无法估算时仍保留股票")
    parser.add_argument("--pre-vol-ratio", type=float, default=cfg.PRE_MIN_VOLUME_RATIO, help="预警层最低量比")
    parser.add_argument("--confirm-vol-ratio", type=float, default=cfg.CONFIRM_VOLUME_RATIO, help="确认层最低量比")
    parser.add_argument("--backtest", action="store_true", help="输出预警层的无未来函数历史回测明细与汇总")
    parser.add_argument("--backtest-days", type=int, default=cfg.BACKTEST_DAYS, help="每只股票回测的最近决策交易日数")
    parser.add_argument("--backtest-horizon", type=int, choices=[1, 2, 3, 5], default=cfg.BACKTEST_HORIZON, help="回测的未来观察窗口")
    parser.add_argument("--top", type=int, default=30, help="终端和推送展示的前 N 条")
    parser.add_argument("--push", action="store_true", help="使用 SENDKEY 通过 Server 酱推送双层结果")
    parser.add_argument("--with-industry", action="store_true", help="指定代码模式下也拉取全市场行业映射；会明显增加启动时间")
    parser.add_argument("--dry-run", action="store_true", help="只校验股票池和行业映射，不执行扫描")
    parser.add_argument("--scan-offset", type=int, default=0, help="当前分片在排序股票池中的起始偏移")
    parser.add_argument("--scan-limit", type=int, default=0, help="当前分片最多扫描股票数；0表示扫描至末尾")
    parser.add_argument("--shard-id", default="single", help="分片标识，例如 a、b、c、d；用于隔离checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="每完成多少只股票保存一次checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="不读取已有checkpoint，从当前分片起点重新扫描")
    parser.add_argument("--reset-checkpoint", action="store_true", help="删除当前分片已有checkpoint及阶段性结果后重新扫描")
    parser.add_argument("--verbose", action="store_true", help="打印调试日志")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> List[str]:
    global cfg, log
    if args.processes < 1:
        raise ValueError("--processes 必须不小于 1")
    if not 0 <= args.min_pre_score <= 100:
        raise ValueError("--min-pre-score 必须在 0-100 之间")
    if args.pre_vol_ratio <= 0 or args.confirm_vol_ratio <= 0:
        raise ValueError("量比阈值必须大于 0")
    if args.min_cap < 0:
        raise ValueError("--min-cap 不能为负数")
    if args.backtest_days < 1:
        raise ValueError("--backtest-days 必须不小于 1")
    if args.top < 1:
        raise ValueError("--top 必须不小于 1")
    if args.scan_offset < 0 or args.scan_limit < 0:
        raise ValueError("--scan-offset 和 --scan-limit 不能为负数")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every 必须不小于 1")
    parse_as_of_date(args.as_of)

    cfg.NUM_PROCESSES = args.processes
    cfg.AS_OF_DATE = args.as_of
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PRE_MIN_SCORE = args.min_pre_score
    cfg.PRE_MIN_VOLUME_RATIO = args.pre_vol_ratio
    cfg.CONFIRM_VOLUME_RATIO = args.confirm_vol_ratio
    cfg.USE_MACD_CONFIRM_FILTER = not args.no_macd_confirm
    cfg.MIN_MARKET_CAP_YI = args.min_cap
    cfg.REQUIRE_MARKET_CAP = not args.allow_missing_cap
    cfg.BACKTEST_DAYS = args.backtest_days
    cfg.BACKTEST_HORIZON = args.backtest_horizon
    cfg.VERBOSE = args.verbose
    cfg.SCAN_OFFSET = args.scan_offset
    cfg.SCAN_LIMIT = args.scan_limit
    cfg.SHARD_ID = args.shard_id
    cfg.CHECKPOINT_EVERY = args.checkpoint_every
    cfg.RESUME = not args.no_resume
    log = configure_logging(args.verbose)

    if not args.symbols:
        return []
    return [item.strip() for item in args.symbols.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    specified_codes = apply_args(args)
    run_started_at = datetime.now()
    output_dir = Path(cfg.OUTPUT_DIR)
    tag = run_started_at.strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print("年线涨停双层预警系统 v7.0")
    print("层 A：T-1 收盘预警候选；层 B：T 日涨停确认；回测严格隔离未来数据。")
    print(f"数据截至：{cfg.AS_OF_DATE or '今天（数据源最近有效交易日）'} | 进程数：{cfg.NUM_PROCESSES}")
    print("=" * 100)

    full_stock_pool = get_stock_pool(specified_codes or None)
    if not full_stock_pool:
        log.error("股票池为空")
        return 2
    shard_end = cfg.SCAN_OFFSET + cfg.SCAN_LIMIT if cfg.SCAN_LIMIT > 0 else None
    stock_pool = full_stock_pool[cfg.SCAN_OFFSET:shard_end]
    if not stock_pool:
        log.error("当前分片为空：offset=%s, limit=%s, 全股票池=%s", cfg.SCAN_OFFSET, cfg.SCAN_LIMIT, len(full_stock_pool))
        return 2
    as_of_key = parse_as_of_date(cfg.AS_OF_DATE).isoformat()
    key = checkpoint_key(stock_pool, as_of_key)
    paths = checkpoint_paths(output_dir, key)
    if args.reset_checkpoint:
        for path in paths.values():
            path.unlink(missing_ok=True)
        log.info("已重置checkpoint：%s", key)
    log.info(
        "分片 %s：全池=%s，offset=%s，计划=%s，checkpoint=%s",
        cfg.SHARD_ID, len(full_stock_pool), cfg.SCAN_OFFSET, len(stock_pool), paths["state"],
    )
    # 全市场扫描默认加载行业映射；指定代码时默认跳过这一慢查询，可用 --with-industry 显式开启。
    industry_map = get_industry_map() if (args.all or args.with_industry) else {}
    log.info("行业映射：%s 条%s", len(industry_map), "" if industry_map else "（指定代码模式已跳过）")
    if args.dry_run:
        log.info("dry-run 完成，未执行股票扫描")
        return 0

    planned_codes = [code for code, _name in stock_pool]
    state = load_checkpoint(paths, key) if cfg.RESUME and not args.reset_checkpoint else None
    completed_codes: Set[str] = set()
    pre_alerts: List[Dict[str, Any]] = []
    confirmations: List[Dict[str, Any]] = []
    backtest_rows: List[Dict[str, Any]] = []
    stock_summaries: List[Dict[str, Any]] = []
    errors: List[str] = []
    latest_data_dates: Dict[str, int] = {}
    stats = {"total": len(stock_pool), "ok": 0, "no_data": 0, "error": 0}
    if state is not None:
        completed_codes = set(map(str, state.get("processed_codes", []))).intersection(planned_codes)
        pre_alerts = load_csv_records(paths["pre_alerts"])
        confirmations = load_csv_records(paths["confirmations"])
        backtest_rows = load_csv_records(paths["backtest_rows"])
        stock_summaries = load_csv_records(paths["summaries"])
        errors = [str(item) for item in state.get("errors", [])]
        latest_data_dates = {str(name): int(count) for name, count in state.get("latest_data_date_counts", {}).items()}
        previous_stats = state.get("statistics", {})
        stats.update({name: int(previous_stats.get(name, stats[name])) for name in stats})
        stats["total"] = len(stock_pool)
        log.info("恢复checkpoint：已完成=%s，待处理=%s", len(completed_codes), len(stock_pool) - len(completed_codes))

    remaining_pool = [(code, name) for code, name in stock_pool if code not in completed_codes]
    if not remaining_pool:
        log.info("当前分片已全部完成，直接使用checkpoint阶段性结果生成最终文件")

    def persist(status: str) -> None:
        write_checkpoint(
            paths,
            key=key,
            status=status,
            processed_codes=completed_codes,
            planned_codes=planned_codes,
            pre_alerts=pre_alerts,
            confirmations=confirmations,
            backtest_rows=backtest_rows,
            stock_summaries=stock_summaries,
            stats=stats,
            errors=errors,
            latest_data_dates=latest_data_dates,
            horizon=cfg.BACKTEST_HORIZON,
        )

    processed_since_checkpoint = 0
    interrupted = False
    outcomes = scan_pool(remaining_pool, industry_map, args.backtest)
    progress: Iterable[Dict[str, Any]] = tqdm(outcomes, total=len(remaining_pool), desc=f"分片{cfg.SHARD_ID}进度", ncols=92) if HAS_TQDM else outcomes
    try:
        for outcome in progress:
            processed_since_checkpoint += 1
            code = normalize_code(outcome.get("code", ""))
            kind = outcome.get("kind")
            if kind == "ok":
                completed_codes.add(code)
                stats["ok"] += 1
                data_date = str(outcome.get("latest_data_date", ""))
                if data_date:
                    latest_data_dates[data_date] = latest_data_dates.get(data_date, 0) + 1
                if outcome.get("pre_alert") is not None:
                    pre_alerts.append(outcome["pre_alert"])
                if outcome.get("confirmation") is not None:
                    confirmations.append(outcome["confirmation"])
                backtest_rows.extend(outcome.get("backtest_rows") or [])
                if outcome.get("backtest_summary") is not None:
                    stock_summaries.append(outcome["backtest_summary"])
            elif kind == "no_data":
                completed_codes.add(code)
                stats["no_data"] += 1
            else:
                # 异常代码不标记为已完成；下一次恢复会重试该代码。
                stats["error"] += 1
                if len(errors) < 200:
                    errors.append(f"{code}: {outcome.get('error', '未知异常')}")
                if args.verbose and outcome.get("traceback"):
                    log.debug("%s", outcome["traceback"])
            if processed_since_checkpoint >= cfg.CHECKPOINT_EVERY:
                persist("running")
                log.info("checkpoint已保存：完成=%s/%s", len(completed_codes), len(stock_pool))
                processed_since_checkpoint = 0
    except KeyboardInterrupt:
        interrupted = True
        log.warning("接收到中断信号，正在保存checkpoint")
    finally:
        persist("interrupted" if interrupted else "completed" if len(completed_codes) == len(stock_pool) else "partial")

    if interrupted:
        log.warning("已保存checkpoint，可使用相同分片参数恢复")
        return 130

    pre_alerts.sort(key=lambda row: (-float(row["pre_score"]), str(row["代码"])))
    confirmations.sort(key=lambda row: (-float(row["confirm_score"]), str(row["代码"])))

    pre_path = output_dir / f"年线预警_{tag}.csv"
    confirm_path = output_dir / f"年线涨停确认_{tag}.csv"
    save_csv(pd.DataFrame(pre_alerts), pre_path, PRE_COLUMNS, ["pre_score", "代码"], [False, True])
    save_csv(pd.DataFrame(confirmations), confirm_path, CONFIRM_COLUMNS, ["confirm_score", "代码"], [False, True])

    generated_files = [str(pre_path), str(confirm_path)]
    aggregate: Dict[str, Any] = {}
    if args.backtest:
        detail_path = output_dir / f"年线预警回测明细_{tag}.csv"
        summary_path = output_dir / f"年线预警回测个股汇总_{tag}.csv"
        summary_json_path = output_dir / f"年线预警回测总汇总_{tag}.json"
        # 列名根据实际 horizon 构造，保证 1/2/3/5 日均可正常输出。
        backtest_columns = [
            "代码", "名称", "板块", "决策日期", "预警评分", "决策收盘价", "未来窗口_交易日", "次日收益_pct",
            "窗口最大涨幅_pct", "窗口最大回撤_pct", "次日涨停", f"{cfg.BACKTEST_HORIZON}日内涨停", f"{cfg.BACKTEST_HORIZON}日内确认",
        ]
        save_csv(pd.DataFrame(backtest_rows), detail_path, backtest_columns, ["决策日期", "预警评分"], [False, False])
        summary_columns = [
            "代码", "名称", "板块", "预警信号数", "次日涨停数", f"{cfg.BACKTEST_HORIZON}日内涨停数", f"{cfg.BACKTEST_HORIZON}日内确认数",
            "次日涨停率_pct", f"{cfg.BACKTEST_HORIZON}日内涨停率_pct", "平均次日收益_pct",
            f"平均{cfg.BACKTEST_HORIZON}日最大涨幅_pct", f"平均{cfg.BACKTEST_HORIZON}日最大回撤_pct",
        ]
        save_csv(pd.DataFrame(stock_summaries), summary_path, summary_columns, [f"{cfg.BACKTEST_HORIZON}日内涨停率_pct", "预警信号数"], [False, False])
        aggregate = aggregate_backtest(backtest_rows, cfg.BACKTEST_HORIZON)
        atomic_write_json(summary_json_path, aggregate)
        generated_files.extend([str(detail_path), str(summary_path), str(summary_json_path)])

    metadata_path = output_dir / f"年线v7运行元数据_{tag}.json"
    metadata = {
        "system": "年线涨停双层预警系统 v7.0",
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "run_finished_at": datetime.now().isoformat(timespec="seconds"),
        "statistics": stats,
        "latest_data_date_counts": latest_data_dates,
        "pre_alert_count": len(pre_alerts),
        "confirmation_count": len(confirmations),
        "backtest_aggregate": aggregate,
        "errors": errors,
        "config": asdict(cfg),
        "files": generated_files,
        "checkpoint": {
            "key": key,
            "state_file": str(paths["state"]),
            "partial_files": {name: str(path) for name, path in paths.items() if name != "state"},
            "planned_count": len(planned_codes),
            "processed_count": len(completed_codes),
            "resumed": state is not None,
        },
        "disclaimer": "预警结果仅为技术形态候选排序，不构成涨停预测或投资建议。",
    }
    atomic_write_json(metadata_path, metadata)

    print("\n扫描统计")
    print("-" * 100)
    print(f"有效处理：{stats['ok']} | 无数据：{stats['no_data']} | 异常：{stats['error']}")
    print(f"T-1 预警候选：{len(pre_alerts)} | T 日涨停确认：{len(confirmations)}")
    print(f"预警文件：{pre_path}")
    print(f"确认文件：{confirm_path}")
    if args.backtest:
        print(f"回测汇总：{json.dumps(aggregate, ensure_ascii=False)}")
    if errors:
        print("异常样本：")
        for item in errors:
            print(f"  - {item}")

    if pre_alerts:
        print(f"\nT-1 收盘预警候选 Top {min(args.top, len(pre_alerts))}（非涨停预测）：")
        display = pd.DataFrame(pre_alerts).head(args.top)[["代码", "名称", "预警评分" if "预警评分" in pd.DataFrame(pre_alerts).columns else "pre_score", "price", "yearline_distance_pct", "resistance_distance_pct", "volume_ratio"]]
        print(display.to_string(index=False))
    if confirmations:
        print(f"\nT 日涨停确认 Top {min(args.top, len(confirmations))}：")
        display = pd.DataFrame(confirmations).head(args.top)[["代码", "名称", "pct_change", "break_pct", "volume_ratio", "confirm_score"]]
        print(display.to_string(index=False))

    if args.push:
        send_serverchan(
            f"年线 v7：预警{len(pre_alerts)}只 / 确认{len(confirmations)}只",
            build_push_content(pre_alerts, confirmations, args.top),
        )
    return 0


if __name__ == "__main__":
    # 修复说明：原上传文件止于 main() 内的 return 0，缺少此入口。
    # 缺少入口时 `python 文件名.py ...` 不会执行扫描，也不会生成输出文件。
    atexit.register(_logout_quietly)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断程序。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        log.error("程序异常终止：%s", exc)
        log.debug("详细堆栈：\n%s", traceback.format_exc())
        raise SystemExit(1)
