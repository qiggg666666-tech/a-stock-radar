#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
年线涨停选股系统  v5.2-快速容错版 (四分片版)
================================================================================
核心策略：
  1. 当日涨停
  2. 收盘价站上250日均线
  3. 前N日曾在年线下方（突破确认）
  4. 涨停日放量
  5. 年线趋势向上

【矩阵防爆改造】
  ① 强制 sorted 股票池，支持 SCAN_OFFSET/SCAN_LIMIT 分段扫描，防乱序。
  ② 新增增量存盘机制，每 200 只自动落盘 checkpoint，防 5 小时白干。
  ③ 分段后续部分自动跳过微信推送，防重复打扰。
================================================================================
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ====================== 可调参数（默认值，可被命令行覆盖） ======================
@dataclass
class Config:
    LOOKBACK_DAYS: int = 320
    MIN_LIST_DAYS: int = 250
    NUM_PROCESSES: int = int(os.getenv("NUM_WORKERS", "2"))
    REQUEST_DELAY: float = 0.10
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "1"))
    AK_QUERY_TIMEOUT_SEC: int = int(os.getenv("AK_FETCH_TIMEOUT_SEC", "12"))
    BS_QUERY_TIMEOUT_SEC: int = int(os.getenv("BS_FETCH_TIMEOUT_SEC", "8"))
    BS_FAILURE_LIMIT: int = int(os.getenv("BS_FAILURE_LIMIT", "5"))
    MAX_RUNTIME_SECONDS: int = int(os.getenv("MAX_RUNTIME_SECONDS", "19200"))
    LOGIN_STAGGER_SEC: float = 0.6

    YEAR_LINE_PERIOD: int = 250
    LIMIT_UP_PCT_MAIN: float = 9.85
    LIMIT_UP_PCT_20: float = 19.85
    YEAR_LINE_BREAK_WINDOW: int = 5
    VOLUME_SURGE_RATIO: float = 1.6
    YEAR_LINE_SLOPE_DAYS: int = 5

    USE_MACD_FILTER: bool = True
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    EXCLUDE_ST: bool = True
    EXCLUDE_KCB: bool = False
    EXCLUDE_CHUANGYE: bool = False
    EXCLUDE_BJ: bool = True
    MIN_PRICE: float = 3.0
    MAX_PRICE: float = 300.0
    MIN_MARKET_CAP: float = 15          # 亿元，估算值；0 表示不过滤
    MIN_AMOUNT: float = 5000            # 万元

    WEIGHT_BREAK: float = 2.0
    WEIGHT_VOL: float = 8.0
    WEIGHT_HIGH_OPEN: float = 0.6
    WEIGHT_FUND: float = 0.00015
    WEIGHT_MACD: float = 5.0

    OUTPUT_DIR: str = "output"
    DRY_RUN: bool = False
    VERBOSE: bool = False


cfg = Config()
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("YearLineLimitUp-BS")

_MAX_ERROR_SAMPLES = 8


@dataclass
class StockResult:
    代码: str
    名称: str
    收盘价: float
    日期: str
    涨跌幅: float
    涨停类型: str
    年线: float
    突破幅度: float
    量比: float
    成交额_万: float
    板块: str
    换手率: float
    次日高开概率: float
    历史信号次数: int
    MACD信号: str
    DIF: float
    DEA: float
    MACD: float
    综合评分: float
    上市天数: int
    流通市值_亿估算: float


# ------------------------- 代码格式转换 -------------------------
def to_bs_code(code: str) -> Optional[str]:
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    if code.startswith(("8", "4")):
        return f"bj.{code}"
    return None


# ------------------------- 无阻塞硬超时与重试 -------------------------
def call_with_timeout(func, *args, timeout: int, **kwargs):
    """不在超时后等待线程结束，避免原线程池上下文管理器隐式阻塞。"""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        raise TimeoutError(f"{func.__name__} 超时({timeout}s)")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def retry_call(func, *args, max_retries=None, base_delay=0.35, timeout=None, **kwargs):
    retries = max_retries if max_retries is not None else cfg.MAX_RETRIES
    last_exc = None
    for attempt in range(retries):
        try:
            return call_with_timeout(func, *args, timeout=timeout or cfg.BS_QUERY_TIMEOUT_SEC, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(base_delay * (attempt + 1))
    raise last_exc

# ------------------------- 子进程：登录/登出/全局状态 -------------------------
_bs = None
_bs_failures = 0
_bs_circuit_open = False
_industry_map: Dict[str, str] = {}
_error_samples: List[str] = []


def _worker_init(industry_map: Dict[str, str]):
    global _bs, _industry_map
    import baostock as bs
    import random
    _bs = bs

    time.sleep(random.uniform(0, cfg.LOGIN_STAGGER_SEC * cfg.NUM_PROCESSES))

    lg = _bs.login()
    pid = os.getpid()
    if lg.error_code != "0":
        log.error(f"[worker pid={pid}] baostock 登录失败: {lg.error_msg}")
    else:
        log.info(f"[worker pid={pid}] baostock 登录成功")

    _industry_map = industry_map

    import atexit
    def _logout():
        try:
            _bs.logout()
        except Exception:
            pass
    atexit.register(_logout)


def _fetch_industry_map() -> Dict[str, str]:
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        log.warning(f"行业映射拉取失败（登录错误）: {lg.error_msg}")
        return {}

    mapping: Dict[str, str] = {}
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            log.warning(f"行业映射拉取失败: {rs.error_msg}")
        else:
            rows = []
            while (rs.error_code == "0") and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=rs.fields)
                for _, r in df.iterrows():
                    bs_code = r.get("code", "")
                    plain = bs_code.split(".")[-1] if "." in bs_code else bs_code
                    mapping[plain] = r.get("industry", "未知") or "未知"
            log.info(f"行业映射拉取完成，共 {len(mapping)} 条")
    finally:
        bs.logout()
    return mapping


def _record_error(context: str, exc: Exception) -> None:
    if len(_error_samples) < _MAX_ERROR_SAMPLES:
        _error_samples.append(f"[{context}] {type(exc).__name__}: {exc}")
    log.debug(f"{context} 失败: {exc}\n{traceback.format_exc()}")


# ------------------------- 数据获取与指标计算 -------------------------
_HIST_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"


def _normalize_history(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "isST"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tradestatus" not in df.columns:
        df["tradestatus"] = "1"
    df = df[df["tradestatus"].astype(str) == "1"]
    return df.dropna(subset=["close", "volume", "open"]).reset_index(drop=True)


def _ak_history(symbol: str, days: int) -> Optional[pd.DataFrame]:
    if not HAS_AK:
        return None
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 80)).strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(symbol=symbol.zfill(6), period="daily", start_date=start, end_date=end, adjust="qfq")
    if raw is None or raw.empty:
        return None
    mapping = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turn", "涨跌幅": "pctChg"}
    df = raw.rename(columns={key: value for key, value in mapping.items() if key in raw.columns})
    df["preclose"] = df["close"].shift(1).fillna(df["close"])
    df["isST"] = 0
    df["tradestatus"] = "1"
    return _normalize_history(df)


def _bs_history(bs_code: str, days: int) -> Optional[pd.DataFrame]:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 80)).strftime("%Y-%m-%d")
    rs = _bs.query_history_k_data_plus(bs_code, _HIST_FIELDS, start_date=start, end_date=end, frequency="d", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"query_history_k_data_plus error: {rs.error_msg}")
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return _normalize_history(pd.DataFrame(rows, columns=rs.fields))


def fetch_stock_hist(code: str, days: int = None) -> Optional[pd.DataFrame]:
    """AkShare 主路径；BaoStock 仅在单标的主路径失败时短超时回退。"""
    global _bs_failures, _bs_circuit_open
    days = days or cfg.LOOKBACK_DAYS
    try:
        df = retry_call(_ak_history, code, days, max_retries=cfg.MAX_RETRIES, timeout=cfg.AK_QUERY_TIMEOUT_SEC)
        if df is not None and len(df) >= cfg.MIN_LIST_DAYS:
            return df
    except Exception as exc:
        _record_error(f"akshare {code}", exc)
    if _bs_circuit_open or _bs is None:
        return None
    try:
        bs_code = to_bs_code(code)
        df = retry_call(_bs_history, bs_code, days, max_retries=cfg.MAX_RETRIES, timeout=cfg.BS_QUERY_TIMEOUT_SEC)
        if df is not None and len(df) >= cfg.MIN_LIST_DAYS:
            _bs_failures = 0
            return df
    except Exception as exc:
        _bs_failures += 1
        _record_error(f"baostock {code}", exc)
        if _bs_failures >= cfg.BS_FAILURE_LIMIT:
            _bs_circuit_open = True
            log.warning("BaoStock 连续失败达到阈值，本 worker 已关闭备用回退")
    return None

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    df["MA250"] = close.rolling(cfg.YEAR_LINE_PERIOD, min_periods=200).mean()
    df["MA5_VOL"] = volume.rolling(5, min_periods=5).mean()

    ema_fast = close.ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    df["DIF"] = dif
    df["DEA"] = dea
    df["MACD"] = (dif - dea) * 2
    return df


def is_limit_up(pct: float, code: str) -> Tuple[bool, str]:
    if pd.isna(pct):
        return False, ""
    code = str(code).zfill(6)
    if code.startswith(("300", "301", "688", "689", "8", "4")):
        if pct >= cfg.LIMIT_UP_PCT_20:
            if code.startswith(("300", "301")):
                return True, "创业板涨停"
            elif code.startswith(("688", "689")):
                return True, "科创板涨停"
            else:
                return True, "北交所涨停"
    else:
        if pct >= cfg.LIMIT_UP_PCT_MAIN:
            return True, "主板涨停"
    return False, ""


def check_condition_at_index(df: pd.DataFrame, idx: int, code: str) -> bool:
    if idx < cfg.YEAR_LINE_PERIOD + 10 or idx >= len(df) - 1:
        return False

    row = df.iloc[idx]
    pct = row.get("pctChg", 0.0)
    is_lu, _ = is_limit_up(pct, code)
    if not is_lu:
        return False

    ma250 = row["MA250"]
    close = row["close"]
    if pd.isna(ma250) or ma250 <= 0 or close < ma250:
        return False

    start = max(0, idx - cfg.YEAR_LINE_BREAK_WINDOW)
    window = df.iloc[start:idx]
    if not any(window["close"] < window["MA250"]):
        return False

    ma5_vol = row["MA5_VOL"]
    if pd.isna(ma5_vol) or ma5_vol <= 0:
        return False
    if row["volume"] / ma5_vol < cfg.VOLUME_SURGE_RATIO:
        return False

    prev_idx = max(0, idx - cfg.YEAR_LINE_SLOPE_DAYS)
    ma250_prev = df.iloc[prev_idx]["MA250"]
    if pd.isna(ma250_prev) or ma250 <= ma250_prev:
        return False

    return True


def calc_next_day_high_open_prob(df: pd.DataFrame, code: str) -> Tuple[float, int]:
    signals = 0
    high_open = 0
    start_idx = max(cfg.YEAR_LINE_PERIOD + 20, len(df) - 500)

    for i in range(start_idx, len(df) - 1):
        if check_condition_at_index(df, i, code):
            signals += 1
            next_open = df.iloc[i + 1]["open"]
            today_close = df.iloc[i]["close"]
            if next_open > today_close:
                high_open += 1

    if signals == 0:
        return 50.0, 0
    return round(high_open / signals * 100, 1), signals


def check_year_line_break(df: pd.DataFrame, code: str) -> Optional[Dict]:
    if len(df) < cfg.YEAR_LINE_PERIOD + 15:
        return None

    df = calc_indicators(df)
    latest = df.iloc[-1]

    if cfg.EXCLUDE_ST and latest.get("isST", 0) == 1:
        return None

    pct = latest.get("pctChg", 0.0)
    is_lu, limit_type = is_limit_up(pct, code)
    if not is_lu:
        return None

    ma250 = latest["MA250"]
    close = latest["close"]
    if pd.isna(ma250) or ma250 <= 0 or close < ma250:
        return None

    window = df.iloc[-(cfg.YEAR_LINE_BREAK_WINDOW + 1):-1]
    if not any(window["close"] < window["MA250"]):
        return None

    ma5_vol = latest["MA5_VOL"]
    if pd.isna(ma5_vol) or ma5_vol <= 0:
        return None
    vol_ratio = latest["volume"] / ma5_vol
    if vol_ratio < cfg.VOLUME_SURGE_RATIO:
        return None

    ma250_prev = df.iloc[-cfg.YEAR_LINE_SLOPE_DAYS]["MA250"]
    if pd.isna(ma250_prev) or ma250 <= ma250_prev:
        return None

    break_pct = (close - ma250) / ma250 * 100

    return {
        "limit_type": limit_type,
        "ma250": round(float(ma250), 2),
        "break_pct": round(float(break_pct), 2),
        "vol_ratio": round(float(vol_ratio), 2),
        "pct": round(float(pct), 2),
        "turn": round(float(latest.get("turn", 0) or 0), 2),
        "amount": float(latest.get("amount", 0) or 0),
        "DIF": round(float(latest["DIF"]), 3),
        "DEA": round(float(latest["DEA"]), 3),
        "MACD": round(float(latest["MACD"]), 3),
        "df": df,
    }


def check_macd_filter(df: pd.DataFrame) -> Tuple[bool, str]:
    if len(df) < 40:
        return False, "无"
    latest = df.iloc[-1]
    if latest["DIF"] > latest["DEA"] and latest["MACD"] > 0:
        return True, "零轴上红柱"
    return False, "无"


def estimate_market_cap(amount_yuan: float, turn_pct: float) -> Optional[float]:
    if not turn_pct or turn_pct <= 0:
        return None
    return round(amount_yuan / (turn_pct / 100) / 1e8, 2)


def calc_score(break_pct: float, vol_ratio: float, high_open_prob: float, macd_ok: bool) -> float:
    score = 0.0
    score += break_pct * cfg.WEIGHT_BREAK
    score += vol_ratio * cfg.WEIGHT_VOL
    score += (high_open_prob - 50) * cfg.WEIGHT_HIGH_OPEN
    if macd_ok:
        score += cfg.WEIGHT_MACD
    return round(score, 2)


# ------------------------- 股票池 -------------------------
def get_stock_pool() -> List[Tuple[str, str]]:
    """AkShare 主路径获取股票池；BaoStock 仅在主路径失败时回退。"""
    if HAS_AK:
        try:
            spot = ak.stock_zh_a_spot_em()
            work = spot[["代码", "名称"]].copy()
            work["代码"] = work["代码"].astype(str).str.zfill(6)
            work = work[~work["名称"].astype(str).str.contains(r"ST|退", na=False, regex=True)]
            work = work[work["代码"].str.match(r"^(00|30|60|68|8|4)", na=False)]
            if cfg.EXCLUDE_KCB:
                work = work[~work["代码"].str.startswith(("688", "689"))]
            if cfg.EXCLUDE_CHUANGYE:
                work = work[~work["代码"].str.startswith(("300", "301"))]
            if cfg.EXCLUDE_BJ:
                work = work[~work["代码"].str.startswith(("8", "4"))]
            pool = list(zip(work["代码"], work["名称"]))
            if pool:
                log.info(f"AkShare 股票池完成: {len(pool)} 只")
                return pool
        except Exception as exc:
            log.warning(f"AkShare 股票池失败，尝试 BaoStock: {exc}")
    import baostock as bs
    try:
        login = bs.login()
        if login.error_code != "0":
            return []
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[(df["type"] == "1") & (df["status"] == "1")]
        df = df[~df["code_name"].astype(str).str.contains(r"ST|退", na=False, regex=True)]
        df["plain_code"] = df["code"].str.split(".").str[-1]
        df = df[df["plain_code"].str.match(r"^(00|30|60|68|8|4)", na=False)]
        return list(zip(df["plain_code"], df["code_name"]))
    except Exception as exc:
        log.warning(f"BaoStock 股票池失败: {exc}")
        return []
    finally:
        try:
            bs.logout()
        except Exception:
            pass


# ------------------------- 单只股票处理（子进程内执行） -------------------------

# ------------------------- 单只股票处理（子进程内执行） -------------------------
def process_one_stock(code: str, name: str) -> Optional[dict]:
    try:
        if to_bs_code(code) is None:
            return None

        df = fetch_stock_hist(code)
        if df is None or len(df) < cfg.MIN_LIST_DAYS:
            return None

        latest_raw = df.iloc[-1]
        price = float(latest_raw["close"])
        if not (cfg.MIN_PRICE <= price <= cfg.MAX_PRICE):
            return None
        amount_yuan = float(latest_raw.get("amount", 0) or 0)
        if amount_yuan < cfg.MIN_AMOUNT * 10000:
            return None

        result = check_year_line_break(df, code)
        if result is None:
            return None

        df = result["df"]

        if cfg.MIN_MARKET_CAP > 0:
            cap = estimate_market_cap(result["amount"], result["turn"])
            if cap is not None and cap < cfg.MIN_MARKET_CAP:
                return None
        else:
            cap = estimate_market_cap(result["amount"], result["turn"])

        macd_ok = False
        macd_signal = "关闭"
        if cfg.USE_MACD_FILTER:
            macd_ok, macd_signal = check_macd_filter(df)
            if not macd_ok:
                return None

        high_open_prob, signal_count = calc_next_day_high_open_prob(df, code)
        industry = _industry_map.get(code, "未知")

        score = calc_score(result["break_pct"], result["vol_ratio"], high_open_prob, macd_ok)

        latest = df.iloc[-1]

        res = StockResult(
            代码=code, 名称=name,
            收盘价=round(float(latest["close"]), 2),
            日期=str(latest.get("date", ""))[:10],
            涨跌幅=result["pct"],
            涨停类型=result["limit_type"],
            年线=result["ma250"],
            突破幅度=result["break_pct"],
            量比=result["vol_ratio"],
            成交额_万=round(amount_yuan / 10000, 1),
            板块=industry,
            换手率=result["turn"],
            次日高开概率=high_open_prob,
            历史信号次数=signal_count,
            MACD信号=macd_signal,
            DIF=result["DIF"], DEA=result["DEA"], MACD=result["MACD"],
            综合评分=score,
            上市天数=len(df),
            流通市值_亿估算=cap if cap is not None else -1.0,
        )
        return asdict(res)
    except Exception as e:
        return {"__error__": f"{code} {name}: {type(e).__name__}: {e}"}


# ------------------------- 输出（仅 CSV，不生成 Excel） -------------------------
def save_checkpoint(results: List[dict], output_dir: Path, today: str, processed: int, total: int, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if frame.empty:
        frame = pd.DataFrame(columns=[f.name for f in fields(StockResult)])
    else:
        frame = frame.sort_values(by="综合评分", ascending=False).reset_index(drop=True)
    frame.to_csv(output_dir / f"年线涨停_评分_{today}_checkpoint_{processed}.csv", index=False, encoding="utf-8-sig")
    status = {"processed": processed, "total": total, "hits": len(results), "reason": reason, "saved_at": datetime.now().isoformat()}
    (output_dir / f"年线涨停_{today}_progress.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"💾 检查点: {processed}/{total}，命中 {len(results)}，原因: {reason}")


def _process_task(task):
    code, name = task
    return code, process_one_stock(code, name)


def save_results(df: pd.DataFrame, output_dir: Path, today: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"年线涨停_评分_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"📄 CSV 已保存: {csv_path}")


def send_wechat_push(title: str, content_md: str) -> None:
    """推送全部内容；过长时自动分页，避免被截断。"""
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        log.info("未设置 SENDKEY 环境变量，跳过 WeChat 推送")
        return
    import requests
    LIMIT = 3800
    chunks, cur, cur_len = [], [], 0
    for ln in content_md.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += lnlen
    if cur:
        chunks.append("\n".join(cur))
    chunks = chunks or [""]
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        try:
            resp = requests.post(
                f"https://sctapi.ftqq.com/{sendkey}.send",
                data={"title": t, "desp": ch},
                timeout=15,
            )
            if resp.status_code == 200:
                log.info(f"📲 WeChat 推送成功 ({i+1}/{len(chunks)})")
            else:
                log.warning(f"WeChat 推送失败: HTTP {resp.status_code}")
        except Exception as e:
            log.warning(f"WeChat 推送异常: {e}")
        if i < len(chunks) - 1:
            time.sleep(1)


def build_push_content(df: pd.DataFrame) -> str:
    """推送全部结果（不限 Top N）。"""
    lines = [f"**共选出 {len(df)} 只（全部）：**", ""]
    lines.append("| 代码 | 名称 | 涨跌幅 | 突破幅度 | 评分 |")
    lines.append("|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['代码']} | {r['名称']} | {r['涨跌幅']}% | {r['突破幅度']}% | {r['综合评分']} |")
    return "\n".join(lines)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║          年线涨停选股系统  v5.2-快速容错版 (四分片版)           ║
║  核心：年线突破 + 涨停 + 放量 + 年线向上                           ║
║  数据源：baostock | 多进程+错峰登录+查询超时保护+增量存盘          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"年线周期 : {cfg.YEAR_LINE_PERIOD}日 | 放量≥{cfg.VOLUME_SURGE_RATIO}倍 | MACD: {'开' if cfg.USE_MACD_FILTER else '关'}")
    print("-" * 66)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="年线涨停选股系统 v5.2-快速容错版")
    p.add_argument("--processes", type=int, default=cfg.NUM_PROCESSES, help="并发进程数（每进程独立登录 baostock）")
    p.add_argument("--output-dir", type=str, default=cfg.OUTPUT_DIR, help="结果输出目录")
    p.add_argument("--no-macd", action="store_true", help="关闭 MACD 过滤")
    p.add_argument("--min-cap", type=float, default=cfg.MIN_MARKET_CAP, help="最小估算流通市值(亿)，0=不过滤")
    p.add_argument("--vol-ratio", type=float, default=cfg.VOLUME_SURGE_RATIO, help="放量倍数阈值")
    p.add_argument("--push", action="store_true", help="推送结果到 WeChat（需设置 SENDKEY 环境变量）")
    p.add_argument("--query-timeout", type=int, default=cfg.BS_QUERY_TIMEOUT_SEC, help="BaoStock备用查询超时(秒)")
    p.add_argument("--dry-run", action="store_true", help="只跑股票池过滤+行业映射，不做全量扫描")
    p.add_argument("--verbose", action="store_true", help="输出 DEBUG 级别日志")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.NUM_PROCESSES = max(1, args.processes)
    
    # 矩阵接入：支持环境变量覆盖输出目录
    cfg.OUTPUT_DIR = os.environ.get("OUTPUT_DIR", args.output_dir)
    cfg.USE_MACD_FILTER = not args.no_macd
    cfg.MIN_MARKET_CAP = args.min_cap
    cfg.VOLUME_SURGE_RATIO = args.vol_ratio
    cfg.BS_QUERY_TIMEOUT_SEC = args.query_timeout
    cfg.DRY_RUN = args.dry_run
    cfg.VERBOSE = args.verbose

    if args.verbose:
        log.setLevel(logging.DEBUG)

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d_%H%M")

    print_banner()

    stock_list = get_stock_pool()
    if not stock_list:
        log.error("股票池为空")
        return

    # 【矩阵接入】保证顺序绝对固定，防止分段乱序
    stock_list = sorted(stock_list, key=lambda x: x[0])
    
    # 【矩阵接入】分段扫描逻辑
    scan_offset = int(os.environ.get("SCAN_OFFSET", "0"))
    scan_limit = int(os.environ.get("SCAN_LIMIT", "0"))
    
    if scan_offset > 0:
        stock_list = stock_list[scan_offset:]
        log.info(f"🚀 分段扫描: 跳过前 {scan_offset} 只, 剩余 {len(stock_list)} 只")
    if scan_limit > 0:
        stock_list = stock_list[:scan_limit]
        log.info(f"🚀 限制扫描: 本段最多扫描 {scan_limit} 只, 实际待扫 {len(stock_list)} 只")

    industry_map = _fetch_industry_map()

    if cfg.DRY_RUN:
        log.info(f"✅ --dry-run 模式：股票池 {len(stock_list)} 只，行业映射 {len(industry_map)} 条，未执行扫描")
        return

    log.info(f"[2/5] 扫描 {len(stock_list)} 只股票 | {cfg.NUM_PROCESSES} 进程 | AkShare主路径、短超时与检查点保护")
    results: List[dict] = []
    error_samples: List[str] = []
    total = len(stock_list)
    processed_count = 0
    save_interval = 25
    deadline = time.monotonic() + cfg.MAX_RUNTIME_SECONDS if cfg.MAX_RUNTIME_SECONDS > 0 else None
    timed_out = False

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=cfg.NUM_PROCESSES, initializer=_worker_init, initargs=(industry_map,))
    pbar = tqdm(total=total, desc="扫描进度", ncols=90, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    try:
        for code, res in pool.imap_unordered(_process_task, stock_list, chunksize=1):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                save_checkpoint(results, output_dir, today, processed_count, total, "安全运行预算到达")
                log.warning("达到安全运行预算，终止未开始任务并保留结果")
                break
            processed_count += 1
            pbar.update(1)
            if res:
                if "__error__" in res:
                    if len(error_samples) < _MAX_ERROR_SAMPLES:
                        error_samples.append(res["__error__"])
                else:
                    results.append(res)
            if processed_count % save_interval == 0:
                save_checkpoint(results, output_dir, today, processed_count, total, "定期保存")
    finally:
        pbar.close()
        if timed_out:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    save_checkpoint(results, output_dir, today, processed_count, total, "完成" if not timed_out else "预算退出")
    log.info(f"[3/5] 扫描完成 | 命中 {len(results)} 只")
    if error_samples:
        log.info(f"异常样本（前 {len(error_samples)} 条，--verbose 可看更多细节）：")
        for s in error_samples:
            log.info(f"  · {s}")

    print("\n[4/5] 结果输出（按综合评分从高到低）")
    print("=" * 66)

    if not results:
        print("\n😔 今日没有符合条件的股票")
        print("   建议：--no-macd 或 --vol-ratio 调低（如 1.3），或 --min-cap 0 关闭市值过滤")
        empty_df = pd.DataFrame(columns=[f.name for f in fields(StockResult)])
        save_results(empty_df, output_dir, today)
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="综合评分", ascending=False).reset_index(drop=True)

    display_cols = [
        "代码", "名称", "收盘价", "涨跌幅", "突破幅度", "量比",
        "次日高开概率", "历史信号次数", "板块", "换手率", "流通市值_亿估算", "综合评分"
    ]
    print(f"\n✅ 共选出 {len(df)} 只股票：\n")
    safe_display_cols = [c for c in display_cols if c in df.columns]
    print(df[safe_display_cols].to_string(index=False))

    save_results(df, output_dir, today)

    print("\n" + "=" * 66)
    print("📈 选股统计摘要")
    print("-" * 66)
    if '涨停类型' in df.columns:
        print(f"  主板涨停       : {(df['涨停类型'].str.contains('主板', na=False)).sum()} 只")
        print(f"  创业板涨停     : {(df['涨停类型'].str.contains('创业板', na=False)).sum()} 只")
        print(f"  科创板涨停     : {(df['涨停类型'].str.contains('科创板', na=False)).sum()} 只")
    if '突破幅度' in df.columns:
        print(f"  平均突破幅度   : {df['突破幅度'].mean():.2f}%")
    if '量比' in df.columns:
        print(f"  平均量比       : {df['量比'].mean():.2f}")
    if '次日高开概率' in df.columns:
        print(f"  平均次日高开概率: {df['次日高开概率'].mean():.1f}%")
    if '综合评分' in df.columns:
        print(f"  平均综合评分   : {df['综合评分'].mean():.1f}")
    print("=" * 66)
    print("提示：流通市值为估算值（成交额/换手率推算），仅供粗筛参考；")
    print("评分同样仅供排序参考，请结合大盘、板块热度、基本面综合决策。")

    # 【矩阵接入】如果是分段扫描的后续部分，不推送，防止重复推送
    if args.push and scan_offset == 0:
        content = build_push_content(df)
        send_wechat_push(f"年线涨停选股 · 命中{len(df)}只", content)
    elif args.push and scan_offset > 0:
        log.info("分段扫描的后续部分，跳过微信推送，结果已落盘。")


if __name__ == "__main__":
    main()
