#!/usr/bin/env python3
"""
实战选股完整版 —— 两路分打 + 分开推送 + 多指数RSRS + 名称补全
============================================================
硬过滤：红筹码 + 获利盘40%~85% + MA20溢价≤18%
A路 宽幅启动 / B路 红筹尖峰 → 各自CSV、各自Server酱
择时：上证/深成/创业板/沪深300
多周期：日线重采样周/月线 → mtf_label / mtf_risk / 小幅加减分
  --mtf-filter off|soft|strict|resonate（默认 off）
  resonate = 周线MA20上 + 日线溢价>=-3% + 月线未极端高位破位

用法：
  python live_stock_selector.py --shard 1
  python live_stock_selector.py --shard all --mtf-filter soft
  python live_stock_selector.py --merge --top 15 --notify --mtf-filter resonate

两段式（推荐）：
  python mtf_resonate_universe.py --shard all
  python live_stock_selector.py --shard all --universe mtf_resonate_universe.csv
  python live_stock_selector.py --merge --top 15 --notify
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import akshare as ak
import numpy as np
import pandas as pd
import requests

try:
    from final_chip_research import analyze, fetch_ohlcv, is_beijing_stock
    FULL_CHIP_AVAILABLE = True
    print("已加载完整 final_chip_research 模块")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：未找到 final_chip_research.py，筹码过滤将失效")

CACHE_DIR = Path("./live_selector_cache")
CACHE_DIR.mkdir(exist_ok=True)

SHARD_MAP = {
    1: ["000", "001"],
    2: ["002"],
    3: ["300"],
    4: ["600"],
    5: ["601"],
    6: ["603"],
    7: ["605", "688"],
    8: ["8", "4", "9"],
}

TOP_N_DEFAULT = 15
PROFIT_MIN = 40.0
PROFIT_MAX = 85.0
TURN_OVER_CAP = 0.15
MIN_BARS = 100
MA20_MAX_PREMIUM = 0.18
INDEX_CRASH_THRESHOLD = -0.03
PEAK_PROFIT_MIN = 70.0
# 多周期硬过滤：
# off=只标注加减分; soft=剔走坏; strict=周线偏多;
# resonate=多周期共振（周线多+日线未深破+月线未极端高位破位）
MTF_FILTER_DEFAULT = "off"

WEIGHTS_WIDE = {
    "wide_score": 0.35,
    "year_first_red": 0.20,
    "VwapClose": 0.15,
    "HighOpen": 0.10,
    "CloseLow": 0.10,
    "Turnover": 0.10,
}

WEIGHTS_PEAK = {
    "profit_pct": 0.30,
    "inv_wide": 0.25,
    "HighOpen": 0.15,
    "VwapClose": 0.15,
    "CloseLow": 0.10,
    "Turnover": 0.05,
}

RSRS_N = 18
RSRS_M = 600
RSRS_BUY_THRESHOLD = 0.7

INDEX_LIST = [
    ("上证综指", "sh000001"),
    ("深证成指", "sz399001"),
    ("创业板指", "sz399006"),
    ("沪深300", "sh000300"),
]

_NAME_MAP: dict[str, str] | None = None


def get_name_map() -> dict[str, str]:
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP
    _NAME_MAP = {}
    try:
        info = ak.stock_info_a_code_name()
        if info is not None and not info.empty:
            codes = info["code"].astype(str).str.zfill(6)
            names = info["name"].astype(str)
            _NAME_MAP = dict(zip(codes, names))
            print(f"名称表加载 {len(_NAME_MAP)} 只")
    except Exception as e:
        print(f"名称表加载失败: {e}")
    return _NAME_MAP


def fill_names(df: pd.DataFrame) -> pd.DataFrame:
    """把 name 为空白/nan 的行用 akshare 名称表补全。"""
    if df.empty:
        return df
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 先统一成普通 str，避免 pandas StringDtype 拒绝 float NaN
    if "name" not in df.columns:
        df["name"] = ""
    df["name"] = df["name"].map(
        lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    )
    df["name"] = df["name"].replace({"nan": "", "None": "", "NaN": ""})
    need = df["name"].str.strip() == ""
    if need.any():
        name_map = get_name_map()
        mapped = df.loc[need, "code"].map(
            lambda c: str(name_map.get(str(c).zfill(6), "") or "")
        )
        df.loc[need, "name"] = mapped.values
    df["name"] = df["name"].fillna("").astype(str)
    return df


def get_all_a_stocks() -> list[str]:
    codes = []
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == "0":
            rs = bs.query_stock_basic()
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                code = row[0]
                stock_type = row[4] if len(row) > 4 else "1"
                if stock_type == "1":
                    pure_code = code.split(".")[-1] if "." in code else code
                    codes.append(pure_code.zfill(6))
            bs.logout()
            if codes:
                codes = sorted(list(set(codes)))
                print(f"baostock 获取到 {len(codes)} 只股票")
                return codes
    except Exception as e:
        print(f"baostock 获取股票列表失败: {e}")

    try:
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        if codes:
            print(f"akshare 获取到 {len(codes)} 只股票")
            return codes
    except Exception as e:
        print(f"akshare 获取股票列表失败: {e}")

    print("错误：无法获取全市场股票列表")
    return []


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def load_universe(path: str | Path) -> list[str]:
    """从共振池 CSV 读取代码列表（需含 code 列，或单列代码）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"universe 文件不存在: {p}")
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        df = pd.read_csv(p, header=None, dtype=str)
    if df.empty:
        return []
    if "code" in df.columns:
        codes = df["code"].astype(str)
    else:
        codes = df.iloc[:, 0].astype(str)
    codes = (
        codes.str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )
    codes = sorted({c for c in codes.tolist() if c and c != "000000"})
    return codes


def get_stock_data(symbol: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if FULL_CHIP_AVAILABLE:
        try:
            tmp, source, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if tmp is not None and not tmp.empty:
                tmp = tmp.copy()
                tmp["symbol"] = symbol
                df = tmp
        except Exception:
            pass

    if df.empty or len(df) < 200:
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y%m%d")
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            if raw is not None and not raw.empty:
                raw = raw.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover",
                })
                raw["date"] = pd.to_datetime(raw["date"])
                raw["symbol"] = symbol
                df = raw.sort_values("date").reset_index(drop=True)
                time.sleep(0.2)
        except Exception:
            pass

    return df if (df is not None and not df.empty) else pd.DataFrame()


def calc_launch_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HighOpen"] = df["high"] / df["open"] - 1
    df["CloseLow"] = df["close"] / df["low"] - 1
    df["vwap"] = np.where(
        (df["volume"] > 0) & (df.get("amount", 0) > 0),
        df["amount"] / (df["volume"] * 100 + 1e-8),
        (df["high"] + df["low"] + df["close"]) / 3,
    )
    df["VwapClose"] = df["vwap"] / df["close"] - 1
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    df["Turnover"] = df.get("turnover", 0) / 100.0

    df["ma20"] = df["close"].rolling(20, min_periods=15).mean()
    df["ma20_premium"] = df["close"] / df["ma20"] - 1

    df["ma250"] = df["close"].rolling(250, min_periods=180).mean()
    df["above_ma250"] = df["close"] > df["ma250"]
    prev_above = df["above_ma250"].shift(1).fillna(False)
    five_days_ago_below = (df["close"].shift(5) < df["ma250"].shift(5)).fillna(False)
    df["year_first_red"] = (
        df["above_ma250"] & (prev_above == False) & five_days_ago_below
    ).astype(float)
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日线 → 周/月线（用日期索引重采样）。"""
    x = df.set_index("date").sort_index()
    ohlc = pd.DataFrame(
        {
            "open": x["open"].resample(rule).first(),
            "high": x["high"].resample(rule).max(),
            "low": x["low"].resample(rule).min(),
            "close": x["close"].resample(rule).last(),
            "volume": x["volume"].resample(rule).sum() if "volume" in x.columns else 0,
        }
    ).dropna(subset=["close"])
    return ohlc


def calc_multi_timeframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    多周期结构（研究用，默认不硬过滤，只输出标签与加分）。
    - 周线 MA20：波段方向
    - 月线相对近 12 月高低位：是否冲高过猛
    - 日线 vs 周线：是否日线弱于周线（回撤）
    """
    out: dict[str, Any] = {
        "w_ma20": None,
        "w_ma20_premium": None,
        "w_above_ma20": None,
        "m_close": None,
        "m_pos_12": None,  # 近12月区间位置 0~1
        "m_stretch": None,  # 相对近12月低点涨幅
        "mtf_label": "数据不足",
        "mtf_bonus": 0.0,
        "mtf_risk": "",
    }
    if df is None or len(df) < 60:
        return out

    try:
        daily = df.sort_values("date").copy()
        daily["date"] = pd.to_datetime(daily["date"])
        close = float(daily["close"].iloc[-1])

        # ----- 周线 -----
        weekly = _resample_ohlc(daily, "W-FRI")
        if len(weekly) >= 20:
            weekly["ma20"] = weekly["close"].rolling(20, min_periods=15).mean()
            w_ma20 = float(weekly["ma20"].iloc[-1])
            w_close = float(weekly["close"].iloc[-1])
            if np.isfinite(w_ma20) and w_ma20 > 0:
                out["w_ma20"] = round(w_ma20, 2)
                out["w_ma20_premium"] = round(w_close / w_ma20 - 1, 4)
                out["w_above_ma20"] = bool(w_close > w_ma20)

        # ----- 月线 -----
        monthly = _resample_ohlc(daily, "ME")
        if len(monthly) >= 6:
            m_close = float(monthly["close"].iloc[-1])
            out["m_close"] = round(m_close, 2)
            tail = monthly.tail(12)
            lo = float(tail["low"].min())
            hi = float(tail["high"].max())
            if hi > lo > 0:
                out["m_pos_12"] = round((m_close - lo) / (hi - lo), 4)
            if lo > 0:
                out["m_stretch"] = round(m_close / lo - 1, 4)

        # ----- 综合标签 -----
        d_prem = float(daily["close"].iloc[-1] / daily["close"].rolling(20, min_periods=15).mean().iloc[-1] - 1) if len(daily) >= 20 else 0.0
        w_up = out.get("w_above_ma20")
        m_pos = out.get("m_pos_12")
        m_stretch = out.get("m_stretch")

        # 风险：月线近一年位置很高 + 日线已跌破日MA20 → 冲高回撤
        risk_bits = []
        if m_pos is not None and m_pos >= 0.75 and d_prem < 0:
            risk_bits.append("月线高位回撤")
        if m_stretch is not None and m_stretch >= 1.0 and d_prem < 0:
            risk_bits.append("较年内低点已翻倍级回落中")
        if w_up is False and d_prem < 0:
            risk_bits.append("周线日线均在MA20下")

        bonus = 0.0
        if w_up is True and d_prem >= -0.03:
            # 周线偏多且日线未深度跌破
            bonus += 0.15
        if w_up is True and d_prem < -0.02 and d_prem > -0.12:
            # 周线仍强、日线小回踩 —— 常见观察结构
            bonus += 0.10
        if m_pos is not None and m_pos >= 0.85 and d_prem < -0.05:
            # 月线过高且日线已弱，降分
            bonus -= 0.15

        if w_up is True and (m_pos is None or m_pos < 0.85) and d_prem >= -0.05:
            label = "周线偏多·日线未过弱"
        elif w_up is True and d_prem < -0.05:
            label = "周线偏多·日线回踩"
        elif w_up is False and d_prem < 0:
            label = "周月偏弱·日线调整"
        elif risk_bits:
            label = "冲高回撤观察"
        else:
            label = "多周期中性"

        out["mtf_label"] = label
        out["mtf_bonus"] = round(bonus, 3)
        out["mtf_risk"] = "；".join(risk_bits)
    except Exception:
        pass
    return out


def pass_mtf_filter(mtf: dict[str, Any], daily_premium: float, mode: str) -> bool:
    """多周期硬过滤。mode: off / soft / strict / resonate。"""
    mode = (mode or "off").strip().lower()
    if mode in ("", "off", "none", "0"):
        return True

    w_up = mtf.get("w_above_ma20")
    m_pos = mtf.get("m_pos_12")
    m_stretch = mtf.get("m_stretch")

    if mode == "soft":
        # 周线在MA20下且日线也明显偏弱
        if w_up is False and daily_premium < -0.05:
            return False
        # 月线近一年高位 + 日线已破位
        if m_pos is not None and m_pos >= 0.85 and daily_premium < -0.05:
            return False
        return True

    if mode == "strict":
        # 必须周线在MA20上（数据不足则不通过，避免误放行）
        if w_up is not True:
            return False
        if m_pos is not None and m_pos >= 0.85 and daily_premium < -0.03:
            return False
        if m_stretch is not None and m_stretch >= 1.5 and daily_premium < -0.05:
            return False
        return True

    if mode in ("resonate", "resonance", "共振"):
        # 多周期共振：
        # 1) 周线收盘在周MA20上
        # 2) 日线MA20溢价 >= -3%（未深度跌破）
        # 3) 月线未处于「近一年极高位 + 日线已弱」
        if w_up is not True:
            return False
        if daily_premium < -0.03:
            return False
        if m_pos is not None and m_pos >= 0.85 and daily_premium < 0:
            return False
        if m_stretch is not None and m_stretch >= 1.5 and daily_premium < 0:
            return False
        return True

    # 未知模式当作 off
    return True


def apply_launch_filter(
    df: pd.DataFrame, code: str, mtf_filter: str = MTF_FILTER_DEFAULT
) -> dict[str, Any]:
    if not FULL_CHIP_AVAILABLE or df.empty or len(df) < MIN_BARS:
        return {}
    try:
        df = calc_launch_factors(df)
        result = analyze(code, "", df)

        if not result.get("is_red_heavy_chip"):
            return {}

        profit = float(result.get("profit_pct") or 0)
        if not (PROFIT_MIN <= profit <= PROFIT_MAX):
            return {}

        latest = df.iloc[-1]
        premium = float(latest.get("ma20_premium") or 0)
        if not np.isfinite(premium) or premium > MA20_MAX_PREMIUM:
            return {}

        ma_signal = str(result.get("ma_signal") or "")
        ma_bonus = 0.0
        if "多头" in ma_signal or "金叉" in ma_signal:
            ma_bonus = 1.0
        elif "趋势健康" in ma_signal:
            ma_bonus = 0.6

        name = str(result.get("name") or "").strip()
        if not name or name.lower() == "nan":
            name = str(get_name_map().get(str(code).zfill(6), "") or "")

        wide_state = result.get("wide_state")
        if wide_state is None or (isinstance(wide_state, float) and np.isnan(wide_state)):
            wide_state = ""
        else:
            wide_state = str(wide_state)

        # 多周期：周线/月线结构
        mtf = calc_multi_timeframe(df)
        if not pass_mtf_filter(mtf, premium, mtf_filter):
            return {}

        return {
            "code": str(code).zfill(6),
            "name": str(name or ""),
            "date": str(
                result.get(
                    "date",
                    latest["date"].date() if hasattr(latest["date"], "date") else latest["date"],
                )
            ),
            "close": round(float(latest["close"]), 2),
            # 与 short_term_note 支撑候选一致：现价下方 8% / 5% 参考位
            "support_092": round(float(latest["close"]) * 0.92, 2),
            "support_095": round(float(latest["close"]) * 0.95, 2),
            "HighOpen": round(float(latest["HighOpen"]), 4),
            "CloseLow": round(float(latest["CloseLow"]), 4),
            "VwapClose": round(float(latest["VwapClose"]), 4),
            "close_pos": round(float(latest["close_pos"]), 4),
            "Turnover": round(float(latest.get("Turnover", 0)), 4),
            "profit_pct": round(profit, 2),
            "ma20_premium": round(premium, 4),
            "wide_score": float(result.get("wide_score") or 0),
            "wide_state": wide_state,
            "is_wide_zone": bool(result.get("is_wide_zone")),
            "ma_signal": str(ma_signal or ""),
            "ma_bonus": ma_bonus,
            "year_first_red": float(latest.get("year_first_red", 0)),
            "ma250": round(float(latest["ma250"]), 2) if pd.notna(latest.get("ma250")) else None,
            "is_red_heavy_chip": True,
            # ----- 多周期 -----
            "w_ma20": mtf.get("w_ma20"),
            "w_ma20_premium": mtf.get("w_ma20_premium"),
            "w_above_ma20": mtf.get("w_above_ma20"),
            "m_pos_12": mtf.get("m_pos_12"),
            "m_stretch": mtf.get("m_stretch"),
            "mtf_label": mtf.get("mtf_label") or "",
            "mtf_bonus": float(mtf.get("mtf_bonus") or 0),
            "mtf_risk": mtf.get("mtf_risk") or "",
        }
    except Exception:
        return {}


def notify_serverchan(title: str, content: str) -> dict[str, Any]:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "missing_SENDKEY"}
    try:
        resp = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"title": title, "desp": content},
            timeout=20,
        )
        result = resp.json()
        if resp.ok and result.get("code") == 0:
            return {"status": "sent"}
        return {"status": "failed", "detail": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200]}


def compute_rsrs(index_df: pd.DataFrame, n: int = RSRS_N, m: int = RSRS_M) -> dict[str, Any]:
    df = index_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < n + 30:
        return {"ok": False, "reason": "index_data_too_short", "rsrs_z": None, "bullish": None}

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    scores = np.full(len(df), np.nan)

    for i in range(n - 1, len(df)):
        y = highs[i - n + 1 : i + 1]
        x = lows[i - n + 1 : i + 1]
        if np.std(x) < 1e-8:
            continue
        x_mean, y_mean = x.mean(), y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        slope = cov / (var + 1e-12)
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = max(0.0, min(1.0, 1.0 - ss_res / (ss_tot + 1e-12)))
        scores[i] = slope * r2

    z = np.full(len(df), np.nan)
    lookback = min(m, len(df) - n)
    for i in range(n - 1, len(df)):
        start = max(0, i - lookback + 1)
        window = scores[start : i + 1]
        window = window[np.isfinite(window)]
        if len(window) < 20:
            continue
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-12:
            continue
        z[i] = (scores[i] - mu) / sigma

    last_z = z[-1]
    if not np.isfinite(last_z):
        return {"ok": False, "reason": "rsrs_nan", "rsrs_z": None, "bullish": None}

    return {
        "ok": True,
        "rsrs_z": round(float(last_z), 3),
        "bullish": bool(last_z >= RSRS_BUY_THRESHOLD),
        "threshold": RSRS_BUY_THRESHOLD,
    }


def get_market_timing() -> dict[str, Any]:
    details = []
    crash_any = False
    rsrs_bull_cnt = 0
    rsrs_valid_cnt = 0
    ret_list = []

    for name, symbol in INDEX_LIST:
        item = {
            "name": name,
            "symbol": symbol,
            "close": None,
            "support_092": None,
            "support_095": None,
            "rsrs_z": None,
            "ret_1d": None,
            "bullish": None,
        }
        try:
            raw = ak.stock_zh_index_daily(symbol=symbol)
            if raw is None or raw.empty:
                details.append(item)
                continue
            df = raw.rename(
                columns={"date": "date", "high": "high", "low": "low", "close": "close"}
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(800).reset_index(drop=True)

            if len(df) >= 1:
                idx_close = float(df["close"].iloc[-1])
                item["close"] = round(idx_close, 2)
                item["support_092"] = round(idx_close * 0.92, 2)
                item["support_095"] = round(idx_close * 0.95, 2)

            if len(df) >= 2:
                ret = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
                item["ret_1d"] = round(ret, 4)
                ret_list.append(ret)
                if ret <= INDEX_CRASH_THRESHOLD:
                    crash_any = True

            rsrs = compute_rsrs(df)
            if rsrs.get("ok"):
                item["rsrs_z"] = rsrs["rsrs_z"]
                item["bullish"] = rsrs["bullish"]
                rsrs_valid_cnt += 1
                if rsrs["bullish"]:
                    rsrs_bull_cnt += 1
        except Exception:
            pass
        details.append(item)

    rsrs_ok = (rsrs_valid_cnt > 0) and (rsrs_bull_cnt * 2 >= rsrs_valid_cnt)
    allow = (not crash_any) and (rsrs_ok if rsrs_valid_cnt > 0 else True)

    notes = []
    if crash_any:
        notes.append("存在指数昨日跌超3%")
    if rsrs_valid_cnt > 0 and not rsrs_ok:
        notes.append(f"RSRS偏多{rsrs_bull_cnt}/{rsrs_valid_cnt}")
    if not notes:
        notes.append("多指数择时偏多")

    avg_ret = round(float(np.mean(ret_list)), 4) if ret_list else None

    return {
        "ok": rsrs_valid_cnt > 0,
        "details": details,
        "rsrs_bull_cnt": rsrs_bull_cnt,
        "rsrs_valid_cnt": rsrs_valid_cnt,
        "index_ret_1d": avg_ret,
        "index_crash": crash_any,
        "allow_new_position": allow,
        "reason": "；".join(notes),
    }


def run_shard(
    shard_id: int,
    mtf_filter: str = MTF_FILTER_DEFAULT,
    universe: list[str] | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"两路分打 - 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)} | mtf_filter={mtf_filter}")
    if universe is not None:
        print(f"universe 限定: {len(universe)} 只")
    print(f"{'='*60}")

    if universe is not None:
        all_codes = list(universe)
    else:
        all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表（或 universe 为空）")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")

    records = []
    n_data = n_red = n_profit = n_ma20 = 0

    for i, code in enumerate(codes):
        if FULL_CHIP_AVAILABLE and is_beijing_stock(code):
            continue

        df = get_stock_data(code)
        if df.empty or len(df) < MIN_BARS:
            continue
        n_data += 1

        try:
            df2 = calc_launch_factors(df)
            result = analyze(code, "", df2)
            if result.get("is_red_heavy_chip"):
                n_red += 1
                profit = float(result.get("profit_pct") or 0)
                if PROFIT_MIN <= profit <= PROFIT_MAX:
                    n_profit += 1
                    prem = float(df2.iloc[-1].get("ma20_premium") or 99)
                    if np.isfinite(prem) and prem <= MA20_MAX_PREMIUM:
                        n_ma20 += 1
        except Exception:
            pass

        row = apply_launch_filter(df, code, mtf_filter=mtf_filter)
        if row:
            records.append(row)

        if (i + 1) % 30 == 0:
            print(
                f"  已处理 {i+1}/{len(codes)} | 有效{n_data} "
                f"红筹{n_red} 获利{n_profit} MA20内{n_ma20} 命中{len(records)}"
            )

    print(
        f"分片 {shard_id} 统计: 有效={n_data} 红筹={n_red} "
        f"获利={n_profit} MA20内={n_ma20} 最终命中={len(records)}"
    )

    if records:
        df_out = pd.DataFrame(records)
        df_out["code"] = df_out["code"].astype(str).str.zfill(6)
        for col in ("name", "date", "wide_state", "ma_signal", "mtf_label", "mtf_risk"):
            if col in df_out.columns:
                df_out[col] = df_out[col].map(
                    lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
                )
        df_out = fill_names(df_out)
        out_file = CACHE_DIR / f"live_shard_{shard_id}.csv"
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id} 完成，命中 {len(records)} 只 → {out_file}")
    else:
        print(f"分片 {shard_id} 无命中股票")


def _safe_z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-8)


def merge_and_select(
    top_n: int = 15,
    do_notify: bool = False,
    mtf_filter: str = MTF_FILTER_DEFAULT,
) -> None:
    files = list(CACHE_DIR.glob("live_shard_*.csv"))
    if not files:
        print("没有找到任何分片结果，请先运行分片")
        return

    dfs = [pd.read_csv(f, dtype={"code": str}) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = fill_names(df)
    print(f"合并后共 {len(df)} 只命中股票 | mtf_filter={mtf_filter}")

    # 合并阶段可再按多周期过滤（分片若用 off 扫全量，这里可 strict 收紧）
    mode = (mtf_filter or "off").strip().lower()
    if mode not in ("", "off", "none", "0") and not df.empty:
        before = len(df)
        keep = []
        for _, row in df.iterrows():
            mtf = {
                "w_above_ma20": (
                    bool(row["w_above_ma20"])
                    if "w_above_ma20" in df.columns and pd.notna(row.get("w_above_ma20"))
                    else None
                ),
                "m_pos_12": (
                    float(row["m_pos_12"])
                    if "m_pos_12" in df.columns and pd.notna(row.get("m_pos_12"))
                    else None
                ),
                "m_stretch": (
                    float(row["m_stretch"])
                    if "m_stretch" in df.columns and pd.notna(row.get("m_stretch"))
                    else None
                ),
            }
            # CSV 里 bool 可能变成字符串
            if "w_above_ma20" in df.columns and pd.notna(row.get("w_above_ma20")):
                v = row.get("w_above_ma20")
                if isinstance(v, str):
                    mtf["w_above_ma20"] = v.strip().lower() in ("1", "true", "yes")
            prem = float(row.get("ma20_premium") or 0)
            keep.append(pass_mtf_filter(mtf, prem, mode))
        df = df.loc[keep].reset_index(drop=True)
        print(f"多周期过滤 {mode}: {before} → {len(df)}")

    timing = get_market_timing()
    print(
        f"择时: allow={timing.get('allow_new_position')} | "
        f"RSRS偏多{timing.get('rsrs_bull_cnt')}/{timing.get('rsrs_valid_cnt')} | "
        f"均涨跌={timing.get('index_ret_1d')} | {timing.get('reason')}"
    )
    for d in timing.get("details") or []:
        print(
            f"  {d['name']}: 收盘={d.get('close')} "
            f"支撑0.92={d.get('support_092')} / 0.95={d.get('support_095')} "
            f"| RSRS={d.get('rsrs_z')} 昨日={d.get('ret_1d')} "
            f"{'偏多' if d.get('bullish') else '偏空' if d.get('bullish') is not None else 'N/A'}"
        )

    if df.empty:
        print("无股票通过过滤")
        if do_notify:
            title = f"选股 {datetime.now().strftime('%Y%m%d')}｜无标的"
            content = "今日无股票通过过滤。\n仅为研究输出，不构成投资建议。"
            print(notify_serverchan(title, content))
        return

    df["wide_z"] = _safe_z(df["wide_score"]) if "wide_score" in df.columns else 0.0
    df["VwapClose_z"] = _safe_z(df["VwapClose"]) if "VwapClose" in df.columns else 0.0
    df["HighOpen_z"] = _safe_z(df["HighOpen"]) if "HighOpen" in df.columns else 0.0
    df["CloseLow_z"] = _safe_z(df["CloseLow"]) if "CloseLow" in df.columns else 0.0
    df["Turnover_z"] = (
        _safe_z(df["Turnover"].clip(upper=TURN_OVER_CAP)) if "Turnover" in df.columns else 0.0
    )
    df["year_first_red"] = df.get("year_first_red", 0).fillna(0)
    df["profit_z"] = _safe_z(df["profit_pct"]) if "profit_pct" in df.columns else 0.0
    df["inv_wide_z"] = -df["wide_z"]

    df["score_wide"] = (
        WEIGHTS_WIDE["wide_score"] * df["wide_z"]
        + WEIGHTS_WIDE["year_first_red"] * df["year_first_red"]
        + WEIGHTS_WIDE["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_WIDE["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_WIDE["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_WIDE["Turnover"] * df["Turnover_z"]
    )
    if "is_wide_zone" in df.columns:
        df.loc[df["is_wide_zone"] == False, "score_wide"] *= 0.6

    df["score_peak"] = (
        WEIGHTS_PEAK["profit_pct"] * df["profit_z"]
        + WEIGHTS_PEAK["inv_wide"] * df["inv_wide_z"]
        + WEIGHTS_PEAK["HighOpen"] * df["HighOpen_z"]
        + WEIGHTS_PEAK["VwapClose"] * df["VwapClose_z"]
        + WEIGHTS_PEAK["CloseLow"] * df["CloseLow_z"]
        + WEIGHTS_PEAK["Turnover"] * df["Turnover_z"]
    )
    df.loc[df["profit_pct"] < PEAK_PROFIT_MIN, "score_peak"] *= 0.5
    if "is_wide_zone" in df.columns:
        df.loc[df["is_wide_zone"] == True, "score_peak"] *= 0.7

    # 多周期小幅加减分（不改变硬过滤，只微调排序）
    if "mtf_bonus" in df.columns:
        df["mtf_bonus"] = pd.to_numeric(df["mtf_bonus"], errors="coerce").fillna(0.0)
        df["score_wide"] = df["score_wide"] + df["mtf_bonus"]
        df["score_peak"] = df["score_peak"] + df["mtf_bonus"]

    df["final_score"] = df[["score_wide", "score_peak"]].max(axis=1)
    df["main_track"] = np.where(df["score_peak"] > df["score_wide"], "尖峰", "宽幅启动")

    top_wide = (
        df[df["main_track"] == "宽幅启动"]
        .sort_values("score_wide", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_peak = (
        df[df["main_track"] == "尖峰"]
        .sort_values("score_peak", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    today = datetime.now().strftime("%Y%m%d")
    top_wide.to_csv(f"live_select_wide_{today}.csv", index=False, encoding="utf-8-sig")
    top_peak.to_csv(f"live_select_peak_{today}.csv", index=False, encoding="utf-8-sig")
    print(f"宽幅启动 {len(top_wide)} 只 → live_select_wide_{today}.csv")
    print(f"红筹尖峰 {len(top_peak)} 只 → live_select_peak_{today}.csv")

    allow = timing.get("allow_new_position", True)
    timing_flag = "可关注" if allow else "建议观望/不新开仓"

    def _print_block(title: str, sub: pd.DataFrame, score_col: str) -> None:
        print("\n" + "=" * 90)
        print(f"{title} | 择时: {timing_flag}")
        print("=" * 90)
        if sub.empty:
            print("（本路无标的）")
            return
        for i, row in sub.iterrows():
            yf = "年线首红" if row.get("year_first_red", 0) > 0.5 else ""
            mtf = str(row.get("mtf_label") or "")
            print(
                f"{i+1:02d}. {row['code']} {row.get('name', '')} "
                f"| 分{row[score_col]:.3f} | 收盘{row['close']} "
                f"| 支撑0.92={row.get('support_092', '-')} "
                f"| 获利{row.get('profit_pct', '-')}% "
                f"| MA20溢价{row.get('ma20_premium', 0):.1%} "
                f"| HO{row.get('HighOpen', 0):.3f} "
                f"| {mtf} {yf}"
            )
        print("=" * 90)

    _print_block("A路 宽幅启动", top_wide, "score_wide")
    _print_block("B路 红筹尖峰", top_peak, "score_peak")

    if do_notify:
        def _timing_lines() -> list[str]:
            lines = [
                f"**择时**：{timing_flag}",
                f"- 综合：{timing.get('reason')}",
                f"- RSRS偏多 {timing.get('rsrs_bull_cnt')}/{timing.get('rsrs_valid_cnt')} | 均涨跌 {timing.get('index_ret_1d')}",
            ]
            for d in timing.get("details") or []:
                flag = (
                    "偏多"
                    if d.get("bullish")
                    else ("偏空" if d.get("bullish") is not None else "N/A")
                )
                lines.append(
                    f"- {d['name']}: 收盘={d.get('close')} "
                    f"支撑0.92={d.get('support_092')} / 0.95={d.get('support_095')} "
                    f"| RSRS={d.get('rsrs_z')} 昨日={d.get('ret_1d')} {flag}"
                )
            return lines

        def _build_desp(track_name: str, sub: pd.DataFrame, score_col: str) -> str:
            lines = [
                f"# {track_name} {today}",
                "",
            ]
            lines.extend(_timing_lines())
            lines.extend([
                "",
                f"共 {len(sub)} 只",
                "",
                "> 说明：获利盘/筹码为自研口径，可能与东财等软件不一致，请以实盘软件复核。",
                "",
            ])
            if not allow:
                lines.append("> 当前择时不建议新开仓，名单仅供观察。")
                lines.append("")
            if sub.empty:
                lines.append("本路无标的。")
            else:
                for i, row in sub.iterrows():
                    yf = "【年线首红】" if row.get("year_first_red", 0) > 0.5 else ""
                    nm = row.get("name") or ""
                    mtf = str(row.get("mtf_label") or "")
                    risk = str(row.get("mtf_risk") or "")
                    risk_s = f" 风险:{risk}" if risk else ""
                    lines.append(
                        f"{i+1:02d}. {row['code']} {nm} "
                        f"| 分{row[score_col]:.3f} | 收盘{row['close']} "
                        f"| 支撑0.92={row.get('support_092', '-')} / 0.95={row.get('support_095', '-')} "
                        f"| 获利{row.get('profit_pct', '-')}% "
                        f"| MA20溢价{row.get('ma20_premium', 0):.1%} "
                        f"| HO{row.get('HighOpen', 0):.3f} "
                        f"| 周期:{mtf}{risk_s} {yf}"
                    )
            lines.append("\n仅为研究输出，不构成投资建议。")
            return "\n".join(lines)

        mtf_tag = {
            "resonate": "共振",
            "strict": "周线多",
            "soft": "软过滤",
            "off": "",
        }.get((mtf_filter or "off").lower(), mtf_filter or "")
        mtf_part = f"|{mtf_tag}" if mtf_tag else ""
        r1 = notify_serverchan(
            f"宽幅启动{today}|{'可做' if allow else '观望'}|{len(top_wide)}只{mtf_part}",
            _build_desp("宽幅启动", top_wide, "score_wide"),
        )
        print(f"Server酱 宽幅推送: {r1}")
        time.sleep(2)
        r2 = notify_serverchan(
            f"红筹尖峰{today}|{'可做' if allow else '观望'}|{len(top_peak)}只{mtf_part}",
            _build_desp("红筹尖峰", top_peak, "score_peak"),
        )
        print(f"Server酱 尖峰推送: {r2}")


def main():
    parser = argparse.ArgumentParser(description="两路分打+分开推送+多指数RSRS+名称补全")
    parser.add_argument("--shard", type=str, help="1-8 或 all")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument(
        "--mtf-filter",
        type=str,
        default=MTF_FILTER_DEFAULT,
        choices=["off", "soft", "strict", "resonate"],
        help="多周期过滤: off/soft/strict/resonate(共振:周线多+日线未深破+月线未极端高位破位)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help="只扫描该 CSV 内的股票（如 mtf_resonate_universe.csv）",
    )
    args = parser.parse_args()

    if args.merge:
        merge_and_select(
            top_n=args.top,
            do_notify=args.notify,
            mtf_filter=args.mtf_filter,
        )
        return
    if not args.shard:
        parser.error("请指定 --shard 或 --merge")

    universe_codes: list[str] | None = None
    if args.universe:
        universe_codes = load_universe(args.universe)
        print(f"已加载 universe: {args.universe} → {len(universe_codes)} 只")
        if not universe_codes:
            print("universe 为空，退出")
            return

    # 已在 universe 内时，分片阶段不必再 resonate（可 off 加速）；仍允许用户指定
    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i, mtf_filter=args.mtf_filter, universe=universe_codes)
        print(
            "\n全部分片完成，请执行："
            f"python live_stock_selector.py --merge --top 15 --notify --mtf-filter {args.mtf_filter}"
        )
    else:
        run_shard(
            int(args.shard),
            mtf_filter=args.mtf_filter,
            universe=universe_codes,
        )


if __name__ == "__main__":
    main()