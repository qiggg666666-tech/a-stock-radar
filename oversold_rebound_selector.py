#!/usr/bin/env python3
"""
超跌反弹选股（研究用，独立于红筹启动池）
========================================
目标形态（参考：大顶后下跌 → 日线止跌反弹，周月线可以仍弱）：
  1) 相对近 60/120 日高点回撤足够深（默认 ≥25%）
  2) 相对近 20 日低点已反弹（默认 ≥5%，且未涨太远 ≤25%）
  3) 日线收盘站上或贴近 MA20（溢价 -2%~12%）
  4) 获利盘偏低（默认 15%~45%，套牢较多）——有 final_chip_research 时启用
  5) 不要求周线在 MA20 上（与 resonate 红筹池刻意区分）
  6) 可选形态标注（加分；--require 可作硬过滤）：
     · 年线长影线：近5日内出现长下影线，探到/贴近250日均线后收回（年线支撑）
     · 倍量柱：近10日内出现低位倍量阳柱，且其后未跌破该柱开盘价/最低价（可带“后缩量守位”）

用法：
  python oversold_rebound_selector.py --shard 1
  python oversold_rebound_selector.py --shard all --exclude-st   # 跳过 ST/退市整理股
  python oversold_rebound_selector.py --shard all
  python oversold_rebound_selector.py --shard all --require ma250     # 只留年线长影线
  python oversold_rebound_selector.py --shard all --require volbar    # 只留倍量柱
  python oversold_rebound_selector.py --shard all --require any       # 二者其一
  python oversold_rebound_selector.py --merge --top 20 --notify
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
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
    print("已加载 final_chip_research")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：无 final_chip_research，将跳过筹码/获利过滤")

    def is_beijing_stock(code: str) -> bool:
        return str(code).startswith(("8", "4", "9"))

try:
    import talib  # type: ignore

    TALIB_AVAILABLE = True
    print("已加载 TA-Lib（指标补全）")
except ImportError:
    TALIB_AVAILABLE = False

try:
    from tsfresh import extract_features  # type: ignore
    from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore
    from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute  # type: ignore

    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False


CACHE_DIR = Path("./oversold_rebound_cache")
CACHE_DIR.mkdir(exist_ok=True)

SHARD_MAP = {
    1: ["000", "001", "003"],
    2: ["002"],
    3: ["300", "301"],
    4: ["600"],
    5: ["601"],
    6: ["603"],
    7: ["605", "688"],
    8: ["8", "4", "9"],
}

MIN_BARS = 120
HIST_DAYS = 800   # akshare 回退取数窗口（日历日）
STALE_DAYS = 15   # 最后一根K线距今超过该天数视为停牌/退市，跳过
# ----- 超跌反弹参数（可按需改） -----
DRAWDOWN_MIN = 0.25          # 相对近60日高点至少回撤 25%
REBOUND_FROM_LOW_MIN = 0.05  # 相对近20日低点至少反弹 5%
REBOUND_FROM_LOW_MAX = 0.25  # 反弹不超过 25%（避免已炒高）
MA20_PREM_MIN = -0.02
MA20_PREM_MAX = 0.12
PROFIT_MIN = 15.0            # 获利偏低 = 套牢多
PROFIT_MAX = 45.0
TOP_N_DEFAULT = 20

# ---- 年线长影线 参数 ----
MA250_LOOKBACK = 5            # 影线K线出现在最近 5 个交易日内
MA250_SHADOW_MIN = 0.03       # 下影线长度 >= 收盘价的 3%
MA250_TOUCH_TOL = 0.01        # 低点在年线上方 1% 以内也算“触线支撑”
MA250_BREAK_TOL = 0.03        # 其后收盘跌破年线超过 3% 视为支撑失败

# ---- 倍量柱 参数 ----
VOLBAR_MULT = 2.0             # 倍量柱量 >= 前一日 2 倍
VOLBAR_MA5_MULT = 1.5         # 且 >= 柱前5日均量 1.5 倍
VOLBAR_MIN_RET = 0.02         # 阳线且涨幅 >= 2%
VOLBAR_MAX_DAYS = 10          # 倍量柱出现在最近 10 个交易日内
VOLBAR_NEAR_LOW = 1.15        # 倍量柱收盘 <= 其前20日最低价的 1.15 倍（低位放量才有止跌意义）
VOLBAR_SUPPORT_TOL = 0.02     # 其后收盘不破柱开盘价、最低不破柱最低价（容差 2%）
VOLBAR_SHRINK = 0.60          # “后缩量守位”：柱后每日量 <= 柱量 60%，最近一日 <= 50%


_ALL_CODES: list[str] | None = None


def get_all_a_stocks() -> list[str]:
    """全市场代码（进程内缓存）。baostock 只取在市股票。"""
    global _ALL_CODES
    if _ALL_CODES is not None:
        return list(_ALL_CODES)
    codes: list[str] = []
    try:
        import baostock as bs

        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_basic()
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    code = row[0]
                    stock_type = row[4] if len(row) > 4 else "1"
                    status = row[5] if len(row) > 5 else "1"  # 1=上市 0=退市
                    if stock_type == "1" and status == "1":
                        pure = code.split(".")[-1] if "." in code else code
                        codes.append(pure.zfill(6))
            finally:
                bs.logout()
            if codes:
                _ALL_CODES = sorted(set(codes))
                return list(_ALL_CODES)
    except Exception as e:
        print(f"baostock 列表失败: {e}")
    try:
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        if codes:
            _ALL_CODES = codes
            return list(codes)
    except Exception as e:
        print(f"akshare 列表失败: {e}")
    return []


def report_uncovered(codes: list[str]) -> None:
    """提示未被任何分片前缀覆盖的代码，避免像 301/003 那样被悄悄漏掉。"""
    covered = tuple(p for ps in SHARD_MAP.values() for p in ps)
    miss = [c for c in codes if not c.startswith(covered)]
    if miss:
        print(f"警告：{len(miss)} 只代码未被任何分片覆盖，例如 {miss[:15]}")


def is_st_name(name: str) -> bool:
    n = str(name or "").upper().replace(" ", "")
    return bool(re.match(r"^(\*?ST|S\*ST|SST)", n)) or "退" in n


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def _normalize_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    """统一：date 为 datetime、升序、去重、OHLC 无空值；缺关键列返回空表。"""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    if any(c not in df.columns for c in ("open", "high", "low", "close")):
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _fetch_akshare_hist(symbol: str, retries: int = 2) -> pd.DataFrame:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=HIST_DAYS)).strftime("%Y%m%d")  # 避开 2/29 的 replace(year=)
    for attempt in range(retries):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            raw = raw.rename(
                columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover",
                }
            )
            raw["symbol"] = symbol
            time.sleep(0.15)
            return _normalize_ohlcv(raw)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def _is_stale(df: pd.DataFrame) -> bool:
    try:
        last = pd.to_datetime(df["date"]).max()
        return (datetime.now() - last.to_pydatetime()).days > STALE_DAYS
    except Exception:
        return False


def get_stock_data(symbol: str) -> pd.DataFrame:
    best = pd.DataFrame()
    if FULL_CHIP_AVAILABLE:
        try:
            tmp, _, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if tmp is not None and not tmp.empty:
                tmp = tmp.copy()
                tmp["symbol"] = symbol
                best = _normalize_ohlcv(tmp)
        except Exception:
            pass
    if best.empty or len(best) < 260:  # 年线(MA250)需要足够长的历史
        fb = _fetch_akshare_hist(symbol)
        if len(fb) > len(best):  # 取更长的一份，避免回退数据覆盖主源
            best = fb
    return best if not best.empty else pd.DataFrame()


_NAME_MAP: dict[str, str] | None = None


def get_name_map() -> dict[str, str]:
    """名称表只拉一次（原 _name_of 每只股票都重新请求全市场名称表）。"""
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP
    _NAME_MAP = {}
    try:
        info = ak.stock_info_a_code_name()
        if info is not None and not info.empty:
            _NAME_MAP = dict(
                zip(info["code"].astype(str).str.zfill(6), info["name"].astype(str))
            )
    except Exception as e:
        print(f"名称表加载失败: {e}")
    return _NAME_MAP


def _name_of(code: str) -> str:
    return str(get_name_map().get(str(code).zfill(6), "") or "")


def calc_ma250_shadow(d: pd.DataFrame) -> dict[str, Any]:
    """
    年线长影线（研究标注）：最近 MA250_LOOKBACK 日内出现一根长下影线K线，
      · 下影线 >= 振幅一半、>= 实体2倍、且 >= 收盘价3%
      · 探线收回：最低价 <= 年线 且 收盘 >= 年线；或 触线支撑：最低价在年线上方1%以内且收盘在年线上方
      · 其后收盘未有效跌破年线，也未跌破该影线最低价
    历史不足 250 根时不判断。
    """
    out: dict[str, Any] = {
        "near_ma250_shadow": False,
        "ma250_label": "",
        "ma250_bonus": 0.0,
        "ma250": None,
        "ma250_premium": None,
        "ma250_bar_date": "",
        "ma250_shadow_pct": None,
        "ma250_days_ago": None,
    }
    try:
        n = len(d)
        if n < 255:
            return out
        o = d["open"].to_numpy(float)
        h = d["high"].to_numpy(float)
        lo = d["low"].to_numpy(float)
        c = d["close"].to_numpy(float)
        ma = pd.Series(c).rolling(250, min_periods=250).mean().to_numpy()
        if np.isfinite(ma[-1]) and ma[-1] > 0:
            out["ma250"] = round(float(ma[-1]), 2)
            out["ma250_premium"] = round(float(c[-1] / ma[-1] - 1.0), 4)

        for k in range(0, MA250_LOOKBACK):
            i = n - 1 - k
            m = ma[i]
            rng = h[i] - lo[i]
            if not (np.isfinite(m) and m > 0 and rng > 0 and c[i] > 0):
                continue
            lower = min(o[i], c[i]) - lo[i]
            body = abs(c[i] - o[i])
            if not (lower / rng >= 0.5 and lower >= 2 * body and lower / c[i] >= MA250_SHADOW_MIN):
                continue
            pierce = lo[i] <= m and c[i] >= m
            touch = (m < lo[i] <= m * (1 + MA250_TOUCH_TOL)) and c[i] > m
            if not (pierce or touch):
                continue
            if k > 0:
                post_c, post_l, post_m = c[i + 1 :], lo[i + 1 :], ma[i + 1 :]
                if np.any(post_c < post_m * (1 - MA250_BREAK_TOL)) or np.any(post_l < lo[i] * 0.99):
                    continue
            kind = "探线收回" if pierce else "触线支撑"
            out.update(
                near_ma250_shadow=True,
                ma250_label=f"年线长影线·{kind}" + (f"·{k}日前" if k else "·当日"),
                ma250_bonus=0.15 if pierce else 0.10,
                ma250=round(float(m), 2),
                ma250_bar_date=str(pd.Timestamp(d["date"].iloc[i]).date()),
                ma250_shadow_pct=round(float(lower / c[i]), 4),
                ma250_days_ago=k,
            )
            return out
    except Exception:
        pass
    return out


def calc_volume_bar(d: pd.DataFrame) -> dict[str, Any]:
    """
    倍量柱（研究标注，超跌反弹语境下看“低位放量止跌”）：最近 VOLBAR_MAX_DAYS 日内，
      · 阳线、涨幅>=2%，量>=前一日2倍，且>=柱前5日均量1.5倍
      · 出现在低位：收盘 <= 柱前20日最低价的 1.15 倍
      · 其后收盘不破柱开盘价、最低不破柱最低价（容差2%）
    若柱后已 >=2 日且持续缩量（<=柱量60%，最近一日<=50%），再标“后缩量守位”。
    """
    out: dict[str, Any] = {
        "near_volbar": False,
        "volbar_label": "",
        "volbar_bonus": 0.0,
        "vb_date": "",
        "vb_days_ago": None,
        "vb_mult": None,
        "vb_support": None,
        "vb_shrink_last": None,
    }
    try:
        if len(d) < 30 or "volume" not in d.columns:
            return out
        v = pd.to_numeric(d["volume"], errors="coerce").to_numpy(float)
        o = d["open"].to_numpy(float)
        lo = d["low"].to_numpy(float)
        c = d["close"].to_numpy(float)
        n = len(d)
        for k in range(0, VOLBAR_MAX_DAYS + 1):
            b = n - 1 - k
            if b < 21:
                break
            vb, vp = v[b], v[b - 1]
            if not (np.isfinite(vb) and np.isfinite(vp) and vb > 0 and vp > 0) or c[b - 1] <= 0:
                continue
            ma5 = np.nanmean(v[b - 5 : b])
            if not (np.isfinite(ma5) and ma5 > 0):
                continue
            if not (
                vb >= VOLBAR_MULT * vp
                and vb >= VOLBAR_MA5_MULT * ma5
                and c[b] > o[b]
                and c[b] / c[b - 1] - 1.0 >= VOLBAR_MIN_RET
            ):
                continue
            if c[b] > lo[b - 20 : b + 1].min() * VOLBAR_NEAR_LOW:
                continue  # 高位放量不算止跌倍量
            shrink = False
            if k > 0:
                post_c, post_l, post_v = c[b + 1 :], lo[b + 1 :], v[b + 1 :]
                if post_c.min() < o[b] * (1 - VOLBAR_SUPPORT_TOL) or post_l.min() < lo[b] * (1 - VOLBAR_SUPPORT_TOL):
                    continue
                if k >= 2 and not np.isnan(post_v).any():
                    shrink = bool(post_v.max() <= VOLBAR_SHRINK * vb and v[-1] <= 0.5 * vb)
            tags = ["今日倍量阳柱" if k == 0 else f"倍量阳柱·{k}日前"]
            bonus = 0.10 if k == 0 else 0.08
            if shrink:
                tags.append("后缩量守位")
                bonus += 0.06
            out.update(
                near_volbar=True,
                volbar_label="+".join(tags),
                volbar_bonus=round(bonus, 3),
                vb_date=str(pd.Timestamp(d["date"].iloc[b]).date()),
                vb_days_ago=k,
                vb_mult=round(float(vb / vp), 2),
                vb_support=round(float(o[b]), 2),
                vb_shrink_last=round(float(v[-1] / vb), 3),
            )
            return out
    except Exception:
        pass
    return out


def _pass_require(sig_ma250: bool, sig_volbar: bool, require: str | None) -> bool:
    if not require:
        return True
    if require == "ma250":
        return sig_ma250
    if require == "volbar":
        return sig_volbar
    return sig_ma250 or sig_volbar  # any


def score_oversold_rebound(
    df: pd.DataFrame, code: str, require: str | None = None
) -> dict[str, Any] | None:
    if df is None or len(df) < MIN_BARS:
        return None
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    close = float(d["close"].iloc[-1])
    if close <= 0:
        return None

    # 均线组 MA/EMA：10 / 20 / 30 / 60 / 120 / 200
    spans = (10, 20, 30, 60, 120, 200)
    for n in spans:
        mp = max(5, n // 2)
        d[f"ma{n}"] = d["close"].rolling(n, min_periods=mp).mean()
        d[f"ema{n}"] = d["close"].ewm(span=n, adjust=False, min_periods=mp).mean()
    ma20 = float(d["ma20"].iloc[-1])
    if not np.isfinite(ma20) or ma20 <= 0:
        return None
    prem = close / ma20 - 1.0
    if prem < MA20_PREM_MIN or prem > MA20_PREM_MAX:
        return None

    emas = {
        n: float(d[f"ema{n}"].iloc[-1])
        if pd.notna(d[f"ema{n}"].iloc[-1]) and np.isfinite(d[f"ema{n}"].iloc[-1])
        else None
        for n in spans
    }
    above_n = sum(1 for n in spans if emas.get(n) and close > emas[n])

    def _chain_up(keys: list[int]) -> bool:
        vals = [emas.get(k) for k in keys]
        if any(v is None for v in vals):
            return False
        return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def _chain_dn(keys: list[int]) -> bool:
        vals = [emas.get(k) for k in keys]
        if any(v is None for v in vals):
            return False
        return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    short_bull = _chain_up([10, 20, 30, 60])
    short_bear = _chain_dn([10, 20, 30, 60])
    long_bull = bool(
        emas.get(60) and emas.get(120) and emas.get(200)
        and emas[60] > emas[120] > emas[200]
    )
    long_bear = bool(
        emas.get(60) and emas.get(120) and emas.get(200)
        and emas[60] < emas[120] < emas[200]
    )
    if short_bull and long_bull and above_n >= 5:
        ema_stack = "全多头排列"
    elif short_bull and above_n >= 4:
        ema_stack = "短多排列"
    elif short_bear and long_bear:
        ema_stack = "空头排列"
    elif short_bear:
        ema_stack = "短空排列"
    else:
        ema_stack = "纠缠"

    cluster_buy_n = cluster_sell_n = 0
    buy_tags: list[str] = []
    sell_tags: list[str] = []
    if above_n >= 4:
        cluster_buy_n += 1
        buy_tags.append(f"站上{above_n}条EMA")
    elif above_n <= 2:
        cluster_sell_n += 1
        sell_tags.append(f"仅上{above_n}条EMA")
    if short_bull:
        cluster_buy_n += 1
        buy_tags.append("短多10>20>30>60")
    if short_bear:
        cluster_sell_n += 1
        sell_tags.append("短空排列")
    if long_bull:
        cluster_buy_n += 1
        buy_tags.append("长多60>120>200")
    if long_bear:
        cluster_sell_n += 1
        sell_tags.append("长空排列")

    if emas.get(20) and len(d) >= 6:
        for i in range(-1, -6, -1):  # 从最近一天往前找，最近一次穿越优先
            c0, c1 = float(d["close"].iloc[i]), float(d["close"].iloc[i - 1])
            a0, a1 = float(d["ema20"].iloc[i]), float(d["ema20"].iloc[i - 1])
            if c1 <= a1 and c0 > a0:
                cluster_buy_n += 1
                buy_tags.append("上穿EMA20")
                break
            if c1 >= a1 and c0 < a0:
                cluster_sell_n += 1
                sell_tags.append("下穿EMA20")
                break

    prem20 = (close / emas[20] - 1.0) if emas.get(20) else None
    prem120 = (close / emas[120] - 1.0) if emas.get(120) else None
    prem200 = (close / emas[200] - 1.0) if emas.get(200) else None
    if prem20 is not None and 0 <= prem20 <= 0.02:
        cluster_buy_n += 1
        buy_tags.append("回踩EMA20")
    if prem20 is not None and -0.02 <= prem20 < 0:
        cluster_sell_n += 1
        sell_tags.append("EMA20受压")

    if cluster_buy_n >= 3 and cluster_buy_n > cluster_sell_n:
        cluster_signal, cluster_label = "聚类买入", "买:" + "+".join(buy_tags[:4])
        cluster_bonus = min(0.22, 0.04 * cluster_buy_n)
    elif cluster_sell_n >= 3 and cluster_sell_n > cluster_buy_n:
        cluster_signal, cluster_label = "聚类卖出", "卖:" + "+".join(sell_tags[:4])
        cluster_bonus = -min(0.22, 0.04 * cluster_sell_n)
    elif cluster_buy_n >= 2 and cluster_buy_n > cluster_sell_n:
        cluster_signal, cluster_label = "偏多观察", "买:" + "+".join(buy_tags[:3])
        cluster_bonus = 0.05
    elif cluster_sell_n >= 2 and cluster_sell_n > cluster_buy_n:
        cluster_signal, cluster_label = "偏空观察", "卖:" + "+".join(sell_tags[:3])
        cluster_bonus = -0.05
    else:
        cluster_signal, cluster_label, cluster_bonus = "中性", ema_stack, 0.0

    high60 = float(d["high"].tail(60).max())
    high120 = float(d["high"].tail(min(120, len(d))).max())
    low20 = float(d["low"].tail(20).min())
    if high60 <= 0 or low20 <= 0:
        return None

    dd60 = 1.0 - close / high60
    dd120 = 1.0 - close / high120
    drawdown = max(dd60, dd120)
    if drawdown < DRAWDOWN_MIN:
        return None

    rebound = close / low20 - 1.0
    if rebound < REBOUND_FROM_LOW_MIN or rebound > REBOUND_FROM_LOW_MAX:
        return None

    # 年线长影线 / 倍量柱：纯价量计算很便宜，硬过滤时先判断，省掉大量 analyze() 调用
    sig_ma = calc_ma250_shadow(d)
    sig_vb = calc_volume_bar(d)
    if not _pass_require(bool(sig_ma["near_ma250_shadow"]), bool(sig_vb["near_volbar"]), require):
        return None

    # 日线止跌：收盘真正站上 MA20。
    # 原写法 close >= ma20*0.98 与前面 prem>=-2% 的过滤完全重复，加分恒成立、没有区分度。
    above_ma20 = close >= ma20

    # 周线仅标注，不硬过滤
    w_above = None
    try:
        w = (
            d.set_index("date")
            .resample("W-FRI")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(subset=["close"])
        )
        if len(w) >= 20:
            w["ma20"] = w["close"].rolling(20, min_periods=15).mean()
            w_ma = float(w["ma20"].iloc[-1])
            w_c = float(w["close"].iloc[-1])
            if np.isfinite(w_ma) and w_ma > 0:
                w_above = bool(w_c > w_ma)
    except Exception:
        pass

    profit = None
    is_red = None
    name = ""
    if FULL_CHIP_AVAILABLE:
        try:
            result = analyze(code, "", d)
            profit = float(result.get("profit_pct") or 0)
            is_red = bool(result.get("is_red_heavy_chip"))
            name = str(result.get("name") or "")
            if not (PROFIT_MIN <= profit <= PROFIT_MAX):
                return None
        except Exception as e:
            print(f"  [{code}] analyze 失败，跳过: {type(e).__name__}: {e}")
            return None
    if not name:
        name = _name_of(code)

    # 综合分：回撤越深、反弹适中、越贴近 MA20 越好；周线仍弱不扣太多
    score = 0.0
    score += min(drawdown, 0.6) * 2.0          # 深跌
    score += (0.15 - abs(rebound - 0.10)) * 3  # 反弹约 10% 附近更优
    score += (0.05 - abs(prem - 0.02)) * 4     # 略高于 MA20
    if above_ma20:
        score += 0.2
    else:
        score -= 0.05  # 仍在 MA20 下方 0~2%：只是“贴近”，略低于已站上的
    if w_above is False:
        score += 0.1  # 超跌反弹允许周线仍弱
    if profit is not None:
        # 获利越低（套牢越多）在区间内略加分
        score += (PROFIT_MAX - profit) / 100.0
    score += cluster_bonus  # 【核心】EMA 聚类加减分
    score += float(sig_ma["ma250_bonus"]) + float(sig_vb["volbar_bonus"])  # 年线长影线 / 倍量柱

    # 【补全】RSI / MACD / 布林（TA-Lib 或 pandas）
    ta_label, ta_bonus = "", 0.0
    rsi14 = macd_hist = bb_pos = None
    try:
        close_a = d["close"].astype(float).values
        if TALIB_AVAILABLE and len(close_a) >= 35:
            rsi_a = talib.RSI(close_a, timeperiod=14)
            macd_a, sig_a, hist_a = talib.MACD(close_a)
            up_a, _, lo_a = talib.BBANDS(close_a, timeperiod=20)
            rsi14 = float(rsi_a[-1]) if np.isfinite(rsi_a[-1]) else None
            macd_hist = float(hist_a[-1]) if np.isfinite(hist_a[-1]) else None
            c = float(close_a[-1])
            if np.isfinite(up_a[-1]) and np.isfinite(lo_a[-1]) and up_a[-1] > lo_a[-1]:
                bb_pos = (c - float(lo_a[-1])) / (float(up_a[-1]) - float(lo_a[-1]) + 1e-12)
        elif len(close_a) >= 35:
            s = pd.Series(close_a)
            delta = s.diff()
            # Wilder 平滑，与 TA-Lib 口径一致
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            _r = (100 - 100 / (1 + gain / (loss + 1e-12))).iloc[-1]
            rsi14 = float(_r) if np.isfinite(_r) else None
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            macd_s = ema12 - ema26
            macd_hist = float((macd_s - macd_s.ewm(span=9, adjust=False).mean()).iloc[-1])
            mid = s.rolling(20).mean()
            std = s.rolling(20).std()
            up, lo = mid + 2 * std, mid - 2 * std
            c = float(s.iloc[-1])
            if float(up.iloc[-1]) > float(lo.iloc[-1]):
                bb_pos = (c - float(lo.iloc[-1])) / (float(up.iloc[-1]) - float(lo.iloc[-1]) + 1e-12)
        tags = []
        if rsi14 is not None and rsi14 < 35:
            tags.append("RSI偏弱区")
            ta_bonus += 0.05
        if rsi14 is not None and rsi14 > 70:
            tags.append("RSI偏强")
            ta_bonus -= 0.04
        if macd_hist is not None and macd_hist > 0:
            tags.append("MACD柱>0")
            ta_bonus += 0.03
        if bb_pos is not None and bb_pos <= 0.2:
            tags.append("近布林下轨")
            ta_bonus += 0.03
        ta_label = "+".join(tags)
        score += ta_bonus
    except Exception:
        pass

    ho = float(d["high"].iloc[-1] / d["open"].iloc[-1] - 1) if d["open"].iloc[-1] > 0 else 0.0
    # 原 d.get("turnover", 0) 在缺列时返回整数 0，.fillna 会直接抛异常
    if "turnover" in d.columns:
        turn = pd.to_numeric(d["turnover"], errors="coerce").fillna(0.0)
    else:
        turn = pd.Series(0.0, index=d.index)
    if float(turn.median(skipna=True) or 0) > 1.5:
        turn = turn / 100.0
    turnover = float(turn.iloc[-1]) if len(turn) else 0.0

    label = "超跌反弹·日线止跌"
    if w_above is False:
        label += "·周线仍弱"
    elif w_above is True:
        label += "·周线转强"

    if sig_ma["near_ma250_shadow"]:
        label += "·年线长影线"
    if sig_vb["near_volbar"]:
        label += "·倍量柱"

    return {
        "code": str(code).zfill(6),
        "name": name or "",
        "date": str(d["date"].iloc[-1].date()),
        "close": round(close, 2),
        "support_092": round(close * 0.92, 2),
        "support_095": round(close * 0.95, 2),
        "ma20": round(ma20, 2),
        "ma20_premium": round(prem, 4),
        "drawdown_60": round(dd60, 4),
        "drawdown_max": round(drawdown, 4),
        "rebound_low20": round(rebound, 4),
        "high60": round(high60, 2),
        "low20": round(low20, 2),
        "w_above_ma20": w_above,
        "profit_pct": round(profit, 2) if profit is not None else None,
        "is_red_heavy_chip": is_red,
        "HighOpen": round(ho, 4),
        "Turnover": round(turnover, 4),
        "ema10": round(emas[10], 2) if emas.get(10) else None,
        "ema20": round(emas[20], 2) if emas.get(20) else None,
        "ema30": round(emas[30], 2) if emas.get(30) else None,
        "ema60": round(emas[60], 2) if emas.get(60) else None,
        "ema120": round(emas[120], 2) if emas.get(120) else None,
        "ema200": round(emas[200], 2) if emas.get(200) else None,
        "ema20_premium": round(prem20, 4) if prem20 is not None else None,
        "ema120_premium": round(prem120, 4) if prem120 is not None else None,
        "ema200_premium": round(prem200, 4) if prem200 is not None else None,
        "ema_stack": ema_stack,
        "above_ema_n": above_n,
        "cluster_buy_n": cluster_buy_n,
        "cluster_sell_n": cluster_sell_n,
        "cluster_signal": cluster_signal,
        "cluster_label": cluster_label,
        "cluster_bonus": round(cluster_bonus, 3),
        "rsi14": round(rsi14, 2) if rsi14 is not None else None,
        "macd_hist": round(macd_hist, 4) if macd_hist is not None else None,
        "bb_pos": round(bb_pos, 4) if bb_pos is not None else None,
        "ta_label": ta_label,
        "ta_bonus": round(ta_bonus, 3),
        "near_ma250_shadow": bool(sig_ma["near_ma250_shadow"]),
        "ma250_label": sig_ma["ma250_label"],
        "ma250_bonus": sig_ma["ma250_bonus"],
        "ma250": sig_ma["ma250"],
        "ma250_premium": sig_ma["ma250_premium"],
        "ma250_bar_date": sig_ma["ma250_bar_date"],
        "ma250_shadow_pct": sig_ma["ma250_shadow_pct"],
        "near_volbar": bool(sig_vb["near_volbar"]),
        "volbar_label": sig_vb["volbar_label"],
        "volbar_bonus": sig_vb["volbar_bonus"],
        "vb_date": sig_vb["vb_date"],
        "vb_days_ago": sig_vb["vb_days_ago"],
        "vb_mult": sig_vb["vb_mult"],
        "vb_support": sig_vb["vb_support"],
        "vb_shrink_last": sig_vb["vb_shrink_last"],
        "label": label,
        "score": round(float(score), 4),
    }


def run_shard(shard_id: int, exclude_st: bool = False, require: str | None = None) -> None:
    print(f"\n{'=' * 60}")
    print(f"超跌反弹 分片 {shard_id} | {SHARD_MAP.get(shard_id)} | exclude_st={exclude_st} | require={require}")
    print(f"{'=' * 60}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # 缓存目录被清理时也能正常写入
    out = CACHE_DIR / f"osr_shard_{shard_id}.csv"
    if out.exists():
        out.unlink()  # 先清旧结果：本次无命中/中途失败时 merge 不会读到过期数据

    codes = filter_shard(get_all_a_stocks(), shard_id)
    print(f"本分片 {len(codes)} 只")
    name_map = get_name_map() if exclude_st else {}

    rows: list[dict[str, Any]] = []
    n_data = n_stale = n_st = n_err = 0
    last_err = ""
    for i, code in enumerate(codes):
        if is_beijing_stock(code):
            continue
        if exclude_st and is_st_name(name_map.get(code, "")):
            n_st += 1
            continue
        try:
            df = get_stock_data(code)
            if df.empty:
                continue
            if _is_stale(df):
                n_stale += 1
                continue
            n_data += 1
            row = score_oversold_rebound(df, code, require=require)
            if row:
                rows.append(row)
        except Exception as e:  # 单只股票出错不拖垮整个分片
            n_err += 1
            last_err = f"{code}: {type(e).__name__}: {e}"
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(codes)} 有效{n_data} 命中 {len(rows)}")

    print(
        f"分片 {shard_id} 统计: 有效={n_data} 命中={len(rows)} "
        f"| 停牌/过期={n_stale} ST跳过={n_st} 异常={n_err}"
    )
    if last_err:
        print(f"  最近一次异常: {last_err}")
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id}: {len(rows)} 只 → {out}")
    else:
        print(f"分片 {shard_id}: 无命中")


def notify_serverchan(title: str, content: str, retries: int = 2) -> str:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return "skipped: no SENDKEY"
    last = ""
    for _ in range(retries):
        try:
            resp = requests.post(
                f"https://sctapi.ftqq.com/{key}.send",
                data={"title": title[:32], "desp": content},  # Server酱标题上限 32 字
                timeout=20,
            )
            try:
                ok = resp.ok and resp.json().get("code") == 0
            except Exception:
                ok = False
            last = f"{'sent' if ok else 'failed'} {resp.status_code} {resp.text[:100]}"
            if ok:
                return last
        except Exception as e:
            last = f"failed: {str(e)[:150]}"
        time.sleep(2)
    return last


def run_tsfresh_research(codes: list[str]) -> None:
    """研究可选：Top 名单 Minimal 特征，不参与打分。"""
    if not TSFRESH_AVAILABLE:
        print("tsfresh 未安装，跳过（pip install tsfresh）")
        return
    rows = []
    for code in codes[:50]:
        try:
            df = get_stock_data(code)
            if df is None or len(df) < 40:
                continue
            d = df.sort_values("date").tail(120).copy()
            d["code"] = str(code).zfill(6)
            d["time"] = range(len(d))
            rows.append(d[["code", "time", "close"]])
        except Exception:
            continue
    if not rows:
        print("tsfresh：无有效序列")
        return
    long_df = pd.concat(rows, ignore_index=True)
    try:
        feat = extract_features(
            long_df,
            column_id="code",
            column_sort="time",
            column_value="close",
            default_fc_parameters=MinimalFCParameters(),
            n_jobs=0,
            disable_progressbar=True,
        )
        tsfresh_impute(feat)
        path = f"tsfresh_minimal_osr_{datetime.now().strftime('%Y%m%d')}.csv"
        feat.to_csv(path, encoding="utf-8-sig")
        print(f"tsfresh 特征 → {path}（仅研究）")
    except Exception as e:
        print(f"tsfresh 失败: {e}")


def _sig_tags(row: Any) -> str:
    """年线长影线 / 倍量柱的展示文本（带防守位）。"""
    parts = []
    ma_l = str(row.get("ma250_label") or "")
    if ma_l and ma_l != "nan":
        parts.append(f"{ma_l}(年线{row.get('ma250', '-')})")
    vb_l = str(row.get("volbar_label") or "")
    if vb_l and vb_l != "nan":
        parts.append(f"{vb_l}(防守{row.get('vb_support', '-')})")
    return (" | " + " | ".join(parts)) if parts else ""


def _fmt_profit(v: Any) -> str:
    """无筹码模块时 profit_pct 为空，避免打印 'nan%'。"""
    x = pd.to_numeric(v, errors="coerce")
    return "-" if pd.isna(x) else f"{float(x):.1f}%"


def _bool_col(s: pd.Series) -> pd.Series:
    """CSV 读回的 bool 可能是 True/'True'/1/NaN，统一解析。"""
    return s.map(
        lambda v: bool(v)
        if isinstance(v, (bool, np.bool_))
        else (False if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip().lower() in ("1", "true", "yes"))
    ).astype(bool)


def merge_and_select(
    top_n: int = TOP_N_DEFAULT,
    do_notify: bool = False,
    tsfresh_research: bool = False,
    require: str | None = None,
) -> None:
    files = sorted(CACHE_DIR.glob("osr_shard_*.csv"))
    if not files:
        print("无分片结果，请先 --shard")
        return
    _today = datetime.now().date()
    _old = [f.name for f in files if datetime.fromtimestamp(f.stat().st_mtime).date() != _today]
    if _old:
        print(f"提示：以下分片结果不是今天生成的（可能是旧数据）: {_old}")
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.sort_values("score", ascending=False).drop_duplicates("code", keep="first")  # 先排序再去重，保留高分
    df = df.reset_index(drop=True)
    # CSV 读回后空字符串变 NaN，打印会出现 "nan"
    for _c in ("name", "cluster_signal", "cluster_label", "ta_label", "label", "ma250_label", "volbar_label", "ma250_bar_date", "vb_date"):
        if _c in df.columns:
            df[_c] = df[_c].fillna("").astype(str).replace({"nan": "", "None": ""})
    if "name" in df.columns:
        _need = df["name"].str.strip() == ""
        if _need.any():
            _m = get_name_map()
            df.loc[_need, "name"] = df.loc[_need, "code"].map(lambda c: _m.get(c, ""))
    if require and not df.empty and {"near_ma250_shadow", "near_volbar"} <= set(df.columns):
        before = len(df)
        _m = _bool_col(df["near_ma250_shadow"])
        _v = _bool_col(df["near_volbar"])
        _keep = _m if require == "ma250" else _v if require == "volbar" else (_m | _v)
        df = df.loc[_keep].reset_index(drop=True)
        print(f"形态过滤 require={require}: {before} → {len(df)}")
    if df.empty:
        print("无有效超跌反弹标的")
        if do_notify:
            print("Server酱:", notify_serverchan(f"超跌反弹{datetime.now():%Y%m%d}|无标的", "今日无标的。\n仅为研究输出，不构成投资建议。"))
        return
    top = df.head(top_n).copy()
    today = datetime.now().strftime("%Y%m%d")
    path = f"oversold_rebound_{today}.csv"
    top.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n超跌反弹 Top{len(top)} → {path}")
    if {"near_ma250_shadow", "near_volbar"} <= set(df.columns):
        _sig = df.loc[_bool_col(df["near_ma250_shadow"]) | _bool_col(df["near_volbar"])].head(top_n)
        if not _sig.empty:
            _sp = f"oversold_rebound_signal_{today}.csv"
            _sig.to_csv(_sp, index=False, encoding="utf-8-sig")
            print(f"年线长影线/倍量柱命中 {len(_sig)} 只 → {_sp}")
        else:
            print("年线长影线/倍量柱：0 只（今日无命中）")
    print("=" * 90)
    for i, row in top.iterrows():
        print(
            f"{i + 1:02d}. {row['code']} {row.get('name', '')} "
            f"| 分{row['score']:.3f} | 收盘{row['close']} "
            f"| 回撤{float(row.get('drawdown_max', 0)):.1%} "
            f"| 低点反弹{float(row.get('rebound_low20', 0)):.1%} "
            f"| MA20溢价{float(row.get('ma20_premium', 0)):.1%} "
            f"| 获利{_fmt_profit(row.get('profit_pct'))} "
            f"| EMA:{row.get('cluster_signal', '')} "
            f"| TA:{row.get('ta_label', '')} "
            f"| {row.get('label', '')}{_sig_tags(row)}"
        )
    print("=" * 90)
    print("说明：核心=超跌+EMA聚类；补全=RSI/MACD/布林。研究用，非投资建议。")

    if tsfresh_research:
        run_tsfresh_research(top["code"].astype(str).str.zfill(6).tolist())

    if do_notify:
        lines = [
            f"# 超跌反弹 {today}",
            "",
            f"共 {len(top)} 只 | 回撤≥{DRAWDOWN_MIN:.0%} 反弹{REBOUND_FROM_LOW_MIN:.0%}~{REBOUND_FROM_LOW_MAX:.0%}",
            f"获利 {PROFIT_MIN}~{PROFIT_MAX}%（有筹码模块时）| 日线贴近 MA20",
            "",
            "> 与红筹启动池不同：允许周线仍弱、获利偏低。",
            "",
        ]
        for i, row in top.iterrows():
            lines.append(
                f"{i + 1:02d}. {row['code']} {row.get('name', '')} "
                f"| 分{row['score']:.3f} | 收盘{row['close']} "
                f"| 回撤{float(row.get('drawdown_max', 0)):.1%} "
                f"| 反弹{float(row.get('rebound_low20', 0)):.1%} "
                f"| MA20溢价{float(row.get('ma20_premium', 0)):.1%} "
                f"| 获利{_fmt_profit(row.get('profit_pct'))} "
                f"| EMA:{row.get('cluster_signal', '')}"
                f"({row.get('cluster_label', '')}) "
                f"| TA:{row.get('ta_label', '')} "
                f"| {row.get('label', '')}{_sig_tags(row)}"
            )
        lines.append("\n仅为研究输出，不构成投资建议。")
        print("Server酱:", notify_serverchan(f"超跌反弹{today}|{len(top)}只", "\n".join(lines)))


def main() -> None:
    p = argparse.ArgumentParser(description="超跌反弹选股（研究）")
    p.add_argument("--shard", type=str, help="1-8 或 all")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    p.add_argument("--notify", action="store_true")
    p.add_argument(
        "--tsfresh-research",
        action="store_true",
        help="研究：对 Top 抽 tsfresh Minimal 特征（不参与打分）",
    )
    p.add_argument("--exclude-st", action="store_true", help="分片扫描时跳过 ST/*ST/退市整理股")
    p.add_argument(
        "--require",
        choices=["ma250", "volbar", "any"],
        default=None,
        help="形态硬过滤：ma250=年线长影线；volbar=倍量柱；any=二者其一（默认只标注加分）",
    )
    args = p.parse_args()

    if args.shard and args.shard != "all":
        try:
            _no = int(args.shard)
        except ValueError:
            p.error("--shard 只能是 1-8 或 all")
        if _no not in SHARD_MAP:
            p.error("--shard 只能是 1-8 或 all")

    if args.merge:
        merge_and_select(
            top_n=args.top,
            do_notify=args.notify,
            tsfresh_research=args.tsfresh_research,
            require=args.require,
        )
        return
    if not args.shard:
        p.error("请指定 --shard 或 --merge")
    if args.shard == "all":
        report_uncovered(get_all_a_stocks())
        for i in range(1, 9):
            run_shard(i, exclude_st=args.exclude_st, require=args.require)
        _req = f" --require {args.require}" if args.require else ""
        print(f"\n完成。执行: python oversold_rebound_selector.py --merge --top 20 --notify{_req}")
    else:
        run_shard(int(args.shard), exclude_st=args.exclude_st, require=args.require)


if __name__ == "__main__":
    main()
