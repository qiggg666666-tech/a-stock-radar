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
     · 底部放量突破200日线（融合自早期拐点猎手）：站上MA200、此前均价在其下方、近5日放量
     · 年线长影线：近5日内出现长下影线，探到/贴近250日均线后收回（年线支撑）
     · 倍量柱：近10日内出现低位倍量阳柱，且其后未跌破该柱开盘价/最低价（可带“后缩量守位”）

用法：
  python oversold_rebound_selector.py --shard 1
  python oversold_rebound_selector.py --shard all --exclude-st   # 跳过 ST/退市整理股
  python oversold_rebound_selector.py --shard all
  python oversold_rebound_selector.py --shard all --require ma250     # 只留年线长影线
  python oversold_rebound_selector.py --shard all --require volbar    # 只留倍量柱
  python oversold_rebound_selector.py --shard all --require bk200     # 只留底部放量突破200日线
  python oversold_rebound_selector.py --shard all --require any       # 三者其一
  python oversold_rebound_selector.py --shard all --max-12m-return 2.0   # 排除近12个月涨幅>200%
  python oversold_rebound_selector.py --merge --top 20 --fundamentals    # 候选池加查基本面领先指标
  python oversold_rebound_selector.py --fund-selftest 600519             # 诊断基本面接口是否可用
  python oversold_rebound_selector.py --shard all --inflection           # 同时记录全市场「放量突破200日线」候选
  python oversold_rebound_selector.py --merge --inflection-only --notify # 生成独立的早期拐点名单（含基本面）
  python oversold_rebound_selector.py --shard 1 --inflection-scan        # 只做全市场「放量突破200日线」扫描（不跑超跌反弹打分）
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


# ====================== 早期拐点：200日线放量突破 + 基本面领先指标 ======================
# 融合自 inflection_hunter.py：
#   扫描阶段（无额外接口）：近12个月涨幅 ret_12m、底部放量突破200日线 near_bk200
#   合并阶段（--fundamentals，只查候选池）：合同负债环比、毛利率连续改善、机构持股低、研报覆盖少、市值区间
# 接口返回 None 表示“数据不可用/未知”，不会被当成“不满足”；用 --fund-selftest 代码 可诊断各接口。
FUND_CAP_MIN = 20.0      # 亿元
FUND_CAP_MAX = 7000.0    # 亿元
BK200_BONUS = 0.10
FUND_W = {"contract": 0.12, "margin": 0.12, "inst_low": 0.08, "analyst_low": 0.08, "cap_sweet": 0.05}


def calc_ma200_breakout(df: pd.DataFrame) -> dict[str, Any]:
    """底部放量突破200日线：收盘站上MA200；此前(近60~5日)均价明显在MA200下方(<=98%)；近5日均量>=前55日均量1.3倍。"""
    out: dict[str, Any] = {
        "near_bk200": False,
        "bk200_label": "",
        "bk200_bonus": 0.0,
        "ma200": None,
        "ret_12m": None,
    }
    try:
        d = df.sort_values("date") if "date" in df.columns else df
        c = pd.to_numeric(d["close"], errors="coerce").to_numpy(float)
        n = len(c)
        if n >= 241 and np.isfinite(c[-1]) and np.isfinite(c[-240]) and c[-240] > 0:
            out["ret_12m"] = round(float(c[-1] / c[-240] - 1.0), 4)
        if n < 260 or "volume" not in d.columns:
            return out
        v = pd.to_numeric(d["volume"], errors="coerce").to_numpy(float)
        ma = pd.Series(c).rolling(200, min_periods=200).mean().to_numpy()
        if not (np.isfinite(ma[-1]) and ma[-1] > 0):
            return out
        out["ma200"] = round(float(ma[-1]), 2)
        if c[-1] <= ma[-1]:
            return out
        if np.nanmean(c[-60:-5]) > np.nanmean(ma[-60:-5]) * 0.98:
            return out
        base, recent = np.nanmean(v[-60:-5]), np.nanmean(v[-5:])
        if not (np.isfinite(base) and base > 0 and np.isfinite(recent) and recent >= 1.3 * base):
            return out
        out.update(near_bk200=True, bk200_label="底部放量突破200日线", bk200_bonus=BK200_BONUS)
    except Exception:
        pass
    return out


def _fd_prefix(code: str) -> str:
    return ("SH" if str(code).startswith(("6", "9")) else "SZ") + str(code)


def _fd_contract_df(code: str) -> pd.DataFrame:
    """资产负债表（东财，按报告期）。symbol 需带交易所前缀；先试带前缀，再试纯代码。"""
    last: Exception | None = None
    for sym in (_fd_prefix(code), str(code)):
        try:
            df = ak.stock_balance_sheet_by_report_em(symbol=sym)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last = e
    if last is not None:
        raise last
    return pd.DataFrame()


def _fd_margin_df(code: str) -> pd.DataFrame:
    try:
        return ak.stock_financial_analysis_indicator(symbol=str(code), start_year=str(datetime.now().year - 2))
    except TypeError:
        return ak.stock_financial_analysis_indicator(symbol=str(code))


def _fd_last_quarter_symbols(k: int = 6) -> list[str]:
    now = datetime.now()
    y, q = now.year, (now.month - 1) // 3 + 1
    syms = []
    for _ in range(k):
        q -= 1
        if q == 0:
            y, q = y - 1, 4
        syms.append(f"{y}{q}")
    return syms


def fd_contract_up(code: str) -> bool | None:
    """合同负债连续两期环比增长（最近3个报告期严格递增）。"""
    try:
        df = _fd_contract_df(code)
        if df.empty or "合同负债" not in df.columns or "REPORT_DATE" not in df.columns:
            return None
        s = pd.to_numeric(df["合同负债"], errors="coerce")
        s.index = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        s = s[s.index.notna()].dropna().sort_index()
        if len(s) < 3:
            return None
        a, b, c = (float(x) for x in s.iloc[-3:])
        return bool(c > b > a)
    except Exception:
        return None


def fd_margin_up(code: str) -> bool | None:
    """销售毛利率连续两期改善。显式按日期升序，避免依赖接口返回顺序。"""
    try:
        df = _fd_margin_df(code)
        if df is None or df.empty or "日期" not in df.columns:
            return None
        col = next((c for c in ("销售毛利率(%)", "销售毛利率", "毛利率") if c in df.columns), None)
        if col is None:
            return None
        g = pd.to_numeric(df[col], errors="coerce")
        g.index = pd.to_datetime(df["日期"], errors="coerce")
        g = g[g.index.notna()].dropna().sort_index()
        if len(g) < 3:
            return None
        a, b, c = (float(x) for x in g.iloc[-3:])
        return bool(c > b > a)
    except Exception:
        return None


def load_institute_map() -> dict[str, float]:
    """全市场机构持股比例（%）。该接口按“季度”取全表（如 20254），一次加载、逐股查表。"""
    for sym in _fd_last_quarter_symbols():
        try:
            df = ak.stock_institute_hold(symbol=sym)
            if df is None or df.empty or "证券代码" not in df.columns or "持股比例" not in df.columns:
                continue
            codes = df["证券代码"].astype(str).str.extract(r"(\d{6})")[0]
            ratio = pd.to_numeric(df["持股比例"], errors="coerce")
            if ratio.dropna().size and float(ratio.dropna().max()) <= 1.0:
                ratio = ratio * 100.0  # 兼容小数口径
            return {c: float(r) for c, r in zip(codes, ratio) if isinstance(c, str) and np.isfinite(r)}
        except Exception:
            continue
    return {}


def fd_analyst_low(code: str) -> bool | None:
    """近12个月覆盖的研报机构数 < 5。无研报 = 覆盖极少 = True；接口失败 = None。"""
    try:
        df = ak.stock_research_report_em(symbol=str(code))
    except Exception:
        return None
    try:
        if df is None:
            return None
        if df.empty:
            return True
        if "机构" not in df.columns or "日期" not in df.columns:
            return None
        dt = pd.to_datetime(df["日期"], errors="coerce")
        recent = df[dt > (datetime.now() - timedelta(days=365))]
        return bool(recent["机构"].nunique() < 5)
    except Exception:
        return None


def fd_market_cap_yi(code: str) -> float | None:
    try:
        df = ak.stock_individual_info_em(symbol=str(code))
        row = df[df["item"] == "总市值"]
        if row.empty:
            return None
        v = float(row["value"].iloc[0])
        return v / 1e8 if v > 1e6 else v  # 元 → 亿元
    except Exception:
        return None


def _fd_truth(v: Any) -> bool | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


import queue
import threading

# 基本面查询的运行参数（可由命令行 --fund-workers / --fund-budget / --fund-no-analyst 覆盖）
FUND_CFG: dict[str, Any] = {
    "workers": 4,          # 并发线程数（接口基本是网络等待，并发能大幅提速）
    "budget": 600.0,       # 整体时间预算（秒）：超时后停止查询，未查的按“未知”处理，保证合并一定能跑完
    "analyst": True,       # 是否查“研报覆盖”（该接口按页翻全部研报，最慢）
    "call_timeout": 25.0,  # 单次接口调用超时（秒），超时记为“未知”
}


def _call_timeout(fn: Any, timeout: float, *a: Any, **k: Any) -> tuple[bool, Any]:
    """在守护线程里调用 fn，超时就放弃（线程在后台自生自灭，不阻塞退出）。返回 (是否按时完成, 结果)。"""
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["v"] = fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive() or "e" in box:
        return False, None
    return True, box.get("v")


def _fund_one(code: str, inst_map: dict[str, float], cap_min: float, cap_max: float) -> dict[str, Any]:
    to = float(FUND_CFG["call_timeout"])
    rec: dict[str, Any] = {"cap": None, "contract": None, "margin": None, "inst_low": None, "analyst_low": None}
    _, rec["cap"] = _call_timeout(fd_market_cap_yi, to, code)
    cap = rec["cap"]
    if cap is not None and not (cap_min <= cap <= cap_max):
        return rec  # 市值不在区间：之后会被剔除，不必再查其余接口
    _, rec["contract"] = _call_timeout(fd_contract_up, to, code)
    _, rec["margin"] = _call_timeout(fd_margin_up, to, code)
    if inst_map:
        r = inst_map.get(code, 0.0)  # 表里没有 = 机构几乎不持有
        rec["inst_low"] = bool(r < 50.0)
    if FUND_CFG["analyst"]:
        _, rec["analyst_low"] = _call_timeout(fd_analyst_low, to, code)
    return rec


def enrich_fundamentals(
    df: pd.DataFrame,
    score_cols: list[str],
    pool_n: int,
    cap_min: float = FUND_CAP_MIN,
    cap_max: float = FUND_CAP_MAX,
    min_leading: int = 0,
    sleep: float = 0.0,
) -> pd.DataFrame:
    """
    只对得分靠前的候选池查基本面。多线程并发 + 单次调用超时 + 整体时间预算：
    接口慢/被限流时不会把整个 workflow 拖到超时，查不到的指标一律记“未知”（不加分也不剔除）。
    新增列：market_cap_yi / fd_contract_up / fd_margin_up / fd_inst_low / fd_analyst_low /
            fd_leading_n / fd_bonus / fd_label / fd_missing。
    min_leading：领先指标（合同负债+毛利率+机构低+研报少+200日线突破）至少满足几项，0=不过滤。
    市值未知的股票不会因市值被剔除。
    """
    if df is None or df.empty:
        return df
    os.environ.setdefault("TQDM_DISABLE", "1")  # 关掉 akshare 的进度条，避免日志被刷屏
    rank = df[score_cols].max(axis=1)
    pool = df.loc[rank.sort_values(ascending=False).index[:pool_n]].copy().reset_index(drop=True)
    codes = pool["code"].astype(str).str.zfill(6).tolist()
    workers = max(1, min(int(FUND_CFG["workers"]), len(codes)))
    budget = float(FUND_CFG["budget"])
    print(
        f"基本面领先指标：候选池 {len(codes)} 只 | 并发{workers} | 时间预算{budget:.0f}s "
        f"| 研报覆盖={'查' if FUND_CFG['analyst'] else '跳过'}"
    )

    ok, inst_map = _call_timeout(load_institute_map, 90.0)
    inst_map = inst_map if ok and inst_map else {}
    if not inst_map:
        print("提示：机构持股数据不可用，该项记为未知（不扣分）")

    q: "queue.Queue[str]" = queue.Queue()
    for c in codes:
        q.put(c)
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def _worker() -> None:
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            try:
                r = _fund_one(c, inst_map, cap_min, cap_max)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None:
                with lock:
                    results[c] = r

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(workers)]
    t0 = time.monotonic()
    for th in threads:
        th.start()
    last_print = 0
    timed_out = False
    while any(th.is_alive() for th in threads):
        time.sleep(0.5)
        with lock:
            done = len(results)
        if done - last_print >= 10:
            last_print = done
            print(f"  基本面 {done}/{len(codes)}（已用 {int(time.monotonic() - t0)}s）")
        if time.monotonic() - t0 > budget:
            timed_out = True
            while not q.empty():  # 清空队列，让线程尽快收尾
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            break
    with lock:
        snap = dict(results)
    if timed_out:
        print(f"提示：基本面查询达到时间预算 {budget:.0f}s，已完成 {len(snap)}/{len(codes)}，"
              "其余按“未查”处理（不加分、不剔除）。可用 --fund-budget 调大。")

    names = {"合同负债": "contract", "毛利率": "margin", "机构持股": "inst_low"}
    if FUND_CFG["analyst"]:
        names["研报覆盖"] = "analyst_low"
    recs = []
    for code in codes:
        r = snap.get(code)
        if r is None:
            recs.append({
                "market_cap_yi": None, "fd_contract_up": None, "fd_margin_up": None,
                "fd_inst_low": None, "fd_analyst_low": None, "fd_bonus": 0.0,
                "fd_label": "", "fd_missing": "未查(超时)",
            })
            continue
        cap = r["cap"]
        flags = {k: r[k] for k in ("contract", "margin", "inst_low", "analyst_low")}
        bonus = sum(FUND_W[k] for k, v in flags.items() if v is True)
        if cap is not None and 30.0 <= cap <= 400.0:
            bonus += FUND_W["cap_sweet"]
        tags = []
        if flags["contract"]:
            tags.append("合同负债连增")
        if flags["margin"]:
            tags.append("毛利率连升")
        if flags["inst_low"]:
            tags.append("机构低配")
        if flags["analyst_low"]:
            tags.append("研报覆盖少")
        skipped_by_cap = cap is not None and not (cap_min <= cap <= cap_max)
        recs.append({
            "market_cap_yi": round(cap, 1) if cap is not None else None,
            "fd_contract_up": flags["contract"],
            "fd_margin_up": flags["margin"],
            "fd_inst_low": flags["inst_low"],
            "fd_analyst_low": flags["analyst_low"],
            "fd_bonus": round(bonus, 3),
            "fd_label": "+".join(tags),
            "fd_missing": "" if skipped_by_cap else ",".join(n for n, k in names.items() if flags[k] is None),
        })

    fd = pd.DataFrame(recs)
    out = pd.concat([pool.reset_index(drop=True), fd], axis=1)
    bk = out["near_bk200"].map(_fd_truth) if "near_bk200" in out.columns else pd.Series(None, index=out.index)
    out["fd_leading_n"] = (
        out[["fd_contract_up", "fd_margin_up", "fd_inst_low", "fd_analyst_low"]]
        .apply(lambda col: col.map(lambda x: 1 if x is True else 0))
        .sum(axis=1)
        + bk.map(lambda x: 1 if x is True else 0)
    ).astype(int)

    n0 = len(out)
    cap_s = pd.to_numeric(out["market_cap_yi"], errors="coerce")
    ok_cap = cap_s.isna() | ((cap_s >= cap_min) & (cap_s <= cap_max))
    out = out.loc[ok_cap].reset_index(drop=True)
    print(f"市值过滤 {cap_min:g}~{cap_max:g}亿（未知市值保留）: {n0} → {len(out)}")
    if min_leading > 0:
        n1 = len(out)
        out = out.loc[out["fd_leading_n"] >= min_leading].reset_index(drop=True)
        print(f"领先指标 ≥{min_leading} 项: {n1} → {len(out)}")
    miss = int((out["fd_missing"].astype(str) != "").sum()) if len(out) else 0
    if miss:
        print(f"提示：{miss} 只存在取不到/未查的指标（fd_missing 列），这些指标按“未知”处理、不加分也不扣分")
    return out


def run_fund_selftest(code: str) -> None:
    """诊断：逐个调用基本面接口，打印行数/列名/异常，确认你的环境（含 GitHub Actions）里哪些接口可用。"""
    code = str(code).strip().zfill(6)
    print(f"基本面接口自检: {code}")
    probes = [
        ("个股信息-市值 stock_individual_info_em", lambda: ak.stock_individual_info_em(symbol=code)),
        ("资产负债表 stock_balance_sheet_by_report_em", lambda: _fd_contract_df(code)),
        ("财务指标 stock_financial_analysis_indicator", lambda: _fd_margin_df(code)),
        ("个股研报 stock_research_report_em", lambda: ak.stock_research_report_em(symbol=code)),
        (f"机构持股 stock_institute_hold({_fd_last_quarter_symbols(1)[0]})",
         lambda: ak.stock_institute_hold(symbol=_fd_last_quarter_symbols(1)[0])),
    ]
    for name, fn in probes:
        try:
            r = fn()
            cols = list(r.columns)[:14] if hasattr(r, "columns") else []
            print(f"  [OK ] {name}: {getattr(r, 'shape', '?')} 列={cols}")
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {str(e)[:160]}")
    print(
        "  结果:",
        {
            "市值(亿)": fd_market_cap_yi(code),
            "合同负债连增": fd_contract_up(code),
            "毛利率连升": fd_margin_up(code),
            "研报覆盖少": fd_analyst_low(code),
            "机构持股(%)": load_institute_map().get(code),
        },
    )


INFL_MAX_12M_DEFAULT = 2.0   # 与原脚本一致：近12个月涨幅>200% 视为已启动，排除


def make_inflection_row(code: str, df: pd.DataFrame, max_12m: float | None) -> dict[str, Any] | None:
    """扫描阶段：不依赖筹码/超跌等任何硬过滤，只要“底部放量突破200日线”且近12个月涨幅不过大，就记录下来。"""
    bk = calc_ma200_breakout(df)
    if not bk.get("near_bk200"):
        return None
    lim = INFL_MAX_12M_DEFAULT if max_12m is None else max_12m
    if bk.get("ret_12m") is not None and bk["ret_12m"] > lim:
        return None
    d = df.sort_values("date") if "date" in df.columns else df
    return {
        "code": str(code).zfill(6),
        "date": str(pd.Timestamp(d["date"].iloc[-1]).date()) if "date" in d.columns else "",
        "close": round(float(d["close"].iloc[-1]), 2),
        "ma200": bk.get("ma200"),
        "ret_12m": bk.get("ret_12m"),
        "near_bk200": True,
        "bk200_label": bk.get("bk200_label"),
    }


def run_inflection(
    cache_dir: Any,
    top_n: int,
    do_notify: bool,
    notify_fn: Any,
    fill_names_fn: Any,
    cap_min: float = FUND_CAP_MIN,
    cap_max: float = FUND_CAP_MAX,
    min_leading: int = 2,
    fund_pool: int = 120,
) -> None:
    """
    早期拐点名单（独立于主选股结果）：
      全市场“底部放量突破200日线”候选（扫描阶段 --inflection 记录）→ 基本面领先指标 → 按满足项数排序。
    领先指标 = 200日线突破(已满足) + 合同负债连增 + 毛利率连升 + 机构低配 + 研报覆盖少。
    """
    files = sorted(Path(cache_dir).glob("inflection_shard_*.csv"))
    if not files:
        print("没有拐点分片结果：请先带 --inflection 重新运行 --shard（旧版本跑出的分片不含这份数据）")
        return
    _today = datetime.now().date()
    _old = [f.name for f in files if datetime.fromtimestamp(f.stat().st_mtime).date() != _today]
    if _old:
        print(f"提示：以下拐点分片不是今天生成的（可能是旧数据）: {_old}")
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.drop_duplicates("code", keep="last").reset_index(drop=True)
    print(f"拐点候选（底部放量突破200日线、近12个月涨幅未过大）共 {len(df)} 只")
    if df.empty:
        return
    df = fill_names_fn(df)
    df["_s"] = -pd.to_numeric(df["ret_12m"], errors="coerce").fillna(0.0)  # 池子大于 fund_pool 时优先查涨幅小的

    out = enrich_fundamentals(
        df, ["_s"], pool_n=fund_pool, cap_min=cap_min, cap_max=cap_max, min_leading=0
    )
    if out is None or out.empty:
        print("市值过滤后无标的")
        return
    dist = out["fd_leading_n"].value_counts().sort_index(ascending=False).to_dict()
    print(f"领先指标满足项数分布（含200日线突破）: {dist}")
    out = out.loc[out["fd_leading_n"] >= min_leading]
    if out.empty:
        print(f"没有标的满足 ≥{min_leading} 项领先指标。可用 --inflection-min-leading 1 放宽；"
              "若 fd_missing 大面积不为空，说明基本面接口取不到数，请先跑 --fund-selftest。")
        return
    out = (
        out.sort_values(["fd_leading_n", "fd_bonus"], ascending=[False, False])
        .head(top_n)
        .drop(columns=["_s"])
        .reset_index(drop=True)
    )
    today = datetime.now().strftime("%Y%m%d")
    path = f"inflection_candidates_{today}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    lines = []
    print("\n" + "=" * 90)
    print(f"早期拐点候选 Top{len(out)} → {path}")
    print("=" * 90)
    for i, r in out.iterrows():
        cap = r.get("market_cap_yi")
        cap_s = f"{cap}亿" if pd.notna(cap) else "市值?"
        line = (
            f"{i + 1:02d}. {r['code']} {r.get('name', '')} | 满足{int(r['fd_leading_n'])}项 | 收盘{r['close']} "
            f"| MA200={r['ma200']} | 12月涨幅{float(r['ret_12m']) * 100:.0f}% | {cap_s} "
            f"| {r.get('fd_label') or '（基本面无满足项）'}"
            + (f" | 缺失:{r['fd_missing']}" if str(r.get("fd_missing") or "") not in ("", "nan") else "")
        )
        print(line)
        lines.append(line)
    print("说明：研究用名单，不是买点；基本面取自东财/新浪公开接口，缺失项不扣分。非投资建议。")
    if do_notify:
        print("Server酱:", notify_fn(f"早期拐点{today}|{len(out)}只", "\n\n".join(lines) + "\n\n仅为研究输出，不构成投资建议。"))


def _fd_tag(row: Any) -> str:
    """200日线突破 / 基本面标签的展示文本。"""
    parts = []
    bk = str(row.get("bk200_label") or "")
    if bk and bk != "nan":
        parts.append(bk)
    fd = str(row.get("fd_label") or "")
    cap = row.get("market_cap_yi")
    cap_ok = cap is not None and str(cap) not in ("", "nan", "None")
    if fd and fd != "nan":
        parts.append(f"基本面:{fd}" + (f"(市值{cap}亿)" if cap_ok else ""))
    elif cap_ok:
        parts.append(f"市值{cap}亿")
    return (" | " + " | ".join(parts)) if parts else ""


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


def _pass_require(sig_ma250: bool, sig_volbar: bool, sig_bk200: bool, require: str | None) -> bool:
    if not require:
        return True
    if require == "ma250":
        return sig_ma250
    if require == "volbar":
        return sig_volbar
    if require == "bk200":
        return sig_bk200
    return sig_ma250 or sig_volbar or sig_bk200  # any


def score_oversold_rebound(
    df: pd.DataFrame, code: str, require: str | None = None, max_12m: float | None = None
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
    sig_bk = calc_ma200_breakout(d)
    if max_12m is not None and sig_bk.get("ret_12m") is not None and sig_bk["ret_12m"] > max_12m:
        return None  # 近12个月已大涨，不属于“早期”
    if not _pass_require(
        bool(sig_ma["near_ma250_shadow"]), bool(sig_vb["near_volbar"]), bool(sig_bk["near_bk200"]), require
    ):
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
    score += float(sig_ma["ma250_bonus"]) + float(sig_vb["volbar_bonus"]) + float(sig_bk["bk200_bonus"])  # 年线长影线 / 倍量柱 / 200日线突破

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
    if sig_bk["near_bk200"]:
        label += "·放量突破200日线"

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
        "near_bk200": bool(sig_bk["near_bk200"]),
        "bk200_label": sig_bk["bk200_label"],
        "bk200_bonus": sig_bk["bk200_bonus"],
        "ma200": sig_bk["ma200"],
        "ret_12m": sig_bk["ret_12m"],
        "label": label,
        "score": round(float(score), 4),
    }


def run_shard(
    shard_id: int,
    exclude_st: bool = False,
    require: str | None = None,
    max_12m: float | None = None,
    inflection: bool = False,
    inflection_only: bool = False,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"超跌反弹 分片 {shard_id} | {SHARD_MAP.get(shard_id)} | exclude_st={exclude_st} | require={require} | max_12m={max_12m}")
    print(f"{'=' * 60}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # 缓存目录被清理时也能正常写入
    out = CACHE_DIR / f"osr_shard_{shard_id}.csv"
    if out.exists() and not inflection_only:  # 只做拐点扫描时不要动超跌反弹的分片结果
        out.unlink()  # 先清旧结果：本次无命中/中途失败时 merge 不会读到过期数据
    infl_file = CACHE_DIR / f"inflection_shard_{shard_id}.csv"
    if infl_file.exists():
        infl_file.unlink()
    infl_rows: list[dict[str, Any]] = []

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
            if inflection:  # 独立于超跌/回撤等条件：全市场记录“底部放量突破200日线”
                _ir = make_inflection_row(code, df, max_12m)
                if _ir:
                    infl_rows.append(_ir)
            if inflection_only:
                continue  # 拐点扫描：不跑超跌反弹打分
            row = score_oversold_rebound(df, code, require=require, max_12m=max_12m)
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
    if inflection:
        print(f"分片 {shard_id} 拐点候选(放量突破200日线): {len(infl_rows)} 只（有效样本 {n_data}）")
        if infl_rows:
            pd.DataFrame(infl_rows).to_csv(infl_file, index=False, encoding="utf-8-sig")
    if inflection_only:
        return
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
    return ((" | " + " | ".join(parts)) if parts else "") + _fd_tag(row)


def _fmt_profit(v: Any) -> str:
    """无筹码模块时 profit_pct 为空，避免打印 'nan%'。"""
    x = pd.to_numeric(v, errors="coerce")
    return "-" if pd.isna(x) else f"{float(x):.1f}%"


def _fill_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    m = get_name_map()
    df["name"] = df["code"].map(lambda c: m.get(c, ""))
    return df


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
    fundamentals: bool = False,
    fund_pool: int | None = None,
    cap_min: float = FUND_CAP_MIN,
    cap_max: float = FUND_CAP_MAX,
    min_leading: int = 0,
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
    if "near_bk200" not in df.columns:
        print("警告：分片结果缺少 near_bk200 列——这是旧版脚本生成的分片，200日线突破/基本面相关数据不会生效，请重新运行 --shard。")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.sort_values("score", ascending=False).drop_duplicates("code", keep="first")  # 先排序再去重，保留高分
    df = df.reset_index(drop=True)
    # CSV 读回后空字符串变 NaN，打印会出现 "nan"
    for _c in ("name", "cluster_signal", "cluster_label", "ta_label", "label", "ma250_label", "volbar_label", "ma250_bar_date", "vb_date", "bk200_label", "fd_label", "fd_missing"):
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
        _b = _bool_col(df["near_bk200"]) if "near_bk200" in df.columns else pd.Series(False, index=df.index)
        _keep = _m if require == "ma250" else _v if require == "volbar" else _b if require == "bk200" else (_m | _v | _b)
        df = df.loc[_keep].reset_index(drop=True)
        print(f"形态过滤 require={require}: {before} → {len(df)}")
    if fundamentals and not df.empty:
        df = enrich_fundamentals(
            df,
            ["score"],
            pool_n=fund_pool or max(top_n * 4, 60),
            cap_min=cap_min,
            cap_max=cap_max,
            min_leading=min_leading,
        )
        if df is not None and not df.empty:
            df["score"] = df["score"] + pd.to_numeric(df["fd_bonus"], errors="coerce").fillna(0.0)
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
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
        _anysig = _bool_col(df["near_ma250_shadow"]) | _bool_col(df["near_volbar"])
        if "near_bk200" in df.columns:
            _anysig = _anysig | _bool_col(df["near_bk200"])
        _sig = df.loc[_anysig].head(top_n)
        if not _sig.empty:
            _sp = f"oversold_rebound_signal_{today}.csv"
            _sig.to_csv(_sp, index=False, encoding="utf-8-sig")
            print(f"年线长影线/倍量柱/200日线突破命中 {len(_sig)} 只 → {_sp}")
        else:
            print("年线长影线/倍量柱/200日线突破：0 只（今日无命中）")
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
        choices=["ma250", "volbar", "bk200", "any"],
        default=None,
        help="形态硬过滤：ma250=年线长影线；volbar=倍量柱；bk200=底部放量突破200日线；any=三者其一（默认只标注加分）",
    )
    p.add_argument("--max-12m-return", type=float, default=None, help="排除近12个月涨幅超过该值的股票（2.0=200%%；默认不过滤）")
    p.add_argument("--fundamentals", action="store_true", help="merge 时对候选池加查基本面领先指标（较慢）")
    p.add_argument("--fund-pool", type=int, default=None, help="基本面候选池大小（默认 max(top*4, 60)）")
    p.add_argument("--cap-min", type=float, default=FUND_CAP_MIN, help="市值下限（亿元，仅 --fundamentals 生效）")
    p.add_argument("--cap-max", type=float, default=FUND_CAP_MAX, help="市值上限（亿元，仅 --fundamentals 生效）")
    p.add_argument("--min-leading", type=int, default=0, help="领先指标至少满足几项（0=不过滤；原脚本为3）")
    p.add_argument(
        "--inflection",
        action="store_true",
        help="--shard 时：全市场记录「底部放量突破200日线」候选；--merge 时：在主结果之外再输出早期拐点名单",
    )
    p.add_argument(
        "--inflection-scan",
        action="store_true",
        help="--shard 时：只做全市场「底部放量突破200日线」扫描并写 inflection_shard_N.csv，不跑超跌反弹打分",
    )
    p.add_argument("--inflection-only", action="store_true", help="--merge 时只生成早期拐点名单，不跑主选股合并")
    p.add_argument(
        "--inflection-min-leading",
        type=int,
        default=2,
        help="拐点名单：领先指标（含200日线突破）至少满足几项，默认2；原脚本为3",
    )
    p.add_argument("--fund-workers", type=int, default=4, help="基本面查询并发线程数（默认4）")
    p.add_argument("--fund-budget", type=float, default=600.0, help="基本面查询总时间预算（秒，默认600；超时后其余记为未知）")
    p.add_argument("--fund-no-analyst", action="store_true", help="跳过“研报覆盖”指标（该接口按页翻全部研报，最慢）")
    p.add_argument("--fund-selftest", type=str, default="", help="诊断：对指定代码逐个测试基本面接口后退出")
    args = p.parse_args()
    FUND_CFG["workers"] = args.fund_workers
    FUND_CFG["budget"] = args.fund_budget
    FUND_CFG["analyst"] = not args.fund_no_analyst

    if args.fund_selftest:
        run_fund_selftest(args.fund_selftest)
        return

    if args.shard and args.shard != "all":
        try:
            _no = int(args.shard)
        except ValueError:
            p.error("--shard 只能是 1-8 或 all")
        if _no not in SHARD_MAP:
            p.error("--shard 只能是 1-8 或 all")

    if args.merge:
        if args.inflection_only:
            run_inflection(
                CACHE_DIR, args.top, args.notify, notify_serverchan, _fill_names,
                args.cap_min, args.cap_max, args.inflection_min_leading, args.fund_pool or 120,
            )
            return
        merge_and_select(
            top_n=args.top,
            do_notify=args.notify,
            tsfresh_research=args.tsfresh_research,
            require=args.require,
            fundamentals=args.fundamentals,
            fund_pool=args.fund_pool,
            cap_min=args.cap_min,
            cap_max=args.cap_max,
            min_leading=args.min_leading,
        )
        if args.inflection:
            run_inflection(
                CACHE_DIR, args.top, args.notify, notify_serverchan, _fill_names,
                args.cap_min, args.cap_max, args.inflection_min_leading, args.fund_pool or 120,
            )
        return
    if not args.shard:
        p.error("请指定 --shard 或 --merge")
    if args.shard == "all":
        report_uncovered(get_all_a_stocks())
        for i in range(1, 9):
            run_shard(i, exclude_st=args.exclude_st, require=args.require, max_12m=args.max_12m_return,
                      inflection=args.inflection or args.inflection_scan, inflection_only=args.inflection_scan)
        _req = (f" --require {args.require}" if args.require else "") + (" --fundamentals" if args.fundamentals else "") + (" --inflection" if args.inflection else "")
        print(f"\n完成。执行: python oversold_rebound_selector.py --merge --top 20 --notify{_req}")
    else:
        run_shard(int(args.shard), exclude_st=args.exclude_st, require=args.require, max_12m=args.max_12m_return,
                  inflection=args.inflection or args.inflection_scan, inflection_only=args.inflection_scan)


if __name__ == "__main__":
    main()
