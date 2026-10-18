#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_event_study.py —— 标签历史回放事件研究（校准阈值、检验标签是否真有用）

做什么
  对一批股票，在过去每隔 step 个交易日的每一天（“as-of 日”），只用该日及以前的数据重新计算我们的标签：
    圆弧底（rb_level）、拐点（near_bk200）、低位破低反包、深跌巨量强收、长下影、
    MACD-V / ADX / 振幅÷ATR / 突破日K线质量 / QQC-lite / 上方密集区放量突破，以及压力位(目标价)。
  再用“次日开盘买入、第 h 个交易日收盘卖出”（T+1，扣交易成本）统计之后的真实表现，
  并与“同期所有 as-of 日的平均表现”（基准）比较，得到各标签的超额(lift)。
  按时间切成 训练 / 测试 两段，并再切成若干个时间块看符号是否稳定，用来回答：
    “这个标签在样本外是不是真的比没有标签好？”

为什么不直接 Optuna 调参
  标签触发的次数往往很少，调参很容易过拟合。这里只做“检验与校准”：先看标签在样本外有没有超额，
  再人工决定要不要改阈值/加分。想改阈值：改动后重新跑一遍，对比样本外结果。

口径与限制（请务必看）
  * 无前视：as-of 日的特征只用该日及以前的行情；大盘 ATR 档位（圆弧底参数）也是按当日回算的。
  * T+1：次日开盘买入，最早第 2 个交易日收盘卖出（默认 horizons=2,3,5,10,20）；次日开盘涨停买不进的样本剔除。
  * 成本：默认往返 0.2%（--cost），请按自己的实际调整；不含冲击成本。
  * 事件去重：同一只股票同一个信号，--cooldown 个交易日内只算第一次，避免连续几天重复计数。
  * 股票池只含“目前仍在上市”的股票（幸存者偏差，会让结果偏乐观）；随机抽样，不是按代码排序。
  * 同时检验很多个信号，必然会有一些“看起来有效”的纯属偶然；看 ★/✓ 之前先看样本量和置信区间。
  * 仅供研究，非投资建议。

用法
  python tag_event_study.py --max-stocks 200 --years 4 --step 5 --workers 4
  python tag_event_study.py --events-in es_out/event_study_events_20261012.csv      # 只重新分析，不重算特征
  python tag_event_study.py --max-stocks 300 --focus 5 --min-n 40 \\
      --combo "near_bk200 and candle_tag == '强阳线'"
耗时：约 每只股票 × 每个 as-of 日 30~60ms；200 只 × 4 年 × step5 ≈ 15~30 分钟（可 --workers 并行）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("FINAL_CHIP_RB_MARKET", "off")   # 圆弧底的大盘档位在本脚本里按当日回算，不联网

WARMUP_DEFAULT = 260          # 需要至少 260 根日线（MA200/年线）才开始回放
WINDOW_BARS = 420             # 每个 as-of 日只取最近 420 根计算特征（足够，且控制耗时）
FEATURE_KEYS = [
    # 变盘 / 形态
    "near_bk200", "low_bullish_outside", "deep_huge_strong", "ret_12m", "turn_dd60", "turn_vr", "turn_clv",
    "daily_long_lower", "lower_shadow_to_body",
    # 圆弧底
    "near_round_bottom", "rb_level", "rb_depth", "rb_bowl_low", "rb_left_peak", "rb_breakout_price", "rb_atr_pct", "rb_market_vol",
    # 动能 / K线 / 密集区
    "macdv", "macdv_tag", "adx", "adx_tag", "tr_atr", "atr_tag", "candle_tag", "body_ratio", "body_rank",
    "qqc_tag", "qqc_green", "vz_tag", "vz_mult",
    # 压力位
    "tp_primary_lo", "tp_primary_hi", "tp_primary_score", "tp_upside_pct",
]
CAT_TAGS = ["adx_tag", "atr_tag", "macdv_tag", "candle_tag", "qqc_tag", "vz_tag"]

_CFG: dict[str, Any] = {}
_IDX: pd.DataFrame | None = None
_MODS: dict[str, Any] = {}


# ----------------------------------------------------------------------------
# 依赖与数据
# ----------------------------------------------------------------------------
def _deps() -> dict[str, Any]:
    if _MODS:
        return _MODS
    import final_chip_research as R
    import signal_backtest as SB
    import target_price as T
    import vol_momentum as V
    _MODS.update(R=R, SB=SB, T=T, V=V)
    return _MODS


def fetch_history(code: str, years: float, cache_dir: Path) -> pd.DataFrame:
    """前复权日线（已标准化：date/open/high/low/close/volume/amount/turnover(小数)/pe_ttm）。当天缓存。"""
    R = _deps()["R"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{code}.csv"
    if path.exists() and datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date():
        try:
            return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            pass
    end = datetime.now().date()
    start = end - timedelta(days=int(years * 365) + 40)
    df = pd.DataFrame()
    for label, fn in (("baostock", R._bs_history), ("akshare", R._ak_history)):
        for _ in range(2):
            try:
                df = R.provider_call(f"{label}:{code}", 40.0, lambda f=fn: f(code, start.isoformat(), end.isoformat()))
                if len(df) >= WARMUP_DEFAULT + 40:
                    break
            except Exception:
                df = pd.DataFrame()
        if len(df) >= WARMUP_DEFAULT + 40:
            break
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def fetch_index(years: float, cache_dir: Path) -> pd.DataFrame:
    """沪深300：用于基准收益和大盘 ATR 档位。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "_index_sh000300.csv"
    if path.exists() and datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date():
        return pd.read_csv(path, parse_dates=["date"])
    import akshare as ak
    raw = ak.stock_zh_index_daily(symbol="sh000300")
    d = raw[["date", "open", "high", "low", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d.to_csv(path, index=False)
    return d


def get_universe(max_stocks: int, seed: int) -> list[str]:
    """全市场在市股票（剔除 ST/退市/北交所），随机抽样 max_stocks 只。"""
    R = _deps()["R"]
    codes: list[str] = []
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == "0":
            try:
                rs = bs.query_stock_basic()
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    code, name = row[0], str(row[1])
                    if len(row) > 5 and row[4] == "1" and row[5] == "1":
                        c = code.split(".")[-1].zfill(6)
                        if "ST" in name.upper() or "退" in name or R.is_beijing_stock(c):
                            continue
                        if c.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")):
                            codes.append(c)
            finally:
                bs.logout()
    except Exception as e:  # noqa: BLE001
        print(f"baostock 股票列表失败: {e}")
    if not codes:
        import akshare as ak
        info = ak.stock_info_a_code_name()
        ok = ~info["name"].astype(str).str.contains("ST|退", na=False)
        codes = [c for c in info.loc[ok, "code"].astype(str).str.zfill(6) if not R.is_beijing_stock(c)]
    codes = sorted(set(codes))
    rng = np.random.default_rng(seed)
    if max_stocks and len(codes) > max_stocks:
        codes = sorted(rng.choice(codes, size=max_stocks, replace=False).tolist())
    return codes


# ----------------------------------------------------------------------------
# 回放：特征（只用 as-of 日及以前）
# ----------------------------------------------------------------------------
def market_state_asof(idx: pd.DataFrame, pos: int) -> dict[str, Any]:
    """与 final_chip_research.rb_market_state 同口径：沪深300 的 ATR(14) 在近 120 日的分位 → 圆弧底参数档位。"""
    normal = {"market_vol": "normal", "PULLBACK_MIN": 0.53, "VOL_EXPAND": 1.95, "DI_LIANG_RATIO": 0.55, "N": 100, "M": 20}
    w = idx.iloc[max(0, pos - 149): pos + 1]
    if len(w) < 100:
        return normal
    tr = pd.concat([w["high"] - w["low"], (w["high"] - w["close"].shift(1)).abs(), (w["low"] - w["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    pct = float((atr.iloc[-120:] < atr.iloc[-1]).mean())
    if pct > 0.70:
        return {"market_vol": "high", "PULLBACK_MIN": 0.48, "VOL_EXPAND": 2.15, "DI_LIANG_RATIO": 0.60, "N": 90, "M": 18}
    if pct < 0.30:
        return {"market_vol": "low", "PULLBACK_MIN": 0.58, "VOL_EXPAND": 1.75, "DI_LIANG_RATIO": 0.48, "N": 115, "M": 22}
    return normal


def compute_features(window: pd.DataFrame, code: str, market: dict[str, Any], with_targets: bool) -> dict[str, Any]:
    m = _deps()
    R, V, T = m["R"], m["V"], m["T"]
    out: dict[str, Any] = {}
    turn = R.compute_turning_features(window)
    out.update({k: turn.get(k) for k in ("near_bk200", "low_bullish_outside", "deep_huge_strong", "ret_12m", "turn_dd60", "turn_vr", "turn_clv")})
    sh = R.compute_shadow_features(window)
    out.update({k: sh.get(k) for k in ("daily_long_lower", "lower_shadow_to_body")})
    rb = R.calc_round_bottom(window, market=market)
    out.update({k: rb.get(k) for k in ("near_round_bottom", "rb_level", "rb_depth", "rb_bowl_low", "rb_left_peak",
                                       "rb_breakout_price", "rb_atr_pct", "rb_market_vol")})
    vm = V.calc_vol_momentum(window)
    out.update({k: vm.get(k) for k in ("macdv", "macdv_tag", "adx", "adx_tag", "tr_atr", "atr_tag", "candle_tag",
                                       "body_ratio", "body_rank", "qqc_tag", "qqc_green", "vz_tag", "vz_mult")})
    if with_targets:
        tp = T.target_fields(window, code, "")
        out.update({k: tp.get(k) for k in ("tp_primary_lo", "tp_primary_hi", "tp_primary_score", "tp_upside_pct")})
    return out


def _init_worker(idx: pd.DataFrame | None, cfg: dict[str, Any]) -> None:
    global _IDX, _CFG
    _IDX, _CFG = idx, cfg
    R = _deps()["R"]
    R.RB_CFG["min_level"] = "daily"      # 记录全部级别（daily/medium/high），分析时再按级别筛选


def process_stock(code: str) -> list[dict[str, Any]]:
    """一只股票的全部 as-of 日：特征 + 前向收益（T+1、含成本）。"""
    cfg = _CFG
    m = _deps()
    SB = m["SB"]
    try:
        df = fetch_history(code, cfg["years"], Path(cfg["cache_dir"]))
    except Exception:
        return []
    n = len(df)
    if n < cfg["warmup"] + 10:
        return []
    horizons: list[int] = cfg["horizons"]
    limit = SB.limit_pct(code, "")
    idx = _IDX
    idx_dates = idx["date"].to_numpy() if idx is not None else None
    rows: list[dict[str, Any]] = []
    ms_cache: dict[int, dict[str, Any]] = {}
    for t in range(cfg["warmup"], n - 1, cfg["step"]):
        window = df.iloc[max(0, t - WINDOW_BARS + 1): t + 1]
        date = pd.Timestamp(df["date"].iloc[t])
        if idx is not None:
            pos = int(np.searchsorted(idx_dates, date.to_datetime64(), side="right") - 1)
            if pos not in ms_cache:
                ms_cache[pos] = market_state_asof(idx, pos) if pos >= 0 else market_state_asof(idx, 0)
            market = ms_cache[pos]
        else:
            market = market_state_asof(pd.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]}), 0)
        try:
            feat = compute_features(window, code, market, cfg["with_targets"])
        except Exception:
            continue
        # 前向收益：信号日 date 收盘后出信号 → 次日开盘买 → 第 h 个交易日收盘卖
        tp_lo = feat.get("tp_primary_lo")
        sup = None
        for c in ("rb_bowl_low",):
            v = feat.get(c)
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                sup = float(v)
        st = SB.forward_stats(
            df, date, horizons, "open", limit, 0,
            tp_lo=float(tp_lo) if tp_lo is not None and pd.notna(tp_lo) else None, sup=sup,
        )
        if not (st.get("buyable") == st.get("buyable")) or st.get("buyable") == 0:     # NaN 或次日开盘涨停：买不进
            continue
        row = {"code": code, "date": date, "bar_idx": t}
        row.update(feat)
        for h in horizons:
            r = st.get(f"ret_{h}")
            row[f"ret_{h}"] = (r - cfg["cost"]) if (r is not None and np.isfinite(r)) else np.nan
            row[f"tp_hit_{h}"] = st.get(f"tp_hit_{h}")
            row[f"sup_break_{h}"] = st.get(f"sup_break_{h}")
        if idx is not None:
            bs = SB.forward_stats(idx, date, horizons, "open", 1.0, 0)
            for h in horizons:
                b = bs.get(f"ret_{h}")
                row[f"bench_{h}"] = b if (b is not None and np.isfinite(b)) else np.nan
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------------
def build_signals(ev: pd.DataFrame, combos: list[str]) -> dict[str, pd.Series]:
    """信号定义：基础形态 + 各动能标签的每个取值 + 组合（形态 × 标签）+ 自定义表达式。"""
    s: dict[str, pd.Series] = {}

    def b(col: str) -> pd.Series:
        return ev[col].astype(str).str.lower().eq("true") if col in ev.columns else pd.Series(False, index=ev.index)

    lv = ev["rb_level"].astype(str) if "rb_level" in ev.columns else pd.Series("", index=ev.index)
    prim: dict[str, pd.Series] = {
        "圆弧底(日线以上)": lv.isin(["daily", "medium", "high"]),
        "圆弧底(日周以上)": lv.isin(["medium", "high"]),
        "圆弧底(日周+年季)": lv.eq("high"),
        "拐点(放量破200日线)": b("near_bk200"),
        "低位破低反包": b("low_bullish_outside"),
        "深跌巨量强收": b("deep_huge_strong"),
        "长下影": b("daily_long_lower"),
        "长下影∩拐点": b("daily_long_lower") & b("near_bk200"),
        "长下影∩破低反包": b("daily_long_lower") & b("low_bullish_outside"),
    }
    s.update(prim)
    tags: dict[str, pd.Series] = {}
    for col in CAT_TAGS:
        if col not in ev.columns:
            continue
        for val in sorted(set(ev[col].dropna().astype(str)) - {"", "nan", "None"}):
            tags[f"{col}={val}"] = ev[col].astype(str).eq(val)
    s.update(tags)
    key_tags = [k for k in tags if k in (
        "candle_tag=强阳线", "adx_tag=趋势启动·偏多", "atr_tag=透支", "vz_tag=放量突破密集区",
        "vz_tag=量能不足突破密集区", "qqc_tag=多头主导", "macdv_tag=超买")]
    for pn in ("圆弧底(日线以上)", "拐点(放量破200日线)", "长下影∩拐点"):
        for kt in key_tags:
            s[f"{pn} & {kt}"] = prim[pn] & tags[kt]
    for expr in combos:
        try:
            s[f"自定义: {expr}"] = ev.eval(expr, engine="python").astype(bool)
        except Exception as e:  # noqa: BLE001
            print(f"自定义表达式无效，已跳过: {expr} ({type(e).__name__}: {e})")
    return s


def apply_cooldown(ev: pd.DataFrame, mask: pd.Series, cooldown: int) -> pd.Series:
    """同一只股票同一个信号，cooldown 个交易日内只保留第一次。"""
    keep = pd.Series(False, index=ev.index)
    sub = ev.loc[mask, ["code", "bar_idx"]].sort_values(["code", "bar_idx"])
    last_code, last_idx = None, -10**9
    for i, c, bi in zip(sub.index, sub["code"], sub["bar_idx"]):
        if c != last_code:
            last_code, last_idx = c, -10**9
        if bi - last_idx >= cooldown:
            keep.at[i] = True
            last_idx = bi
    return keep


def _stats(d: pd.DataFrame, h: int, base_excess: float) -> dict[str, Any]:
    r = d[f"ret_{h}"].dropna()
    ex = (d[f"ret_{h}"] - d[f"bench_{h}"]).dropna() if f"bench_{h}" in d.columns else r
    n = int(len(ex))
    if n == 0:
        return {"n": 0}
    sd = float(ex.std(ddof=1)) if n > 1 else np.nan
    ci = 1.96 * sd / np.sqrt(n) if n > 1 and np.isfinite(sd) else np.nan
    tp = d[f"tp_hit_{h}"].dropna() if f"tp_hit_{h}" in d.columns else pd.Series(dtype=float)
    sp = d[f"sup_break_{h}"].dropna() if f"sup_break_{h}" in d.columns else pd.Series(dtype=float)
    return {
        "n": n,
        "mean": float(r.mean()) if len(r) else np.nan,
        "win": float((r > 0).mean()) if len(r) else np.nan,
        "excess": float(ex.mean()),
        "lift": float(ex.mean()) - base_excess,
        "ci": ci,
        "tp": float(tp.mean()) if len(tp) else np.nan,
        "sup": float(sp.mean()) if len(sp) else np.nan,
    }


def analyze(ev: pd.DataFrame, args: argparse.Namespace, horizons: list[int]) -> pd.DataFrame:
    h = args.focus if args.focus in horizons else horizons[0]
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.sort_values(["date", "code"]).reset_index(drop=True)
    dates = np.sort(ev["date"].unique())
    split = pd.Timestamp(args.split_date) if args.split_date else pd.Timestamp(dates[int(len(dates) * 0.7)])
    ev["_period"] = np.where(ev["date"] < split, "train", "test")
    blocks = np.array_split(dates, max(args.blocks, 1))
    block_of = {d: i for i, blk in enumerate(blocks) for d in blk}
    ev["_block"] = ev["date"].map(block_of)

    def base_excess(d: pd.DataFrame) -> float:
        ex = (d[f"ret_{h}"] - d[f"bench_{h}"]).dropna() if f"bench_{h}" in d.columns else d[f"ret_{h}"].dropna()
        return float(ex.mean()) if len(ex) else 0.0

    base_all = _stats(ev, h, 0.0)
    base = {"train": base_excess(ev[ev["_period"] == "train"]), "test": base_excess(ev[ev["_period"] == "test"])}
    base_blocks = {i: base_excess(ev[ev["_block"] == i]) for i in range(len(blocks))}
    print(f"\n事件 {len(ev)} 条 | 股票 {ev['code'].nunique()} 只 | as-of 日 {len(dates)} 个 | 切分日 {split.date()} "
          f"(训练 {int((ev['_period'] == 'train').sum())} / 测试 {int((ev['_period'] == 'test').sum())}) | 聚焦持有 {h} 日")
    print(f"基准（所有 as-of 日、次日开盘买入）: 持有{h}日 平均净收益 {base_all['mean'] * 100:.2f}% | 相对沪深300 超额 {base_all['excess'] * 100:.2f}% | "
          f"胜率 {base_all['win'] * 100:.0f}%")

    sigs = build_signals(ev, args.combo)
    rows = []
    for name, mask in sigs.items():
        mask = mask.fillna(False)
        if int(mask.sum()) < args.min_n:
            continue
        keep = apply_cooldown(ev, mask, args.cooldown)
        d = ev[keep]
        tr, te = d[d["_period"] == "train"], d[d["_period"] == "test"]
        s_tr, s_te = _stats(tr, h, base["train"]), _stats(te, h, base["test"])
        if s_tr.get("n", 0) < 1 or s_te.get("n", 0) < 1:
            continue
        blk_pos = blk_valid = 0
        for i in range(len(blocks)):
            sb = d[d["_block"] == i]
            if len(sb) >= 5:
                blk_valid += 1
                st = _stats(sb, h, base_blocks[i])
                blk_pos += int(st["lift"] > 0)
        stable = ""
        if s_tr["n"] >= args.min_n and s_te["n"] >= args.min_n and s_tr["lift"] > 0 and s_te["lift"] > 0 \
                and blk_valid >= 2 and blk_pos / blk_valid >= 0.6:
            stable = "★" if (np.isfinite(s_te["ci"]) and s_te["lift"] - s_te["ci"] > 0) else "✓"
        rows.append({
            "信号": name, "稳定": stable,
            "n训练": s_tr["n"], "训练lift%": round(s_tr["lift"] * 100, 2),
            "n测试": s_te["n"], "测试lift%": round(s_te["lift"] * 100, 2), "测试±95%": round(s_te["ci"] * 100, 2) if np.isfinite(s_te["ci"]) else np.nan,
            "测试均值%": round(s_te["mean"] * 100, 2), "测试胜率%": round(s_te["win"] * 100, 1),
            "触目标%": round(s_te["tp"] * 100, 1) if np.isfinite(s_te["tp"]) else np.nan,
            "破支撑%": round(s_te["sup"] * 100, 1) if np.isfinite(s_te["sup"]) else np.nan,
            "正块/有效块": f"{blk_pos}/{blk_valid}",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        print("没有任何信号达到最小样本量（--min-n），请增加股票数/年数或降低 --min-n。")
        return out
    out = out.sort_values("测试lift%", ascending=False).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="标签历史回放事件研究（只用 as-of 日及以前的数据；T+1；样本外检验）")
    p.add_argument("--universe", default="", help="股票代码 CSV（含 code 列）；空=全市场随机抽样")
    p.add_argument("--max-stocks", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--years", type=float, default=4.0, help="回放的历史年数（含 260 根预热）")
    p.add_argument("--step", type=int, default=5, help="每隔多少个交易日取一个 as-of 日")
    p.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    p.add_argument("--horizons", default="2,3,5,10,20", help="持有交易日（T+1：最小 2）")
    p.add_argument("--focus", type=int, default=5)
    p.add_argument("--cost", type=float, default=0.002, help="往返交易成本（小数，0.002=0.2%）")
    p.add_argument("--cooldown", type=int, default=10, help="同一股票同一信号的去重间隔（交易日）")
    p.add_argument("--split-date", default="", help="训练/测试切分日 YYYYMMDD；空=按事件日期 70/30")
    p.add_argument("--blocks", type=int, default=4, help="稳定性检查：把时间均分成几块")
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--combo", action="append", default=[], help="自定义组合表达式（pandas eval），可重复")
    p.add_argument("--skip-targets", action="store_true", help="不算压力位(目标价)，更快")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--cache-dir", default="./es_cache")
    p.add_argument("--out-dir", default="./es_out")
    p.add_argument("--events-in", default="", help="直接读取已有的事件 CSV，只做分析")
    args = p.parse_args()

    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if horizons[0] < 2:
        print("提示：A 股 T+1，持有 1 日等于当日买当日卖，已自动从 horizons 中去掉 1。")
        horizons = [x for x in horizons if x >= 2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    if args.events_in:
        ev = pd.read_csv(args.events_in, parse_dates=["date"], dtype={"code": str})
        print(f"读取事件 {len(ev)} 条 ← {args.events_in}")
    else:
        idx = None
        try:
            idx = fetch_index(args.years, Path(args.cache_dir))
        except Exception as e:  # noqa: BLE001
            print(f"沪深300 获取失败（无基准、圆弧底按“正常”档）: {e}")
        if args.universe:
            codes = pd.read_csv(args.universe, dtype=str)["code"].astype(str).str.zfill(6).tolist()
        else:
            codes = get_universe(args.max_stocks, args.seed)
        print(f"股票池 {len(codes)} 只 | 回放 {args.years} 年 | step={args.step} | 持有 {horizons} | 成本 {args.cost:.2%} | workers={args.workers}")
        cfg = {"years": args.years, "cache_dir": args.cache_dir, "warmup": args.warmup, "step": args.step, "horizons": horizons,
               "cost": args.cost, "with_targets": not args.skip_targets}
        t0 = time.time()
        rows: list[dict[str, Any]] = []
        if args.workers <= 1:
            _init_worker(idx, cfg)
            for i, c in enumerate(codes, 1):
                rows.extend(process_stock(c))
                if i % 10 == 0:
                    print(f"  {i}/{len(codes)} 只，事件 {len(rows)}，用时 {time.time() - t0:.0f}s")
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(idx, cfg)) as ex:
                for i, r in enumerate(ex.map(process_stock, codes, chunksize=1), 1):
                    rows.extend(r)
                    if i % 10 == 0:
                        print(f"  {i}/{len(codes)} 只，事件 {len(rows)}，用时 {time.time() - t0:.0f}s")
        ev = pd.DataFrame(rows)
        if ev.empty:
            print("没有产生任何事件（数据获取失败？）")
            return
        ev_path = out_dir / f"event_study_events_{today}.csv"
        ev.to_csv(ev_path, index=False, encoding="utf-8-sig")
        print(f"事件明细 → {ev_path}")

    res = analyze(ev, args, horizons)
    if res.empty:
        return
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print("\n" + "=" * 120)
    print("各信号样本外表现（lift = 相对同期所有 as-of 日基准的超额；★=训练/测试/各时间块均为正，且测试 lift 的 95% 区间下限>0；✓=符号稳定但区间含 0）")
    print("=" * 120)
    print(res.to_string(index=False))
    path = out_dir / f"event_study_summary_{today}.csv"
    res.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n汇总 → {path}")
    print("提醒：同时检验的信号很多，个别“稳定”可能是偶然；请结合样本量、置信区间，并在更长/更大的股票池上复核。非投资建议。")


if __name__ == "__main__":
    main()
