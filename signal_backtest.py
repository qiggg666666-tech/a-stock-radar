#!/usr/bin/env python3
"""
选股名单回测统计（研究用）
==========================
读取历史选股输出 CSV，统计命中名单"之后"的真实表现，回答：
  - 各路名单（宽幅/尖峰/冲板/超跌反弹）次日~N日的胜率、均值、中位数
  - 相对沪深300的超额收益
  - 之后 N 日内出现涨停的比例；N 日内翻倍的比例
  - 哪些标签/分层/条件组合真的有超额，哪些只是看起来好看

读取的文件（按文件名前缀识别来源，日期取 date 列，缺失则取文件名 YYYYMMDD）：
  live_select_wide_YYYYMMDD.csv      → 宽幅启动
  live_select_peak_YYYYMMDD.csv      → 红筹尖峰
  live_select_limitup_YYYYMMDD.csv   → 冲板结构
  oversold_rebound_YYYYMMDD.csv      → 超跌反弹
  （live_select_layer_*.csv 是 wide+peak 的合并，已忽略避免重复计算）

重要前提：
  1) 这些 CSV 每天生成后必须保留下来（GitHub Actions 里请 commit 或上传 artifact），
     否则历史丢失，没有可统计的样本。
  2) 样本少时结论不可靠（见输出中的 n 和 ±95% 区间）。分组越多，越容易偶然出现
     "看起来显著"的组合，请用更长时间的样本复核后再相信。

成交假设（贴近实盘，可用参数调整）：
  --entry open   信号日收盘后出名单，次日开盘价买入（默认）
  --entry close  以信号日收盘价买入（偏乐观，名单通常收盘后才出）
  次日开盘即涨停（买不进）的样本默认剔除，--include-unbuyable 可保留
  收益统计：信号日后第 h 个交易日收盘价 / 买入价 - 1，不含手续费与滑点

用法：
  python signal_backtest.py
  python signal_backtest.py --dir . ./archive --horizons 1,3,5,10 --focus 5
  python signal_backtest.py --since 20260901 --group-by source,layer,mtf_label
  python signal_backtest.py --entry close --double-days 60 --min-n 8
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # 允许只做离线检查
    ak = None

SOURCES = [
    ("live_select_wide_", "宽幅启动"),
    ("live_select_peak_", "红筹尖峰"),
    ("live_select_limitup_", "冲板结构"),
    ("oversold_rebound_", "超跌反弹"),
]
DATE_RE = re.compile(r"(\d{8})\.csv$")
SCORE_COL = {
    "宽幅启动": "score_wide",
    "红筹尖峰": "score_peak",
    "冲板结构": "limitup_score",
    "超跌反弹": "score",
}
DEFAULT_GROUPS = [
    "source",
    "layer",
    "score_band",
    "mtf_label",
    "long_label",
    "cluster_signal",
    "ema_stack",
    "year_first_red",
    "near_breakout",
    "near_limitup_setup",
    "washout_label",
    "ta_label",
    "label",
]
BENCH_SYMBOL = "sh000300"
LIMIT_TOL = 0.005  # 涨停判断容差（前复权价格有舍入误差）


# --------------------------------------------------------------------------- #
# 读取历史名单
# --------------------------------------------------------------------------- #
def load_signals(
    dirs: list[str], since: str | None, until: str | None
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    for d in dirs:
        base = Path(d)
        if not base.exists():
            print(f"目录不存在，跳过: {base}")
            continue
        for prefix, label in SOURCES:
            for f in sorted(base.rglob(f"{prefix}*.csv")):
                rp = f.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                m = DATE_RE.search(f.name)
                try:
                    df = pd.read_csv(f, dtype={"code": str})
                except Exception as e:
                    print(f"读取失败 {f}: {e}")
                    continue
                if df.empty or "code" not in df.columns:
                    continue
                df = df.copy()
                df["code"] = (
                    df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                )
                file_date = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce") if m else pd.NaT
                if "date" in df.columns:
                    sd = pd.to_datetime(df["date"], errors="coerce")
                else:
                    sd = pd.Series(pd.NaT, index=df.index)
                df["signal_date"] = sd.fillna(file_date)
                df["source"] = label
                df["src_file"] = f.name
                frames.append(df)
    if not frames:
        return pd.DataFrame()

    sig = pd.concat(frames, ignore_index=True)
    sig = sig.dropna(subset=["signal_date"])
    sig["signal_date"] = pd.to_datetime(sig["signal_date"]).dt.normalize()
    if since:
        sig = sig[sig["signal_date"] >= pd.to_datetime(since)]
    if until:
        sig = sig[sig["signal_date"] <= pd.to_datetime(until)]
    # 同一来源、同一天、同一只股票只算一次
    sig = sig.sort_values(["source", "signal_date", "src_file"]).drop_duplicates(
        ["source", "code", "signal_date"], keep="last"
    )

    # 统一分数列 + 同一天内的分数分位
    sig["score_val"] = np.nan
    for src, col in SCORE_COL.items():
        if col in sig.columns:
            m = sig["source"] == src
            sig.loc[m, "score_val"] = pd.to_numeric(sig.loc[m, col], errors="coerce")
    sig["score_band"] = "n/a"
    for (_, _), g in sig.groupby(["source", "signal_date"]):
        if len(g) < 3 or g["score_val"].notna().sum() < 3:
            continue
        pct = g["score_val"].rank(ascending=False, pct=True)
        band = np.where(pct <= 0.34, "前1/3", np.where(pct <= 0.67, "中1/3", "后1/3"))
        sig.loc[g.index, "score_band"] = band
    return sig.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 行情
# --------------------------------------------------------------------------- #
def fetch_prices(
    code: str, start: pd.Timestamp, cache_dir: Path, refresh: bool
) -> pd.DataFrame:
    """前复权日线。缓存当天有效；换日后整份重拉，避免新旧复权口径混用。"""
    path = cache_dir / f"{code}.csv"
    if path.exists() and not refresh:
        fresh = datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()
        if fresh:
            try:
                c = pd.read_csv(path, parse_dates=["date"])
                if not c.empty and c["date"].min() <= start + timedelta(days=5):
                    return c
            except Exception:
                pass
    if ak is None:
        return pd.DataFrame()
    s = (start - timedelta(days=40)).strftime("%Y%m%d")
    e = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=s, end_date=e, adjust="qfq"
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            raw = raw.rename(
                columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low"}
            )
            out = raw[["date", "open", "high", "low", "close"]].copy()
            out["date"] = pd.to_datetime(out["date"])
            out = out.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            time.sleep(0.15)
            return out
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return pd.DataFrame()


def fetch_benchmark() -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    try:
        raw = ak.stock_zh_index_daily(symbol=BENCH_SYMBOL)
        out = raw[["date", "open", "high", "low", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"基准指数获取失败（超额收益将为空）: {e}")
        return pd.DataFrame()


def limit_pct(code: str, name: str = "", st_limit: float = 0.05) -> float:
    n = str(name or "").upper().replace(" ", "")
    if re.match(r"^\*?ST", n):
        return st_limit
    if code.startswith(("688", "689", "300", "301")):
        return 0.20
    if code.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


# --------------------------------------------------------------------------- #
# 单个信号的后续表现
# --------------------------------------------------------------------------- #
def forward_stats(
    px: pd.DataFrame | None,
    sig_date: pd.Timestamp,
    horizons: list[int],
    entry_mode: str,
    limit: float,
    double_days: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {"entry_px": np.nan, "buyable": np.nan, "open_pct": np.nan}
    for h in horizons:
        out[f"ret_{h}"] = np.nan
        out[f"maxup_{h}"] = np.nan
        out[f"maxdd_{h}"] = np.nan
        out[f"lu_{h}"] = np.nan
    out["double"] = np.nan
    if px is None or px.empty:
        return out

    dates = px["date"].values
    i0 = int(np.searchsorted(dates, sig_date.to_datetime64(), side="right") - 1)
    if i0 < 0:
        return out
    # 信号日对应K线不存在（长期停牌/数据缺失）时不硬匹配到很旧的K线
    if (sig_date - pd.Timestamp(dates[i0])).days > 5:
        return out

    o = px["open"].to_numpy(float)
    hi = px["high"].to_numpy(float)
    lo = px["low"].to_numpy(float)
    c = px["close"].to_numpy(float)
    n = len(c)

    if entry_mode == "open":
        e = i0 + 1
        if e >= n:
            return out  # 次日K线还不存在
        entry = o[e]
        if c[i0] > 0:
            out["open_pct"] = entry / c[i0] - 1
            out["buyable"] = float(not (out["open_pct"] >= limit - LIMIT_TOL))
    else:
        e = i0
        entry = c[i0]
        buyable = 1.0
        if i0 >= 1 and c[i0 - 1] > 0 and c[i0] / c[i0 - 1] - 1 >= limit - LIMIT_TOL:
            buyable = 0.0  # 信号日本身收在涨停，收盘价买不到
        out["buyable"] = buyable
    if not np.isfinite(entry) or entry <= 0:
        return out
    out["entry_px"] = float(entry)

    for h in horizons:
        j = i0 + h
        if j >= n:
            continue
        out[f"ret_{h}"] = float(c[j] / entry - 1)
        out[f"maxup_{h}"] = float(hi[e : j + 1].max() / entry - 1)
        out[f"maxdd_{h}"] = float(lo[e : j + 1].min() / entry - 1)
        ks = np.arange(i0 + 1, j + 1)
        prev = c[ks - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = c[ks] / prev - 1
        hit = (pct >= limit - LIMIT_TOL) & (c[ks] >= hi[ks] * 0.998)
        out[f"lu_{h}"] = float(bool(hit.any()))

    if double_days > 0 and i0 + double_days < n:
        out["double"] = float(c[i0 + 1 : i0 + double_days + 1].max() / entry >= 2.0)
    return out


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def _stat_row(g: pd.DataFrame, h: int) -> dict[str, Any] | None:
    r = g[f"ret_{h}"].dropna()
    if r.empty:
        return None
    ex = g[f"excess_{h}"].dropna()
    lu = g[f"lu_{h}"].dropna()
    n = len(r)
    sd = float(r.std(ddof=1)) if n > 1 else np.nan
    ci = 1.96 * sd / np.sqrt(n) if n > 1 and np.isfinite(sd) else np.nan
    return {
        "n": n,
        "胜率%": round(float((r > 0).mean() * 100), 1),
        "均值%": round(float(r.mean() * 100), 2),
        "±95%": round(float(ci * 100), 2) if np.isfinite(ci) else np.nan,
        "中位数%": round(float(r.median() * 100), 2),
        "超额均值%": round(float(ex.mean() * 100), 2) if len(ex) else np.nan,
        "最差%": round(float(r.min() * 100), 2),
        "最好%": round(float(r.max() * 100), 2),
        "涨停率%": round(float(lu.mean() * 100), 1) if len(lu) else np.nan,
    }


def summarize_group(df: pd.DataFrame, by: str, h: int, min_n: int) -> pd.DataFrame:
    d = df
    if by == "ta_label":  # 多标签用 + 连接，拆开分别统计
        d = df.copy()
        d["ta_label"] = d["ta_label"].fillna("").astype(str).replace("", "（无标签）")
        d["ta_label"] = d["ta_label"].str.split("+")
        d = d.explode("ta_label")
    rows = []
    for key, g in d.groupby(by, dropna=False):
        row = _stat_row(g, h)
        if row is None or row["n"] < min_n:
            continue
        key_s = "（空）" if (isinstance(key, float) and np.isnan(key)) or str(key) in ("", "nan") else str(key)
        rows.append({"分组": f"{by}={key_s}", **row})
    return pd.DataFrame(rows)


def print_table(title: str, tbl: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if tbl.empty:
        print("（样本不足）")
    else:
        print(tbl.to_string(index=False))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    focus = args.focus if args.focus in horizons else horizons[0]
    if focus != args.focus:
        print(f"--focus {args.focus} 不在 horizons 中，改用 {focus}")

    sig = load_signals(args.dir, args.since, args.until)
    if sig.empty:
        print("没有找到任何历史名单 CSV。请确认已保留每天的输出文件，并用 --dir 指向存放目录。")
        return
    print(
        f"读取名单 {len(sig)} 条 | 日期 {sig['signal_date'].min():%Y-%m-%d} ~ "
        f"{sig['signal_date'].max():%Y-%m-%d} | 来源 {sig['source'].value_counts().to_dict()}"
    )

    cache_dir = Path(args.cache_dir)
    bench = fetch_benchmark()

    price_cache: dict[str, pd.DataFrame] = {}
    min_date = sig.groupby("code")["signal_date"].min()
    codes = sorted(sig["code"].unique())
    miss = 0
    for i, code in enumerate(codes):
        px = fetch_prices(code, min_date[code], cache_dir, args.refresh)
        if px.empty:
            miss += 1
        price_cache[code] = px
        if (i + 1) % 25 == 0:
            print(f"  行情 {i + 1}/{len(codes)}")
    if miss:
        print(f"提示：{miss} 只股票行情获取失败，已从统计中剔除")

    rows: list[dict[str, Any]] = []
    for rd in sig.to_dict("records"):  # 比 itertuples 稳：不会因列名不合法而被改名
        code = rd["code"]
        name = rd.get("name", "") if isinstance(rd.get("name", ""), str) else ""
        lim = limit_pct(code, name, args.st_limit)
        st = forward_stats(
            price_cache.get(code), rd["signal_date"], horizons, args.entry, lim, args.double_days
        )
        bs = forward_stats(bench, rd["signal_date"], horizons, args.entry, 9.0) if not bench.empty else {}
        row = dict(rd)
        row.update(st)
        for h in horizons:
            b = bs.get(f"ret_{h}", np.nan) if bs else np.nan
            row[f"bench_{h}"] = b
            rr = st.get(f"ret_{h}", np.nan)
            row[f"excess_{h}"] = rr - b if np.isfinite(rr) and np.isfinite(b) else np.nan
        rows.append(row)
    det = pd.DataFrame(rows)

    n_all = len(det)
    n_noprice = int(det["entry_px"].isna().sum())
    unbuy = det["buyable"] == 0
    n_unbuy = int(unbuy.sum())
    if not args.include_unbuyable:
        det_use = det[~unbuy].copy()
    else:
        det_use = det.copy()
    print(
        f"信号 {n_all} 条 | 无有效行情/次日K线未出 {n_noprice} | "
        f"买不进（{'开盘涨停' if args.entry == 'open' else '收盘涨停'}）{n_unbuy}"
        f"{'（已保留）' if args.include_unbuyable else '（已剔除）'}"
    )

    # ---------- 总览：每个 horizon ----------
    ov = []
    for h in horizons:
        row = _stat_row(det_use, h)
        if row:
            ov.append({"持有日": h, **row})
    print_table(f"总览（全部来源，买入方式={args.entry}，收益=信号后第N个交易日收盘）", pd.DataFrame(ov))

    # ---------- 各来源 ----------
    src_rows = []
    for src, g in det_use.groupby("source"):
        for h in horizons:
            row = _stat_row(g, h)
            if row and row["n"] >= 1:
                src_rows.append({"来源": src, "持有日": h, **row})
    print_table("按来源 × 持有日", pd.DataFrame(src_rows))

    # ---------- 分组（聚焦 focus 日） ----------
    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    summary_frames = []
    for col in group_cols:
        if col not in det_use.columns:
            continue
        tbl = summarize_group(det_use, col, focus, args.min_n)
        if not tbl.empty:
            tbl.insert(0, "持有日", focus)
            summary_frames.append(tbl)
            print_table(f"分组：{col}（持有 {focus} 日，n≥{args.min_n}）", tbl.drop(columns=["持有日"]))

    # ---------- 翻倍 ----------
    if args.double_days > 0:
        d = det_use["double"].dropna()
        print("\n" + "=" * 100)
        print(f"翻倍统计：信号后 {args.double_days} 个交易日内收盘价曾达到买入价 2 倍")
        print("=" * 100)
        if d.empty:
            print(f"（暂无满 {args.double_days} 个交易日的样本，无法统计）")
        else:
            print(f"已满期样本 {len(d)} 条，翻倍 {int(d.sum())} 条，比例 {d.mean() * 100:.2f}%")
            by_src = det_use.dropna(subset=["double"]).groupby("source")["double"].agg(["count", "sum", "mean"])
            by_src["mean"] = (by_src["mean"] * 100).round(2)
            print(by_src.rename(columns={"count": "n", "sum": "翻倍数", "mean": "比例%"}).to_string())

    print(
        "\n说明：样本少时（n<30）仅供参考；±95% 是均值的近似置信区间，区间跨过 0 说明均值与 0 无显著差异。"
        "\n分组很多时偶然出现“好看”的组合很常见，请用更长样本复核。不含手续费/滑点，非投资建议。"
    )

    # ---------- 输出文件 ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    keep_front = ["source", "signal_date", "code", "name", "entry_px", "open_pct", "buyable"]
    ret_cols = [c for c in det.columns if re.match(r"(ret|bench|excess|maxup|maxdd|lu)_\d+$", c)] + ["double"]
    other = [c for c in det.columns if c not in keep_front + ret_cols]
    det_out = det[[c for c in keep_front if c in det.columns] + ret_cols + other]
    p1 = out_dir / f"backtest_detail_{today}.csv"
    det_out.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"\n明细 → {p1}")
    if summary_frames:
        p2 = out_dir / f"backtest_summary_{today}.csv"
        pd.concat(summary_frames, ignore_index=True).to_csv(p2, index=False, encoding="utf-8-sig")
        print(f"汇总 → {p2}")


def main() -> None:
    p = argparse.ArgumentParser(description="选股名单回测统计（研究）")
    p.add_argument("--dir", nargs="+", default=["."], help="历史 CSV 所在目录（可多个，递归查找）")
    p.add_argument("--horizons", default="1,3,5,10,20", help="持有交易日，逗号分隔")
    p.add_argument("--focus", type=int, default=5, help="分组统计使用的持有日")
    p.add_argument("--entry", choices=["open", "close"], default="open", help="买入价：次日开盘/信号日收盘")
    p.add_argument("--group-by", default=",".join(DEFAULT_GROUPS), help="分组列，逗号分隔")
    p.add_argument("--min-n", type=int, default=5, help="分组最少样本数")
    p.add_argument("--since", default="", help="起始信号日 YYYYMMDD")
    p.add_argument("--until", default="", help="截止信号日 YYYYMMDD")
    p.add_argument("--double-days", type=int, default=60, help="翻倍统计窗口（交易日），0 关闭")
    p.add_argument("--st-limit", type=float, default=0.05, help="ST 股涨跌停幅度（请按当前交易所规则核对）")
    p.add_argument("--include-unbuyable", action="store_true", help="保留买不进（涨停）的样本")
    p.add_argument("--refresh", action="store_true", help="忽略行情缓存，强制重拉")
    p.add_argument("--cache-dir", default="./backtest_cache", help="行情缓存目录")
    p.add_argument("--out-dir", default=".", help="明细/汇总输出目录")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
