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

用法：
  python oversold_rebound_selector.py --shard 1
  python oversold_rebound_selector.py --shard all
  python oversold_rebound_selector.py --merge --top 20 --notify
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
    print("已加载 final_chip_research")
except ImportError:
    FULL_CHIP_AVAILABLE = False
    print("警告：无 final_chip_research，将跳过筹码/获利过滤")

    def is_beijing_stock(code: str) -> bool:
        return str(code).startswith(("8", "4", "9"))


CACHE_DIR = Path("./oversold_rebound_cache")
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

MIN_BARS = 120
# ----- 超跌反弹参数（可按需改） -----
DRAWDOWN_MIN = 0.25          # 相对近60日高点至少回撤 25%
REBOUND_FROM_LOW_MIN = 0.05  # 相对近20日低点至少反弹 5%
REBOUND_FROM_LOW_MAX = 0.25  # 反弹不超过 25%（避免已炒高）
MA20_PREM_MIN = -0.02
MA20_PREM_MAX = 0.12
PROFIT_MIN = 15.0            # 获利偏低 = 套牢多
PROFIT_MAX = 45.0
TOP_N_DEFAULT = 20


def get_all_a_stocks() -> list[str]:
    codes: list[str] = []
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
                    pure = code.split(".")[-1] if "." in code else code
                    codes.append(pure.zfill(6))
            bs.logout()
            if codes:
                return sorted(set(codes))
    except Exception as e:
        print(f"baostock 列表失败: {e}")
    try:
        df = ak.stock_info_a_code_name()
        return df["code"].astype(str).str.zfill(6).tolist()
    except Exception as e:
        print(f"akshare 列表失败: {e}")
    return []


def filter_shard(codes: list[str], shard_id: int) -> list[str]:
    prefixes = SHARD_MAP.get(shard_id, [])
    return [c for c in codes if any(c.startswith(p) for p in prefixes)]


def get_stock_data(symbol: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if FULL_CHIP_AVAILABLE:
        try:
            tmp, _, _ = fetch_ohlcv(symbol, timeout_seconds=25, retries=2)
            if tmp is not None and not tmp.empty:
                df = tmp.copy()
                df["symbol"] = symbol
        except Exception:
            pass
    if df.empty or len(df) < 200:
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y%m%d")
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            if raw is not None and not raw.empty:
                raw = raw.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "换手率": "turnover",
                    }
                )
                raw["date"] = pd.to_datetime(raw["date"])
                raw["symbol"] = symbol
                df = raw.sort_values("date").reset_index(drop=True)
                time.sleep(0.15)
        except Exception:
            pass
    return df if (df is not None and not df.empty) else pd.DataFrame()


def _name_of(code: str) -> str:
    try:
        info = ak.stock_info_a_code_name()
        m = dict(
            zip(
                info["code"].astype(str).str.zfill(6),
                info["name"].astype(str),
            )
        )
        return str(m.get(str(code).zfill(6), "") or "")
    except Exception:
        return ""


def score_oversold_rebound(df: pd.DataFrame, code: str) -> dict[str, Any] | None:
    if df is None or len(df) < MIN_BARS:
        return None
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    close = float(d["close"].iloc[-1])
    if close <= 0:
        return None

    d["ma20"] = d["close"].rolling(20, min_periods=15).mean()
    d["ema120"] = d["close"].ewm(span=120, adjust=False, min_periods=60).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False, min_periods=100).mean()
    ma20 = float(d["ma20"].iloc[-1])
    if not np.isfinite(ma20) or ma20 <= 0:
        return None
    prem = close / ma20 - 1.0
    if prem < MA20_PREM_MIN or prem > MA20_PREM_MAX:
        return None

    # EMA120/200 聚类信号
    e120 = float(d["ema120"].iloc[-1]) if pd.notna(d["ema120"].iloc[-1]) else None
    e200 = float(d["ema200"].iloc[-1]) if pd.notna(d["ema200"].iloc[-1]) else None
    prem120 = (close / e120 - 1.0) if e120 and e120 > 0 else None
    prem200 = (close / e200 - 1.0) if e200 and e200 > 0 else None
    ema_stack = ""
    if e120 and e200 and e120 > 0 and e200 > 0:
        if close > e120 > e200:
            ema_stack = "多头排列"
        elif close < e120 < e200:
            ema_stack = "空头排列"
        else:
            ema_stack = "纠缠"
    cluster_buy_n = 0
    cluster_sell_n = 0
    buy_tags: list[str] = []
    sell_tags: list[str] = []
    if e120 and close > e120:
        cluster_buy_n += 1
        buy_tags.append("站上EMA120")
    elif e120:
        cluster_sell_n += 1
        sell_tags.append("跌破EMA120")
    if e200 and close > e200:
        cluster_buy_n += 1
        buy_tags.append("站上EMA200")
    elif e200:
        cluster_sell_n += 1
        sell_tags.append("跌破EMA200")
    if ema_stack == "多头排列":
        cluster_buy_n += 1
        buy_tags.append("多头排列")
    elif ema_stack == "空头排列":
        cluster_sell_n += 1
        sell_tags.append("空头排列")
    # 近 5 日上穿/下穿 EMA120
    if e120 and len(d) >= 6:
        for i in range(-5, 0):
            c0, c1 = float(d["close"].iloc[i]), float(d["close"].iloc[i - 1])
            a0, a1 = float(d["ema120"].iloc[i]), float(d["ema120"].iloc[i - 1])
            if c1 <= a1 and c0 > a0:
                cluster_buy_n += 1
                buy_tags.append("上穿EMA120")
                break
            if c1 >= a1 and c0 < a0:
                cluster_sell_n += 1
                sell_tags.append("下穿EMA120")
                break
    if prem120 is not None and 0 <= prem120 <= 0.03:
        cluster_buy_n += 1
        buy_tags.append("回踩EMA120")
    if prem120 is not None and -0.03 <= prem120 < 0:
        cluster_sell_n += 1
        sell_tags.append("反抽EMA120受压")

    if cluster_buy_n >= 3 and cluster_buy_n > cluster_sell_n:
        cluster_signal, cluster_label = "聚类买入", "买:" + "+".join(buy_tags[:4])
        cluster_bonus = min(0.20, 0.05 * cluster_buy_n)
    elif cluster_sell_n >= 3 and cluster_sell_n > cluster_buy_n:
        cluster_signal, cluster_label = "聚类卖出", "卖:" + "+".join(sell_tags[:4])
        cluster_bonus = -min(0.20, 0.05 * cluster_sell_n)
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

    # 日线需止跌：收盘站上 MA20 或刚上穿（溢价>=-2% 已在上面）
    above_ma20 = close >= ma20 * 0.98

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
        except Exception:
            return None
    if not name:
        name = _name_of(code)

    # 综合分：回撤越深、反弹适中、越贴近 MA20 越好；周线仍弱不扣太多
    score = 0.0
    score += min(drawdown, 0.6) * 2.0          # 深跌
    score += (0.15 - abs(rebound - 0.10)) * 3  # 反弹约 10% 附近更优
    score += (0.05 - abs(prem - 0.02)) * 4     # 略高于 MA20
    if above_ma20:
        score += 0.3
    if w_above is False:
        score += 0.1  # 超跌反弹允许周线仍弱
    if profit is not None:
        # 获利越低（套牢越多）在区间内略加分
        score += (PROFIT_MAX - profit) / 100.0
    score += cluster_bonus  # EMA 聚类加减分

    ho = float(d["high"].iloc[-1] / d["open"].iloc[-1] - 1) if d["open"].iloc[-1] > 0 else 0.0
    turn = pd.to_numeric(d.get("turnover", 0), errors="coerce").fillna(0.0)
    if float(turn.median(skipna=True) or 0) > 1.5:
        turn = turn / 100.0
    turnover = float(turn.iloc[-1]) if len(turn) else 0.0

    label = "超跌反弹·日线止跌"
    if w_above is False:
        label += "·周线仍弱"
    elif w_above is True:
        label += "·周线转强"

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
        "ema120": round(e120, 2) if e120 else None,
        "ema200": round(e200, 2) if e200 else None,
        "ema120_premium": round(prem120, 4) if prem120 is not None else None,
        "ema200_premium": round(prem200, 4) if prem200 is not None else None,
        "ema_stack": ema_stack,
        "cluster_buy_n": cluster_buy_n,
        "cluster_sell_n": cluster_sell_n,
        "cluster_signal": cluster_signal,
        "cluster_label": cluster_label,
        "cluster_bonus": round(cluster_bonus, 3),
        "label": label,
        "score": round(float(score), 4),
    }


def run_shard(shard_id: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"超跌反弹 分片 {shard_id} | {SHARD_MAP.get(shard_id)}")
    print(f"{'=' * 60}")
    codes = filter_shard(get_all_a_stocks(), shard_id)
    print(f"本分片 {len(codes)} 只")
    rows: list[dict[str, Any]] = []
    for i, code in enumerate(codes):
        if is_beijing_stock(code):
            continue
        df = get_stock_data(code)
        if df.empty:
            continue
        row = score_oversold_rebound(df, code)
        if row:
            rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(codes)} 命中 {len(rows)}")
    out = CACHE_DIR / f"osr_shard_{shard_id}.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id}: {len(rows)} 只 → {out}")
    else:
        print(f"分片 {shard_id}: 无命中")


def notify_serverchan(title: str, content: str) -> str:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return "skipped: no SENDKEY"
    try:
        resp = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"title": title[:100], "desp": content},
            timeout=20,
        )
        return f"{resp.status_code} {resp.text[:100]}"
    except Exception as e:
        return str(e)


def merge_and_select(top_n: int = TOP_N_DEFAULT, do_notify: bool = False) -> None:
    files = list(CACHE_DIR.glob("osr_shard_*.csv"))
    if not files:
        print("无分片结果，请先 --shard")
        return
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = df.drop_duplicates("code")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    top = df.head(top_n).copy()
    today = datetime.now().strftime("%Y%m%d")
    path = f"oversold_rebound_{today}.csv"
    top.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n超跌反弹 Top{len(top)} → {path}")
    print("=" * 90)
    for i, row in top.iterrows():
        print(
            f"{i + 1:02d}. {row['code']} {row.get('name', '')} "
            f"| 分{row['score']:.3f} | 收盘{row['close']} "
            f"| 回撤{float(row.get('drawdown_max', 0)):.1%} "
            f"| 低点反弹{float(row.get('rebound_low20', 0)):.1%} "
            f"| MA20溢价{float(row.get('ma20_premium', 0)):.1%} "
            f"| 获利{row.get('profit_pct', '-')}% "
            f"| EMA:{row.get('cluster_signal', '')} "
            f"| {row.get('label', '')}"
        )
    print("=" * 90)
    print("说明：周线可以仍弱；含 EMA120/200 聚类信号。研究用，非投资建议。")

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
                f"| 获利{row.get('profit_pct', '-')}% "
                f"| EMA:{row.get('cluster_signal', '')}"
                f"({row.get('cluster_label', '')}) "
                f"| {row.get('label', '')}"
            )
        lines.append("\n仅为研究输出，不构成投资建议。")
        print("Server酱:", notify_serverchan(f"超跌反弹{today}|{len(top)}只", "\n".join(lines)))


def main() -> None:
    p = argparse.ArgumentParser(description="超跌反弹选股（研究）")
    p.add_argument("--shard", type=str, help="1-8 或 all")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    p.add_argument("--notify", action="store_true")
    args = p.parse_args()

    if args.merge:
        merge_and_select(top_n=args.top, do_notify=args.notify)
        return
    if not args.shard:
        p.error("请指定 --shard 或 --merge")
    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i)
        print("\n完成。执行: python oversold_rebound_selector.py --merge --top 20 --notify")
    else:
        run_shard(int(args.shard))


if __name__ == "__main__":
    main()
