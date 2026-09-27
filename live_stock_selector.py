#!/usr/bin/env python3
"""
实战选股完整版（修复数据长度 + 放宽过滤）
========================================
硬过滤：红筹码占优 + 获利盘 40%\~85%
宽幅 / 年线首红：加分，不否决
数据：fetch_ohlcv 不足 200 根时用 akshare 补全
择时：沪深300 RSRS 轻量

用法：
  python live_stock_selector.py --shard 1
  python live_stock_selector.py --shard all
  python live_stock_selector.py --merge --top 20 --notify
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

TOP_N_DEFAULT = 20
PROFIT_MIN = 40.0
PROFIT_MAX = 85.0
TURN_OVER_CAP = 0.15
MIN_BARS = 100          # 与筹码 LOOKBACK 对齐；年线不足时 year_first_red=0

FACTOR_WEIGHTS = {
    "wide_score": 0.25,
    "VwapClose": 0.20,
    "HighOpen": 0.15,
    "CloseLow": 0.15,
    "year_first_red": 0.15,
    "Turnover": 0.10,
    "ma_bonus": 0.00,
}

RSRS_N = 18
RSRS_M = 600
RSRS_BUY_THRESHOLD = 0.7


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


def get_stock_data(symbol: str) -> pd.DataFrame:
    """优先 fetch_ohlcv；不足 200 根则用 akshare 补约 1 年数据。"""
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

    df["ma250"] = df["close"].rolling(250, min_periods=180).mean()
    df["above_ma250"] = df["close"] > df["ma250"]
    prev_above = df["above_ma250"].shift(1).fillna(False)
    five_days_ago_below = (df["close"].shift(5) < df["ma250"].shift(5)).fillna(False)
    df["year_first_red"] = (
        df["above_ma250"] & (prev_above == False) & five_days_ago_below
    ).astype(float)
    return df


def apply_launch_filter(df: pd.DataFrame, code: str) -> dict[str, Any]:
    """硬过滤：红筹码 + 获利盘区间；宽幅/年线只加分。"""
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
        ma_signal = str(result.get("ma_signal") or "")
        ma_bonus = 0.0
        if "多头" in ma_signal or "金叉" in ma_signal:
            ma_bonus = 1.0
        elif "趋势健康" in ma_signal:
            ma_bonus = 0.6

        return {
            "code": str(code).zfill(6),
            "name": result.get("name", ""),
            "date": str(result.get("date", latest["date"].date() if hasattr(latest["date"], "date") else latest["date"])),
            "close": round(float(latest["close"]), 2),
            "HighOpen": round(float(latest["HighOpen"]), 4),
            "CloseLow": round(float(latest["CloseLow"]), 4),
            "VwapClose": round(float(latest["VwapClose"]), 4),
            "close_pos": round(float(latest["close_pos"]), 4),
            "Turnover": round(float(latest.get("Turnover", 0)), 4),
            "profit_pct": round(profit, 2),
            "wide_score": float(result.get("wide_score") or 0),
            "wide_state": result.get("wide_state"),
            "is_wide_zone": bool(result.get("is_wide_zone")),
            "ma_signal": ma_signal,
            "ma_bonus": ma_bonus,
            "year_first_red": float(latest.get("year_first_red", 0)),
            "ma250": round(float(latest["ma250"]), 2) if pd.notna(latest.get("ma250")) else None,
            "total_score": result.get("total_score"),
            "is_red_heavy_chip": True,
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
        x_mean = x.mean()
        y_mean = y.mean()
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
        mu = window.mean()
        sigma = window.std()
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


def get_hs300_rsrs() -> dict[str, Any]:
    try:
        raw = ak.stock_zh_index_daily(symbol="sh000300")
        if raw is None or raw.empty:
            return {"ok": False, "reason": "empty_hs300"}
        df = raw.rename(columns={"date": "date", "high": "high", "low": "low", "close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(800).reset_index(drop=True)
        return compute_rsrs(df)
    except Exception as e:
        return {"ok": False, "reason": f"hs300_error:{type(e).__name__}:{str(e)[:120]}"}


def run_shard(shard_id: int) -> None:
    print(f"\n{'='*60}")
    print(f"选股 - 分片 {shard_id} | 前缀 {SHARD_MAP.get(shard_id)}")
    print(f"{'='*60}")

    all_codes = get_all_a_stocks()
    if not all_codes:
        print("无法获取股票列表")
        return

    codes = filter_shard(all_codes, shard_id)
    print(f"本分片股票数: {len(codes)}")

    records = []
    n_data = 0
    n_red = 0
    n_profit = 0

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
        except Exception:
            pass

        row = apply_launch_filter(df, code)
        if row:
            records.append(row)

        if (i + 1) % 30 == 0:
            print(
                f"  已处理 {i+1}/{len(codes)} | 有效数据{n_data} "
                f"红筹{n_red} 获利区间{n_profit} 命中{len(records)}"
            )

    print(
        f"分片 {shard_id} 统计: 有效数据={n_data} 红筹码={n_red} "
        f"获利盘区间={n_profit} 最终命中={len(records)}"
    )

    if records:
        df_out = pd.DataFrame(records)
        df_out["code"] = df_out["code"].astype(str).str.zfill(6)
        out_file = CACHE_DIR / f"live_shard_{shard_id}.csv"
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"分片 {shard_id} 完成，命中 {len(records)} 只 → {out_file}")
    else:
        print(f"分片 {shard_id} 无命中股票")


def merge_and_select(top_n: int = 20, do_notify: bool = False) -> None:
    files = list(CACHE_DIR.glob("live_shard_*.csv"))
    if not files:
        print("没有找到任何分片结果，请先运行分片")
        return

    dfs = [pd.read_csv(f, dtype={"code": str}) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    print(f"合并后共 {len(df)} 只命中股票")

    rsrs = get_hs300_rsrs()
    if rsrs.get("ok"):
        print(f"RSRS标准分: {rsrs['rsrs_z']} | 阈值: {rsrs['threshold']} | 偏多: {rsrs['bullish']}")
    else:
        print(f"RSRS计算失败: {rsrs.get('reason')}，默认按偏多处理")
        rsrs = {"ok": False, "rsrs_z": None, "bullish": True, "threshold": RSRS_BUY_THRESHOLD}

    if df.empty:
        print("无股票通过过滤")
        if do_notify:
            title = f"选股 {datetime.now().strftime('%Y%m%d')}｜无标的"
            content = "今日无股票通过过滤。\n仅为研究输出，不构成投资建议。"
            print(notify_serverchan(title, content))
        return

    def safe_z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-8)

    df["wide_z"] = safe_z(df["wide_score"]) if "wide_score" in df.columns else 0.0
    df["VwapClose_z"] = safe_z(df["VwapClose"]) if "VwapClose" in df.columns else 0.0
    df["HighOpen_z"] = safe_z(df["HighOpen"]) if "HighOpen" in df.columns else 0.0
    df["CloseLow_z"] = safe_z(df["CloseLow"]) if "CloseLow" in df.columns else 0.0
    if "Turnover" in df.columns:
        df["Turnover_z"] = safe_z(df["Turnover"].clip(upper=TURN_OVER_CAP))
    else:
        df["Turnover_z"] = 0.0
    df["ma_bonus"] = df.get("ma_bonus", 0).fillna(0)
    df["year_first_red"] = df.get("year_first_red", 0).fillna(0)

    df["final_score"] = (
        FACTOR_WEIGHTS["wide_score"] * df["wide_z"]
        + FACTOR_WEIGHTS["VwapClose"] * df["VwapClose_z"]
        + FACTOR_WEIGHTS["HighOpen"] * df["HighOpen_z"]
        + FACTOR_WEIGHTS["CloseLow"] * df["CloseLow_z"]
        + FACTOR_WEIGHTS["year_first_red"] * df["year_first_red"]
        + FACTOR_WEIGHTS["Turnover"] * df["Turnover_z"]
        + FACTOR_WEIGHTS["ma_bonus"] * df["ma_bonus"]
    )

    if "is_wide_zone" in df.columns:
        df["_wide_flag"] = df["is_wide_zone"].astype(int)
    else:
        df["_wide_flag"] = 0

    df = df.sort_values(
        ["final_score", "_wide_flag", "year_first_red", "wide_score", "profit_pct"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    top_df = df.head(top_n)

    today = datetime.now().strftime("%Y%m%d")
    out_csv = f"live_select_launch_{today}.csv"
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"今日优选前 {len(top_df)} 只 → {out_csv}")

    timing_flag = "偏多·可关注" if rsrs.get("bullish") else "偏空·建议观望"
    print("\n" + "=" * 90)
    print(f"RSRS择时: {timing_flag} (标准分={rsrs.get('rsrs_z')})")
    print("选股结果（红筹码必须，宽幅/年线加分）")
    print("=" * 90)
    for i, row in top_df.iterrows():
        yf = "年线首红" if row.get("year_first_red", 0) > 0.5 else ""
        wz = "宽幅" if row.get("is_wide_zone") else ""
        print(
            f"{i+1:02d}. {row['code']} {row.get('name', '')} "
            f"| 收盘{row['close']} | 得分{row['final_score']:.3f} "
            f"| 获利盘{row.get('profit_pct', '-')}% | 宽幅分{row.get('wide_score', '-')} "
            f"| HO{row.get('HighOpen', 0):.3f} VC{row.get('VwapClose', 0):.3f} "
            f"| {wz} {yf} {row.get('ma_signal', '')}"
        )
    print("=" * 90)

    if do_notify:
        lines = [
            f"# 选股 {today}",
            "",
            f"**RSRS择时**：{timing_flag}（标准分 {rsrs.get('rsrs_z')}，阈值 {rsrs.get('threshold')}）",
            "",
            "硬过滤：红筹码占优 + 获利盘 40%\~85%",
            "加分：宽幅 / 年线首红 / VwapClose / HighOpen / CloseLow",
            f"共推荐 {len(top_df)} 只",
            "",
        ]
        if not rsrs.get("bullish"):
            lines.append("> 当前 RSRS 偏空，建议观望或轻仓。")
            lines.append("")
        for i, row in top_df.iterrows():
            yf = "【年线首红】" if row.get("year_first_red", 0) > 0.5 else ""
            wz = "【宽幅】" if row.get("is_wide_zone") else ""
            lines.append(
                f"{i+1:02d}. {row['code']} {row.get('name', '')} "
                f"| 收盘{row['close']} | 得分{row['final_score']:.3f} "
                f"| 获利盘{row.get('profit_pct', '-')}% | 宽幅分{row.get('wide_score', '-')} "
                f"| HO{row.get('HighOpen', 0):.3f} VC{row.get('VwapClose', 0):.3f} {wz}{yf}"
            )
        lines.append("\n仅为研究输出，不构成投资建议。")
        title = f"选股{today}|RSRS{'偏多' if rsrs.get('bullish') else '偏空'}|推荐{len(top_df)}只"
        print(f"Server酱推送结果: {notify_serverchan(title, chr(10).join(lines))}")


def main():
    parser = argparse.ArgumentParser(description="实战选股完整版")
    parser.add_argument("--shard", type=str, help="1-8 或 all")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_and_select(top_n=args.top, do_notify=args.notify)
        return
    if not args.shard:
        parser.error("请指定 --shard 或 --merge")
    if args.shard == "all":
        for i in range(1, 9):
            run_shard(i)
        print("\n全部分片完成，请执行：python live_stock_selector.py --merge --top 20 --notify")
    else:
        run_shard(int(args.shard))


if __name__ == "__main__":
    main()
