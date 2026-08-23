#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定海神针 —— 多周期极速扫描器
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import os
import sys
import time
import argparse
import json
from tqdm import tqdm

try:
    from serverchan_sdk import sc_send
except ImportError:
    sc_send = None

PERIODS = {
    "daily":   {"ak": "daily",   "label": "日", "lookback": 25, "days": 120,  "hold": [5], "distill_m": 6},
    "weekly":  {"ak": "weekly",  "label": "周", "lookback": 12, "days": 730,  "hold": [2], "distill_m": 18},
    "monthly": {"ak": "monthly", "label": "月", "lookback": 6,  "days": 1095, "hold": [1], "distill_m": 36},
    "quarterly": {"ak": "monthly", "label": "季", "lookback": 4, "days": 1460, "hold": [1], "distill_m": 60, "resample": "Q"},
    "yearly":  {"ak": "monthly", "label": "年", "lookback": 3,  "days": 2190, "hold": [1], "distill_m": 120, "resample": "Y"},
}

SIGNAL_DIR = "./signals/dinghai"
CACHE_DIR = "./cache"
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def get_all_codes():
    df = ak.stock_info_a_code_name()
    mask = ~df["name"].str.contains("ST|退|N|C|B", na=False, regex=True)
    return df[mask]["code"].astype(str).str.zfill(6).tolist()


def get_industry_map():
    cache = f"{CACHE_DIR}/industry.json"
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    print("[行业] 构建映射...")
    m = {}
    for name in tqdm(ak.stock_board_industry_name_em()["板块名称"].tolist(), desc="行业"):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=name)
            if cons is not None and not cons.empty:
                cc = "代码" if "代码" in cons.columns else cons.columns[1]
                for c in cons[cc].astype(str).str.zfill(6):
                    m[c] = name
            time.sleep(0.12)
        except Exception:
            pass
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False)
    return m


def resample_df(df, rule):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["日期"])
    df = df.set_index("dt").sort_index()
    r = df.resample(rule).agg({"开盘": "first", "最高": "max", "最低": "min", "收盘": "last"}).dropna()
    r = r.reset_index()
    r["日期"] = r["dt"].dt.strftime("%Y-%m-%d")
    return r.drop(columns=["dt"])


def get_hist(code, period, start, end):
    meta = PERIODS[period]
    df = ak.stock_zh_a_hist(symbol=code, period=meta["ak"], start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return None
    if "resample" in meta:
        df = resample_df(df, meta["resample"])
    return df


def is_dinghai(df, params):
    if len(df) < 30:
        return False
    latest, prev = df.iloc[-1], df.iloc[-2]
    o, h, l, c = latest["开盘"], latest["最高"], latest["最低"], latest["收盘"]
    entity = abs(c - o)
    lower = min(c, o) - l
    upper = h - max(c, o)
    if entity < 1e-6 or upper < 1e-6 or c <= 0 or prev["收盘"] <= 0:
        return False
    lb = params["recent_low_lookback"]
    if len(df) < lb + 1:
        return False
    return (
        lower > entity * params["lower_shadow_to_entity"] and
        lower > upper * params["lower_shadow_to_upper"] and
        entity / c < params["entity_to_close_max"] and
        c > o and
        (h - l) / prev["收盘"] * 100 >= params["amplitude_pct_min"] and
        abs(l - df["最低"].iloc[-lb:].min()) < 1e-6
    )


def distill(period):
    meta = PERIODS[period]
    label = meta["label"]
    cache = f"{CACHE_DIR}/params_{period}.json"
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            d = json.load(f)
            if (datetime.now() - datetime.fromisoformat(d["ts"])).total_seconds() < 21600:
                print(f"[{label}线] 使用缓存参数")
                return d["params"]

    print(f"\n[{label}线] 蒸馏参数...")
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=30 * meta["distill_m"])).strftime("%Y%m%d")
    try:
        codes = ak.index_stock_cons_weight_csindex(symbol="000300")["成分券代码"].astype(str).str.zfill(6).tolist()[:60]
    except Exception:
        codes = get_all_codes()[:60]

    space = {
        "lower_shadow_to_entity": [2.0, 3.0, 4.0],
        "lower_shadow_to_upper": [2.0, 2.5, 3.5],
        "entity_to_close_max": [0.008, 0.012, 0.018],
        "amplitude_pct_min": [2.0, 2.5, 3.5],
    }
    best_score, best_params = -999, None

    for combo in tqdm(list(product(*space.values())), desc=f"{label}线蒸馏"):
        p = dict(zip(["lower_shadow_to_entity", "lower_shadow_to_upper", "entity_to_close_max", "amplitude_pct_min"], combo))
        p["min_data_days"] = 30
        p["recent_low_lookback"] = meta["lookback"]
        signals = []
        for code in codes:
            try:
                df = get_hist(code, period, start, end)
                if df is None or len(df) < 35:
                    continue
                for i in range(30, len(df) - meta["hold"][0]):
                    if is_dinghai(df.iloc[:i + 1], p):
                        ret = (df.iloc[i + meta["hold"][0]]["收盘"] - df.iloc[i]["收盘"]) / df.iloc[i]["收盘"] * 100
                        signals.append(ret)
            except Exception:
                pass
        if len(signals) < 3:
            continue
        wins = [r for r in signals if r > 0]
        wr = len(wins) / len(signals) if signals else 0
        ar = np.mean(signals)
        losses = [r for r in signals if r <= 0]
        pl = abs(np.mean(wins)) / abs(np.mean(losses)) if losses else 999
        score = 0.35 * wr + 0.30 * (ar / 10) + 0.25 * min(pl / 3, 1) + 0.10 * min(len(signals) / 30, 1)
        if score > best_score:
            best_score, best_params = score, p

    if best    if best_params is None:
        best_params = {"lower_shadow_to_entity": 3.0, "lower_shadow_to_upper": 2.5, "entity_to_close_max": 0.012, "amplitude_pct_min": 2.5, "recent_low_lookback": meta["lookback"]}

    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"params": best_params, "ts": datetime.now().isoformat()}, f)
    print(f"[{label}线] 最优: 下影/实体={best_params['lower_shadow_to_entity']}, 下影/上影={best_params['lower_shadow_to_upper']}, 实体/收盘={best_params['entity_to_close_max']}, 振幅>={best_params['amplitude_pct_min']}%")
    return best_params


def prefilter_daily(codes, params):
    try:
        spot = ak.stock_zh_a_spot_em()
    except Exception:
        return codes
    cc = "代码" if "代码" in spot.columns else spot.columns[0]
    spot = spot.rename(columns={cc: "代码"})
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    spot = spot[spot["代码"].isin(set(codes))].copy()
    if spot.empty:
        return []

    cm = {"open": "今开" if "今开" in spot.columns else "开盘价",
          "close": "最新价" if "最新价" in spot.columns else "收盘价",
          "high": "最高" if "最高" in spot.columns else "最高价",
          "low": "最低" if "最低" in spot.columns else "最低价",
          "prev": "昨收" if "昨收" in spot.columns else "昨收价"}
    for c in cm.values():
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")

    o, c, h, l, prev = spot[cm["open"]], spot[cm["close"]], spot[cm["high"]], spot[cm["low"]], spot[cm["prev"]]
    entity = (c - o).abs()
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    amp = (h - l) / prev * 100
    mask = (c > o) & (lower > entity * params["lower_shadow_to_entity"]) & (lower > upper * params["lower_shadow_to_upper"]) & (entity / c < params["entity_to_close_max"]) & (amp >= params["amplitude_pct_min"])
    return spot.loc[mask, "代码"].tolist()


def scan(codes, industry_map, params, period, date_str):
    meta = PERIODS[period]
    label = meta["label"]
    lb = params["recent_low_lookback"]

    if date_str:
        end = date_str
        start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=meta["days"])).strftime("%Y%m%d")
    else:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=meta["days"])).strftime("%Y%m%d")

    candidates = prefilter_daily(codes, params) if period == "daily" else codes
    print(f"[{label}线] 候选 {len(candidates)} 只")

    results = []
    for code in tqdm(candidates, desc=f"{label}线扫描"):
        try:
            df = get_hist(code, period, start, end)
            if df is None or len(df) < lb + 1:
                continue
            if is_dinghai(df, params):
                latest = df.iloc[-1]
                results.append({
                    "代码": code, "板块": industry_map.get(code, "未知"),
                    "周期": label, "日期": str(latest["日期"]),
                    "收盘": round(float(latest["收盘"]), 2),
                    "最低": round(float(latest["最低"]), 2),
                    "振幅%": round((latest["最高"] - latest["最低"]) / df.iloc[-2]["收盘"] * 100, 2)
                })
        except Exception:
            pass
        time.sleep(0.15)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).drop_duplicates(subset=["代码"])


def push(title, content):
    key = os.environ.get("SERVERCHAN_SEND_KEY", "")
    if not key or sc_send is None:
        print("[推送] 跳过")
        return
    try:
        sc_send(key, title, content, tags="定海神针")
        print("[推送] 已发送")
    except Exception as e:
        print(f"[推送] 失败: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="+", default=["daily"])
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    periods = [p for p in args.periods if p in PERIODS]
    if not periods:
        sys.exit("无有效周期")

    print(f"🚀 定海神针扫描 {date_str}")
    codes = get_all_codes()
    industry = get_industry_map()
    print(f"[市场] {len(codes)} 只")

    all_results = {}
    for period in periods:
        params = distill(period)
        df = scan(codes, industry, params, period, date_str)
        all_results[period] = df
        if not df.empty:
            fpath = f"{SIGNAL_DIR}/dinghai_{period}_{date_str}.csv"
            df.to_csv(fpath, index=False, encoding="utf-8-sig")
            print(f"🏆 [{PERIODS[period]['label']}线] {len(df)} 只 → {fpath}")

    lines = [f"## 📌 定海神针 {date_str}", ""]
    total = 0
    for p in periods:
        df = all_results[p]
        if df.empty:
            continue
        lb = PERIODS[p]["label"]
        total += len(df)
        lines.append(f"### [{lb}线] {len(df)} 只")
        lines.append("| 代码 | 板块 | 日期 | 收盘 | 最低 | 振幅% |")
        lines.append("|------|------|------|------|------|-------|")
        for _, r in df.iterrows():
            lines.append(f"| {r['代码']} | {r['板块']} | {r['日期']} | {r['收盘']} | {r['最低']} | {r['振幅%']} |")
        lines.append("")

    if total == 0:
        lines.append("> 📭 今日无信号")
    else:
        lines.append(f"**合计: {total} 只**")

    md = "\n".join(lines)
    report = f"{SIGNAL_DIR}/report_{date_str}.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write(md)

    if not args.no_push:
        push(f"定海神针 {date_str} | 共{total}只", md)

    print("✅ 完成")


if __name__ == "__main__":
    main()
