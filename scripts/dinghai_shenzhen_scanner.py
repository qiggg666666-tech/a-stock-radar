#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定海神针 —— GitHub Actions 多周期极速扫描器
支持: 日线/周线/月线/季线/年线，结果推送 Server酱 + 本地存档

环境变量:
    SERVERCHAN_SEND_KEY  - Server酱 SendKey
    GH_REPO_NAME         - 仓库名(可选，用于推送链接)
    GH_RUN_ID            - GitHub Run ID(可选，用于推送链接)

用法:
    python scripts/dinghai_shenzhen_scanner.py --periods daily
    python scripts/dinghai_shenzhen_scanner.py --periods daily weekly --parallel 2
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from typing import List, Dict, Optional, Set
import time
import os
import sys
import argparse
import warnings
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Server酱推送
try:
    from serverchan_sdk import sc_send
except ImportError:
    sc_send = None

warnings.filterwarnings("ignore")

# ==================== 周期配置 ====================
PERIOD_META = {
    "daily": {
        "ak_period": "daily", "label": "日", "lookback_default": 25,
        "history_days": 120, "hold_days_list": [5], "distill_months": 6,
    },
    "weekly": {
        "ak_period": "weekly", "label": "周", "lookback_default": 12,
        "history_days": 730, "hold_days_list": [2], "distill_months": 18,
    },
    "monthly": {
        "ak_period": "monthly", "label": "月", "lookback_default": 6,
        "history_days": 1095, "hold_days_list": [1], "distill_months": 36,
    },
    "quarterly": {
        "ak_period": "monthly", "resample": "Q", "label": "季",
        "lookback_default": 4, "history_days": 1460, "hold_days_list": [1], "distill_months": 60,
    },
    "yearly": {
        "ak_period": "monthly", "resample": "Y", "label": "年",
        "lookback_default": 3, "history_days": 2190, "hold_days_list": [1], "distill_months": 120,
    },
}

CONFIG = {
    "sleep_sec": 0.15,
    "parallel_sleep": 0.30,
    "batch_progress_size": 50,
    "cache_dir": "./cache",
    "signal_dir": "./signals/dinghai",
    "industry_cache_file": "./cache/industry_map_cache.json",
    "industry_cache_ttl_hours": 48,
    "spot_cache_sec": 300,
    "distill": {
        "sample_pool": "hs300", "sample_size": 60,
        "param_space": {
            "lower_shadow_to_entity": [2.0, 3.0, 4.0],
            "lower_shadow_to_upper": [2.0, 2.5, 3.5],
            "entity_to_close_max": [0.008, 0.012, 0.018],
            "amplitude_pct_min": [2.0, 2.5, 3.5],
        },
        "weights": {"win_rate": 0.35, "avg_return": 0.30, "profit_loss_ratio": 0.25, "signal_freq": 0.10},
    },
}

_SPOT_CACHE = None
_SPOT_CACHE_TIME = 0


def ensure_dirs():
    for d in [CONFIG["cache_dir"], CONFIG["signal_dir"]]:
        os.makedirs(d, exist_ok=True)


def load_json_cache(filepath: str, ttl_hours: int) -> Optional[dict]:
    if not os.path.exists(filepath):
        return None
    try:
        mtime = os.path.getmtime(filepath)
        if (time.time() - mtime) / 3600 > ttl_hours:
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json_cache(filepath: str, data: dict):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_all_a_stocks() -> List[str]:
    try:
        df = ak.stock_info_a_code_name()
        mask = ~df["name"].str.contains("ST|退|N|C|B", na=False, regex=True)
        return df[mask]["code"].astype(str).str.zfill(6).tolist()
    except Exception as e:
        print(f"[错误] 获取股票列表失败: {e}")
        return []


def build_industry_map(use_cache: bool = True) -> dict:
    if use_cache:
        cached = load_json_cache(CONFIG["industry_cache_file"], CONFIG["industry_cache_ttl_hours"])
        if cached is not None:
            return cached
    print("[行业] 构建映射表...")
    code_to_industry = {}
    try:
        boards = ak.stock_board_industry_name_em()
        for name in tqdm(boards["板块名称"].tolist(), desc="行业"):
            try:
                cons = ak.stock_board_industry_cons_em(symbol=name)
                if cons is not None and not cons.empty:
                    code_col = "代码" if "代码" in cons.columns else cons.columns[1]
                    for code in cons[code_col].astype(str).str.zfill(6):
                        code_to_industry[code] = name
                time.sleep(0.12)
            except Exception:
                continue
        print(f"[行业] 完成: {len(code_to_industry)} 只")
        save_json_cache(CONFIG["industry_cache_file"], code_to_industry)
    except Exception as e:
        print(f"[行业] 失败: {e}")
    return code_to_industry


def resample_kline(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df["日期_dt"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期_dt").sort_index()
    resampled = df.resample(rule).agg({
        "开盘": "first", "最高": "max", "最低": "min", "收盘": "last",
        "成交量": "sum", "成交额": "sum",
    }).dropna()
    prev_close = resampled["收盘"].shift(1)
    resampled["振幅"] = (resampled["最高"] - resampled["最低"]) / prev_close * 100
    resampled = resampled.reset_index()
    resampled["日期"] = resampled["日期_dt"].dt.strftime("%Y-%m-%d")
    return resampled.drop(columns=["日期_dt"])


def get_kline_data(code: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
    meta = PERIOD_META[period]
    try:
        df = ak.stock_zh_a_hist(symbol=code, period=meta["ak_period"],
                                start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            return df
        if "resample" in meta:
            df = resample_kline(df, meta["resample"])
        return df
    except Exception:
        return None


def is_dinghai_shenzhen(df: pd.DataFrame, params: dict) -> bool:
    if len(df) < params.get("min_data_days", 30):
        return False
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = latest["开盘"], latest["最高"], latest["最低"], latest["收盘"]
    entity = abs(c - o)
    lower_shadow = min(c, o) - l
    upper_shadow = h - max(c, o)
    if entity < 1e-6 or upper_shadow < 1e-6 or c <= 0 or prev["收盘"] <= 0:
        return False
    cond1 = lower_shadow > entity * params["lower_shadow_to_entity"]
    cond2 = lower_shadow > upper_shadow * params["lower_shadow_to_upper"]
    cond3 = entity / c < params["entity_to_close_max"]
    cond4 = c > o
    cond5 = (h - l) / prev["收盘"] * 100 >= params["amplitude_pct_min"]
    lookback = params["recent_low_lookback"]
    if len(df) < lookback + 1:
        return False
    recent_low = df["最低"].iloc[-lookback:].min()
    cond6 = abs(l - recent_low) < 1e-6
    return all([cond1, cond2, cond3, cond4, cond5, cond6])


def backtest_single_period(code: str, params: dict, hold_days: int,
                           period: str, start_date: str, end_date: str) -> List[dict]:
    results = []
    try:
        df = get_kline_data(code, period, start_date, end_date)
        if df is None or len(df) < params.get("min_data_days", 30) + hold_days:
            return results
        df = df.reset_index(drop=True)
        for i in range(params.get("min_data_days", 30), len(df) - hold_days):
            if is_dinghai_shenzhen(df.iloc[:i+1], params):
                entry = df.iloc[i]["收盘"]
                exit_p = df.iloc[i + hold_days]["收盘"]
                results.append({"return_pct": (exit_p - entry) / entry * 100})
    except Exception:
        pass
    return results


def evaluate_param_set_period(param_combo: tuple, hold_days: int, codes: List[str],
                              period: str, start_date: str, end_date: str,
                              weights: dict, lookback: int) -> dict:
    keys = ["lower_shadow_to_entity", "lower_shadow_to_upper",
            "entity_to_close_max", "amplitude_pct_min"]
    params = dict(zip(keys, param_combo))
    params["min_data_days"] = 30
    params["recent_low_lookback"] = lookback
    all_signals = []
    for code in codes:
        all_signals.extend(backtest_single_period(code, params, hold_days, period, start_date, end_date))
    if len(all_signals) < 3:
        return {"params": params, "score": -999, "signal_count": 0,
                "win_rate": 0, "avg_return": 0, "profit_loss_ratio": 0}
    returns = [s["return_pct"] for s in all_signals]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns) * 100 if returns else 0
    avg_ret = np.mean(returns)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1e-6
    plr = avg_win / avg_loss if avg_loss > 0 else 999
    freq_score = min(len(all_signals) / 20, 1.0)
    score = (weights["win_rate"] * (win_rate / 100) +
             weights["avg_return"] * (avg_ret / 10) +
             weights["profit_loss_ratio"] * min(plr / 3, 1.0) +
             weights["signal_freq"] * freq_score)
    return {"params": params, "score": round(score, 4), "signal_count": len(all_signals),
            "win_rate": round(win_rate, 2), "avg_return": round(avg_ret, 3),
            "profit_loss_ratio": round(plr, 2)}


def distill_period(period: str) -> dict:
    meta = PERIOD_META[period]
    label = meta["label"]
    cache_file = f"{CONFIG['cache_dir']}/adaptive_params_{period}.json"
    cached = load_json_cache(cache_file, 6)
    if cached and "params" in cached:
        print(f"[{label}线] 使用缓存参数 (评分: {cached.get('score', 'N/A')})")
        return cached["params"]

    print(f"\n[{label}线] 蒸馏参数...")
    cfg = CONFIG["distill"]
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30 * meta["distill_months"])).strftime("%Y%m%d")
    codes = []
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        codes = df["成分券代码"].astype(str).str.zfill(6).tolist()[:cfg["sample_size"]]
    except Exception:
        codes = get_all_a_stocks()[:cfg["sample_size"]]

    space = cfg["param_space"]
    all_combos = list(product(*list(space.values())))
    tasks = [(combo, hd, codes, period, start_date, end_date, cfg["weights"], meta["lookback_default"])
             for combo in all_combos for hd in meta["hold_days_list"]]
    results = []
    for task in tqdm(tasks, desc=f"{label}线蒸馏"):
        results.append(evaluate_param_set_period(*task))
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    best_params = best["params"]
    best_params["hold_days"] = best.get("hold_days", meta["hold_days_list"][0])
    print(f"[{label}线] 最优: 下影/实体={best_params['lower_shadow_to_entity']}, 下影/上影={best_params['lower_shadow_to_upper']}, 实体/收盘={best_params['entity_to_close_max']}, 振幅>={best_params['amplitude_pct_min']}%, 回看{meta['lookback_default']}{label}")
    save_json_cache(cache_file, {"params": best_params, **best,
                    "timestamp": datetime.now().isoformat(), "period": period})
    return best_params


def get_spot_em() -> Optional[pd.DataFrame]:
    global _SPOT_CACHE, _SPOT_CACHE_TIME
    now = time.time()
    if _SPOT_CACHE is not None and (now - _SPOT_CACHE_TIME) < CONFIG["spot_cache_sec"]:
        return _SPOT_CACHE
    try:
        df = ak.stock_zh_a_spot_em()
        _SPOT_CACHE = df
        _SPOT_CACHE_TIME = now
        return df
    except Exception as e:
        print(f"[预过滤] 失败: {e}")
        return None


def prefilter_candidates_daily(codes: List[str], params: dict) -> List[str]:
    spot_df = get_spot_em()
    if spot_df is None or spot_df.empty:
        return codes
    code_col = "代码" if "代码" in spot_df.columns else spot_df.columns[0]
    spot_df = spot_df.rename(columns={code_col: "代码"})
    spot_df["代码"] = spot_df["代码"].astype(str).str.zfill(6)
    spot_df = spot_df[spot_df["代码"].isin(set(codes))].copy()
    if spot_df.empty:
        return []

    col_map = {
        "open": "今开" if "今开" in spot_df.columns else "开盘价",
        "close": "最新价" if "最新价" in spot_df.columns else "收盘价",
        "high": "最高" if "最高" in spot_df.columns else "最高价",
        "low": "最低" if "最低" in spot_df.columns else "最低价",
        "prev_close": "昨收" if "昨收" in spot_df.columns else "昨收价",
    }
    for col in col_map.values():
        if col in spot_df.columns:
            spot_df[col] = pd.to_numeric(spot_df[col], errors="coerce")

    o = spot_df[col_map["open"]]
    c = spot_df[col_map["close"]]
    h = spot_df[col_map["high"]]
    l = spot_df[col_map["low"]]
    prev = spot_df[col_map["prev_close"]]
    entity = (c - o).abs()
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    amplitude = (h - l) / prev * 100

    mask = (
        (c > o) &
        (lower_shadow > entity * params["lower_shadow_to_entity"]) &
        (lower_shadow > upper_shadow * params["lower_shadow_to_upper"]) &
        (entity / c < params["entity_to_close_max"]) &
        (amplitude >= params["amplitude_pct_min"])
    )
    candidates = spot_df.loc[mask, "代码"].tolist()
    print(f"[预过滤] {len(spot_df)}只 → 候选{len(candidates)}只 (筛掉{len(spot_df)-len(candidates)}只)")
    return candidates


def scan_period(codes: List[str], industry_map: dict, params: dict,
                period: str, scan_date: str, sleep_sec: float) -> pd.DataFrame:
    meta = PERIOD_META[period]
    label = meta["label"]
    lookback = params["recent_low_lookback"]
    needed_days = meta["history_days"]

    if scan_date:
        end_date = scan_date
        start_date = (datetime.strptime(scan_date, "%Y%m%d") - timedelta(days=needed_days)).strftime("%Y%m%d")
    else:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=needed_days)).strftime("%Y%m%d")

    # 日线预过滤，其他周期全量
    if period == "daily":
        candidates = prefilter_candidates_daily(codes, params)
    else:
        candidates = codes

    results = []
    for code in tqdm(candidates, desc=f"{label}线扫描"):
        try:
            df = get_kline_data(code, period, start_date, end_date)
            if df is None or len(df) < lookback + 1:
                continue
            if is_dinghai_shenzhen(df, params):
                latest = df.iloc[-1]
                results.append({
                    "代码": code,
                    "板块": industry_map.get(code, "未知板块"),
                    "周期": label,
                    "日期": str(latest["日期"]),
                    "收盘": round(float(latest["收盘"]), 2),
                    "最低": round(float(latest["最低"]), 2),
                    "振幅%": round((latest["最高"] - latest["最低"]) / df.iloc[-2]["收盘"] * 100, 2)
                })
        except Exception:
            pass
        time.sleep(sleep_sec)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).drop_duplicates(subset=["代码"])


def save_signal(df: pd.DataFrame, period: str, scan_date: str):
    if df.empty:
        return None
    fname = f"{CONFIG['signal_dir']}/dinghai_{period}_{scan_date}.csv"
    os.makedirs(CONFIG["signal_dir"], exist_ok=True)
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"[存档] {fname} ({len(df)}条)")
    return fname


def build_markdown_report(all_results: Dict[str, pd.DataFrame], scan_date: str) -> str:
    lines = [f"## 📌 定海神针信号报告 ({scan_date})", ""]
    total = 0
    for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
        if period not in all_results or all_results[period].empty:
            continue
        df = all_results[period]
        label = PERIOD_META[period]["label"]
        total += len(df)
        lines.append(f"### [{label}线] 命中 {len(df)} 只")
        lines.append("| 代码 | 板块 | 日期 | 收盘 | 最低 | 振幅% |")
        lines.append("|------|------|------|------|------|-------|")
        for _, row in df.iterrows():
            lines.append(f"| {row['代码']} | {row['板块']} | {row['日期']} | {row['收盘']} | {row['最低']} | {row['振幅%']} |")
        lines.append("")

    if total == 0:
        lines.append("> 📭 今日全周期无符合定海神针条件的股票。")
    else:
        lines.append(f"**全周期合计命中: {total} 只**")

    # 添加仓库链接（如果环境变量存在）
    repo = os.environ.get("GH_REPO_NAME", "")
    run_id = os.environ.get("GH_RUN_ID", "")
    if repo and run_id:
        lines.append(f"")
        lines.append(f"[查看详细日志](https://github.com/{repo}/actions/runs/{run_id})")
    return "\n".join(lines)


def push_serverchan(title: str, content: str):
    sendkey = os.environ.get("SERVERCHAN_SEND_KEY", "")
    if not sendkey:
        print("[推送] 未配置 SERVERCHAN_SEND_KEY，跳过")
        return
    if sc_send is None:
        print("[推送] serverchan_sdk 未安装，跳过")
        return
    try:
        # 方糖 Server酱推送
        resp = sc_send(sendkey, title, content, tags="定海神针")
        print(f"[推送] Server酱已发送: {resp}")
    except Exception as e:
        print(f"[推送] 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="定海神针 GitHub Actions 扫描器")
    parser.add_argument("--periods", nargs="+", default=["daily"],
                        help="扫描周期: daily weekly monthly quarterly yearly all")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--no-industry", action="store_true")
    parser.add_argument("--no-push", action="store_true", help="跳过Server酱推送")
    args = parser.parse_args()

    ensure_dirs()
    scan_date = args.date if args.date else datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("🚀 定海神针 —— GitHub Actions 多周期扫描")
    print(f"📅 日期: {scan_date}")
    print("=" * 60)

    periods = args.periods
    if "all" in periods:
        periods = list(PERIOD_META.keys())

    # 行业映射
    industry_map = {} if args.no_industry else build_industry_map(use_cache=True)

    # 全市场股票
    all_codes = get_all_a_stocks()
    if not all_codes:
        print("[错误] 无法获取股票列表")
        sys.exit(1)
    print(f"[市场] 共 {len(all_codes)} 只A股")

    all_results = {}

    for period in periods:
        if period not in PERIOD_META:
            continue
        meta = PERIOD_META[period]
        label = meta["label"]

        # 蒸馏参数
        params = distill_period(period)

        print(f"\n🔍 [{label}线] 开始扫描...")
        sleep_sec = CONFIG["parallel_sleep"] if args.parallel > 1 else CONFIG["sleep_sec"]

        df = scan_period(all_codes, industry_map, params, period, scan_date, sleep_sec)
        all_results[period] = df

        if not df.empty:
            save_signal(df, period, scan_date)
            print(f"🏆 [{label}线] 命中 {len(df)} 只")
        else:
            print(f"📭 [{label}线] 无命中")

    # 生成报告
    md_report = build_markdown_report(all_results, scan_date)

    # 保存 markdown 报告
    report_file = f"{CONFIG['signal_dir']}/report_{scan_date}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"\n[报告] 已保存: {report_file}")

    # 推送 Server酱
    if not args.no_push:
        total = sum(len(v) for v in all_results.values() if not v.empty)
        title = f"定海神针 {scan_date} | 共{total}只信号"
        push_serverchan(title, md_report)

    # 退出码：有信号返回0，无信号也返回0（GitHub Actions不失败）
    print("\n✅ 扫描完成")


if __name__ == "__main__":
    main()
