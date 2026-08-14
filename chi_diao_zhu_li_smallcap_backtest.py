#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小流通市值趋势信号的无前视偏差回测模块。

核心约束：
1. 小市值资格来自 ``--universe-snapshots`` 历史快照；每个信号日只可使用该日或更早的最近快照。
2. 信号以 T 日收盘后可知；交易以 T+1 开盘价进入；5/10/20日持有期以 T+h 收盘价退出。
3. 个股价格使用未复权日线，避免未来复权因子进入过去决策；公司行为、停牌、涨跌停、容量和真实成交价
   仍需要在研究结论中单独审阅。

快照CSV最低字段：date,code,float_market_cap。流通市值默认单位为人民币元。
代码CSV最低字段：code（可选 name）。历史股票池覆盖不完整或仅包含当前存续股票时，结果可能仍有幸存者偏差。
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chi_diao_zhu_li_optimized import DataSourceError, PRESETS, compute_indicator, fetch_with_hard_timeout, normalize_ohlc


LOGGER = logging.getLogger("chi_diao_smallcap_backtest")
HORIZONS = (5, 10, 20)


@dataclass(frozen=True)
class Config:
    codes_file: Path
    snapshots_file: Path
    start: pd.Timestamp
    end: pd.Timestamp
    mode: str
    min_cap_yi: float
    max_cap_yi: float
    snapshot_unit: str
    ak_timeout: int
    bao_timeout: int
    output_dir: Path
    offset: int
    limit: int


def normalize_code(value: Any) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(6) if len(digits) <= 6 and digits else ""


def read_codes(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    aliases = {"代码": "code", "股票代码": "code", "code": "code", "名称": "name", "name": "name"}
    raw = raw.rename(columns=aliases)
    if "code" not in raw.columns: raise ValueError("代码文件缺少 code 或 代码 列")
    raw["code"] = raw["code"].map(normalize_code)
    if "name" not in raw.columns: raw["name"] = ""
    output = raw[raw["code"].str.startswith(("0", "3", "6"))].drop_duplicates("code").sort_values("code")
    if output.empty: raise ValueError("代码文件没有有效沪深A股代码")
    return output[["code", "name"]].reset_index(drop=True)


def read_snapshots(path: Path, unit: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"code": str, "代码": str})
    aliases = {"日期": "date", "date": "date", "快照日期": "date", "代码": "code", "code": "code", "股票代码": "code", "流通市值": "float_market_cap", "float_market_cap": "float_market_cap"}
    raw = raw.rename(columns=aliases)
    required = {"date", "code", "float_market_cap"}
    if not required.issubset(raw.columns): raise ValueError("历史快照缺少 date、code 或 float_market_cap/流通市值 列")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["code"] = raw["code"].map(normalize_code)
    raw["float_market_cap"] = pd.to_numeric(raw["float_market_cap"], errors="coerce")
    if unit == "yi": raw["float_market_cap"] *= 100_000_000
    output = raw.dropna(subset=["date", "float_market_cap"])
    output = output[output["code"].str.startswith(("0", "3", "6")) & output["float_market_cap"].gt(0)]
    output = output.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    if output.empty: raise ValueError("历史快照清洗后为空")
    return output.reset_index(drop=True)


def cap_as_of(snapshots: pd.DataFrame, code: str, signal_date: pd.Timestamp) -> float | None:
    """只返回信号日及之前已有的最近快照，禁止使用未来快照。"""
    rows = snapshots[(snapshots["code"] == code) & (snapshots["date"] <= signal_date.normalize())]
    if rows.empty: return None
    return float(rows.iloc[-1]["float_market_cap"])


def fetch_history(code: str, start: date, end: date, config: Config) -> tuple[pd.DataFrame, str]:
    try:
        return normalize_ohlc(fetch_with_hard_timeout("stock_ak_raw", code, start, end, config.ak_timeout), "AkShare个股未复权"), "akshare_raw"
    except DataSourceError as ak_error:
        LOGGER.warning("%s AkShare失败，改BaoStock：%s", code, ak_error)
        return normalize_ohlc(fetch_with_hard_timeout("stock_bao", code, start, end, config.bao_timeout), "BaoStock个股"), "baostock"


def fetch_index(start: date, end: date, config: Config) -> pd.DataFrame:
    return normalize_ohlc(fetch_with_hard_timeout("index_ak", "sh000001", start, end, config.ak_timeout), "上证指数")


def signal_trades(code: str, name: str, stock: pd.DataFrame, index: pd.DataFrame, snapshots: pd.DataFrame, config: Config, source: str) -> list[dict[str, Any]]:
    indicator = compute_indicator(stock, index, PRESETS[config.mode])
    trades: list[dict[str, Any]] = []
    min_cap, max_cap = config.min_cap_yi * 100_000_000, config.max_cap_yi * 100_000_000
    for position, (signal_date, row) in enumerate(indicator.iterrows()):
        if signal_date < config.start or signal_date > config.end or int(row["买进"]) != 1: continue
        cap = cap_as_of(snapshots, code, signal_date)
        if cap is None or not (min_cap <= cap <= max_cap): continue
        # T日收盘产生信号；T+1开盘才可进场，T+h收盘才可退出。
        entry_position = position + 1
        if entry_position >= len(indicator): continue
        entry_price = float(indicator.iloc[entry_position]["open"])
        if not np.isfinite(entry_price) or entry_price <= 0: continue
        for horizon in HORIZONS:
            exit_position = position + horizon
            if exit_position >= len(indicator): continue
            exit_price = float(indicator.iloc[exit_position]["close"])
            if not np.isfinite(exit_price) or exit_price <= 0: continue
            window = indicator.iloc[entry_position:exit_position + 1]
            mae = float(window["low"].min() / entry_price - 1.0)
            index_entry = float(index.loc[indicator.index[entry_position], "open"])
            index_exit = float(index.loc[indicator.index[exit_position], "close"])
            index_return = index_exit / index_entry - 1.0 if index_entry > 0 else np.nan
            stock_return = exit_price / entry_price - 1.0
            trades.append({
                "code": code, "name": name, "signal_date": signal_date.strftime("%Y-%m-%d"), "entry_date": indicator.index[entry_position].strftime("%Y-%m-%d"), "exit_date": indicator.index[exit_position].strftime("%Y-%m-%d"),
                "horizon_days": horizon, "float_market_cap_yi": round(cap / 100_000_000, 2), "entry_open": round(entry_price, 4), "exit_close": round(exit_price, 4),
                "stock_return": stock_return, "index_return": index_return, "excess_return": stock_return - index_return if np.isfinite(index_return) else np.nan,
                "max_adverse_excursion": mae, "signal_score": int(row["信号评分"]), "source": source,
            })
    return trades


def summarize(trades: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["horizon_days", "signals", "win_rate", "avg_return", "median_return", "avg_excess_return", "avg_mae", "net_avg_return_after_cost"])
    cost = cost_bps / 10_000
    rows = []
    for horizon, group in trades.groupby("horizon_days", sort=True):
        returns = group["stock_return"].astype(float)
        rows.append({"horizon_days": int(horizon), "signals": int(len(group)), "win_rate": float((returns > 0).mean()), "avg_return": float(returns.mean()), "median_return": float(returns.median()), "avg_excess_return": float(group["excess_return"].mean()), "avg_mae": float(group["max_adverse_excursion"].mean()), "net_avg_return_after_cost": float(returns.mean() - cost)})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "无满足完整未来持有期的历史买点。"
    shown = frame.copy()
    for column in ["win_rate", "avg_return", "median_return", "avg_excess_return", "avg_mae", "net_avg_return_after_cost"]:
        if column in shown: shown[column] = (shown[column] * 100).round(2).astype(str) + "%"
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
    for row in shown.itertuples(index=False, name=None): lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def run(config: Config, cost_bps: float) -> int:
    codes = read_codes(config.codes_file); snapshots = read_snapshots(config.snapshots_file, config.snapshot_unit)
    selected = codes.iloc[config.offset:config.offset + config.limit] if config.limit else codes.iloc[config.offset:]
    fetch_start = (config.start - pd.Timedelta(days=220)).date(); fetch_end = (config.end + pd.Timedelta(days=max(HORIZONS) + 10)).date()
    index = fetch_index(fetch_start, fetch_end, config)
    records: list[dict[str, Any]] = []; failures: list[dict[str, str]] = []
    for number, item in selected.reset_index(drop=True).iterrows():
        code, name = str(item["code"]), str(item["name"])
        try:
            stock, source = fetch_history(code, fetch_start, fetch_end, config)
            records.extend(signal_trades(code, name, stock, index, snapshots, config, source))
        except DataSourceError as error:
            failures.append({"code": code, "reason": str(error)[:250]})
        if (number + 1) % 25 == 0: LOGGER.info("回测进度：%s/%s，交易记录%s，失败%s", number + 1, len(selected), len(records), len(failures))
    trades = pd.DataFrame(records); summary = summarize(trades, cost_bps)
    config.output_dir.mkdir(parents=True, exist_ok=True); stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"); prefix = config.output_dir / f"chi_diao_smallcap_backtest_{stamp}"
    trades.to_csv(prefix.with_suffix(".trades.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(prefix.with_suffix(".summary.csv"), index=False, encoding="utf-8-sig")
    metadata = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, "cost_bps": cost_bps, "codes_requested": len(selected), "trades": len(trades), "failures": failures, "disclosure": "小市值资格仅使用信号日或此前快照；若快照或代码表不覆盖退市/历史成分，仍存在幸存者偏差风险。"}
    prefix.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    prefix.with_suffix(".md").write_text(
        f"# 小流通市值趋势信号回测\n\n- 信号区间：`{config.start:%Y-%m-%d}` 至 `{config.end:%Y-%m-%d}`\n- 代码数：`{len(selected)}`；完整交易记录：`{len(trades)}`；数据失败：`{len(failures)}`\n- 交易规则：信号日 T 收盘后确认，`T+1` 开盘进入，`T+h` 收盘退出。\n- 交易成本敏感性：单次往返成本 ` {cost_bps:.1f} bps` 从平均收益中扣除。\n- 市值资格：只使用信号日或此前最近的历史快照；禁止未来快照。\n\n## 汇总\n\n{markdown_table(summary)}\n\n> 此结果没有模拟涨跌停无法成交、容量约束或退市复权的全部影响；快照/代码覆盖不完整时仍可能有幸存者偏差。\n",
        encoding="utf-8",
    )
    print(f"BACKTEST_DONE trades={len(trades)} failures={len(failures)} output={prefix}")
    return 0


def self_test() -> int:
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    close = pd.Series(np.linspace(10, 20, len(dates)), index=dates)
    stock = pd.DataFrame({"open": close * .995, "high": close * 1.01, "low": close * .99, "close": close, "volume": 1_000_000}, index=dates)
    index = pd.DataFrame({"open": np.linspace(3000, 3200, len(dates)), "high": np.linspace(3020, 3220, len(dates)), "low": np.linspace(2980, 3180, len(dates)), "close": np.linspace(3000, 3200, len(dates)), "volume": 1}, index=dates)
    snapshots = pd.DataFrame({"date": [dates[0], dates[100]], "code": ["600000", "600000"], "float_market_cap": [10e9, 12e9]})
    assert cap_as_of(snapshots, "600000", dates[50]) == 10e9
    assert cap_as_of(snapshots, "600000", dates[99]) == 10e9
    assert cap_as_of(snapshots, "600000", dates[0] - pd.Timedelta(days=1)) is None
    test_config = Config(Path("codes.csv"), Path("snapshots.csv"), dates[130], dates[220], "normal", 20, 200, "yuan", 12, 8, Path("output"), 0, 0)
    rows = signal_trades("600000", "测试", stock, index, snapshots, test_config, "test")
    assert all(pd.Timestamp(item["entry_date"]) > pd.Timestamp(item["signal_date"]) for item in rows)
    print("SELF_TEST_OK"); return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无前视偏差的小市值趋势信号历史回测")
    parser.add_argument("--codes-file", type=Path, help="历史代码CSV，至少含 code")
    parser.add_argument("--universe-snapshots", type=Path, help="历史股票池快照CSV，至少含 date,code,float_market_cap")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--mode", choices=sorted(PRESETS), default="normal")
    parser.add_argument("--min-cap-yi", type=float, default=20.0); parser.add_argument("--max-cap-yi", type=float, default=200.0)
    parser.add_argument("--snapshot-unit", choices=("yuan", "yi"), default="yuan")
    parser.add_argument("--ak-timeout", type=int, default=12); parser.add_argument("--bao-timeout", type=int, default=8)
    parser.add_argument("--cost-bps", type=float, default=20.0, help="单次往返成本，单位bps")
    parser.add_argument("--offset", type=int, default=0); parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output")); parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.self_test: return self_test()
    if args.codes_file is None or args.universe_snapshots is None: raise SystemExit("必须提供 --codes-file 和 --universe-snapshots；否则无法避免用未来市值快照筛选历史信号。")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end <= start: raise SystemExit("end必须晚于start")
    config = Config(args.codes_file, args.universe_snapshots, start, end, args.mode, args.min_cap_yi, args.max_cap_yi, args.snapshot_unit, max(1, args.ak_timeout), max(1, args.bao_timeout), args.output_dir, max(0, args.offset), max(0, args.limit))
    try: return run(config, max(0, args.cost_bps))
    except (ValueError, DataSourceError) as error:
        LOGGER.error("回测未执行：%s", error); return 2


if __name__ == "__main__": raise SystemExit(main())
