#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格预备首粉：20个交易日无前视回测分片器。

规则：信号仅使用信号日及此前日线；下一交易日开盘入场；第20个交易日收盘退出。
本脚本只做研究回放，不发送通知，也不改变每日生产扫描器。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chi_diao_zhu_li_optimized import (
    DataSourceError,
    PRESETS,
    compute_indicator,
    fetch_with_hard_timeout,
    normalize_ohlc,
)


TRADE_COLUMNS = [
    "代码", "名称", "信号类型", "信号日期", "入场日期", "退出日期", "入场价", "退出价",
    "毛收益%", "净收益%", "最大不利波动%", "最大有利波动%", "持有交易日", "信号评分",
    "预备首粉评分", "粉红准备度", "多周期评分", "多周期通过数", "多周期缺失条件", "量比5", "三日涨幅%", "数据源",
]


@dataclass(frozen=True)
class Config:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    hold_days: int
    cost_bps_each_side: float
    mode: str
    signal_scope: str
    scan_offset: int
    scan_limit: int
    shard_name: str
    history_buffer_days: int
    future_buffer_days: int
    ak_timeout: int
    bao_timeout: int
    index_timeout: int
    universe_file: Path
    universe_status_file: Path | None
    output_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="严格预备首粉20交易日无前视回测分片器")
    parser.add_argument("--universe-file", type=Path, required=True, help="当次股票池CSV；历史回放存在当前存续股票池偏差")
    parser.add_argument("--universe-status", type=Path, default=None)
    parser.add_argument("--start-date", required=True, help="允许信号的首日，YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="允许信号的末日，YYYY-MM-DD")
    parser.add_argument("--hold-days", type=int, default=20, help="次日开盘入场后，第N个交易日收盘退出")
    parser.add_argument("--cost-bps-each-side", type=float, default=10.0, help="单边交易成本，基点")
    parser.add_argument("--mode", choices=sorted(PRESETS), default="normal")
    parser.add_argument("--signal-scope", choices=("pre_pink_a", "confirmed_first_pink", "multi_pre_pink", "multi_confirmed_first_pink", "both", "all"), default="all")
    parser.add_argument("--scan-offset", type=int, default=0)
    parser.add_argument("--scan-limit", type=int, default=0, help="0表示本分片从offset起扫描全部")
    parser.add_argument("--shard-name", default="a")
    parser.add_argument("--history-buffer-days", type=int, default=420, help="信号计算的前置自然日缓冲")
    parser.add_argument("--future-buffer-days", type=int, default=60, help="末端持仓结算的自然日缓冲")
    parser.add_argument("--ak-timeout", type=int, default=12)
    parser.add_argument("--bao-timeout", type=int, default=8)
    parser.add_argument("--index-timeout", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start_date).normalize(), pd.Timestamp(args.end_date).normalize()
    if end < start:
        raise ValueError("end-date必须不早于start-date")
    if args.hold_days < 1:
        raise ValueError("hold-days必须至少为1")
    if args.cost_bps_each_side < 0:
        raise ValueError("cost-bps-each-side不能为负数")
    return Config(
        start_date=start, end_date=end, hold_days=args.hold_days, cost_bps_each_side=float(args.cost_bps_each_side),
        mode=args.mode, signal_scope=args.signal_scope, scan_offset=max(0, args.scan_offset), scan_limit=max(0, args.scan_limit),
        shard_name=(args.shard_name or "a").lower(), history_buffer_days=max(130, args.history_buffer_days),
        future_buffer_days=max(30, args.future_buffer_days), ak_timeout=max(1, args.ak_timeout), bao_timeout=max(1, args.bao_timeout),
        index_timeout=max(1, args.index_timeout), universe_file=args.universe_file, universe_status_file=args.universe_status,
        output_dir=args.output_dir,
    )


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_universe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"股票池文件不存在：{path}")
    frame = pd.read_csv(path, dtype={"代码": str})
    if not {"代码", "名称"}.issubset(frame.columns):
        raise ValueError("股票池CSV必须包含代码、名称列")
    frame = frame.copy()
    frame["代码"] = frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    return frame.dropna(subset=["代码"]).drop_duplicates("代码").sort_values("代码").reset_index(drop=True)


def fetch_history(code: str, start: date, end: date, config: Config) -> tuple[pd.DataFrame, str]:
    try:
        return normalize_ohlc(fetch_with_hard_timeout("stock_ak_raw", code, start, end, config.ak_timeout), "AkShare个股未复权"), "akshare_raw"
    except DataSourceError:
        return normalize_ohlc(fetch_with_hard_timeout("stock_bao", code, start, end, config.bao_timeout), "BaoStock个股"), "baostock"


def selected_signal_types(row: pd.Series, scope: str) -> list[str]:
    types: list[str] = []
    is_multi_pre = int(row.get("多周期预粉", 0) or 0) == 1
    is_multi_confirmed = int(row.get("多周期首粉确认", 0) or 0) == 1
    if scope in {"multi_pre_pink", "all"} and is_multi_pre:
        types.append("多周期预粉M")
    if scope in {"pre_pink_a", "both", "all"} and int(row.get("预备首粉", 0) or 0) == 1 and not is_multi_pre:
        types.append("预备首粉A")
    if scope in {"multi_confirmed_first_pink", "all"} and is_multi_confirmed:
        types.append("多周期确认首粉")
    if scope in {"confirmed_first_pink", "both", "all"} and int(row.get("首粉确认", 0) or 0) == 1 and not is_multi_confirmed:
        types.append("确认首粉")
    return types


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def extract_trades(result: pd.DataFrame, code: str, name: str, source: str, config: Config) -> tuple[list[dict[str, Any]], Counter[str]]:
    trades: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    cost_rate = config.cost_bps_each_side / 10_000.0
    for signal_position, (signal_date, row) in enumerate(result.iterrows()):
        day = pd.Timestamp(signal_date).normalize()
        if day < config.start_date or day > config.end_date:
            continue
        types = selected_signal_types(row, config.signal_scope)
        if not types:
            continue
        entry_position, exit_position = signal_position + 1, signal_position + config.hold_days
        if exit_position >= len(result):
            skipped["insufficient_future_sessions"] += len(types)
            continue
        entry, exit_row = result.iloc[entry_position], result.iloc[exit_position]
        entry_price, exit_price = entry.get("open"), exit_row.get("close")
        if not finite(entry_price) or not finite(exit_price) or float(entry_price) <= 0:
            skipped["invalid_entry_or_exit_price"] += len(types)
            continue
        holding = result.iloc[entry_position:exit_position + 1]
        low_after_entry, high_after_entry = holding["low"].min(), holding["high"].max()
        gross = float(exit_price) / float(entry_price) - 1.0
        net = (1.0 - cost_rate) * (1.0 + gross) * (1.0 - cost_rate) - 1.0
        for signal_type in types:
            trades.append({
                "代码": code, "名称": name, "信号类型": signal_type, "信号日期": day.strftime("%Y-%m-%d"),
                "入场日期": pd.Timestamp(result.index[entry_position]).strftime("%Y-%m-%d"),
                "退出日期": pd.Timestamp(result.index[exit_position]).strftime("%Y-%m-%d"),
                "入场价": round(float(entry_price), 4), "退出价": round(float(exit_price), 4),
                "毛收益%": round(gross * 100.0, 4), "净收益%": round(net * 100.0, 4),
                "最大不利波动%": round((float(low_after_entry) / float(entry_price) - 1.0) * 100.0, 4),
                "最大有利波动%": round((float(high_after_entry) / float(entry_price) - 1.0) * 100.0, 4),
                "持有交易日": config.hold_days, "信号评分": int(row.get("信号评分", 0) or 0),
                "预备首粉评分": int(row.get("预备首粉评分", 0) or 0), "粉红准备度": int(row.get("粉红准备度", 0) or 0),
                "多周期评分": int(row.get("多周期评分", 0) or 0), "多周期通过数": int(row.get("多周期通过数", 0) or 0), "多周期缺失条件": str(row.get("多周期缺失条件", "") or ""),
                "量比5": round(float(row["量比5"]), 4) if finite(row.get("量比5")) else None,
                "三日涨幅%": round(float(row["三日涨幅%"]), 4) if finite(row.get("三日涨幅%")) else None,
                "数据源": source,
            })
    return trades, skipped


def summarize_trades(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate_pct": None, "mean_net_return_pct": None, "median_net_return_pct": None, "worst_trade_mae_pct": None, "best_trade_mfe_pct": None}
    returns = pd.to_numeric(frame["净收益%"], errors="coerce").dropna()
    return {
        "trades": int(len(frame)), "win_rate_pct": round(float((returns > 0).mean() * 100.0), 4),
        "mean_net_return_pct": round(float(returns.mean()), 4), "median_net_return_pct": round(float(returns.median()), 4),
        "worst_trade_mae_pct": round(float(pd.to_numeric(frame["最大不利波动%"], errors="coerce").min()), 4),
        "best_trade_mfe_pct": round(float(pd.to_numeric(frame["最大有利波动%"], errors="coerce").max()), 4),
    }


def run(config: Config) -> int:
    universe = load_universe(config.universe_file)
    shard = universe.iloc[config.scan_offset: config.scan_offset + config.scan_limit if config.scan_limit else None].copy()
    status = load_json(config.universe_status_file)
    fetch_start = (config.start_date - pd.Timedelta(days=config.history_buffer_days)).date()
    fetch_end = (config.end_date + pd.Timedelta(days=config.future_buffer_days)).date()
    benchmark = None
    benchmark_source = "unavailable"
    try:
        benchmark = normalize_ohlc(fetch_with_hard_timeout("index_ak", "sh000001", fetch_start, fetch_end, config.index_timeout), "AkShare上证指数")
        benchmark_source = "akshare_index"
    except DataSourceError:
        # 相对强弱会降级为不可用；个股日线仍可按同一无前视交易规则完成回放。
        benchmark_source = "unavailable_degraded"
    trades: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for item in shard.to_dict("records"):
        code, name = str(item["代码"]), str(item["名称"])
        try:
            stock, source = fetch_history(code, fetch_start, fetch_end, config)
            result = compute_indicator(stock, benchmark, PRESETS[config.mode])
            found, skipped_for_symbol = extract_trades(result, code, name, source, config)
            trades.extend(found); skipped.update(skipped_for_symbol); source_counts[source] += 1
        except Exception as error:
            errors.append({"代码": code, "名称": name, "错误": f"{type(error).__name__}:{str(error)[:160]}"})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS).sort_values(["信号日期", "信号类型", "净收益%"], ascending=[True, True, False]) if trades else pd.DataFrame(columns=TRADE_COLUMNS)
    prefix = f"pre_pink_backtest_{config.shard_name}"
    trade_frame.to_csv(config.output_dir / f"{prefix}.trades.csv", index=False, encoding="utf-8-sig")
    report = {
        "state": "completed", "shard": config.shard_name, "scan_offset": config.scan_offset, "scanned_codes": int(len(shard)), "universe_count": int(len(universe)),
        "processed_codes": int(sum(source_counts.values())), "source_counts": dict(source_counts), "benchmark_source": benchmark_source, "errors": errors,
        "skipped": dict(skipped), "period": {"signal_start": str(config.start_date.date()), "signal_end": str(config.end_date.date()), "entry_rule": "信号日后下一交易日开盘", "exit_rule": f"入场后第{config.hold_days}个交易日收盘", "cost_bps_each_side": config.cost_bps_each_side},
        "signal_scope": config.signal_scope, "mode": config.mode, "universe_basis": {"file": str(config.universe_file), "run_id": status.get("universe_run_id"), "as_of": status.get("universe_as_of"), "warning": "当前股票池回放存在存续股票偏差；价格路径无前视，但不能代表历史全市场存续池。"},
        "summary": summarize_trades(trade_frame), "by_signal_type": {kind: summarize_trades(group) for kind, group in trade_frame.groupby("信号类型", dropna=False)},
    }
    (config.output_dir / f"{prefix}.state.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


def self_test() -> int:
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    result = pd.DataFrame({"open": np.linspace(10, 12.9, 30), "close": np.linspace(10.1, 13.0, 30), "low": np.linspace(9.8, 12.7, 30), "high": np.linspace(10.3, 13.2, 30), "预备首粉": [0] * 5 + [1] + [0] * 24, "多周期预粉": [0] * 5 + [1] + [0] * 24, "首粉确认": [0] * 30, "多周期首粉确认": [0] * 30, "信号评分": [70] * 30, "预备首粉评分": [70] * 30, "粉红准备度": [3] * 30, "多周期评分": [85] * 30, "多周期通过数": [5] * 30, "多周期缺失条件": [""] * 30, "量比5": [1.1] * 30, "三日涨幅%": [1.0] * 30}, index=dates)
    config = Config(dates[0], dates[-1], 20, 10.0, "normal", "all", 0, 0, "test", 420, 60, 12, 8, 20, Path("universe.csv"), None, Path("output"))
    trades, skipped = extract_trades(result, "000001", "测试", "test", config)
    assert len(trades) == 1 and trades[0]["信号类型"] == "多周期预粉M" and not skipped
    assert trades[0]["入场日期"] == dates[6].strftime("%Y-%m-%d")
    assert trades[0]["退出日期"] == dates[25].strftime("%Y-%m-%d")
    assert trades[0]["净收益%"] < trades[0]["毛收益%"]
    print("SELF_TEST_OK")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(self_test() if "--self-test" in sys.argv else run(parse_args()))
