#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吃掉主力：单标的趋势与相对强弱研究脚本（优化单文件版）。

本版保留原脚本的“中线趋势 + 短均线方向 + 操纵/趋势线”框架，但将其明确为
价格行为与相对强弱的技术研究指标；它不观测主力账户、持仓或真实资金意图，不能据此断言“主力吸筹”。

关键改进：
* 修复原 EMA 行中“反斜杠加波浪号”造成的 Python 语法错误；所有指标在预热期保留 NaN，不把缺失值静默前推。
* 对 OHLC、日期、重复数据和基准日期进行校验；股票与指数按共同交易日精确对齐。
* 买点改为“操纵线上穿趋势线”事件，并使用冷却期抑制连续重复信号，而非仅依赖5日滚动计数。
* AkShare 主数据源、BaoStock 单股回退；两个请求使用可终止子进程硬超时。
* 输出 CSV、JSON、Markdown 与终端摘要，说明信号条件、数据质量和当前状态。

示例：
  python chi_diao_zhu_li_optimized.py 300650 normal
  python chi_diao_zhu_li_optimized.py 300650 sensitive --days 900
  python chi_diao_zhu_li_optimized.py 600519 stable --no-index
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("chi_diao_zhu_li")
CN_TZ = "Asia/Shanghai"
EPS = 1e-12


@dataclass(frozen=True)
class Parameters:
    ema_control: int
    ema_trend: int
    trend_weight: float
    cooldown_days: int


PRESETS: dict[str, Parameters] = {
    "sensitive": Parameters(ema_control=9, ema_trend=2, trend_weight=0.50, cooldown_days=3),
    "normal": Parameters(ema_control=13, ema_trend=3, trend_weight=0.618, cooldown_days=5),
    "stable": Parameters(ema_control=18, ema_trend=5, trend_weight=0.70, cooldown_days=8),
}


@dataclass(frozen=True)
class RuntimeConfig:
    code: str
    mode: str
    index_code: str
    fetch_index: bool
    strict_index: bool
    days: int
    ak_timeout_seconds: int
    bao_timeout_seconds: int
    output_dir: Path
    params: Parameters


class DataSourceError(RuntimeError):
    """外部数据源返回错误、超时或数据质量不满足计算要求。"""


def get_context() -> Any:
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def recursive_ema(values: pd.Series, period: int) -> pd.Series:
    """与常用图表软件一致的 adjust=False 递推 EMA；预热缺失值保持缺失。"""
    if period <= 0:
        raise ValueError("EMA 周期必须为正数")
    data = pd.to_numeric(values, errors="coerce").astype("float64")
    output = np.full(len(data), np.nan, dtype="float64")
    valid = np.flatnonzero(np.isfinite(data.to_numpy()))
    if valid.size == 0:
        return pd.Series(output, index=data.index, name=values.name)
    alpha = 2.0 / (period + 1.0)
    first = int(valid[0])
    output[first] = float(data.iloc[first])
    for index in range(first + 1, len(data)):
        current = data.iloc[index]
        output[index] = output[index - 1] if not np.isfinite(current) else alpha * current + (1.0 - alpha) * output[index - 1]
    return pd.Series(output, index=data.index, name=values.name)


def chinese_sma(values: pd.Series, period: int, weight: int = 1) -> pd.Series:
    """中国公式语言常用 SMA：Y=(M*X+(N-M)*Y')/N。"""
    if not (period > 0 and 0 < weight <= period):
        raise ValueError("SMA 参数非法")
    return pd.to_numeric(values, errors="coerce").astype("float64").ewm(alpha=weight / period, adjust=False).mean()


def enforce_cooldown(events: pd.Series, cooldown_days: int) -> pd.Series:
    """仅保留相隔至少 cooldown_days 个交易日的首次事件。"""
    accepted = np.zeros(len(events), dtype=bool)
    last_signal = -cooldown_days - 1
    for position, is_event in enumerate(events.fillna(False).astype(bool).to_numpy()):
        if is_event and position - last_signal > cooldown_days:
            accepted[position] = True
            last_signal = position
    return pd.Series(accepted, index=events.index)


def normalize_ohlc(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {
        "日期": "date", "date": "date", "开盘": "open", "open": "open", "最高": "high", "high": "high",
        "最低": "low", "low": "low", "收盘": "close", "close": "close", "成交量": "volume", "volume": "volume",
    }
    frame = raw.rename(columns=aliases).copy()
    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataSourceError(f"{source} 缺少字段：{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:] + (["volume"] if "volume" in frame.columns else []):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates("date", keep="last").sort_values("date")
    valid = (frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)
    valid &= frame["high"] >= frame[["open", "close", "low"]].max(axis=1)
    valid &= frame["low"] <= frame[["open", "close", "high"]].min(axis=1)
    frame = frame.loc[valid, [column for column in ["date", "open", "high", "low", "close", "volume"] if column in frame.columns]]
    if len(frame) < 130:
        raise DataSourceError(f"{source} 有效日线不足 130 根：{len(frame)}")
    return frame.set_index("date").sort_index()


def _fetch_worker(result_queue: Any, kind: str, code: str, start: str, end: str) -> None:
    try:
        import akshare as ak
        if kind == "stock_ak":
            raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        elif kind == "stock_ak_raw":
            raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
        elif kind == "index_ak":
            raw = ak.stock_zh_index_daily(symbol=code)
            raw = raw[(pd.to_datetime(raw["date"], errors="coerce") >= pd.Timestamp(start)) & (pd.to_datetime(raw["date"], errors="coerce") <= pd.Timestamp(end))]
        elif kind == "stock_bao":
            import baostock as bs
            market = "sh" if code.startswith(("6", "9")) else "sz"
            login = bs.login()
            if getattr(login, "error_code", "1") != "0":
                raise DataSourceError(f"BaoStock登录失败：{getattr(login, 'error_msg', '')}")
            try:
                query = bs.query_history_k_data_plus(f"{market}.{code}", "date,open,high,low,close,volume,tradestatus", start_date=f"{start[:4]}-{start[4:6]}-{start[6:]}", end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}", frequency="d", adjustflag="2")
                if getattr(query, "error_code", "1") != "0":
                    raise DataSourceError(f"BaoStock日线失败：{getattr(query, 'error_msg', '')}")
                rows: list[list[str]] = []
                while query.next():
                    rows.append(query.get_row_data())
                raw = pd.DataFrame(rows, columns=query.fields)
                if "tradestatus" in raw.columns:
                    raw = raw[raw["tradestatus"].astype(str) == "1"]
            finally:
                try:
                    bs.logout()
                except Exception:
                    pass
        else:
            raise ValueError(f"未知数据请求：{kind}")
        if raw is None or raw.empty:
            raise DataSourceError(f"{kind} 返回空数据")
        result_queue.put({"ok": True, "data": raw})
    except Exception as error:
        result_queue.put({"ok": False, "reason": f"{type(error).__name__}: {str(error)[:240]}"})


def fetch_with_hard_timeout(kind: str, code: str, start: date, end: date, timeout_seconds: int) -> pd.DataFrame:
    """用可终止子进程包裹外部调用；超过时间即终止，避免线程池伪超时。"""
    ctx = get_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_fetch_worker, args=(result_queue, kind, code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    process.start()
    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as error:
        if process.is_alive():
            process.terminate()
        process.join(5)
        result_queue.close()
        raise DataSourceError(f"{kind} 超过 {timeout_seconds} 秒") from error
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    result_queue.close()
    if not payload.get("ok"):
        raise DataSourceError(str(payload.get("reason", f"{kind} 未知错误")))
    return payload["data"]


def fetch_data(config: RuntimeConfig) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, str]]:
    end = date.today()
    start = end - timedelta(days=config.days)
    quality: dict[str, str] = {}
    try:
        stock = normalize_ohlc(fetch_with_hard_timeout("stock_ak", config.code, start, end, config.ak_timeout_seconds), "AkShare个股")
        quality["stock_source"] = "akshare"
    except DataSourceError as ak_error:
        LOGGER.warning("AkShare个股失败，切换BaoStock：%s", ak_error)
        stock = normalize_ohlc(fetch_with_hard_timeout("stock_bao", config.code, start, end, config.bao_timeout_seconds), "BaoStock个股")
        quality["stock_source"] = "baostock"
        quality["stock_fallback_reason"] = str(ak_error)
    benchmark: Optional[pd.DataFrame] = None
    if config.fetch_index:
        try:
            benchmark = normalize_ohlc(fetch_with_hard_timeout("index_ak", config.index_code, start, end, config.ak_timeout_seconds), "AkShare指数")
            quality["index_source"] = "akshare"
        except DataSourceError as index_error:
            quality["index_source"] = "unavailable"
            quality["index_reason"] = str(index_error)
            if config.strict_index:
                raise
            LOGGER.warning("基准指数不可用；本次不计算相对强弱：%s", index_error)
    return stock, benchmark, quality


def compute_indicator(stock: pd.DataFrame, benchmark: Optional[pd.DataFrame], params: Parameters) -> pd.DataFrame:
    """计算趋势、相对强弱和一次性买点；输入均为已校验且按日期排序的日线。"""
    frame = stock.copy()
    if benchmark is not None:
        common = frame.index.intersection(benchmark.index)
        if len(common) < 130:
            raise DataSourceError(f"股票与指数共同交易日不足 130 根：{len(common)}")
        frame = frame.loc[common].copy()
        benchmark_close = benchmark.loc[common, "close"].astype(float)
    else:
        benchmark_close = None
    if len(frame) < 130:
        raise DataSourceError("计算窗口不足 130 根")

    close, high, low = (frame[column].astype(float) for column in ("close", "high", "low"))
    ma5, ma10 = close.rolling(5, min_periods=5).mean(), close.rolling(10, min_periods=10).mean()
    low125, high125 = low.rolling(125, min_periods=125).min(), high.rolling(125, min_periods=125).max()
    position125 = (close - low125) / (high125 - low125).replace(0, np.nan) * 100
    mid_fast = chinese_sma(position125, 72, 1)
    mid_slow = chinese_sma(mid_fast, 34, 1)
    medium_trend = 3.0 * mid_fast - 2.0 * mid_slow

    typical = (2.0 * close + high + low) / 4.0
    low27, high27 = low.rolling(27, min_periods=27).min(), high.rolling(27, min_periods=27).max()
    control = recursive_ema((typical - low27) / (high27 - low27).replace(0, np.nan) * 100 - 50.0, params.ema_control)
    trend_input = params.trend_weight * control.shift(1) + (1.0 - params.trend_weight) * control
    trend_input.iloc[0] = control.iloc[0]
    trend = recursive_ema(trend_input, params.ema_trend)

    ma5_up = ma5.gt(ma5.shift(1))
    ma10_up = ma10.gt(ma10.shift(1))
    medium_up = medium_trend.gt(medium_trend.shift(1))
    cross_up = control.gt(trend) & control.shift(1).le(trend.shift(1))
    negative_zone = trend.lt(0)

    if benchmark_close is not None:
        stock_curve = (1.0 + close.pct_change(fill_method=None).fillna(0.0)).cumprod()
        index_curve = (1.0 + benchmark_close.pct_change(fill_method=None).fillna(0.0)).cumprod()
        relative_strength = stock_curve / index_curve - 1.0
        relative_leader = relative_strength.ge(relative_strength.rolling(20, min_periods=20).mean())
    else:
        relative_strength = pd.Series(np.nan, index=frame.index)
        relative_leader = pd.Series(True, index=frame.index)

    bull_state = ma5_up & ma10_up & medium_up & relative_leader
    raw_buy = cross_up & ma5_up & ma10_up & medium_up & negative_zone & relative_leader
    buy_signal = enforce_cooldown(raw_buy, params.cooldown_days)
    score = (
        ma5_up.astype(int) * 18 + ma10_up.astype(int) * 18 + medium_up.astype(int) * 24
        + cross_up.astype(int) * 20 + negative_zone.astype(int) * 10 + relative_leader.astype(int) * 10
    )
    frame["MA5"] = ma5
    frame["MA10"] = ma10
    frame["中线趋势"] = medium_trend
    frame["操纵"] = control
    frame["趋势"] = trend
    frame["相对强弱%"] = relative_strength * 100.0
    frame["股牛股"] = bull_state.astype(int) * 4
    frame["粉色柱"] = np.where(bull_state, control + 6.0, np.nan)
    frame["操纵上穿"] = cross_up.astype(int)
    frame["买进"] = buy_signal.astype(int)
    frame["信号评分"] = score.astype(int)
    frame["均五升"] = ma5_up.astype(int)
    frame["均十升"] = ma10_up.astype(int)
    frame["中线趋势升"] = medium_up.astype(int)
    frame["领涨"] = relative_leader.astype(int) if benchmark_close is not None else np.nan
    frame["数据预热完成"] = (np.arange(len(frame)) >= 124).astype(int)
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    """不依赖tabulate，将DataFrame转为适合报告的基础Markdown表格。"""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                text = f"{value:.3f}" if np.isfinite(value) else ""
            elif isinstance(value, pd.Timestamp):
                text = value.strftime("%Y-%m-%d")
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(result: pd.DataFrame, quality: dict[str, str], config: RuntimeConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz=CN_TZ).strftime("%Y%m%d_%H%M%S")
    stem = f"chi_diao_zhu_li_{config.code}_{config.mode}_{stamp}"
    csv_path = config.output_dir / f"{stem}.csv"
    json_path = config.output_dir / f"{stem}.json"
    markdown_path = config.output_dir / f"{stem}.md"
    export = result.reset_index().rename(columns={"index": "date"})
    export.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    latest = export.iloc[-1].replace({np.nan: None}).to_dict()
    buy_dates = export.loc[export["买进"] == 1, "date"].dt.strftime("%Y-%m-%d").tail(10).tolist()
    payload = {"generated_at": pd.Timestamp.now(tz=CN_TZ).isoformat(), "code": config.code, "mode": config.mode, "parameters": asdict(config.params), "data_quality": quality, "latest": latest, "recent_buy_dates": buy_dates, "rows": len(export)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    state = "买点触发" if int(latest.get("买进") or 0) else ("多项趋势条件成立" if int(latest.get("股牛股") or 0) else "无买点")
    relative_text = "未计算（基准不可用或已关闭）" if pd.isna(latest.get("相对强弱%")) else f"{float(latest['相对强弱%']):.2f}%"
    recent = export.tail(15)[["date", "close", "股牛股", "操纵", "趋势", "买进", "信号评分"]].copy()
    for column in ["close", "操纵", "趋势"]:
        recent[column] = recent[column].round(3)
    markdown_path.write_text(
        f"# 吃掉主力技术研究报告\n\n"
        f"- 标的：`{config.code}`\n- 模式：`{config.mode}`\n- 最新交易日：`{pd.Timestamp(latest['date']):%Y-%m-%d}`\n- 当前状态：**{state}**\n- 信号评分：`{int(latest.get('信号评分') or 0)}/100`\n- 相对强弱：`{relative_text}`\n- 最近10次去重买点：`{', '.join(buy_dates) if buy_dates else '无'}`\n\n"
        f"> 本指标根据价格、均线和相对表现生成技术研究信号，不代表可观测的主力账户行为，也不构成投资建议。\n\n"
        f"## 最近15日\n\n{markdown_table(recent)}\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def run_self_test() -> int:
    dates = pd.date_range("2023-01-02", periods=220, freq="B")
    close = pd.Series(np.linspace(10.0, 18.0, len(dates)), index=dates)
    stock = pd.DataFrame({"open": close * 0.995, "high": close * 1.015, "low": close * 0.985, "close": close, "volume": 1_000_000}, index=dates)
    benchmark = pd.DataFrame({"open": np.linspace(3000, 3200, len(dates)), "high": np.linspace(3020, 3220, len(dates)), "low": np.linspace(2980, 3180, len(dates)), "close": np.linspace(3000, 3200, len(dates)), "volume": 1}, index=dates)
    result = compute_indicator(stock, benchmark, PRESETS["normal"])
    required = {"股牛股", "操纵", "趋势", "买进", "信号评分", "相对强弱%"}
    assert required.issubset(result.columns)
    buy_positions = np.flatnonzero(result["买进"].to_numpy() == 1)
    assert all(np.diff(buy_positions) > PRESETS["normal"].cooldown_days)
    assert result.index.equals(stock.index)
    print("SELF_TEST_OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单标的趋势与相对强弱技术研究脚本")
    parser.add_argument("code", nargs="?", default="300650", help="六位A股代码，例如 300650")
    parser.add_argument("mode", nargs="?", choices=sorted(PRESETS), default="normal", help="sensitive / normal / stable")
    parser.add_argument("--index", default="sh000001", help="AkShare指数代码，默认 sh000001")
    parser.add_argument("--no-index", action="store_true", help="不计算相对强弱；信号报告会明确标注")
    parser.add_argument("--strict-index", action="store_true", help="基准不可用时终止，而非降级运行")
    parser.add_argument("--days", type=int, default=900, help="下载自然日数，至少建议 700")
    parser.add_argument("--ak-timeout", type=int, default=12, help="AkShare硬超时秒数")
    parser.add_argument("--bao-timeout", type=int, default=8, help="BaoStock硬超时秒数")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录")
    parser.add_argument("--self-test", action="store_true", help="不联网的离线计算自检")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.self_test:
        return run_self_test()
    code = "".join(character for character in args.code if character.isdigit())
    if len(code) != 6:
        raise SystemExit("股票代码必须是6位数字")
    config = RuntimeConfig(code=code, mode=args.mode, index_code=args.index, fetch_index=not args.no_index, strict_index=args.strict_index, days=max(700, args.days), ak_timeout_seconds=max(1, args.ak_timeout), bao_timeout_seconds=max(1, args.bao_timeout), output_dir=args.output_dir, params=PRESETS[args.mode])
    try:
        stock, benchmark, quality = fetch_data(config)
        result = compute_indicator(stock, benchmark, config.params)
        paths = write_outputs(result, quality, config)
    except DataSourceError as error:
        LOGGER.error("数据处理失败：%s", error)
        return 2
    latest = result.iloc[-1]
    print("\n===== 吃掉主力优化版 · 最新状态 =====")
    print(f"标的 {config.code} | 模式 {config.mode} | 数据源 {quality.get('stock_source')} | 基准 {quality.get('index_source', 'disabled')}")
    print(result[["close", "股牛股", "操纵", "趋势", "相对强弱%", "买进", "信号评分"]].tail(15).round(3).to_string())
    print(f"\n最新评分：{int(latest['信号评分'])}/100；买进信号：{'是' if int(latest['买进']) else '否'}")
    for name, path in paths.items():
        print(f"{name.upper()}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
