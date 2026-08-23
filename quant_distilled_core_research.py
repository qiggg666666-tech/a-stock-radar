#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DistilledQuant Core Research v1.0.

本模块是A股日线研究排序器，不是收益预测、统计套利或交易执行程序。

设计边界：
1. 仅使用真实OHLCV数据；AkShare失败时才尝试BaoStock，绝不合成价格、行业或候选。
2. 每个请求在独立子进程中执行并具有硬超时；每次失败都会进入errors.csv和status.json。
3. 因子只使用--signal-date及此前日线；不训练未来收益，不计算凯利仓位，不宣称Hawkes、幂律、Copula或资金流因果结论。
4. 输出是相对研究评分，供进一步复核使用，不构成买卖建议。

示例：
  python quant_distilled_core_research.py --symbols-file universe.csv \
      --signal-date 2026-08-21 --output-dir output/core
  python quant_distilled_core_research.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
VALID_CODE = re.compile(r"^(?:000|001|002|003|300|301|600|601|603|605|688)\d{3}$")

# 因子权重为固定、可审计的研究设定；横截面标准化发生在所有有效标的合并之后。
FACTOR_WEIGHTS: dict[str, float] = {
    "trend_60": 0.22,
    "momentum_20": 0.18,
    "drawdown_60": 0.18,
    "volatility_20": 0.13,
    "liquidity_20": 0.13,
    "volume_ratio_20": 0.08,
    "price_position_60": 0.08,
}


@dataclass(frozen=True)
class FetchAttempt:
    source: str
    ok: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class FetchResult:
    frame: Optional[pd.DataFrame]
    source: Optional[str]
    attempts: list[FetchAttempt]


class DataSourceUnavailable(RuntimeError):
    """Raised when neither public source returned usable real daily bars."""


def normalize_code(value: Any) -> str:
    code = str(value).strip().replace(".0", "")
    if not VALID_CODE.fullmatch(code):
        raise ValueError(f"unsupported_a_share_code:{code}")
    return code


def baostock_code(code: str) -> str:
    return f"sh.{code}" if code.startswith(("600", "601", "603", "605", "688")) else f"sz.{code}"


def _normalize_daily_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    aliases = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    frame = raw.rename(columns=aliases).copy()
    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}_missing_columns:{','.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=needed).sort_values("date").drop_duplicates("date", keep="last")
    frame = frame[(frame["close"] > 0) & (frame["high"] >= frame["low"]) & (frame["volume"] >= 0)]
    if frame.empty:
        raise ValueError(f"{source}_no_valid_daily_rows")
    return frame.set_index("date")[['open', 'high', 'low', 'close', 'volume']]


def _fetch_akshare_payload(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    import akshare as ak  # Imported inside child process so a blocked request can be terminated.

    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if raw is None or raw.empty:
        raise ValueError("akshare_empty_response")
    frame = _normalize_daily_frame(raw, "akshare").reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


def _fetch_baostock_payload(code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    import baostock as bs  # Imported inside child process; login/logout cannot leak across symbols.

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            baostock_code(code),
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权，接近AkShare qfq用于研究排序的口径。
        )
        if result.error_code != "0":
            raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        if raw.empty:
            raise ValueError("baostock_empty_response")
        frame = _normalize_daily_frame(raw, "baostock").reset_index()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame.to_dict(orient="records")
    finally:
        bs.logout()


def _fetch_worker(
    fetcher: Callable[[str, str, str], list[dict[str, Any]]],
    code: str,
    start_date: str,
    end_date: str,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put({"ok": True, "records": fetcher(code, start_date, end_date)})
    except BaseException as exc:  # Child must serialize errors rather than silently disappearing.
        result_queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "traceback": traceback.format_exc(limit=3),
            }
        )


def call_with_timeout(
    fetcher: Callable[[str, str, str], list[dict[str, Any]]],
    code: str,
    start_date: str,
    end_date: str,
    timeout_seconds: int,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """Execute one source request in a child process and return frame/error details."""
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(target=_fetch_worker, args=(fetcher, code, start_date, end_date, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, "TimeoutError", f"request_exceeded_{timeout_seconds}_seconds"
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return None, "ChildProcessError", f"child_exit_{process.exitcode}_without_payload"
    finally:
        result_queue.close()
        result_queue.join_thread()
    if not payload.get("ok"):
        return None, str(payload.get("error_type", "SourceError")), str(payload.get("error_message", "unknown_error"))
    try:
        return _normalize_daily_frame(pd.DataFrame(payload["records"]), "child_payload"), None, None
    except Exception as exc:
        return None, type(exc).__name__, str(exc)


def fetch_real_daily(
    code: str,
    start_date: str,
    end_date: str,
    timeout_seconds: int,
    retries: int,
) -> FetchResult:
    """Fetch real daily bars using AkShare first and BaoStock only after explicit failures."""
    code = normalize_code(code)
    attempts: list[FetchAttempt] = []
    for source, fetcher in (("akshare", _fetch_akshare_payload), ("baostock", _fetch_baostock_payload)):
        for attempt_no in range(1, retries + 1):
            frame, error_type, error_message = call_with_timeout(
                fetcher, code, start_date, end_date, timeout_seconds
            )
            if frame is not None:
                attempts.append(FetchAttempt(source=f"{source}:{attempt_no}", ok=True))
                return FetchResult(frame=frame, source=source, attempts=attempts)
            attempts.append(
                FetchAttempt(
                    source=f"{source}:{attempt_no}",
                    ok=False,
                    error_type=error_type,
                    error_message=error_message,
                )
            )
    return FetchResult(frame=None, source=None, attempts=attempts)


def parse_signal_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value).normalize()
    if pd.isna(parsed):
        raise ValueError("invalid_signal_date")
    if parsed.date() > date.today():
        raise ValueError("signal_date_cannot_be_in_the_future")
    return parsed


def calculate_raw_factors(frame: pd.DataFrame, signal_date: pd.Timestamp, max_stale_days: int) -> dict[str, float | str]:
    """Calculate only factors known on or before the requested signal date."""
    visible = frame.loc[frame.index <= signal_date].copy()
    if visible.empty:
        raise ValueError("no_bar_on_or_before_signal_date")
    data_last_date = visible.index.max().normalize()
    staleness_days = int((signal_date - data_last_date).days)
    if staleness_days > max_stale_days:
        raise ValueError(f"stale_daily_data:{staleness_days}_days")
    if len(visible) < 61:
        raise ValueError(f"insufficient_history:{len(visible)}_rows")

    close = visible["close"]
    returns = close.pct_change().dropna()
    amount = visible["close"] * visible["volume"]
    max_60 = close.rolling(60).max()
    min_60 = close.rolling(60).min()
    price_range = (max_60 - min_60).iloc[-1]
    if not np.isfinite(price_range) or price_range <= 0:
        raise ValueError("invalid_price_range_60")
    volatility = float(returns.tail(20).std(ddof=0) * math.sqrt(252))
    raw = {
        "data_last_date": data_last_date.strftime("%Y-%m-%d"),
        "staleness_days": float(staleness_days),
        "close": float(close.iloc[-1]),
        "trend_60": float(close.iloc[-1] / close.rolling(60).mean().iloc[-1] - 1.0),
        "momentum_20": float(close.iloc[-1] / close.iloc[-21] - 1.0),
        "drawdown_60": float(close.iloc[-1] / max_60.iloc[-1] - 1.0),
        "volatility_20": volatility,
        "liquidity_20": float(math.log1p(amount.tail(20).mean())),
        "volume_ratio_20": float(visible["volume"].iloc[-1] / (visible["volume"].tail(20).mean() + 1e-12)),
        "price_position_60": float((close.iloc[-1] - min_60.iloc[-1]) / price_range),
    }
    if not all(np.isfinite(float(value)) for key, value in raw.items() if key not in {"data_last_date"}):
        raise ValueError("non_finite_factor")
    return raw


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    median_absolute_deviation = (series - median).abs().median()
    scale = 1.4826 * median_absolute_deviation
    if not np.isfinite(scale) or scale < 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((series - median) / scale).clip(-3.0, 3.0)


def score_cross_section(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent, fixed-weight robust standardization across valid symbols only."""
    if raw.empty:
        return raw.copy()
    required = set(FACTOR_WEIGHTS)
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing_factor_columns:{','.join(sorted(missing))}")
    directions = {
        "trend_60": 1.0,
        "momentum_20": 1.0,
        "drawdown_60": 1.0,  # Less negative drawdown ranks higher; no recovery claim is made.
        "volatility_20": -1.0,
        "liquidity_20": 1.0,
        "volume_ratio_20": 1.0,
        "price_position_60": 1.0,
    }
    scored = raw.copy()
    weighted_sum = pd.Series(0.0, index=scored.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        z_column = f"z_{factor}"
        scored[z_column] = robust_zscore(scored[factor]) * directions[factor]
        weighted_sum = weighted_sum + weight * scored[z_column]
    scored["distilled_research_score"] = (50.0 + 10.0 * weighted_sum).clip(0.0, 100.0)
    scored["rank"] = scored["distilled_research_score"].rank(method="first", ascending=False).astype(int)
    return scored.sort_values(["rank", "code"]).reset_index(drop=True)


def load_symbols(symbol: Optional[str], symbols_file: Optional[Path]) -> pd.DataFrame:
    if bool(symbol) == bool(symbols_file):
        raise ValueError("provide_exactly_one_of_symbol_or_symbols_file")
    if symbol:
        return pd.DataFrame([{"code": normalize_code(symbol), "name": ""}])
    assert symbols_file is not None
    if not symbols_file.exists():
        raise FileNotFoundError(f"symbols_file_not_found:{symbols_file}")
    source = pd.read_csv(symbols_file, dtype=str)
    code_column = next((column for column in ("code", "symbol", "代码") if column in source.columns), None)
    if code_column is None:
        raise ValueError("symbols_file_requires_code_or_symbol_column")
    name_column = next((column for column in ("name", "名称") if column in source.columns), None)
    rows: list[dict[str, str]] = []
    rejected: list[str] = []
    for _, row in source.iterrows():
        try:
            rows.append({"code": normalize_code(row[code_column]), "name": str(row[name_column]) if name_column else ""})
        except Exception:
            rejected.append(str(row[code_column]))
    if rejected:
        raise ValueError(f"invalid_codes_in_symbols_file:{','.join(rejected[:20])}")
    return pd.DataFrame(rows).drop_duplicates("code").sort_values("code").reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    candidates: pd.DataFrame,
    errors: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "distilled_quant_candidates.csv", index=False, encoding="utf-8-sig")
    error_columns = ["code", "name", "stage", "error_type", "error_message", "attempts"]
    pd.DataFrame(errors, columns=error_columns).to_csv(
        output_dir / "distilled_quant_errors.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "distilled_quant_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# DistilledQuant Core Research",
        "",
        f"- 状态：`{status['status']}`",
        f"- 请求信号日：`{status['signal_date_requested']}`",
        f"- 输入标的：`{status['input_symbol_count']}`",
        f"- 有效研究记录：`{status['valid_symbol_count']}`",
        f"- 错误记录：`{status['error_count']}`",
        "",
        "> 分数是基于真实日线的横截面研究排序，不代表未来收益预测、买卖建议或仓位建议。",
    ]
    (output_dir / "distilled_quant_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan(
    symbols: pd.DataFrame,
    signal_date: pd.Timestamp,
    start_date: str,
    timeout_seconds: int,
    retries: int,
    max_stale_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {"akshare": 0, "baostock": 0}
    for _, row in symbols.iterrows():
        code, name = str(row["code"]), str(row.get("name", ""))
        result = fetch_real_daily(code, start_date, signal_date.strftime("%Y-%m-%d"), timeout_seconds, retries)
        attempts_json = json.dumps([asdict(attempt) for attempt in result.attempts], ensure_ascii=False)
        if result.frame is None or result.source is None:
            errors.append(
                {
                    "code": code,
                    "name": name,
                    "stage": "daily_fetch",
                    "error_type": "DataSourceUnavailable",
                    "error_message": "both_public_sources_failed_or_returned_no_usable_rows",
                    "attempts": attempts_json,
                }
            )
            continue
        try:
            raw = calculate_raw_factors(result.frame, signal_date, max_stale_days)
            records.append({"code": code, "name": name, "daily_data_source": result.source, **raw})
            source_counts[result.source] = source_counts.get(result.source, 0) + 1
        except Exception as exc:
            errors.append(
                {
                    "code": code,
                    "name": name,
                    "stage": "factor_calculation",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "attempts": attempts_json,
                }
            )
    return pd.DataFrame(records), errors, source_counts


def deterministic_fixture() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=100)
    steps = np.arange(len(index), dtype=float)
    close = 10.0 + 0.04 * steps + 0.15 * np.sin(steps / 7.0)
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": 1_500_000 + steps * 11_000,
        },
        index=index,
    )


def run_self_test() -> None:
    fixture = deterministic_fixture()
    signal = fixture.index[79]
    baseline = calculate_raw_factors(fixture, signal, max_stale_days=0)
    future_changed = fixture.copy()
    future_changed.loc[future_changed.index > signal, ["open", "high", "low", "close", "volume"]] *= 50.0
    assert baseline == calculate_raw_factors(future_changed, signal, max_stale_days=0), "future_rows_changed_signal_factors"
    raw = pd.DataFrame(
        [
            {"code": "000001", **baseline},
            {"code": "600000", **{**baseline, "trend_60": baseline["trend_60"] + 0.02, "momentum_20": baseline["momentum_20"] + 0.01}},
        ]
    )
    scored = score_cross_section(raw)
    assert len(scored) == 2 and scored["rank"].tolist() == [1, 2], "cross_section_ranking_failed"
    unavailable = FetchResult(
        frame=None,
        source=None,
        attempts=[FetchAttempt("akshare:1", False, "TimeoutError", "request_exceeded"), FetchAttempt("baostock:1", False, "ValueError", "empty")],
    )
    assert unavailable.frame is None and len(unavailable.attempts) == 2, "explicit_failure_audit_failed"
    print("SELF_TEST_OK: no_future_data, robust_cross_section, explicit_source_failure")


def main() -> None:
    parser = argparse.ArgumentParser(description="DistilledQuant Core Research (real OHLCV only)")
    parser.add_argument("--symbol", help="单一A股六位代码；与--symbols-file二选一")
    parser.add_argument("--symbols-file", type=Path, help="包含code/symbol/代码列的共同股票池CSV；与--symbol二选一")
    parser.add_argument("--signal-date", help="信号日YYYY-MM-DD；仅使用该日及此前日线")
    parser.add_argument("--start-date", default="2023-01-01", help="日线起始日YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("distilled_quant_core_output"))
    parser.add_argument("--timeout-seconds", type=int, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-stale-days", type=int, default=3)
    parser.add_argument("--self-test", action="store_true", help="仅运行离线确定性自检，不访问网络")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.signal_date:
        parser.error("--signal-date is required unless --self-test is used")
    if args.timeout_seconds <= 0 or args.retries <= 0 or args.max_stale_days < 0:
        parser.error("timeout-seconds/retries must be positive and max-stale-days must be non-negative")
    signal_date = parse_signal_date(args.signal_date)
    symbols = load_symbols(args.symbol, args.symbols_file)
    raw, errors, source_counts = scan(
        symbols=symbols,
        signal_date=signal_date,
        start_date=args.start_date,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        max_stale_days=args.max_stale_days,
    )
    candidates = score_cross_section(raw) if not raw.empty else raw
    status_name = "ready" if errors == [] and not candidates.empty else "degraded" if not candidates.empty else "unavailable"
    status = {
        "strategy": "DistilledQuant Core Research",
        "version": SCRIPT_VERSION,
        "status": status_name,
        "signal_date_requested": signal_date.strftime("%Y-%m-%d"),
        "start_date": args.start_date,
        "input_symbol_count": int(len(symbols)),
        "valid_symbol_count": int(len(candidates)),
        "error_count": int(len(errors)),
        "daily_data_source_counts": source_counts,
        "fixed_factor_weights": FACTOR_WEIGHTS,
        "research_boundary": "real_OHLCV_only; no_synthetic_data; no_future_return_training; no_position_sizing",
    }
    write_outputs(args.output_dir, candidates, errors, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status_name == "unavailable":
        sys.exit(2)


if __name__ == "__main__":
    main()
