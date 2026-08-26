#!/usr/bin/env python3
"""筹码模型独立全市场A-D扫描器。保留用户提供的筹码分布计算，不直接发送通知。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    universe: str
    shard_index: int
    shard_total: int
    output_dir: str
    history_days: int = 60
    dispose_k: float = 0.8
    provider_timeout_seconds: int = 35
    retries: int = 1
    request_pause_seconds: float = 0.2


class Chip:
    """原始蒸馏筹码分布模型；仅把它封装到可审计的全市场运行框架。"""

    def __init__(self, k_dispose: float = 0.8, decay_base: float = 1.0) -> None:
        self.k = k_dispose
        self.decay_base = decay_base
        self.prices: np.ndarray | None = None
        self.chip: np.ndarray | None = None
        self.total = 0.0

    def _grid(self, low: float, high: float, step: float) -> None:
        if self.prices is None:
            pad = max(0.8, (high - low) * 0.25)
            self.prices = np.round(np.arange(max(0.01, low - pad), high + pad + step, step), 2)
            self.chip = np.zeros(len(self.prices))
            return
        if low < self.prices[0] or high > self.prices[-1]:
            new_prices = np.round(np.arange(min(self.prices[0], low - 0.3), max(self.prices[-1], high + 0.3) + step, step), 2)
            new_chip = np.zeros(len(new_prices))
            indices = np.searchsorted(new_prices, self.prices)
            valid = (indices >= 0) & (indices < len(new_prices))
            new_chip[indices[valid]] = self.chip[valid]
            self.prices, self.chip = new_prices, new_chip

    def _triangle(self, low: float, high: float, average: float, volume: float) -> np.ndarray:
        assert self.chip is not None and self.prices is not None
        if high <= low or volume <= 0:
            return np.zeros_like(self.chip)
        mask = (self.prices >= low) & (self.prices <= high)
        if not np.any(mask):
            return np.zeros_like(self.chip)
        prices = self.prices[mask]
        height = 2.0 / (high - low)
        density = np.zeros_like(prices)
        left = prices < average
        if np.any(left) and average > low:
            density[left] = height / (average - low) * (prices[left] - low)
        right = ~left
        if np.any(right) and high > average:
            density[right] = height / (high - average) * (high - prices[right])
        total = density.sum()
        if total > 0:
            density = density / total * volume
        result = np.zeros_like(self.chip)
        result[mask] = density
        return result

    def _move_out(self, close: float, moved: float) -> None:
        if self.chip is None or self.prices is None or self.total <= 0 or moved <= 0:
            return
        if self.k <= 0:
            self.chip *= 1.0 - moved
            return
        pnl = (close - self.prices) / (self.prices + 1e-8)
        weights = (1.0 + self.k * np.maximum(pnl, 0.0)) * (self.chip > 0)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            self.chip *= 1.0 - moved
            return
        out = (weights / weight_sum) * (self.chip.sum() * moved)
        self.chip = np.maximum(self.chip - np.minimum(out, self.chip), 0.0)

    def update(self, high: float, low: float, average: float, volume: float, turnover: float, close: float) -> None:
        step = max(0.01, round(close * 0.001, 2)) if close > 0 else 0.01
        self._grid(low, high, step)
        moved = min(max(turnover, 0.0) * self.decay_base, 1.0)
        self._move_out(close, moved)
        assert self.chip is not None
        self.chip += self._triangle(low, high, average, volume) * moved
        if self.prices is not None and len(self.prices) > 3000:
            count = len(self.prices) // 2 * 2
            self.prices = np.round((self.prices[:count:2] + self.prices[1:count:2]) / 2, 2)
            self.chip = self.chip[:count:2] + self.chip[1:count:2]
        self.total = float(self.chip.sum())

    def winner(self, price: float) -> float:
        if self.chip is None or self.prices is None or self.total <= 0:
            return 0.0
        return float(self.chip[self.prices < price].sum() / self.total)

    def avg_cost(self) -> float:
        if self.chip is None or self.prices is None or self.total <= 0:
            return 0.0
        return float(np.average(self.prices, weights=self.chip))

    def concentration(self, center: float, pct: float = 0.10) -> float:
        if self.chip is None or self.prices is None or self.total <= 0 or center <= 0:
            return 0.0
        mask = (self.prices >= center * (1 - pct)) & (self.prices <= center * (1 + pct))
        return float(self.chip[mask].sum() / self.total)

    def peak(self) -> float:
        if self.chip is None or self.prices is None or self.total <= 0:
            return 0.0
        return float(self.prices[np.argmax(self.chip)])


def _child(conn: Any, func: Callable[[], Any]) -> None:
    try:
        conn.send(("ok", func()))
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", f"{type(exc).__name__}:{str(exc)[:240]}"))
    finally:
        conn.close()


def isolated_call(label: str, timeout_seconds: int, func: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return func()
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, func), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=3)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds}s")
        state, payload = parent.recv()
        process.join(timeout=3)
        if state != "ok":
            raise RuntimeError(f"provider_error:{label}:{payload}")
        return payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        parent.close()


def akshare_daily(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")


def baostock_daily(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        symbol = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
        query = bs.query_history_k_data_plus(symbol, "date,open,high,low,close,volume,turn", start_date=start, end_date=end, frequency="d", adjustflag="2")
        rows: list[list[str]] = []
        while query.error_code == "0" and query.next():
            rows.append(query.get_row_data())
        if query.error_code != "0":
            raise RuntimeError(f"baostock_daily:{query.error_code}:{query.error_msg}")
        return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "turn"])
    finally:
        bs.logout()


def normalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "akshare":
        rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "换手率": "turn"}
        frame = frame.rename(columns=rename)
    required = ["date", "open", "high", "low", "close", "volume", "turn"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"schema_missing:{','.join(missing)}")
    frame = frame[required].copy()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    frame = frame[(frame["high"] >= frame["low"]) & (frame["close"] > 0) & (frame["volume"] >= 0)]
    if source == "baostock":
        frame["turn"] = frame["turn"] / 100.0
    else:
        frame["turn"] = frame["turn"] / 100.0
    frame["average"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    return frame.reset_index(drop=True)


def fetch_daily(code: str, start: str, end: str, cfg: Config) -> tuple[pd.DataFrame | None, str | None, str | None]:
    errors: list[str] = []
    for source, loader in (("akshare", akshare_daily), ("baostock", baostock_daily)):
        for attempt in range(cfg.retries + 1):
            try:
                frame = normalize(isolated_call(f"{source}_daily:{code}", cfg.provider_timeout_seconds, lambda: loader(code, start, end)), source)
                if len(frame) >= cfg.history_days:
                    return frame.tail(cfg.history_days).reset_index(drop=True), source, None
                errors.append(f"{source}:attempt{attempt + 1}:insufficient_history:{len(frame)}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}:attempt{attempt + 1}:{type(exc).__name__}:{str(exc)[:160]}")
    return None, None, " | ".join(errors)


def scan_one(code: str, name: str, cfg: Config, start: str, end: str) -> tuple[dict[str, Any] | None, str | None]:
    frame, source, error = fetch_daily(code, start, end, cfg)
    if frame is None:
        return None, error
    chip = Chip(k_dispose=cfg.dispose_k)
    for _, row in frame.iterrows():
        chip.update(float(row["high"]), float(row["low"]), float(row["average"]), float(row["volume"]), float(row["turn"]), float(row["close"]))
    average_cost = chip.avg_cost()
    close = float(frame.iloc[-1]["close"])
    premium = (close - average_cost) / average_cost * 100 if average_cost > 0 else 0.0
    return {
        "code": code,
        "name": name,
        "signal_date": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "close": round(close, 2),
        "avg_cost": round(average_cost, 2),
        "winner_pct": round(chip.winner(close) * 100, 2),
        "concentration_10_pct": round(chip.concentration(average_cost, 0.10) * 100, 2) if average_cost > 0 else 0.0,
        "peak_price": round(chip.peak(), 2),
        "premium_pct": round(premium, 2),
        "history_days": len(frame),
        "daily_data_source": source,
    }, None


def load_universe(path: str) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "chip-model-universe/v1" or payload.get("strategy_id") != "chip-model" or payload.get("status") != "ready":
        raise ValueError("invalid_chip_model_universe")
    return list(payload.get("stocks", []))


def write_outputs(output_dir: Path, records: list[dict[str, Any]], errors: list[dict[str, str]], audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates = candidates.sort_values(["concentration_10_pct", "winner_pct", "premium_pct"], ascending=[False, False, True]).reset_index(drop=True)
        candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    candidates.to_csv(output_dir / "chip_model_candidates.csv", index=False, encoding="utf-8-sig")
    candidates.to_json(output_dir / "chip_model_candidates.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(errors, columns=["code", "name", "stage", "reason"]).to_csv(output_dir / "chip_model_errors.csv", index=False, encoding="utf-8-sig")
    (output_dir / "chip_model_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 独立全市场筹码模型分片", f"- 状态：`{audit['status']}`", f"- 分片：`{audit['shard_index']}/{audit['shard_total']}`", f"- 扫描：`{audit['processed_count']}`", f"- 有效记录：`{audit['candidate_count']}`", f"- 错误：`{audit['error_count']}`", "", "> 此输出是筹码分布统计研究，不构成买卖建议或收益承诺。"]
    if not candidates.empty:
        display = ["rank", "code", "name", "concentration_10_pct", "winner_pct", "premium_pct", "daily_data_source"]
        lines.extend(["", "## 前30条原始排序", candidates.head(30)[display].to_markdown(index=False)])
    (output_dir / "chip_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    chip = Chip(k_dispose=0.8)
    for index in range(60):
        close = 10.0 + index * 0.02
        chip.update(close * 1.02, close * 0.98, close, 1_000_000.0, 0.02, close)
    assert chip.total > 0 and 0 <= chip.winner(11.0) <= 1 and chip.avg_cost() > 0
    sample = [{"code": "000001"}, {"code": "000002"}, {"code": "600000"}, {"code": "600001"}]
    assert [item["code"] for item in sample[1::2]] == ["000002", "600001"]
    print("chip-model scanner self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="独立全市场筹码模型A-D扫描器")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--universe", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--dispose-k", type=float, default=0.8)
    parser.add_argument("--provider-timeout-seconds", type=int, default=35)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--request-pause-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.universe or args.shard_total < 1 or not 0 <= args.shard_index < args.shard_total:
        raise SystemExit("必须提供有效共同股票池与分片参数")
    cfg = Config(args.universe, args.shard_index, args.shard_total, args.output_dir, args.history_days, args.dispose_k, args.provider_timeout_seconds, args.retries, args.request_pause_seconds)
    universe = load_universe(cfg.universe)
    shard = universe[cfg.shard_index :: cfg.shard_total]
    end = date.today()
    start = end - timedelta(days=max(cfg.history_days * 4, 300))
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for position, item in enumerate(shard, start=1):
        code, name = str(item["code"]), str(item["name"])
        record, error = scan_one(code, name, cfg, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        if record is None:
            errors.append({"code": code, "name": name, "stage": "daily", "reason": error or "unknown"})
        else:
            records.append(record)
            source = str(record["daily_data_source"])
            source_counts[source] = source_counts.get(source, 0) + 1
        if position < len(shard):
            time.sleep(cfg.request_pause_seconds)
    status = "ready" if records else "degraded"
    audit = {"schema_version": "chip-model-scan/v1", "strategy_id": "chip-model", "status": status, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "shard_index": cfg.shard_index, "shard_total": cfg.shard_total, "universe_count": len(universe), "processed_count": len(shard), "candidate_count": len(records), "error_count": len(errors), "daily_data_source_counts": source_counts, "config": asdict(cfg), "research_disclaimer": "筹码分布为日线成交与换手率的统计代理，不等同主力资金、逐笔成交或收益预测，不构成投资建议。"}
    write_outputs(Path(cfg.output_dir), records, errors, audit)
    print(f"chip-model shard={cfg.shard_index} status={status} records={len(records)} errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
