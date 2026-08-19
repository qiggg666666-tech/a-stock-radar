#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""“吃掉主力”小流通市值全市场扫描器。

候选池仅基于**运行当日**可取得的流通市值和成交额快照：
  * 流通市值：20–200亿元；
  * 非ST、非退市整理标的、代码为沪深A股；
  * 当日成交额不低于默认2,000万元；
  * 单标的信号由 chi_diao_zhu_li_optimized.py 计算。

该筛选器输出技术研究候选，不识别任何账户主体或“主力”真实交易意图。历史回测不得将
当前股票池快照用于过去日期；请使用配套的历史快照回测模块。
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from chi_diao_zhu_li_optimized import (
    DataSourceError,
    PRESETS,
    Parameters,
    compute_indicator,
    fetch_with_hard_timeout,
    normalize_ohlc,
)


LOGGER = logging.getLogger("chi_diao_zhu_li_smallcap")
EXPORT_COLUMNS = [
    "代码", "名称", "信号日期", "信号类型", "流通市值(亿)", "当日成交额(亿)", "最新价",
    "信号评分", "股牛股", "买进", "操纵", "趋势", "中线趋势", "相对强弱%", "数据源",
]


@dataclass(frozen=True)
class Config:
    scan_offset: int = 0
    scan_limit: int = 1000
    shard_name: str = "a"
    mode: str = "normal"
    min_float_cap_yi: float = 20.0
    max_float_cap_yi: float = 200.0
    min_amount_yi: float = 0.20
    min_watch_score: int = 70
    workers: int = 2
    ak_timeout: int = 12
    bao_timeout: int = 8
    per_symbol_timeout: int = 36
    max_runtime_seconds: int = 19_200
    max_failures: int = 200
    checkpoint_every: int = 25
    history_days: int = 900
    output_dir: Path = Path("output")
    universe_file: Path | None = None
    resume: bool = True
    sendkey: str = ""
    enable_notify: bool = False

    @property
    def params(self) -> Parameters:
        return PRESETS[self.mode]


def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default


def env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def make_config() -> Config:
    mode = os.getenv("SIGNAL_MODE", "normal").strip().lower()
    if mode not in PRESETS: mode = "normal"
    minimum = env_float("MIN_FLOAT_CAP_YI", 20.0); maximum = env_float("MAX_FLOAT_CAP_YI", 200.0)
    if maximum <= minimum: raise ValueError("MAX_FLOAT_CAP_YI必须大于MIN_FLOAT_CAP_YI")
    universe_path = os.getenv("UNIVERSE_FILE", "").strip()
    return Config(
        scan_offset=max(0, env_int("SCAN_OFFSET", 0)), scan_limit=max(0, env_int("SCAN_LIMIT", 1000)),
        shard_name=os.getenv("SCAN_SHARD", "a").lower() or "a", mode=mode,
        min_float_cap_yi=minimum, max_float_cap_yi=maximum, min_amount_yi=max(0, env_float("MIN_AMOUNT_YI", 0.20)),
        min_watch_score=max(0, min(100, env_int("MIN_WATCH_SCORE", 70))), workers=max(1, min(4, env_int("NUM_WORKERS", 2))),
        ak_timeout=max(1, env_int("AK_TIMEOUT", 12)), bao_timeout=max(1, env_int("BAO_TIMEOUT", 8)),
        per_symbol_timeout=max(10, env_int("PER_SYMBOL_TIMEOUT", 36)), max_runtime_seconds=max(60, env_int("MAX_RUNTIME_SECONDS", 19_200)),
        max_failures=max(1, env_int("MAX_FAILURES", 200)), checkpoint_every=max(1, env_int("CHECKPOINT_EVERY", 25)),
        history_days=max(700, env_int("HISTORY_DAYS", 900)), output_dir=Path(os.getenv("OUTPUT_DIR", "output")), universe_file=Path(universe_path) if universe_path else None,
        resume=env_bool("RESUME", True), sendkey=os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY", ""), enable_notify=env_bool("ENABLE_NOTIFY", False),
    )


def ctx() -> Any:
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def spot_worker(result_queue: Any, source: str) -> None:
    try:
        import akshare as ak
        if source == "akshare_em":
            raw = ak.stock_zh_a_spot_em()
        elif source == "akshare_tx_fallback":
            # 腾讯快照字段为代码/名称、ltsz(亿元)、turnover(万元)、zxj(分)，统一换算成既有股票池契约。
            tx = ak.stock_zh_a_spot_tx()
            if tx is None or tx.empty:
                raise DataSourceError("腾讯快照为空")
            required = {"code", "name", "ltsz", "turnover", "zxj"}
            if not required.issubset(tx.columns):
                raise DataSourceError("腾讯快照字段不完整：" + ",".join(sorted(required.difference(tx.columns))))
            raw = pd.DataFrame({
                "代码": tx["code"].astype(str).str.extract(r"(\d{6})", expand=False),
                "名称": tx["name"],
                "流通市值": pd.to_numeric(tx["ltsz"], errors="coerce") * 100_000_000,
                "成交额": pd.to_numeric(tx["turnover"], errors="coerce") * 10_000,
                "最新价": pd.to_numeric(tx["zxj"], errors="coerce") / 10.0,
            })
        else:
            raise DataSourceError(f"未知快照来源：{source}")
        if raw is None or raw.empty: raise DataSourceError("AkShare实时快照为空")
        result_queue.put({"ok": True, "data": raw, "source": source})
    except Exception as error:
        result_queue.put({"ok": False, "reason": f"{type(error).__name__}:{str(error)[:200]}"})


def get_spot_source_hard(config: Config, source: str) -> pd.DataFrame:
    context = ctx(); result_queue = context.Queue(maxsize=1); process = context.Process(target=spot_worker, args=(result_queue, source)); process.start()
    try: payload = result_queue.get(timeout=config.ak_timeout)
    except queue.Empty as error:
        if process.is_alive(): process.terminate()
        process.join(5); result_queue.close(); raise DataSourceError(f"{source}股票池快照超时{config.ak_timeout}秒") from error
    process.join(5)
    if process.is_alive(): process.terminate(); process.join(5)
    result_queue.close()
    if not payload.get("ok"): raise DataSourceError(str(payload.get("reason", "股票池快照失败")))
    LOGGER.info("股票池快照来源：%s", payload.get("source", "unknown"))
    return payload["data"]


def get_spot_hard(config: Config) -> pd.DataFrame:
    try:
        return get_spot_source_hard(config, "akshare_em")
    except DataSourceError as em_error:
        LOGGER.warning("东财股票池快照失败，切换腾讯：%s", em_error)
        return get_spot_source_hard(config, "akshare_tx_fallback")


def build_universe(spot: pd.DataFrame, config: Config) -> pd.DataFrame:
    aliases = {"代码": "代码", "名称": "名称", "流通市值": "流通市值", "成交额": "成交额", "最新价": "最新价"}
    missing = [source for source in aliases if source not in spot.columns]
    if missing: raise DataSourceError("股票池字段缺失：" + ",".join(missing))
    frame = spot[list(aliases)].rename(columns=aliases).copy()
    frame["代码"] = frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["名称"] = frame["名称"].astype(str).str.strip()
    for column in ["流通市值", "成交额", "最新价"]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame["代码"].notna() & frame["代码"].str.startswith(("0", "3", "6")) & ~frame["名称"].str.contains("ST|退", case=False, na=False, regex=True)
    valid &= frame["流通市值"].between(config.min_float_cap_yi * 100_000_000, config.max_float_cap_yi * 100_000_000)
    valid &= frame["成交额"].ge(config.min_amount_yi * 100_000_000) & frame["最新价"].gt(0)
    output = frame.loc[valid].drop_duplicates("代码", keep="last").sort_values("代码").reset_index(drop=True)
    if output.empty: raise DataSourceError("小市值候选池为空；请检查实时字段单位、时点或阈值")
    return output


def load_shared_universe(path: Path) -> pd.DataFrame:
    """读取准备任务生成的共同股票池，防止各分片各自快照造成遗漏或重叠。"""
    if not path.is_file():
        raise DataSourceError(f"共享股票池文件不存在：{path}")
    try:
        frame = pd.read_csv(path, dtype={"代码": str})
    except Exception as error:
        raise DataSourceError(f"共享股票池读取失败：{type(error).__name__}:{str(error)[:160]}") from error
    required = ["代码", "名称", "流通市值", "成交额", "最新价"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataSourceError("共享股票池字段缺失：" + ",".join(missing))
    frame = frame[required].copy()
    frame["代码"] = frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["名称"] = frame["名称"].astype(str).str.strip()
    for column in ["流通市值", "成交额", "最新价"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates("代码", keep="last").sort_values("代码").reset_index(drop=True)
    if frame.empty:
        raise DataSourceError("共享股票池为空")
    return frame


def symbol_worker(result_queue: Any, item: dict[str, Any], benchmark: pd.DataFrame, config_dict: dict[str, Any]) -> None:
    config = Config(**{**config_dict, "output_dir": Path(config_dict["output_dir"])})
    code = str(item["代码"])
    end, start = date.today(), date.today() - timedelta(days=config.history_days)
    try:
        try:
            stock = normalize_ohlc(fetch_with_hard_timeout("stock_ak_raw", code, start, end, config.ak_timeout), "AkShare个股未复权")
            source = "akshare_raw"
        except DataSourceError as ak_error:
            stock = normalize_ohlc(fetch_with_hard_timeout("stock_bao", code, start, end, config.bao_timeout), "BaoStock个股")
            source = "baostock"
        result = compute_indicator(stock, benchmark, config.params)
        latest = result.iloc[-1]
        score = int(latest["信号评分"])
        if int(latest["买进"]) == 1:
            signal_type = "买点"
        elif int(latest["股牛股"]) == 4 and score >= config.min_watch_score:
            signal_type = "趋势观察"
        else:
            result_queue.put({"kind": "no_signal", "code": code}); return
        row = {
            "代码": code, "名称": str(item["名称"]), "信号日期": result.index[-1].strftime("%Y-%m-%d"), "信号类型": signal_type,
            "流通市值(亿)": round(float(item["流通市值"]) / 100_000_000, 2), "当日成交额(亿)": round(float(item["成交额"]) / 100_000_000, 2),
            "最新价": round(float(latest["close"]), 2), "信号评分": score, "股牛股": int(latest["股牛股"]), "买进": int(latest["买进"]),
            "操纵": round(float(latest["操纵"]), 3), "趋势": round(float(latest["趋势"]), 3), "中线趋势": round(float(latest["中线趋势"]), 3),
            "相对强弱%": round(float(latest["相对强弱%"]) if pd.notna(latest["相对强弱%"]) else np.nan, 3), "数据源": source,
        }
        result_queue.put({"kind": "candidate", "code": code, "row": row})
    except DataSourceError as error:
        result_queue.put({"kind": "source_error", "code": code, "reason": str(error)[:200]})
    except Exception as error:
        result_queue.put({"kind": "logic_error", "code": code, "reason": f"{type(error).__name__}:{str(error)[:200]}"})


def output_paths(config: Config) -> tuple[Path, Path, Path, Path]:
    prefix = f"chi_diao_smallcap_{config.shard_name}"
    return tuple(config.output_dir / f"{prefix}{suffix}" for suffix in (".checkpoint.json", ".csv", ".json", ".md"))  # type: ignore[return-value]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"); temp.replace(path)


def load_checkpoint(path: Path, config: Config) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    defaults = {"source_error": 0, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}
    if not config.resume or not path.exists(): return set(), [], defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("processed", [])), list(payload.get("results", [])), {**defaults, **{key: int(value) for key, value in payload.get("stats", {}).items()}}
    except Exception: return set(), [], defaults


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "本分片没有新的候选。"
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None): lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def save_outputs(config: Config, universe_size: int, processed: set[str], results: list[dict[str, Any]], stats: dict[str, int], reason: str, started: float, universe_as_of: str) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True); checkpoint, csv_path, json_path, md_path = output_paths(config)
    ordered = sorted(results, key=lambda row: (0 if row["信号类型"] == "买点" else 1, -int(row["信号评分"])))
    frame = pd.DataFrame(ordered, columns=EXPORT_COLUMNS); frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "strategy": "smallcap_trend_relative_strength", "universe_as_of": universe_as_of, "market_cap_basis": "运行当日东方财富快照中的流通市值；不能用于历史回测", "shard": config.shard_name,
        "scan_offset": config.scan_offset, "scan_limit": config.scan_limit, "universe_size": universe_size, "processed": len(processed), "candidates": len(ordered), "stats": stats, "stop_reason": reason,
        "elapsed_seconds": round(time.monotonic() - started, 1), "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items() if key != "sendkey"}, "results": ordered,
    }
    atomic_json(json_path, payload); atomic_json(checkpoint, {"processed": sorted(processed), "results": ordered, "stats": stats})
    md_path.write_text(
        f"# 小流通市值趋势候选（分片 {config.shard_name.upper()}）\n\n- 股票池快照：`{universe_as_of}`\n- 流通市值过滤：`{config.min_float_cap_yi}–{config.max_float_cap_yi}亿元`\n- 成交额过滤：不少于 `{config.min_amount_yi}亿元`\n- 运行状态：`{reason}`\n- 已处理：`{len(processed)}/{universe_size}`；候选：`{len(ordered)}`\n- 异常统计：`{json.dumps(stats, ensure_ascii=False)}`\n\n> 此处市值为当日快照，仅用于当日候选筛选；不得反推用于历史表现结论。\n\n## 候选\n\n{markdown_table(frame)}\n",
        encoding="utf-8",
    )


def notify(config: Config, results: list[dict[str, Any]], reason: str) -> None:
    if not config.enable_notify or not config.sendkey: return
    top = sorted(results, key=lambda row: (0 if row["信号类型"] == "买点" else 1, -int(row["信号评分"])))[:20]
    lines = [f"- {row['名称']}({row['代码']}) {row['信号类型']} 分{row['信号评分']} 流通市值{row['流通市值(亿)']}亿" for row in top] or ["本分片无候选。"]
    try: requests.post(f"https://sctapi.ftqq.com/{config.sendkey}.send", data={"title": f"小市值趋势[{config.shard_name.upper()}] {len(results)}只", "desp": f"状态：{reason}\n\n" + "\n".join(lines)}, timeout=15).raise_for_status()
    except requests.RequestException as error: LOGGER.warning("通知失败：%s", error)


def run(config: Config) -> int:
    started = time.monotonic(); config.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        universe = load_shared_universe(config.universe_file) if config.universe_file else build_universe(get_spot_hard(config), config)
        end = date.today(); start = end - timedelta(days=config.history_days)
        benchmark = normalize_ohlc(fetch_with_hard_timeout("index_ak", "sh000001", start, end, config.ak_timeout), "上证指数")
    except DataSourceError as error:
        LOGGER.warning("股票池或基准不可用，安全结束：%s", error)
        save_outputs(config, 0, set(), [], {"source_error": 1, "logic_error": 0, "timeout": 0, "no_signal": 0, "worker_crash": 0}, "universe_or_benchmark_unavailable", started, pd.Timestamp.now().strftime("%Y-%m-%d")); return 0
    shard = universe.iloc[config.scan_offset:config.scan_offset + config.scan_limit] if config.scan_limit else universe.iloc[config.scan_offset:]
    checkpoint, _, _, _ = output_paths(config); processed, results, stats = load_checkpoint(checkpoint, config); tasks = [row.to_dict() for _, row in shard.iterrows() if row["代码"] not in processed]
    LOGGER.info("小市值分片%s：快照候选%s，本片%s，待处理%s", config.shard_name.upper(), len(universe), len(shard), len(tasks))
    context = ctx(); active: dict[str, tuple[Any, Any, float]] = {}; index = 0; stop_reason = "completed"; last_save = len(processed); config_dict = asdict(config); config_dict["output_dir"] = str(config.output_dir)
    def launch(item: dict[str, Any]) -> None:
        result_queue = context.Queue(maxsize=1); process = context.Process(target=symbol_worker, args=(result_queue, item, benchmark, config_dict)); process.start(); active[str(item["代码"])] = (process, result_queue, time.monotonic())
    while index < len(tasks) or active:
        if time.monotonic() - started >= config.max_runtime_seconds and stop_reason == "completed": stop_reason = "runtime_budget_reached"
        if stats["source_error"] >= config.max_failures and stop_reason == "completed": stop_reason = "circuit_breaker_open"
        while stop_reason == "completed" and len(active) < config.workers and index < len(tasks): launch(tasks[index]); index += 1
        for code, (process, result_queue, launched) in list(active.items()):
            try: payload = result_queue.get_nowait()
            except queue.Empty: payload = None
            if payload is not None:
                process.join(2); result_queue.close(); active.pop(code, None); processed.add(code); kind = payload.get("kind", "worker_crash")
                if kind == "candidate": results.append(payload["row"])
                elif kind in stats: stats[kind] += 1
                else: stats["worker_crash"] += 1
            elif time.monotonic() - launched >= config.per_symbol_timeout:
                if process.is_alive(): process.terminate()
                process.join(5); result_queue.close(); active.pop(code, None); processed.add(code); stats["timeout"] += 1
            elif process.exitcode is not None:
                result_queue.close(); active.pop(code, None); processed.add(code); stats["worker_crash"] += 1
        if len(processed) - last_save >= config.checkpoint_every:
            save_outputs(config, len(shard), processed, results, stats, "in_progress" if stop_reason == "completed" else stop_reason, started, pd.Timestamp.now().strftime("%Y-%m-%d")); last_save = len(processed)
        if stop_reason != "completed":
            for process, result_queue, _ in active.values():
                if process.is_alive(): process.terminate()
                process.join(5); result_queue.close()
            active.clear(); break
        time.sleep(0.1)
    as_of_value = os.getenv("UNIVERSE_AS_OF", "").strip()
    try:
        as_of = pd.Timestamp(as_of_value).strftime("%Y-%m-%d") if as_of_value else pd.Timestamp.now().strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        as_of = pd.Timestamp.now().strftime("%Y-%m-%d")
    save_outputs(config, len(shard), processed, results, stats, stop_reason, started, as_of); notify(config, results, stop_reason)
    LOGGER.info("分片%s结束：%s，处理%s/%s，候选%s，统计%s", config.shard_name.upper(), stop_reason, len(processed), len(shard), len(results), stats)
    return 0


def self_test() -> int:
    raw = pd.DataFrame({"代码": ["600000", "300001", "000001", "600002"], "名称": ["测试甲", "测试乙", "ST测试", "测试丁"], "流通市值": [5e9, 15e9, 10e9, 15e9], "成交额": [5e7, 1e7, 5e7, 5e7], "最新价": [10, 8, 5, 12]})
    config = Config(min_float_cap_yi=20, max_float_cap_yi=200, min_amount_yi=0.2)
    universe = build_universe(raw, config)
    assert universe["代码"].tolist() == ["600000", "600002"]
    print("SELF_TEST_OK"); return 0


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    return self_test() if "--self-test" in sys.argv else run(make_config())


if __name__ == "__main__": raise SystemExit(main())
