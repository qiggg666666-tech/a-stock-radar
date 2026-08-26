# -*- coding: utf-8 -*-
"""汇总Q01 A-D分片，并生成参数指纹、公开快照和主备单条通知。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


MAX_SERVERCHAN_CONTENT = 3800
MAX_TELEGRAM_CONTENT = 3800


def notify_serverchan(title: str, content: str) -> str:
    key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
    if not key:
        return "skipped:no_sendkey"
    if len(content) > MAX_SERVERCHAN_CONTENT:
        return f"failed:content_too_long:{len(content)}"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15)
        payload = response.json()
        business_code = payload.get("code") if isinstance(payload, dict) else None
        return "sent" if response.ok and business_code == 0 else f"failed:http_{response.status_code}:code_{business_code}"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def resolve_telegram_chat_id(token: str) -> tuple[str | None, str]:
    configured = os.getenv("TELEGRAM_CHAT_ID") or ""
    if configured.strip():
        return configured.strip(), "configured"
    try:
        response = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15)
        payload = response.json()
        updates = payload.get("result") if response.ok and isinstance(payload, dict) and payload.get("ok") is True else []
        if not isinstance(updates, list):
            return None, "unavailable:invalid_updates"
        for update in reversed(updates):
            message = update.get("message") if isinstance(update, dict) else None
            chat = message.get("chat") if isinstance(message, dict) else None
            chat_id = chat.get("id") if isinstance(chat, dict) else None
            if chat_id is not None:
                return str(chat_id), "auto_detected"
        return None, "unavailable:no_message_update"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


def notify_telegram(title: str, content: str) -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    if not token:
        return "skipped:telegram_not_configured", "not_configured"
    if len(content) > MAX_TELEGRAM_CONTENT:
        return f"failed:telegram_content_too_long:{len(content)}", "not_attempted:content_too_long"
    chat_id, target = resolve_telegram_chat_id(token)
    if not chat_id:
        return f"skipped:telegram_chat_id_{target}", target
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": f"{title}\n\n{content}", "disable_web_page_preview": "true"},
            timeout=15,
        )
        payload = response.json()
        telegram_ok = payload.get("ok") is True if isinstance(payload, dict) else False
        return ("sent" if response.ok and telegram_ok else f"failed:http_{response.status_code}:ok_{telegram_ok}"), target
    except Exception as exc:
        return f"failed:{type(exc).__name__}", target


def notify_with_telegram_fallback(title: str, content: str) -> dict[str, str]:
    serverchan = notify_serverchan(title, content)
    result = {"serverchan": serverchan, "telegram": "not_attempted:serverchan_sent", "telegram_target": "not_attempted:serverchan_sent", "delivered_by": "serverchan"}
    if serverchan == "sent":
        return result
    telegram, target = notify_telegram(title, content)
    result["telegram"] = telegram
    result["telegram_target"] = target
    result["delivered_by"] = "telegram" if telegram == "sent" else "none"
    return result


def format_candidate_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Create compact, human-readable full-candidate rows for one notification."""
    lines = ["## 全部候选（按研究评分排序）"]
    for index, row in enumerate(rows, start=1):
        code = str(row.get("code", "")).zfill(6)
        name = str(row.get("name", ""))
        score = int(float(row.get("research_score", 0) or 0))
        close = float(row.get("close", 0) or 0)
        ret_1 = float(row.get("ret_1_pct", 0) or 0)
        ret_20 = float(row.get("ret_20_pct", 0) or 0)
        sector = str(row.get("sector", "板块待补充"))
        source = str(row.get("data_source", "未记录"))
        lines.append(
            f"{index:02d}. `{code}` {name}｜评分{score}｜收{close:.2f}｜1日{ret_1:+.2f}%｜20日{ret_20:+.2f}%｜{sector}｜{source}"
        )
    return lines


def format_notification_candidate_lines(rows: list[dict[str, Any]], max_items: int = 20) -> list[str]:
    """Render at most 20 mobile-readable candidates; keep all rows in artifacts."""
    lines = [f"## 推送候选（最多{max_items}条；完整排序见artifact）"]
    for index, row in enumerate(rows[:max_items], start=1):
        code = str(row.get("code", "")).zfill(6)
        name = str(row.get("name", ""))
        score = int(float(row.get("research_score", 0) or 0))
        lines.append(f"{index:02d}. `{code}`｜{name}｜{score}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Q01分片汇总")
    parser.add_argument("--input-root", default="collected")
    parser.add_argument("--universe-status")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--notify", choices=["true", "false"], default="true")
    args = parser.parse_args()
    root, output = Path(args.input_root), Path(args.output_dir)
    states = [payload for path in sorted(root.rglob("multi_factor_ohlcv_state.json")) if (payload := read_json(path))]
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("multi_factor_ohlcv_candidates.csv")):
        try:
            frames.append(pd.read_csv(path, dtype={"code": str}))
        except Exception:
            pass
    errors = 0
    for payload in states:
        errors += int(payload.get("errors") or 0)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not merged.empty:
        merged["code"] = merged["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        merged = merged.sort_values(["research_score", "signed_flow_ratio_5", "volume_ratio"], ascending=False).drop_duplicates("code").head(max(0, args.top)).copy()
        if "sector" not in merged.columns:
            merged["sector"] = ""
        if "sector_source" not in merged.columns:
            merged["sector_source"] = ""
        sector_missing = merged["sector"].astype(str).str.strip().isin({"", "nan", "None"})
        merged.loc[sector_missing, "sector"] = "板块待补充"
        source_missing = merged["sector_source"].astype(str).str.strip().isin({"", "nan", "None"})
        merged.loc[source_missing, "sector_source"] = "sector_not_queried_at_summary"
        merged["score_label"] = "研究总评分"
    completed = [item for item in states if item.get("state") == "completed"]
    state = "ready" if len(completed) == args.expected_shards else "partial"
    config = states[0].get("config", {}) if states else {}
    fingerprint = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16] if config else "unavailable"
    signal_date = str(states[0].get("signal_date") or "") if states else ""
    universe = read_json(Path(args.universe_status)) if args.universe_status else None
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    candidate_rows = [] if merged.empty else merged.to_dict("records")
    snapshot = {"schema_version": "q01-snapshot/v1", "strategy": {"key": "multi-factor-ohlcv", "label": "Q01多因子", "score_label": "研究总评分"}, "run_date": signal_date, "generated_at": generated_at, "status": state, "candidate_count": len(candidate_rows), "shards": [str(item.get("shard_index")) for item in states], "source_files": [path.name for path in root.rglob("multi_factor_ohlcv_candidates.csv")], "read_errors": [], "candidates": [{"code": str(row.get("code", "")), "name": str(row.get("name", "")), "score": int(row.get("research_score", 0)), "tag": "通过", "extras": {"板块": row.get("sector", "板块待补充"), "收盘": row.get("close"), "20日涨幅%": row.get("ret_20_pct"), "20日均成交额": row.get("turnover_ma20_yuan"), "数据源": row.get("data_source"), "参数指纹": fingerprint}} for row in candidate_rows], "data_provenance": {"state": "cache" if (universe or {}).get("state") == "degraded_cache" else "live", "source": (universe or {}).get("source", "未记录"), "detail": "共同股票池与A-D同日分片产物汇总。"}, "parameter_fingerprint": fingerprint, "config": config, "processed": sum(int(item.get("processed") or 0) for item in states), "errors": errors, "disclaimer": "Q01为日线OHLCV研究筛选，板块为运行时元数据；不构成投资建议。"}
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "multi_factor_ohlcv_candidates.csv", index=False, encoding="utf-8-sig")
    (output / "multi_factor_ohlcv_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Q01全市场多因子汇总", f"- 状态：`{state}`", f"- 信号日：`{signal_date or '未记录'}`", f"- 分片：`{len(completed)}/{args.expected_shards}`", f"- 处理：`{snapshot['processed']}`", f"- 候选：`{snapshot['candidate_count']}`", f"- 错误：`{errors}`", f"- 参数指纹：`{fingerprint}`", "", "> 仅以信号日及此前OHLCV计算；板块为运行时显示元数据，不参与评分或历史回测。", ""]
    lines.extend(format_candidate_lines(candidate_rows))
    notification_rows = candidate_rows[:20]
    notification_lines = lines[:9] + ["", f"> 完整指标和全部{len(candidate_rows)}条排序见运行artifact；以下仅展示前{len(notification_rows)}条。", ""] + format_notification_candidate_lines(notification_rows)
    notification_content = "\n".join(notification_lines)
    snapshot["notification_parts"] = 1
    snapshot["notification_candidate_count"] = len(notification_rows)
    snapshot["notification_total_candidate_count"] = len(candidate_rows)
    snapshot["notification_content_length"] = len(notification_content)
    if args.notify == "true":
        outcomes = notify_with_telegram_fallback(f"Q01全市场：{state} / 候选{snapshot['candidate_count']}只", notification_content)
        snapshot["notification_serverchan"] = outcomes["serverchan"]
        snapshot["notification_telegram"] = outcomes["telegram"]
        snapshot["notification_telegram_target"] = outcomes["telegram_target"]
        snapshot["notification_delivered_by"] = outcomes["delivered_by"]
        snapshot["notification"] = "sent:serverchan" if outcomes["delivered_by"] == "serverchan" else "sent:telegram_fallback" if outcomes["delivered_by"] == "telegram" else "failed:all_channels"
    else:
        snapshot["notification"] = "skipped:notify_disabled"
        snapshot["notification_serverchan"] = "skipped:notify_disabled"
        snapshot["notification_telegram"] = "not_attempted:notify_disabled"
        snapshot["notification_telegram_target"] = "not_attempted:notify_disabled"
        snapshot["notification_delivered_by"] = "none"
    (output / "multi_factor_ohlcv_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "multi_factor_ohlcv_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "multi_factor_ohlcv_notification_preview.md").write_text(notification_content, encoding="utf-8")
    print(json.dumps({"state": state, "candidates": snapshot["candidate_count"], "fingerprint": fingerprint}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
