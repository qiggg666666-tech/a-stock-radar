# -*- coding: utf-8 -*-
"""汇总Q01 A-D分片，并生成参数指纹、公开快照和单条Server酱通知。"""
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

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def sector_for(code: str) -> tuple[str, str]:
    if ak is None:
        return "板块待补充", "akshare_unavailable"
    try:
        profile = ak.stock_individual_info_em(symbol=str(code).zfill(6))
        item_column = "item" if "item" in profile else "项目" if "项目" in profile else ""
        value_column = "value" if "value" in profile else "值" if "值" in profile else ""
        if item_column and value_column:
            for _, row in profile.iterrows():
                if str(row[item_column]).strip() in {"所属行业", "行业"}:
                    label = str(row[value_column]).strip()
                    if label and label.lower() not in {"nan", "none", "-", "--"}:
                        return label, "akshare.stock_individual_info_em:所属行业"
        return "板块待补充", "akshare_profile_missing_industry"
    except Exception:
        return "板块待补充", "akshare_sector_lookup_failed"


def notify(title: str, content: str) -> str:
    key = os.getenv("SENDKEY") or os.getenv("SERVERCHAN_SENDKEY") or ""
    if not key:
        return "skipped:no_sendkey"
    try:
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content[:3800]}, timeout=15)
        payload = response.json()
        return "sent" if response.ok and payload.get("code") == 0 else f"failed:http_{response.status_code}"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


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
        sectors = [sector_for(code) for code in merged["code"]]
        merged["sector"] = [item[0] for item in sectors]
        merged["sector_source"] = [item[1] for item in sectors]
        merged["score_label"] = "研究总评分"
    completed = [item for item in states if item.get("state") == "completed"]
    state = "ready" if len(completed) == args.expected_shards else "partial"
    config = states[0].get("config", {}) if states else {}
    fingerprint = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16] if config else "unavailable"
    signal_date = str(states[0].get("signal_date") or "") if states else ""
    universe = read_json(Path(args.universe_status)) if args.universe_status else None
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    candidate_rows = [] if merged.empty else merged.to_dict("records")
    snapshot = {"schema_version": "q01-snapshot/v1", "strategy": {"key": "multi-factor-ohlcv", "label": "Q01多因子", "score_label": "研究总评分"}, "run_date": signal_date, "generated_at": generated_at, "status": state, "candidate_count": len(candidate_rows), "shards": [str(item.get("shard_index")) for item in states], "source_files": [path.name for path in root.rglob("multi_factor_ohlcv_candidates.csv")], "read_errors": [], "candidates": [{"code": str(row.get("code", "")), "name": str(row.get("name", "")), "score": int(row.get("research_score", 0)), "tag": "通过", "extras": {"板块": row.get("sector", "板块待补充"), "收盘": row.get("close"), "20日涨幅%": row.get("ret_20_pct"), "20日均成交额": row.get("turnover_ma20_yuan"), "数据源": row.get("data_source"), "参数指纹": fingerprint}} for row in candidate_rows], "data_provenance": {"state": "cache" if (universe or {}).get("state") == "degraded_cache" else "live", "source": (universe or {}).get("source", "未记录"), "detail": "共同股票池与A-D同日分片产物汇总。"}, "parameter_fingerprint": fingerprint, "config": config, "processed": sum(int(item.get("processed") or 0) for item in states), "errors": errors, "disclaimer": "Q01为日线OHLCV研究筛选，板块为运行时AkShare行业快照；不构成投资建议。"}
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "multi_factor_ohlcv_candidates.csv", index=False, encoding="utf-8-sig")
    (output / "multi_factor_ohlcv_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Q01全市场多因子汇总", f"- 状态：`{state}`", f"- 信号日：`{signal_date or '未记录'}`", f"- 分片：`{len(completed)}/{args.expected_shards}`", f"- 处理：`{snapshot['processed']}`", f"- 候选：`{snapshot['candidate_count']}`", f"- 错误：`{errors}`", f"- 参数指纹：`{fingerprint}`", "", "> 仅以信号日及此前OHLCV计算；板块为运行时显示元数据，不参与评分或历史回测。"]
    if args.notify == "true":
        snapshot["notification"] = notify(f"Q01全市场：{state} / 候选{snapshot['candidate_count']}只", "\n".join(lines))
        (output / "multi_factor_ohlcv_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "multi_factor_ohlcv_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": state, "candidates": snapshot["candidate_count"], "fingerprint": fingerprint}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
