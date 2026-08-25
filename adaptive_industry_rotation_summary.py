#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总自适应行业轮动A-D分片，并在完整截面上统一排序。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from a_share_adaptive_industry_rotation_research import AdaptiveWeights, FACTOR_COLUMNS, MarketRegime


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_shard_dir(root: Path, shard_index: int) -> Path | None:
    expected = root / f"shard-{shard_index}"
    if (expected / "adaptive_industry_rotation_audit.json").exists():
        return expected
    candidates: list[Path] = []
    for audit_path in root.rglob("adaptive_industry_rotation_audit.json"):
        try:
            audit = read_json(audit_path)
            if int(audit.get("shard_index", -1)) == shard_index:
                candidates.append(audit_path.parent)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return sorted(candidates)[0] if candidates else None


def read_csv_or_empty(path: Path, **kwargs: Any) -> tuple[pd.DataFrame, bool]:
    """读取扫描产物；零字节CSV表示该分片没有该类记录，而非汇总器故障。"""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(), True
    try:
        return pd.read_csv(path, **kwargs), False
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), True


def merge(root: Path, expected_shards: int, top: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates: list[pd.DataFrame] = []
    errors: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    missing: list[int] = []
    empty_candidate_artifacts: list[str] = []
    empty_error_artifacts: list[str] = []
    for shard_index in range(expected_shards):
        shard = find_shard_dir(root, shard_index)
        if shard is None:
            missing.append(shard_index)
            continue
        audit = read_json(shard / "adaptive_industry_rotation_audit.json")
        audit["artifact_shard_dir"] = str(shard)
        audits.append(audit)
        candidate_path = shard / "adaptive_industry_rotation_candidates.csv"
        if candidate_path.exists():
            frame, is_empty = read_csv_or_empty(candidate_path, dtype={"code": str})
            if is_empty:
                empty_candidate_artifacts.append(f"shard-{shard_index}/{candidate_path.name}")
            if not frame.empty:
                frame["code"] = frame["code"].str.zfill(6)
                frame["source_shard_index"] = shard_index
                candidates.append(frame)
        error_path = shard / "adaptive_industry_rotation_errors.csv"
        if error_path.exists():
            frame, is_empty = read_csv_or_empty(error_path, dtype=str)
            if is_empty:
                empty_error_artifacts.append(f"shard-{shard_index}/{error_path.name}")
            frame = frame.fillna("")
            if not frame.empty:
                frame["source_shard_index"] = str(shard_index)
                errors.append(frame)
    merged = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    merged_errors = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=["code", "name", "industry", "stage", "reason", "source_shard_index"])
    signals = sorted({str(audit.get("signal_date_requested", "")) for audit in audits})
    regimes = sorted({str(audit.get("market_regime", "")) for audit in audits})
    status = "ready" if not missing and audits else "degraded"
    if not merged.empty:
        missing_factors = [factor for factor in FACTOR_COLUMNS if factor not in merged.columns]
        if missing_factors:
            raise ValueError(f"candidate_factor_columns_missing:{','.join(missing_factors)}")
        if len(regimes) != 1:
            status = "degraded"
        regime = MarketRegime(regimes[0]) if regimes and regimes[0] in {value.value for value in MarketRegime} else MarketRegime.NEUTRAL
        weights, separability = AdaptiveWeights().allocate(merged, regime)
        correlations = AdaptiveWeights.correlations(merged)
        merged["research_score"] = 100.0 * sum(merged[factor] * weights[factor] for factor in FACTOR_COLUMNS)
        merged = merged.sort_values(["research_score", "trend", "average_turnover_20d_yuan"], ascending=False)
        merged = merged.drop_duplicates("code", keep="first").reset_index(drop=True)
        merged.insert(0, "rank", range(1, len(merged) + 1))
        merged["rank_in_industry"] = merged.groupby("industry")["research_score"].rank(method="min", ascending=False).astype(int)
        result = merged.head(top).copy() if top > 0 else merged
    else:
        weights, separability, correlations, result = {}, {}, {}, merged
    output_audit = {
        "schema_version": "a-share-adaptive-industry-rotation-summary/v1",
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "expected_shards": expected_shards,
        "received_shards": [audit.get("shard_index") for audit in audits],
        "missing_shards": missing,
        "empty_candidate_artifacts": empty_candidate_artifacts,
        "empty_error_artifacts": empty_error_artifacts,
        "signal_dates": signals,
        "market_regimes": regimes,
        "source_candidate_count": int(len(merged)),
        "candidate_count": int(len(result)),
        "error_count": int(len(merged_errors)),
        "adaptive_weights": weights,
        "cross_sectional_separability_adjustment": separability,
        "factor_cross_sectional_correlation": correlations,
        "per_shard": audits,
        "research_disclaimer": "权重在合并后的全市场候选截面统一计算；不以未来收益估计IC。行业快照为运行时截面，历史signal_date不能据此做行业时点回测。结果不构成投资建议。",
    }
    return result, merged_errors, output_audit


def completed_shard_count(audit: dict[str, Any]) -> int:
    failed = {"", "missing", "unavailable", "failed", "error"}
    return sum(str(shard.get("status", "")).lower() not in failed for shard in audit.get("per_shard", []))


def notification_body(audit: dict[str, Any], candidates: pd.DataFrame) -> str:
    lines = [
        f"- 状态：`{audit['status']}`",
        f"- 信号日：`{', '.join(audit['signal_dates']) or '未记录'}`",
        f"- 已完成分片：`{completed_shard_count(audit)}/{audit['expected_shards']}`",
        f"- 候选：`{audit['candidate_count']}`",
        f"- 错误：`{audit['error_count']}`",
    ]
    fields = [field for field in ["rank", "code", "name", "industry", "research_score"] if field in candidates.columns]
    if fields:
        lines.extend(["", "## 前列研究候选"])
        for _, row in candidates[fields].head(20).iterrows():
            score = row.get("research_score")
            score_text = f" / {float(score):.2f}" if pd.notna(score) else ""
            lines.append(f"{row.get('rank', '-')}. {row.get('code', '')} {row.get('name', '')} {row.get('industry', '')}{score_text}")
    else:
        lines.extend(["", "当日没有符合当前研究条件的候选；完整状态和错误审计见artifact。"])
    lines.extend(["", "> 该输出为研究筛选，不构成投资建议。"])
    return "\n".join(lines)


def notify_serverchan(audit: dict[str, Any], candidates: pd.DataFrame) -> str:
    if completed_shard_count(audit) == 0:
        return "skipped:no_completed_shards"
    sendkey = os.getenv("SENDKEY", "").strip()
    if not sendkey:
        return "skipped:no_sendkey"
    title = f"自适应行业轮动：{audit['status']} / 候选{audit['candidate_count']}只"
    try:
        response = requests.post(
            f"https://sctapi.ftqq.com/{sendkey}.send",
            data={"title": title, "desp": notification_body(audit, candidates)[:3800]},
            timeout=15,
        )
        payload = response.json()
        code = payload.get("code") if isinstance(payload, dict) else None
        return "sent" if response.ok and code == 0 else f"failed:http_{response.status_code}:serverchan_code_{code}"
    except (requests.RequestException, ValueError) as exc:
        return f"failed:{type(exc).__name__}"


def write_outputs(output_dir: Path, candidates: pd.DataFrame, errors: pd.DataFrame, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "adaptive_industry_rotation_candidates.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(output_dir / "adaptive_industry_rotation_errors.csv", index=False, encoding="utf-8-sig")
    (output_dir / "adaptive_industry_rotation_snapshot.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    display = [column for column in ["rank", "code", "name", "industry", "research_score", "close", "return_20d_pct", "average_turnover_20d_wan", "source_shard_index"] if column in candidates]
    lines = [
        "# A股自适应行业轮动全市场汇总",
        f"- 状态：`{audit['status']}`",
        f"- 信号日：`{', '.join(audit['signal_dates']) or '未记录'}`",
        f"- 已收到分片：`{len(audit['received_shards'])}/{audit['expected_shards']}`",
        f"- 候选：`{audit['candidate_count']}`",
        f"- 错误：`{audit['error_count']}`",
        f"- 通知：`{audit.get('notification', 'skipped:notify_not_requested')}`",
        "",
        "> 统一权重仅在全部分片合并后生成；不使用未来收益。该研究筛选不构成投资建议。",
    ]
    if not candidates.empty and display:
        lines.extend(["", "## 候选", candidates[display].head(100).to_markdown(index=False, floatfmt=".4f")])
    (output_dir / "adaptive_industry_rotation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="自适应行业轮动A-D分片汇总器")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("output_adaptive_industry_rotation_summary"))
    parser.add_argument("--notify", action="store_true", help="仅在至少一个完成分片后尝试发送一条Server酱消息")
    args = parser.parse_args()
    if args.expected_shards < 1 or args.top < 0:
        raise ValueError("expected-shards必须为正数，top不能为负数")
    candidates, errors, audit = merge(args.artifacts_dir, args.expected_shards, args.top)
    audit["notification"] = notify_serverchan(audit, candidates) if args.notify else "skipped:notify_not_requested"
    write_outputs(args.output_dir, candidates, errors, audit)
    print(json.dumps({"status": audit["status"], "candidate_count": audit["candidate_count"], "missing_shards": audit["missing_shards"], "notification": audit["notification"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
