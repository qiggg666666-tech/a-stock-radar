#!/usr/bin/env python3
"""FINAL Chip专属汇总器；只在有完成分片时尝试单条通知。
只处理"现价下方长红柱"(is_below_spike)这一条线——不再计算/排序/推送 is_tradeable
那套"可交易"榜单。全部命中分批推送，避免单条消息过长被截断。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() and path.stat().st_size else pd.DataFrame()


def notify(title: str, body: str) -> dict[str, object]:
    key = os.getenv("SENDKEY", "").strip()
    if not key: return {"status": "skipped", "reason": "missing_sendkey"}
    try:
        import requests
        response = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body}, timeout=20); result = response.json()
        return {"status": "sent", "http_status": response.status_code} if response.ok and result.get("code") == 0 else {"status": "failed", "http_status": response.status_code, "code": result.get("code")}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:250]}"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--shard-total", type=int, default=4); parser.add_argument("--notify", action="store_true"); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--batch-size", type=int, default=40, help="每条Server酱消息最多放多少只，避免单条过长被截断"); args = parser.parse_args()
    if args.self_test: assert read_csv(Path("/missing.csv")).empty; print("FINAL_CHIP_SUMMARY_SELF_TEST_OK"); return 0
    if args.input_dir is None or args.output_dir is None: parser.error("--input-dir and --output-dir are required")
    statuses, frames, error_frames = [], [], []
    for status_path in args.input_dir.rglob("status.json"):
        try:
            statuses.append(json.loads(status_path.read_text(encoding="utf-8"))); frames.append(read_csv(status_path.parent / "records.csv")); error_frames.append(read_csv(status_path.parent / "errors.csv"))
        except Exception as exc: statuses.append({"state": "artifact_read_error", "error": f"{type(exc).__name__}:{str(exc)[:180]}"})
    completed = [item for item in statuses if item.get("state") in {"completed", "completed_zero_records"}]; args.output_dir.mkdir(parents=True, exist_ok=True)
    if not completed:
        result = {"schema_version": "final-chip-summary/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "skipped:no_completed_shards", "completed_shards": [], "below_spike_count": 0, "notification": {"status": "skipped", "reason": "no_completed_shards"}, "disclosure": "无完成分片，不生成研究结论或通知。"}; (args.output_dir / "final_chip_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (args.output_dir / "final_chip_report.md").write_text("# FINAL Chip汇总\n\n无完成分片，未生成研究结论或通知。\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0
    records = pd.concat([item for item in frames if not item.empty], ignore_index=True) if any(not item.empty for item in frames) else pd.DataFrame(); errors = pd.concat([item for item in error_frames if not item.empty], ignore_index=True) if any(not item.empty for item in error_frames) else pd.DataFrame(columns=["code", "name", "error_type", "error_message"])
    errors.to_csv(args.output_dir / "final_chip_errors.csv", index=False, encoding="utf-8-sig")

    below_spike = pd.DataFrame()
    if not records.empty and "is_below_spike" in records.columns:
        records["is_below_spike"] = records["is_below_spike"].astype(str).str.lower().eq("true")
        records["below_score"] = pd.to_numeric(records["below_score"], errors="coerce").fillna(0) if "below_score" in records else 0.0
        below_spike = records[records["is_below_spike"] == True].sort_values("below_score", ascending=False)
    below_spike.to_csv(args.output_dir / "final_chip_below_spike.csv", index=False, encoding="utf-8-sig")

    wide_zone = pd.DataFrame()
    if not records.empty and "is_wide_zone" in records.columns:
        records["is_wide_zone"] = records["is_wide_zone"].astype(str).str.lower().eq("true")
        records["wide_score"] = pd.to_numeric(records["wide_score"], errors="coerce").fillna(0) if "wide_score" in records else 0.0
        wide_zone = records[records["is_wide_zone"] == True].sort_values("wide_score", ascending=False)
    wide_zone.to_csv(args.output_dir / "final_chip_wide_zone.csv", index=False, encoding="utf-8-sig")

    below_lines = []
    for number, (_, row) in enumerate(below_spike.iterrows(), 1):
        below_lines.append(
            f"{number:03d}. {row.get('code','')} {row.get('name','')}｜评分{row.get('below_score','')}"
            f"｜收盘{row.get('close','')}｜下方峰{row.get('below_peak','')}｜峰距{row.get('below_dist_pct','')}%"
            f"｜峰带占比{row.get('below_band_ratio_pct','')}%｜占比{row.get('below_peak_ratio_pct','')}%"
            f"｜峰隙比{row.get('below_peak_gap','')}｜{row.get('below_state','')}"
        )
    wide_lines = []
    for number, (_, row) in enumerate(wide_zone.iterrows(), 1):
        wide_lines.append(
            f"{number:03d}. {row.get('code','')} {row.get('name','')}｜评分{row.get('wide_score','')}"
            f"｜收盘{row.get('close','')}｜区间{row.get('wide_zone_low','')}~{row.get('wide_zone_high','')}"
            f"｜宽度{row.get('wide_width_pct','')}%｜占比{row.get('wide_zone_ratio_pct','')}%｜距上沿{row.get('wide_dist_pct','')}%"
            f"｜{row.get('wide_state','')}"
        )

    lines = [
        "# FINAL Chip 汇总（现价下方长红柱 + 宽幅堆积区）", "",
        f"- 完成分片：{len(completed)}/{args.shard_total}", f"- 有效记录：{len(records)}", f"- 错误台账：{len(errors)}",
        f"- 现价下方长红柱(is_below_spike)：{len(below_spike)}只", f"- 宽幅堆积区(is_wide_zone)：{len(wide_zone)}只",
        f"- 均全部推送，每条消息最多{args.batch_size}只", "",
        "## 现价下方长红柱（按评分排序，全部）",
    ]
    lines.extend(below_lines if below_lines else ["当日无命中记录。"])
    lines.extend(["", "## 宽幅堆积区（按评分排序，全部）"])
    lines.extend(wide_lines if wide_lines else ["当日无命中记录。"])
    lines.extend(["", "仅为FINAL Chip规则研究输出，不构成投资建议。"])

    # 分批推送：每种信号各自独立分批，每条 Server酱 消息最多 batch_size 只。
    def push_batches(label: str, row_lines: list[str]) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        batch_size = max(args.batch_size, 1)
        if not row_lines:
            title = f"FINAL Chip｜{label} 0只｜{len(completed)}/{args.shard_total}分片"
            body = "\n".join([f"# FINAL Chip {label}", "", "当日无命中记录。", "", "仅为FINAL Chip规则研究输出，不构成投资建议。"])
            results.append(notify(title, body))
            return results
        total_batches = (len(row_lines) + batch_size - 1) // batch_size
        for batch_index in range(total_batches):
            chunk = row_lines[batch_index * batch_size: (batch_index + 1) * batch_size]
            title = f"FINAL Chip｜{label} 批次{batch_index + 1}/{total_batches}（共{len(row_lines)}只）"
            body_lines = [f"# FINAL Chip {label} 批次{batch_index + 1}/{total_batches}", "", f"共{len(row_lines)}只，本批{len(chunk)}只", ""] + chunk + ["", "仅为FINAL Chip规则研究输出，不构成投资建议。"]
            results.append(notify(title, "\n".join(body_lines)))
        return results

    notifications: dict[str, list[dict[str, object]]] = {"below_spike": [], "wide_zone": []}
    if args.notify:
        notifications["below_spike"] = push_batches("现价下方长红柱", below_lines)
        notifications["wide_zone"] = push_batches("宽幅堆积区", wide_lines)
    else:
        notifications["below_spike"] = [{"status": "not_requested"}]
        notifications["wide_zone"] = [{"status": "not_requested"}]

    result = {"schema_version": "final-chip-summary/v1", "generated_at": datetime.now(timezone.utc).isoformat(), "state": "ready" if len(completed) == args.shard_total else "partial", "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed), "record_count": len(records), "error_count": len(errors), "below_spike_count": len(below_spike), "wide_zone_count": len(wide_zone), "batch_size": args.batch_size, "notifications": notifications, "disclosure": "FINAL Chip专属汇总(现价下方长红柱+宽幅堆积区)；通知失败不阻断artifact。"}
    (args.output_dir / "final_chip_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (args.output_dir / "final_chip_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
