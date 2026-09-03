#!/usr/bin/env python3
"""FINAL Chip 洗盘专属汇总器。

只处理 final_chip_research.py 里移植自筹码尖峰蒸馏引擎 classify_stage()
判定出的"洗盘"阶段(is_washout/stage=='洗盘')——不再输出买入/高优先级/
宽幅堆积区等其它信号。按 washout_score 从高到低排序，全部命中分批推送。

用法：
  python final_chip_summary.py --input-dir ./shards --output-dir ./out --notify
  python final_chip_summary.py --self-test
"""
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
    if not key:
        return {"status": "skipped", "reason": "missing_sendkey"}
    try:
        import requests
        response = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"title": title, "desp": body},
            timeout=20,
        )
        result = response.json()
        return (
            {"status": "sent", "http_status": response.status_code}
            if response.ok and result.get("code") == 0
            else {"status": "failed", "http_status": response.status_code, "code": result.get("code")}
        )
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:250]}"}


def _pe_str(row: pd.Series) -> str:
    pe = row.get("pe_ttm", None)
    if pe is None or (isinstance(pe, float) and pd.isna(pe)):
        return ""
    try:
        return f"｜PE{float(pe):.1f}"
    except (TypeError, ValueError):
        return ""


def format_washout_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜洗盘分{row.get('washout_score', '')}"
            f"｜收盘{row.get('close', '')}"
            f"｜{row.get('stage_note', '')}"
            f"｜穿透率{row.get('cross_ratio_pct', '')}%"
            f"｜90%成本区{row.get('cost90_low', '')}~{row.get('cost90_high', '')}"
            f"｜区间宽度{row.get('conc90_width_pct', '')}%"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="FINAL Chip 汇总（只输出洗盘阶段）")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=40,
        help="每条 Server酱 消息最多放多少只，避免单条过长被截断",
    )
    args = parser.parse_args()

    if args.self_test:
        assert read_csv(Path("/missing.csv")).empty
        print("FINAL_CHIP_SUMMARY_SELF_TEST_OK")
        return 0

    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required")

    statuses, frames, error_frames = [], [], []
    for status_path in args.input_dir.rglob("status.json"):
        try:
            statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
            frames.append(read_csv(status_path.parent / "records.csv"))
            error_frames.append(read_csv(status_path.parent / "errors.csv"))
        except Exception as exc:
            statuses.append({
                "state": "artifact_read_error",
                "error": f"{type(exc).__name__}:{str(exc)[:180]}",
            })

    completed = [item for item in statuses if item.get("state") in {"completed", "completed_zero_records"}]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not completed:
        result = {
            "schema_version": "final-chip-summary/v4-washout",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "skipped:no_completed_shards",
            "completed_shards": [],
            "washout_count": 0,
            "notification": {"status": "skipped", "reason": "no_completed_shards"},
            "disclosure": "无完成分片，不生成研究结论或通知。",
        }
        (args.output_dir / "final_chip_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "final_chip_report.md").write_text(
            "# FINAL Chip 洗盘汇总\n\n无完成分片，未生成研究结论或通知。\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0

    records = (
        pd.concat([item for item in frames if not item.empty], ignore_index=True)
        if any(not item.empty for item in frames)
        else pd.DataFrame()
    )
    errors = (
        pd.concat([item for item in error_frames if not item.empty], ignore_index=True)
        if any(not item.empty for item in error_frames)
        else pd.DataFrame(columns=["code", "name", "error_type", "error_message"])
    )
    errors.to_csv(args.output_dir / "final_chip_errors.csv", index=False, encoding="utf-8-sig")

    if not records.empty and "code" in records.columns:
        records["code"] = records["code"].astype(str).str.zfill(6)

    # ---------- 只保留洗盘阶段(is_washout / stage=='洗盘') ----------
    washout = pd.DataFrame()
    if not records.empty and "is_washout" in records.columns:
        records["is_washout"] = records["is_washout"].astype(str).str.lower().eq("true")
        records["washout_score"] = (
            pd.to_numeric(records["washout_score"], errors="coerce").fillna(0)
            if "washout_score" in records.columns else 0.0
        )
        washout = records[records["is_washout"] == True].sort_values(
            "washout_score", ascending=False
        )
    washout.to_csv(args.output_dir / "final_chip_washout.csv", index=False, encoding="utf-8-sig")

    washout_lines = format_washout_lines(washout)

    lines = [
        "# FINAL Chip 洗盘汇总（移植自筹码尖峰蒸馏引擎 classify_stage）",
        "",
        f"- 完成分片：{len(completed)}/{args.shard_total}",
        f"- 有效记录：{len(records)}",
        f"- 错误台账：{len(errors)}",
        f"- 洗盘阶段(is_washout)：{len(washout)}只，按 washout_score 从高到低全部推送",
        f"- 每条消息最多{args.batch_size}只",
        "",
        "## 洗盘专区（按 washout_score 从高到低排序）",
    ]
    lines.extend(washout_lines if washout_lines else ["当日无洗盘阶段记录。"])
    lines.extend(["", "仅为 FINAL Chip 规则研究输出，不构成投资建议。"])

    def push_batches(label: str, row_lines: list[str]) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        batch_size = max(args.batch_size, 1)
        if not row_lines:
            title = f"FINAL Chip｜{label} 0只｜{len(completed)}/{args.shard_total}分片"
            body = "\n".join([
                f"# FINAL Chip {label}", "", "当日无命中记录。", "",
                "仅为 FINAL Chip 规则研究输出，不构成投资建议。",
            ])
            results.append(notify(title, body))
            return results
        total_batches = (len(row_lines) + batch_size - 1) // batch_size
        for batch_index in range(total_batches):
            chunk = row_lines[batch_index * batch_size: (batch_index + 1) * batch_size]
            title = f"FINAL Chip｜{label} 批次{batch_index + 1}/{total_batches}（共{len(row_lines)}只）"
            body_lines = (
                [f"# FINAL Chip {label} 批次{batch_index + 1}/{total_batches}", "",
                 f"共{len(row_lines)}只，本批{len(chunk)}只", ""]
                + chunk
                + ["", "仅为 FINAL Chip 规则研究输出，不构成投资建议。"]
            )
            results.append(notify(title, "\n".join(body_lines)))
        return results

    notifications: dict[str, list[dict[str, object]]] = {"washout": []}
    if args.notify:
        notifications["washout"] = push_batches("洗盘专区", washout_lines)
    else:
        notifications["washout"] = [{"status": "not_requested"}]

    result = {
        "schema_version": "final-chip-summary/v4-washout",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ready" if len(completed) == args.shard_total else "partial",
        "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed),
        "record_count": len(records),
        "error_count": len(errors),
        "washout_count": len(washout),
        "batch_size": args.batch_size,
        "notifications": notifications,
        "disclosure": (
            "FINAL Chip 洗盘专属汇总：只输出/推送 is_washout(stage=='洗盘') 的股票，"
            "判定逻辑移植自筹码尖峰蒸馏引擎 classify_stage；通知失败不阻断 artifact。"
        ),
    }
    (args.output_dir / "final_chip_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "final_chip_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
