#!/usr/bin/env python3
"""FINAL Chip 专属汇总器（新逻辑 + 高优先级综合榜）。

推送内容：
1. 高优先级综合榜（筹码+均线+PE+周线粘合 综合信号）
2. 现价下方长红柱 · 买入 / 洗盘
3. 宽幅堆积区 · 买入 / 洗盘

与 final_chip_research 对齐的参数约定：
- 周线5/10 即将粘合：间距 ≤ 6%；已粘合 ≤ 3%
- 筹码集中度确认：conc90 ≤ 30%
- total_score 含 glue（周线粘合分，权重 0.10）

兼容字段：pe_ttm / pe_score / total_score / weekly_ma_* / signal / ma_signal 等。
分批推送，避免单条 Server酱 消息过长被截断。

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


def _total_str(row: pd.Series) -> str:
    if "total_score" not in row or pd.isna(row.get("total_score")):
        return ""
    try:
        return f"｜综合{float(row['total_score']):.1f}"
    except (TypeError, ValueError):
        return ""


def is_high_priority_row(row: pd.Series) -> bool:
    """与扫描器一致的高优先级判定。"""
    sig = str(row.get("signal", "") or "")
    ma_sig = str(row.get("ma_signal", "") or "")
    try:
        score = float(row.get("total_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0

    def _bool(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    if sig in ("高潜力·筹码+均线共振", "可交易·接近尖峰+趋势确认"):
        return True
    if ma_sig.startswith("强多") and score >= 70:
        return True
    if _bool(row.get("is_tradeable")) and score >= 65:
        return True
    below_state = str(row.get("below_state", "") or "")
    if _bool(row.get("is_below_spike")) and below_state.startswith("买入"):
        return True
    wide_state = str(row.get("wide_state", "") or "")
    if _bool(row.get("is_wide_zone")) and wide_state.startswith("买入"):
        return True
    # 周线5/10即将粘合（或已粘合）+ 综合分门槛
    if _bool(row.get("weekly_ma_glue_near")) and score >= 60:
        return True
    if _bool(row.get("weekly_ma_glue")) and score >= 55:
        return True
    return False


def format_high_priority_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        glue = str(row.get("weekly_ma_glue_signal", "") or "")
        glue_part = f"｜{glue}" if glue and glue != "无" else ""
        gap = row.get("weekly_ma_gap_pct", None)
        gap_part = ""
        if gap is not None and not (isinstance(gap, float) and pd.isna(gap)):
            try:
                gap_part = f"｜周距{float(gap):.2f}%"
            except (TypeError, ValueError):
                pass
        gscore = row.get("weekly_ma_glue_score", None)
        gscore_part = ""
        if gscore is not None and not (isinstance(gscore, float) and pd.isna(gscore)):
            try:
                gscore_part = f"｜粘合分{float(gscore):.1f}"
            except (TypeError, ValueError):
                pass
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"{_total_str(row)}"
            f"｜收盘{row.get('close', '')}"
            f"{_pe_str(row)}"
            f"｜{row.get('signal', '') or '无'}"
            f"｜{row.get('ma_signal', '') or '无'}"
            f"{glue_part}{gap_part}{gscore_part}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"｜获利{row.get('profit_pct', '')}%"
        )
    return lines


def format_below_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜评分{row.get('below_score', '')}"
            f"{_total_str(row)}"
            f"｜收盘{row.get('close', '')}"
            f"｜下方峰{row.get('below_peak', '')}"
            f"｜峰距{row.get('below_dist_pct', '')}%"
            f"｜峰带占比{row.get('below_band_ratio_pct', '')}%"
            f"｜占比{row.get('below_peak_ratio_pct', '')}%"
            f"｜峰隙比{row.get('below_peak_gap', '')}"
            f"{_pe_str(row)}"
            f"｜{row.get('below_state', '')}"
        )
    return lines


def format_wide_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜评分{row.get('wide_score', '')}"
            f"{_total_str(row)}"
            f"｜收盘{row.get('close', '')}"
            f"｜区间{row.get('wide_zone_low', '')}~{row.get('wide_zone_high', '')}"
            f"｜宽度{row.get('wide_width_pct', '')}%"
            f"｜占比{row.get('wide_zone_ratio_pct', '')}%"
            f"｜距上沿{row.get('wide_dist_pct', '')}%"
            f"{_pe_str(row)}"
            f"｜{row.get('wide_state', '')}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="FINAL Chip 汇总（高优先级 + 下方长红柱 + 宽幅区）")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=40,
        help="每条 Server酱 消息最多放多少只",
    )
    parser.add_argument(
        "--high-min-score", type=float, default=0.0,
        help="高优先级额外最低综合分过滤（默认0=不额外过滤）",
    )
    args = parser.parse_args()

    if args.self_test:
        assert read_csv(Path("/missing.csv")).empty
        # 简易判定自检
        s = pd.Series({
            "signal": "高潜力·筹码+均线共振",
            "ma_signal": "强多·多头排列+金叉+量能确认",
            "total_score": 72,
            "is_tradeable": False,
            "is_below_spike": False,
            "is_wide_zone": False,
            "below_state": "无",
            "wide_state": "无",
        })
        assert is_high_priority_row(s) is True
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

    completed = [
        item for item in statuses
        if item.get("state") in {"completed", "completed_zero_records"}
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not completed:
        result = {
            "schema_version": "final-chip-summary/v4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "skipped:no_completed_shards",
            "completed_shards": [],
            "below_spike_count": 0,
            "high_priority_count": 0,
            "notification": {"status": "skipped", "reason": "no_completed_shards"},
            "disclosure": "无完成分片，不生成研究结论或通知。",
        }
        (args.output_dir / "final_chip_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "final_chip_report.md").write_text(
            "# FINAL Chip汇总\n\n无完成分片，未生成研究结论或通知。\n", encoding="utf-8"
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
    if not records.empty and "total_score" in records.columns:
        records["total_score"] = pd.to_numeric(records["total_score"], errors="coerce").fillna(0)

    # ---------- 高优先级综合榜 ----------
    high_priority = pd.DataFrame()
    if not records.empty:
        mask = records.apply(is_high_priority_row, axis=1)
        high_priority = records[mask].copy()
        if args.high_min_score > 0 and "total_score" in high_priority.columns:
            high_priority = high_priority[high_priority["total_score"] >= args.high_min_score]
        if "total_score" in high_priority.columns:
            high_priority = high_priority.sort_values("total_score", ascending=False)
        else:
            high_priority = high_priority.reset_index(drop=True)
    high_priority.to_csv(
        args.output_dir / "final_chip_high_priority.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 下方长红柱 ----------
    below_spike = pd.DataFrame()
    if not records.empty and "is_below_spike" in records.columns:
        records["is_below_spike"] = records["is_below_spike"].astype(str).str.lower().eq("true")
        if "below_score" in records.columns:
            records["below_score"] = pd.to_numeric(records["below_score"], errors="coerce").fillna(0)
        else:
            records["below_score"] = 0.0
        below_spike = records[records["is_below_spike"] == True].sort_values(
            "below_score", ascending=False
        )
    below_spike.to_csv(args.output_dir / "final_chip_below_spike.csv", index=False, encoding="utf-8-sig")

    # ---------- 宽幅堆积区 ----------
    wide_zone = pd.DataFrame()
    if not records.empty and "is_wide_zone" in records.columns:
        records["is_wide_zone"] = records["is_wide_zone"].astype(str).str.lower().eq("true")
        if "wide_score" in records.columns:
            records["wide_score"] = pd.to_numeric(records["wide_score"], errors="coerce").fillna(0)
        else:
            records["wide_score"] = 0.0
        wide_zone = records[records["is_wide_zone"] == True].sort_values(
            "wide_score", ascending=False
        )
    wide_zone.to_csv(args.output_dir / "final_chip_wide_zone.csv", index=False, encoding="utf-8-sig")

    # 洗盘 / 买入 专区
    below_washout = pd.DataFrame()
    below_buy = pd.DataFrame()
    if not below_spike.empty and "below_state" in below_spike.columns:
        st = below_spike["below_state"].astype(str)
        below_washout = below_spike[st.str.startswith("洗盘")].sort_values(
            "below_score", ascending=False
        )
        below_buy = below_spike[st.str.startswith("买入")].sort_values(
            "below_score", ascending=False
        )
    below_washout.to_csv(
        args.output_dir / "final_chip_below_washout.csv", index=False, encoding="utf-8-sig"
    )
    below_buy.to_csv(
        args.output_dir / "final_chip_below_buy.csv", index=False, encoding="utf-8-sig"
    )

    wide_washout = pd.DataFrame()
    wide_buy = pd.DataFrame()
    if not wide_zone.empty and "wide_state" in wide_zone.columns:
        st = wide_zone["wide_state"].astype(str)
        wide_washout = wide_zone[st.str.startswith("洗盘")].sort_values(
            "wide_score", ascending=False
        )
        wide_buy = wide_zone[st.str.startswith("买入")].sort_values(
            "wide_score", ascending=False
        )
    wide_washout.to_csv(
        args.output_dir / "final_chip_wide_washout.csv", index=False, encoding="utf-8-sig"
    )
    wide_buy.to_csv(
        args.output_dir / "final_chip_wide_buy.csv", index=False, encoding="utf-8-sig"
    )

    # 格式化文案
    high_lines = format_high_priority_lines(high_priority)
    below_lines = format_below_lines(below_spike)
    wide_lines = format_wide_lines(wide_zone)
    below_washout_lines = format_below_lines(below_washout)
    wide_washout_lines = format_wide_lines(wide_washout)
    below_buy_lines = format_below_lines(below_buy)
    wide_buy_lines = format_wide_lines(wide_buy)

    lines = [
        "# FINAL Chip 汇总（高优先级综合榜 + 下方长红柱 + 宽幅堆积区）",
        "",
        f"- 完成分片：{len(completed)}/{args.shard_total}",
        f"- 有效记录：{len(records)}",
        f"- 错误台账：{len(errors)}",
        f"- 高优先级综合榜：{len(high_priority)}只（按 total_score 降序推送）",
        f"- 现价下方长红柱：{len(below_spike)}只（买入{len(below_buy)}只，洗盘{len(below_washout)}只）",
        f"- 宽幅堆积区：{len(wide_zone)}只（买入{len(wide_buy)}只，洗盘{len(wide_washout)}只）",
        f"- 每条消息最多{args.batch_size}只；兼容 pe_ttm / total_score / weekly_ma_*",
        "- 参数约定：周线即将粘合≤6%、已粘合≤3%；筹码集中度≤30%；total 含 glue 权重0.10",
        "",
        "## 高优先级综合榜（按综合分从高到低，推送）",
    ]
    lines.extend(high_lines if high_lines else ["当日无高优先级记录。"])
    lines.extend(["", "## 现价下方长红柱·买入（推送）"])
    lines.extend(below_buy_lines if below_buy_lines else ["当日无买入状态记录。"])
    lines.extend(["", "## 现价下方长红柱·洗盘专区（推送）"])
    lines.extend(below_washout_lines if below_washout_lines else ["当日无洗盘状态记录。"])
    lines.extend(["", "## 宽幅堆积区·买入（推送）"])
    lines.extend(wide_buy_lines if wide_buy_lines else ["当日无买入状态记录。"])
    lines.extend(["", "## 宽幅堆积区·洗盘专区（推送）"])
    lines.extend(wide_washout_lines if wide_washout_lines else ["当日无洗盘状态记录。"])
    lines.extend(["", "## 现价下方长红柱（全部，仅供参考，不推送）"])
    lines.extend(below_lines if below_lines else ["当日无命中记录。"])
    lines.extend(["", "## 宽幅堆积区（全部，仅供参考，不推送）"])
    lines.extend(wide_lines if wide_lines else ["当日无命中记录。"])
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

    notifications: dict[str, list[dict[str, object]]] = {
        "high_priority": [],
        "below_buy": [],
        "below_washout": [],
        "wide_buy": [],
        "wide_washout": [],
    }
    if args.notify:
        notifications["high_priority"] = push_batches("高优先级综合榜", high_lines)
        notifications["below_buy"] = push_batches("现价下方长红柱·买入", below_buy_lines)
        notifications["below_washout"] = push_batches("现价下方长红柱·洗盘", below_washout_lines)
        notifications["wide_buy"] = push_batches("宽幅堆积区·买入", wide_buy_lines)
        notifications["wide_washout"] = push_batches("宽幅堆积区·洗盘", wide_washout_lines)
    else:
        for key in notifications:
            notifications[key] = [{"status": "not_requested"}]

    result = {
        "schema_version": "final-chip-summary/v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ready" if len(completed) == args.shard_total else "partial",
        "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed),
        "record_count": len(records),
        "error_count": len(errors),
        "high_priority_count": len(high_priority),
        "below_spike_count": len(below_spike),
        "wide_zone_count": len(wide_zone),
        "below_buy_count": len(below_buy),
        "below_washout_count": len(below_washout),
        "wide_buy_count": len(wide_buy),
        "wide_washout_count": len(wide_washout),
        "batch_size": args.batch_size,
        "high_min_score": args.high_min_score,
        "params_note": {
            "weekly_ma_glue_near_pct": 0.06,
            "weekly_ma_glue_gap_pct": 0.03,
            "conc_threshold": 0.30,
            "glue_weight_in_total": 0.10,
        },
        "notifications": notifications,
        "disclosure": (
            "FINAL Chip 汇总 v4：高优先级综合榜 + 下方长红柱 + 宽幅区；"
            "周线即将粘合≤6%、集中度≤30%、total 含 glue；"
            "推送高优先级/买入/洗盘；兼容 pe_ttm/total_score/weekly_ma_*；通知失败不阻断 artifact。"
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
