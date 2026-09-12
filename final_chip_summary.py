#!/usr/bin/env python3
"""FINAL Chip 洗盘 + 首红确认 + 红筹码占优 + 长影线 汇总器。

处理四类信号：
1. is_washout(stage=='洗盘')，按 washout_score 从高到低排序。
2. is_first_red_confirmed（日线首红 is_first_red_daily + 盘中MA5拐头都确认），按盘中MA5斜率从高到低排序。
3. 红筹码占优：is_red_heavy_chip（获利盘/红色筹码占比达到阈值，套牢盘/绿色筹码很轻），
   按获利占比(profit_pct)从高到低排序。取代了原来的"尖峰筹码柱"(is_chip_spike/is_below_spike)专区——
   信息量太大，改成只看红绿筹码整体占比这一个更直观的维度。
4. 长影线：has_long_shadow（当日长上影或长下影），按 max(lower_shadow_to_body, upper_shadow_to_body) 从高到低排序。
四类互不排斥，同一只股票可能同时命中多类。每类各自全部命中分批推送。

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


def _fund_flow_str(row: pd.Series) -> str:
    """主力资金二次确认标签：只在 main_fund_ok=True 时显示，False/缺失(未对该候选调用)都留空，
    跟 PE 标签一样——没数据就不硬凑一句，避免把"未调用"和"调用了但不达标"混为一谈。"""
    ok = row.get("main_fund_ok", None)
    if ok is None or (isinstance(ok, float) and pd.isna(ok)):
        return ""
    if str(ok).strip().lower() not in {"true", "1"}:
        return ""
    try:
        today_wan = float(row.get("main_fund_today_net", 0)) / 10000.0
        days3_wan = float(row.get("main_fund_days3_net", 0)) / 10000.0
        return f"｜主力净流入(今{today_wan:.0f}万/3日{days3_wan:.0f}万)"
    except (TypeError, ValueError):
        return "｜主力资金确认"


def _bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    """安全地把字符串/缺失列统一转换为bool Series。

    CSV往返之后布尔值会变成字符串"True"/"False"；如果某个分片是用旧版
    final_chip_research.py跑的、还没有新字段（is_chip_spike/has_long_shadow等），
    缺失列一律按False处理，不因为字段不存在而报错或漏推。
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.strip().str.lower().eq("true")


def _numeric_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


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
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_first_red_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜收盘{row.get('close', '')}"
            f"｜盘中MA5斜率{row.get('intraday_ma5_slope', '')}"
            f"｜{row.get('intraday_ma5_note', '')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
        )
    return lines


def format_red_heavy_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜获利盘{row.get('profit_pct', '')}%"
            f"｜套牢盘{row.get('green_chip_pct', '')}%"
            f"｜收盘{row.get('close', '')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_long_shadow_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        directions = []
        if bool(row.get("daily_long_lower")):
            directions.append(f"长下影(影/实体{row.get('lower_shadow_to_body', '')})")
        if bool(row.get("daily_long_upper")):
            directions.append(f"长上影(影/实体{row.get('upper_shadow_to_body', '')})")
        direction_str = "、".join(directions) if directions else "长影线"
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{direction_str}"
            f"｜收盘{row.get('close', '')}"
            f"｜当日振幅{row.get('shadow_amplitude_pct', '')}%"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="FINAL Chip 汇总（洗盘 + 首红确认 + 红筹码占优 + 长影线）")
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
            "schema_version": "final-chip-summary/v7-red-heavy",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "skipped:no_completed_shards",
            "completed_shards": [],
            "washout_count": 0,
            "first_red_confirmed_count": 0,
            "red_heavy_count": 0,
            "long_shadow_count": 0,
            "notification": {"status": "skipped", "reason": "no_completed_shards"},
            "disclosure": "无完成分片，不生成研究结论或通知。",
        }
        (args.output_dir / "final_chip_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "final_chip_report.md").write_text(
            "# FINAL Chip 汇总\n\n无完成分片，未生成研究结论或通知。\n", encoding="utf-8"
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

    # ---------- 洗盘阶段(is_washout / stage=='洗盘') ----------
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

    # ---------- 首红确认(is_first_red_confirmed，日线首红+盘中MA5拐头都确认) ----------
    first_red = pd.DataFrame()
    if not records.empty and "is_first_red_confirmed" in records.columns:
        records["is_first_red_confirmed"] = records["is_first_red_confirmed"].astype(str).str.lower().eq("true")
        records["intraday_ma5_slope"] = (
            pd.to_numeric(records["intraday_ma5_slope"], errors="coerce").fillna(0)
            if "intraday_ma5_slope" in records.columns else 0.0
        )
        first_red = records[records["is_first_red_confirmed"] == True].sort_values(
            "intraday_ma5_slope", ascending=False
        )
    first_red.to_csv(args.output_dir / "final_chip_first_red_confirmed.csv", index=False, encoding="utf-8-sig")

    # ---------- 红筹码占优(is_red_heavy_chip)，取代原尖峰筹码柱专区 ----------
    red_heavy = pd.DataFrame()
    if not records.empty:
        red_heavy_mask = _bool_col(records, "is_red_heavy_chip")
        candidate = records[red_heavy_mask].copy()
        if not candidate.empty:
            rank = _numeric_col(candidate, "profit_pct")
            red_heavy = candidate.assign(_rank=rank).sort_values(
                "_rank", ascending=False
            ).drop(columns="_rank")
    red_heavy.to_csv(args.output_dir / "final_chip_red_heavy.csv", index=False, encoding="utf-8-sig")

    # ---------- 长影线(has_long_shadow) ----------
    long_shadow = pd.DataFrame()
    if not records.empty:
        shadow_mask = _bool_col(records, "has_long_shadow")
        candidate = records[shadow_mask].copy()
        if not candidate.empty:
            rank = pd.concat(
                [_numeric_col(candidate, "lower_shadow_to_body"), _numeric_col(candidate, "upper_shadow_to_body")],
                axis=1,
            ).max(axis=1)
            long_shadow = candidate.assign(_shadow_rank=rank).sort_values(
                "_shadow_rank", ascending=False
            ).drop(columns="_shadow_rank")
    long_shadow.to_csv(args.output_dir / "final_chip_long_shadow.csv", index=False, encoding="utf-8-sig")

    washout_lines = format_washout_lines(washout)
    first_red_lines = format_first_red_lines(first_red)
    red_heavy_lines = format_red_heavy_lines(red_heavy)
    long_shadow_lines = format_long_shadow_lines(long_shadow)

    lines = [
        "# FINAL Chip 汇总（洗盘 + 首红确认 + 红筹码占优 + 长影线）",
        "",
        f"- 完成分片：{len(completed)}/{args.shard_total}",
        f"- 有效记录：{len(records)}",
        f"- 错误台账：{len(errors)}",
        f"- 洗盘阶段(is_washout)：{len(washout)}只，按 washout_score 从高到低全部推送",
        f"- 首红确认(is_first_red_confirmed)：{len(first_red)}只，按盘中MA5斜率从高到低全部推送",
        f"- 红筹码占优(is_red_heavy_chip)：{len(red_heavy)}只，按获利占比从高到低全部推送",
        f"- 长影线(has_long_shadow)：{len(long_shadow)}只，按影线/实体比从高到低全部推送",
        f"- 每条消息最多{args.batch_size}只",
        "",
        "## 洗盘专区（按 washout_score 从高到低排序）",
    ]
    lines.extend(washout_lines if washout_lines else ["当日无洗盘阶段记录。"])
    lines.extend(["", "## 首红确认专区（日线首红+盘中MA5拐头，按斜率从高到低排序）"])
    lines.extend(first_red_lines if first_red_lines else ["当日无首红确认记录。"])
    lines.extend(["", "## 红筹码占优专区（获利盘占比达标、套牢盘轻，按获利占比从高到低排序）"])
    lines.extend(red_heavy_lines if red_heavy_lines else ["当日无红筹码占优记录。"])
    lines.extend(["", "## 长影线专区（长上影/长下影，按影线/实体比从高到低排序）"])
    lines.extend(long_shadow_lines if long_shadow_lines else ["当日无长影线记录。"])
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
        "washout": [], "first_red_confirmed": [], "red_heavy": [], "long_shadow": [],
    }
    if args.notify:
        notifications["washout"] = push_batches("洗盘专区", washout_lines)
        notifications["first_red_confirmed"] = push_batches("首红确认专区", first_red_lines)
        notifications["red_heavy"] = push_batches("红筹码占优专区", red_heavy_lines)
        notifications["long_shadow"] = push_batches("长影线专区", long_shadow_lines)
    else:
        notifications["washout"] = [{"status": "not_requested"}]
        notifications["first_red_confirmed"] = [{"status": "not_requested"}]
        notifications["red_heavy"] = [{"status": "not_requested"}]
        notifications["long_shadow"] = [{"status": "not_requested"}]

    result = {
        "schema_version": "final-chip-summary/v7-red-heavy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ready" if len(completed) == args.shard_total else "partial",
        "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed),
        "record_count": len(records),
        "error_count": len(errors),
        "washout_count": len(washout),
        "first_red_confirmed_count": len(first_red),
        "red_heavy_count": len(red_heavy),
        "long_shadow_count": len(long_shadow),
        "batch_size": args.batch_size,
        "notifications": notifications,
        "disclosure": (
            "FINAL Chip 汇总：输出/推送 is_washout(stage=='洗盘')、is_first_red_confirmed"
            "（日线首红+盘中MA5拐头都确认）、红筹码占优(is_red_heavy_chip，获利盘占优/套牢盘轻)、"
            "长影线(has_long_shadow) 四类股票，互不排斥；通知失败不阻断 artifact。"
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
