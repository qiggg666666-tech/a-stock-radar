#!/usr/bin/env python3
"""FINAL Chip 洗盘 + 首红确认 + 红筹码占优∩宽幅堆积区 + 长影线 + 均线强多 汇总器。

处理五类信号（红筹码占优与宽幅堆积区不再分开推送，只推送二者同时命中的交集）：
1. is_washout(stage=='洗盘')，按 washout_score 从高到低排序。
2. is_first_red_confirmed（日线首红 is_first_red_daily + 盘中MA5拐头都确认），按盘中MA5斜率从高到低排序。
3. 红筹码占优 ∩ 宽幅堆积区：is_red_heavy_chip 且 is_wide_zone 同时为真，
   按 wide_score 从高到低排序（次级用 profit_pct）。单独只命中其一的股票不再进入任何专区推送。
4. 长影线：daily_long_lower（当日长下影，探底回升的买入型态；长上影冲高回落是卖出/压力信号，
   不再进这个专区），按lower_shadow_to_body从高到低排序。
5. 均线强多：ma_signal以"强多"开头、但没有命中以上任一专区的股票，按ma_score从高到低排序。
五类互不排斥（除了均线强多本身定义为"不与前四类重叠"），同一只股票可能同时命中多类。
每类各自全部命中分批推送。

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
            f"｜90%成本区{row.get('cost90_low', '')}\~{row.get('cost90_high', '')}"
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


def format_red_heavy_wide_zone_lines(df: pd.DataFrame) -> list[str]:
    """红筹码占优 ∩ 宽幅堆积区 同时命中。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜获利盘{row.get('profit_pct', '')}%"
            f"｜套牢盘{row.get('green_chip_pct', '')}%"
            f"｜{row.get('wide_state', '') or '无'}"
            f"｜宽幅区{row.get('wide_zone_low', '')}\~{row.get('wide_zone_high', '')}"
            f"｜距上沿{row.get('wide_dist_pct', '')}%"
            f"｜宽幅分{row.get('wide_score', '')}"
            f"｜收盘{row.get('close', '')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_long_shadow_lines(df: pd.DataFrame) -> list[str]:
    """现在这个专区只收daily_long_lower==True的(长下影买入型态)，长上影不再进来。
    极少数情况下当天同时满足长下影和长上影，顺带把长上影也标出来，但入选原因
    永远是长下影。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        direction_str = f"长下影买入(影/实体{row.get('lower_shadow_to_body', '')})"
        if bool(row.get("daily_long_upper")):
            direction_str += f"、同时有长上影(影/实体{row.get('upper_shadow_to_body', '')})"
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


def format_ma_strong_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{row.get('ma_signal', '')}"
            f"｜均线分{row.get('ma_score', '')}"
            f"｜收盘{row.get('close', '')}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FINAL Chip 汇总（洗盘 + 首红确认 + 红筹码占优∩宽幅堆积区 + 长影线 + 均线强多）"
    )
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
            "schema_version": "final-chip-summary/v10-red-wide-intersect",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "skipped:no_completed_shards",
            "completed_shards": [],
            "washout_count": 0,
            "first_red_confirmed_count": 0,
            "red_heavy_wide_zone_count": 0,
            "long_shadow_count": 0,
            "ma_strong_count": 0,
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

    # ---------- 红筹码占优 ∩ 宽幅堆积区（只推送同时命中，不再分开推送） ----------
    red_heavy_wide = pd.DataFrame()
    if not records.empty:
        red_mask = _bool_col(records, "is_red_heavy_chip")
        wide_mask = _bool_col(records, "is_wide_zone")
        candidate = records[red_mask & wide_mask].copy()
        if not candidate.empty:
            # 主排序 wide_score，次级 profit_pct
            candidate = candidate.assign(
                _wide_rank=_numeric_col(candidate, "wide_score"),
                _profit_rank=_numeric_col(candidate, "profit_pct"),
            ).sort_values(
                ["_wide_rank", "_profit_rank"], ascending=[False, False]
            ).drop(columns=["_wide_rank", "_profit_rank"])
            red_heavy_wide = candidate
    red_heavy_wide.to_csv(
        args.output_dir / "final_chip_red_heavy_wide_zone.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 长影线(daily_long_lower，长下影买入型态) ----------
    long_shadow = pd.DataFrame()
    if not records.empty:
        shadow_mask = _bool_col(records, "daily_long_lower")
        candidate = records[shadow_mask].copy()
        if not candidate.empty:
            rank = _numeric_col(candidate, "lower_shadow_to_body")
            long_shadow = candidate.assign(_shadow_rank=rank).sort_values(
                "_shadow_rank", ascending=False
            ).drop(columns="_shadow_rank")
    long_shadow.to_csv(args.output_dir / "final_chip_long_shadow.csv", index=False, encoding="utf-8-sig")

    # ---------- 均线强多，但没有落进上面任何一个专区 ----------
    # 排除口径与改前一致：单独命中 is_red_heavy_chip 或 is_wide_zone 也算已覆盖，
    # 不进入均线强多；仅推送层面把红筹码与宽幅改为只推交集，其它逻辑不变。
    ma_strong = pd.DataFrame()
    if not records.empty:
        ma_strong_signal = records.get("ma_signal", pd.Series("", index=records.index)).astype(str).str.startswith("强多")
        already_covered = (
            _bool_col(records, "is_washout")
            | _bool_col(records, "is_first_red_confirmed")
            | _bool_col(records, "is_red_heavy_chip")
            | _bool_col(records, "daily_long_lower")
            | _bool_col(records, "is_wide_zone")
        )
        candidate = records[ma_strong_signal & \~already_covered].copy()
        if not candidate.empty:
            rank = _numeric_col(candidate, "ma_score")
            ma_strong = candidate.assign(_rank=rank).sort_values(
                "_rank", ascending=False
            ).drop(columns="_rank")
    ma_strong.to_csv(args.output_dir / "final_chip_ma_strong.csv", index=False, encoding="utf-8-sig")

    washout_lines = format_washout_lines(washout)
    first_red_lines = format_first_red_lines(first_red)
    red_heavy_wide_lines = format_red_heavy_wide_zone_lines(red_heavy_wide)
    long_shadow_lines = format_long_shadow_lines(long_shadow)
    ma_strong_lines = format_ma_strong_lines(ma_strong)

    lines = [
        "# FINAL Chip 汇总（洗盘 + 首红确认 + 红筹码占优∩宽幅堆积区 + 长影线 + 均线强多）",
        "",
        f"- 完成分片：{len(completed)}/{args.shard_total}",
        f"- 有效记录：{len(records)}",
        f"- 错误台账：{len(errors)}",
        f"- 洗盘阶段(is_washout)：{len(washout)}只，按 washout_score 从高到低全部推送",
        f"- 首红确认(is_first_red_confirmed)：{len(first_red)}只，按盘中MA5斜率从高到低全部推送",
        f"- 红筹码占优∩宽幅堆积区(is_red_heavy_chip 且 is_wide_zone)：{len(red_heavy_wide)}只，按 wide_score 从高到低全部推送",
        f"- 长影线(daily_long_lower买入型态)：{len(long_shadow)}只，按下影/实体比从高到低全部推送",
        f"- 均线强多(未落入以上任一专区)：{len(ma_strong)}只，按均线分从高到低全部推送",
        f"- 每条消息最多{args.batch_size}只",
        "",
        "说明：红筹码占优与宽幅堆积区不再分开推送，仅推送二者同时命中的股票。",
        "",
        "## 洗盘专区（按 washout_score 从高到低排序）",
    ]
    lines.extend(washout_lines if washout_lines else ["当日无洗盘阶段记录。"])
    lines.extend(["", "## 首红确认专区（日线首红+盘中MA5拐头，按斜率从高到低排序）"])
    lines.extend(first_red_lines if first_red_lines else ["当日无首红确认记录。"])
    lines.extend([
        "",
        "## 红筹码占优∩宽幅堆积区专区（二者同时命中，按 wide_score 从高到低排序）",
    ])
    lines.extend(
        red_heavy_wide_lines if red_heavy_wide_lines else ["当日无红筹码占优且宽幅堆积区同时命中记录。"]
    )
    lines.extend(["", "## 长影线专区（长下影买入型态，按下影/实体比从高到低排序）"])
    lines.extend(long_shadow_lines if long_shadow_lines else ["当日无长影线记录。"])
    lines.extend(["", "## 均线强多专区（均线强多信号，但没有命中以上任一专区，按均线分从高到低排序）"])
    lines.extend(ma_strong_lines if ma_strong_lines else ["当日无均线强多(且未入其它专区)记录。"])
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
        "washout": [],
        "first_red_confirmed": [],
        "red_heavy_wide_zone": [],
        "long_shadow": [],
        "ma_strong": [],
    }
    if args.notify:
        notifications["washout"] = push_batches("洗盘专区", washout_lines)
        notifications["first_red_confirmed"] = push_batches("首红确认专区", first_red_lines)
        notifications["red_heavy_wide_zone"] = push_batches(
            "红筹码占优∩宽幅堆积区专区", red_heavy_wide_lines
        )
        notifications["long_shadow"] = push_batches("长影线专区", long_shadow_lines)
        notifications["ma_strong"] = push_batches("均线强多专区", ma_strong_lines)
    else:
        notifications["washout"] = [{"status": "not_requested"}]
        notifications["first_red_confirmed"] = [{"status": "not_requested"}]
        notifications["red_heavy_wide_zone"] = [{"status": "not_requested"}]
        notifications["long_shadow"] = [{"status": "not_requested"}]
        notifications["ma_strong"] = [{"status": "not_requested"}]

    result = {
        "schema_version": "final-chip-summary/v10-red-wide-intersect",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ready" if len(completed) == args.shard_total else "partial",
        "completed_shards": sorted(int(item.get("shard_index", -1)) for item in completed),
        "record_count": len(records),
        "error_count": len(errors),
        "washout_count": len(washout),
        "first_red_confirmed_count": len(first_red),
        "red_heavy_wide_zone_count": len(red_heavy_wide),
        "long_shadow_count": len(long_shadow),
        "ma_strong_count": len(ma_strong),
        "batch_size": args.batch_size,
        "notifications": notifications,
        "disclosure": (
            "FINAL Chip 汇总：输出/推送 is_washout(stage=='洗盘')、is_first_red_confirmed"
            "（日线首红+盘中MA5拐头都确认）、红筹码占优∩宽幅堆积区(is_red_heavy_chip 且 is_wide_zone，"
            "不再分开推送单独命中其一的股票)、长影线(daily_long_lower，长下影买入型态)、"
            "均线强多(ma_signal以'强多'开头且未落入以上任一专区) 五类股票；"
            "通知失败不阻断 artifact。"
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
