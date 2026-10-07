#!/usr/bin/env python3
"""FINAL Chip 洗盘 + 首红确认 + 红筹码占优∩宽幅堆积区 + 长影线 + 均线强多 汇总器。

处理信号专区（红筹码占优与宽幅堆积区只推送交集；变盘三态单独推送）：
1. is_washout(stage=='洗盘')，按 washout_score 从高到低排序。
2. is_first_red_confirmed（日线首红 + 盘中MA5拐头），按盘中MA5斜率从高到低排序。
3. 红筹码占优 ∩ 宽幅堆积区：is_red_heavy_chip 且 is_wide_zone，按 wide_score 排序。
4. 长影线·变盘共振：daily_long_lower 且 (near_bk200 或 low_bullish_outside)，按下影/实体比排序后推送。
   仅有长下影、没有命中拐点/低位破低反包的股票不会推送（只写入 long_shadow_all.csv 供研究）。
   如果这个专区仍然偏多，可再叠加：--long-shadow-need-fund（要求主力资金确认）、
   --long-shadow-min-vr（量比下限）、--long-shadow-top（只推前 N 只，按下影/实体比排序）。
5. 拐点(near_bk200)：底部放量突破200日线，按 ret_12m 升序（偏低优先）再 volume_ratio。
6. 低位破低反包(low_bullish_outside)：研究观察，按 |turn_dd60| 与 turn_vr 排序。
7. 深跌巨量强收(deep_huge_strong)：研究观察，按 turn_vr 与 |turn_dd60| 排序。
8. 均线强多：ma_signal以"强多"开头且未落入以上任一专区，按 ma_score 排序。
变盘三类为观察推送（参考 Hanai：高原始频率≠独立 alpha），不构成买卖建议。
每类各自全部命中分批推送。

推送说明：
  * Server酱标题有长度上限（32字），本脚本会把标题压到 32 字以内，避免被截断/拒绝。
  * 默认每个专区即使 0 只也推一条“当日无命中”；如担心每日推送条数上限，可加 --skip-empty 只推有命中的专区。
  * 连续推送之间默认间隔 1 秒（--push-interval），降低被限流概率；失败会重试一次。

用法：
  python final_chip_summary.py --input-dir ./shards --output-dir ./out --notify
  python final_chip_summary.py --input-dir ./shards --output-dir ./out --notify --skip-empty
  python final_chip_summary.py --input-dir ./shards --output-dir ./out --notify \
      --long-shadow-need-fund --long-shadow-min-vr 1.5 --long-shadow-top 15
  python final_chip_summary.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TITLE_MAX_CHARS = 32  # Server酱标题长度上限


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() and path.stat().st_size else pd.DataFrame()


def notify(title: str, body: str, retries: int = 2) -> dict[str, object]:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "missing_sendkey"}
    title = title[:TITLE_MAX_CHARS]
    last: dict[str, object] = {}
    for attempt in range(1, max(retries, 1) + 1):
        try:
            import requests
            response = requests.post(
                f"https://sctapi.ftqq.com/{key}.send",
                data={"title": title, "desp": body},
                timeout=20,
            )
            result = response.json()
            if response.ok and result.get("code") == 0:
                return {"status": "sent", "http_status": response.status_code}
            last = {
                "status": "failed",
                "http_status": response.status_code,
                "code": result.get("code"),
                "message": str(result.get("message", ""))[:120],
            }
        except Exception as exc:
            last = {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:250]}"}
        if attempt < max(retries, 1):
            time.sleep(2)
    return last


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


def _boll_macd_str(row: pd.Series) -> str:
    """布林带 + MACD 简短标签，缺字段时静默跳过。"""
    parts: list[str] = []
    mid = row.get("boll_mid", None)
    upper = row.get("boll_upper", None)
    lower = row.get("boll_lower", None)
    if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
        try:
            parts.append(f"布林{float(lower):.2f}/{float(mid):.2f}/{float(upper):.2f}")
        except (TypeError, ValueError):
            pass
    pos = row.get("boll_pos", None)
    if pos is not None and not (isinstance(pos, float) and pd.isna(pos)):
        try:
            parts.append(f"轨位{float(pos):.2f}")
        except (TypeError, ValueError):
            pass
    if str(row.get("boll_squeeze", "")).strip().lower() in {"true", "1"}:
        parts.append("布林收窄")
    dif = row.get("macd_dif", None)
    dea = row.get("macd_dea", None)
    hist = row.get("macd_hist", None)
    if dif is not None and not (isinstance(dif, float) and pd.isna(dif)):
        try:
            parts.append(f"MACD{float(dif):.3f}/{float(dea):.3f}/{float(hist):.3f}")
        except (TypeError, ValueError):
            pass
    tags = []
    if str(row.get("macd_golden_cross", "")).strip().lower() in {"true", "1"}:
        tags.append("金叉")
    if str(row.get("macd_dif_above_zero", "")).strip().lower() in {"true", "1"}:
        tags.append("零上")
    if str(row.get("macd_hist_expanding", "")).strip().lower() in {"true", "1"}:
        tags.append("柱放大")
    if tags:
        parts.append("MACD" + "+".join(tags))
    return ("｜" + "｜".join(parts)) if parts else ""


def format_red_heavy_wide_zone_lines(df: pd.DataFrame) -> list[str]:
    """红筹码占优 ∩ 宽幅堆积区 同时命中；附加布林带与 MACD 摘要。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜获利盘{row.get('profit_pct', '')}%"
            f"｜套牢盘{row.get('green_chip_pct', '')}%"
            f"｜{row.get('wide_state', '') or '无'}"
            f"｜宽幅区{row.get('wide_zone_low', '')}~{row.get('wide_zone_high', '')}"
            f"｜距上沿{row.get('wide_dist_pct', '')}%"
            f"｜宽幅分{row.get('wide_score', '')}"
            f"｜收盘{row.get('close', '')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_boll_macd_str(row)}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_long_shadow_lines(df: pd.DataFrame) -> list[str]:
    """长影线·变盘共振：必须 daily_long_lower，且同时命中拐点或低位破低反包。
    长上影仅作附加标注，不单独入选。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        direction_str = f"长下影买入(影/实体{row.get('lower_shadow_to_body', '')})"
        if str(row.get("daily_long_upper", "")).strip().lower() in {"true", "1"}:
            direction_str += f"、同时有长上影(影/实体{row.get('upper_shadow_to_body', '')})"
        tags = []
        if str(row.get("near_bk200", "")).strip().lower() in {"true", "1"}:
            tags.append(row.get("bk200_label") or "拐点")
        if str(row.get("low_bullish_outside", "")).strip().lower() in {"true", "1"}:
            tags.append(row.get("outside_label") or "低位破低反包")
        tag_str = ("｜" + "+".join(str(x) for x in tags)) if tags else ""
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{direction_str}{tag_str}"
            f"｜收盘{row.get('close', '')}"
            f"｜当日振幅{row.get('shadow_amplitude_pct', '')}%"
            f"｜回撤{row.get('turn_dd60', '-')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_bk200_lines(df: pd.DataFrame) -> list[str]:
    """拐点：底部放量突破200日线。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        ret = row.get("ret_12m", "")
        try:
            ret_s = f"{float(ret)*100:.1f}%" if ret != "" and ret is not None and str(ret) not in ("nan", "None") else "-"
        except (TypeError, ValueError):
            ret_s = "-"
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{row.get('bk200_label', '') or '拐点·底部放量突破200日线'}"
            f"｜收盘{row.get('close', '')}"
            f"｜MA200={row.get('ma200', '-')}"
            f"｜近12月{ret_s}"
            f"｜回撤{row.get('turn_dd60', '-')}"
            f"｜量比{row.get('volume_ratio', '')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_outside_lines(df: pd.DataFrame) -> list[str]:
    """低位破低反包（观察）。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{row.get('outside_label', '') or '低位破低反包'}"
            f"｜收盘{row.get('close', '')}"
            f"｜回撤{row.get('turn_dd60', '-')}"
            f"｜量比VR{row.get('turn_vr', row.get('volume_ratio', ''))}"
            f"｜CLV{row.get('turn_clv', '-')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
            f"{_pe_str(row)}"
            f"{_fund_flow_str(row)}"
        )
    return lines


def format_deep_strong_lines(df: pd.DataFrame) -> list[str]:
    """深跌巨量强收（观察）。"""
    lines = []
    for number, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"{number:03d}. {row.get('code', '')} {row.get('name', '')}"
            f"｜{row.get('deep_strong_label', '') or '深跌巨量强收'}"
            f"｜收盘{row.get('close', '')}"
            f"｜回撤{row.get('turn_dd60', '-')}"
            f"｜量比VR{row.get('turn_vr', row.get('volume_ratio', ''))}"
            f"｜CLV{row.get('turn_clv', '-')}"
            f"｜均线{row.get('ma_signal', '') or '无'}"
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
    parser.add_argument(
        "--skip-empty", action="store_true",
        help="0 只命中的专区不推送“当日无命中”（默认每个专区都推一条，8 个专区每天至少 8 条）",
    )
    parser.add_argument(
        "--long-shadow-need-fund", action="store_true",
        help="长影线·变盘共振专区：额外要求主力资金二次确认通过（main_fund_ok=True）",
    )
    parser.add_argument(
        "--long-shadow-min-vr", type=float, default=0.0,
        help="长影线·变盘共振专区：额外要求量比 >= 该值（0=不限制）",
    )
    parser.add_argument(
        "--long-shadow-top", type=int, default=0,
        help="长影线·变盘共振专区：只推前 N 只（按下影/实体比从高到低；0=不限制）。完整名单仍写入 CSV",
    )
    parser.add_argument(
        "--push-interval", type=float, default=1.0,
        help="连续两次推送之间等待的秒数，降低被限流概率（默认1秒）",
    )
    args = parser.parse_args()

    if args.self_test:
        assert read_csv(Path("/missing.csv")).empty
        assert len(f"FINAL｜{'红筹码占优∩宽幅堆积区专区'} 12/12（共120只）") <= TITLE_MAX_CHARS
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
            "bk200_count": 0,
            "low_outside_count": 0,
            "deep_strong_count": 0,
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
    if not errors.empty and "code" in errors.columns:
        errors["code"] = errors["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    errors.to_csv(args.output_dir / "final_chip_errors.csv", index=False, encoding="utf-8-sig")

    if not records.empty and "code" in records.columns:
        records["code"] = records["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

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

    # ---------- 长影线·变盘共振：长下影 ∩ (拐点 或 低位破低反包) ----------
    # 全量长下影仍落盘便于研究；推送/专区只保留与变盘信号的交集。
    long_shadow_all = pd.DataFrame()
    long_shadow_turn = pd.DataFrame()  # 长下影 ∩ (拐点|破低反包)，未叠加额外过滤
    long_shadow = pd.DataFrame()       # 实际推送：在 long_shadow_turn 上再叠加可选过滤
    if not records.empty:
        shadow_mask = _bool_col(records, "daily_long_lower")
        all_cand = records[shadow_mask].copy()
        if not all_cand.empty:
            rank = _numeric_col(all_cand, "lower_shadow_to_body")
            long_shadow_all = all_cand.assign(_shadow_rank=rank).sort_values(
                "_shadow_rank", ascending=False
            ).drop(columns="_shadow_rank")
            turn_mask = (
                _bool_col(long_shadow_all, "near_bk200")
                | _bool_col(long_shadow_all, "low_bullish_outside")
            )
            long_shadow_turn = long_shadow_all[turn_mask].copy()
            long_shadow = long_shadow_turn
            if args.long_shadow_need_fund:
                long_shadow = long_shadow[_bool_col(long_shadow, "main_fund_ok")]
            if args.long_shadow_min_vr > 0:
                long_shadow = long_shadow[_numeric_col(long_shadow, "volume_ratio") >= args.long_shadow_min_vr]
            if args.long_shadow_top > 0:
                long_shadow = long_shadow.head(args.long_shadow_top)  # 已按下影/实体比降序
            long_shadow = long_shadow.copy()
    long_shadow_all.to_csv(
        args.output_dir / "final_chip_long_shadow_all.csv", index=False, encoding="utf-8-sig"
    )
    long_shadow_turn.to_csv(
        args.output_dir / "final_chip_long_shadow_turn.csv", index=False, encoding="utf-8-sig"
    )
    long_shadow.to_csv(
        args.output_dir / "final_chip_long_shadow.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 拐点 near_bk200 ----------
    bk200 = pd.DataFrame()
    if not records.empty:
        m = _bool_col(records, "near_bk200")
        candidate = records[m].copy()
        if not candidate.empty:
            candidate = candidate.assign(
                _r12=_numeric_col(candidate, "ret_12m"),
                _vr=_numeric_col(candidate, "volume_ratio"),
            ).sort_values(["_r12", "_vr"], ascending=[True, False]).drop(columns=["_r12", "_vr"])
            bk200 = candidate
    bk200.to_csv(args.output_dir / "final_chip_bk200.csv", index=False, encoding="utf-8-sig")

    # ---------- 低位破低反包 ----------
    outside = pd.DataFrame()
    if not records.empty:
        m = _bool_col(records, "low_bullish_outside")
        candidate = records[m].copy()
        if not candidate.empty:
            candidate = candidate.assign(
                _dd=_numeric_col(candidate, "turn_dd60").abs(),
                _vr=_numeric_col(candidate, "turn_vr"),
            ).sort_values(["_dd", "_vr"], ascending=[False, False]).drop(columns=["_dd", "_vr"])
            outside = candidate
    outside.to_csv(args.output_dir / "final_chip_low_outside.csv", index=False, encoding="utf-8-sig")

    # ---------- 深跌巨量强收 ----------
    deep_strong = pd.DataFrame()
    if not records.empty:
        m = _bool_col(records, "deep_huge_strong")
        candidate = records[m].copy()
        if not candidate.empty:
            candidate = candidate.assign(
                _vr=_numeric_col(candidate, "turn_vr"),
                _dd=_numeric_col(candidate, "turn_dd60").abs(),
            ).sort_values(["_vr", "_dd"], ascending=[False, False]).drop(columns=["_vr", "_dd"])
            deep_strong = candidate
    deep_strong.to_csv(args.output_dir / "final_chip_deep_strong.csv", index=False, encoding="utf-8-sig")

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
            | _bool_col(records, "near_bk200")
            | _bool_col(records, "low_bullish_outside")
            | _bool_col(records, "deep_huge_strong")
        )
        candidate = records[ma_strong_signal & ~already_covered].copy()
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
    bk200_lines = format_bk200_lines(bk200)
    outside_lines = format_outside_lines(outside)
    deep_strong_lines = format_deep_strong_lines(deep_strong)
    ma_strong_lines = format_ma_strong_lines(ma_strong)

    lines = [
        "# FINAL Chip 汇总（洗盘 + 首红 + 红筹码∩宽幅 + 长影线 + 拐点 + 破低反包 + 深跌巨量 + 均线强多）",
        "",
        f"- 完成分片：{len(completed)}/{args.shard_total}",
        f"- 有效记录：{len(records)}",
        f"- 错误台账：{len(errors)}",
        f"- 洗盘阶段(is_washout)：{len(washout)}只，按 washout_score 从高到低全部推送",
        f"- 首红确认(is_first_red_confirmed)：{len(first_red)}只，按盘中MA5斜率从高到低全部推送",
        f"- 红筹码占优∩宽幅堆积区(is_red_heavy_chip 且 is_wide_zone)：{len(red_heavy_wide)}只，按 wide_score 从高到低全部推送",
        f"- 长影线·变盘共振(长下影∩(拐点|破低反包))：{len(long_shadow)}只推送（长下影∩变盘{len(long_shadow_turn)}只；全量长下影{len(long_shadow_all)}只已写入 long_shadow_all.csv）",
        f"- 拐点(near_bk200)：{len(bk200)}只，观察推送",
        f"- 低位破低反包(low_bullish_outside)：{len(outside)}只，观察推送",
        f"- 深跌巨量强收(deep_huge_strong)：{len(deep_strong)}只，观察推送",
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
    lines.extend(["", "## 长影线·变盘共振专区（长下影 ∩ (拐点 或 低位破低反包)）"])
    lines.extend(
        long_shadow_lines if long_shadow_lines
        else ["当日无「长下影且拐点/破低反包」记录。"]
    )
    lines.extend(["", "## 拐点专区（底部放量突破200日线，观察）"])
    lines.extend(bk200_lines if bk200_lines else ["当日无拐点记录。"])
    lines.extend(["", "## 低位破低反包专区（观察，非已验证买点）"])
    lines.extend(outside_lines if outside_lines else ["当日无低位破低反包记录。"])
    lines.extend(["", "## 深跌巨量强收专区（观察，非已验证买点）"])
    lines.extend(deep_strong_lines if deep_strong_lines else ["当日无深跌巨量强收记录。"])
    lines.extend(["", "## 均线强多专区（均线强多信号，但没有命中以上任一专区，按均线分从高到低排序）"])
    lines.extend(ma_strong_lines if ma_strong_lines else ["当日无均线强多(且未入其它专区)记录。"])
    lines.extend(["", "仅为 FINAL Chip 规则研究输出，不构成投资建议。"])

    push_state = {"count": 0}

    def _send(title: str, body: str) -> dict[str, object]:
        """统一发送：控制推送间隔，避免连发被限流。"""
        if push_state["count"] > 0 and args.push_interval > 0:
            time.sleep(args.push_interval)
        push_state["count"] += 1
        return notify(title, body)

    def push_batches(label: str, row_lines: list[str]) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        batch_size = max(args.batch_size, 1)
        if not row_lines:
            if args.skip_empty:
                return [{"status": "skipped", "reason": "empty_zone"}]
            # 标题 <= 32 字：Server酱标题有长度上限
            title = f"FINAL｜{label} 0只｜{len(completed)}/{args.shard_total}分片"
            body = "\n".join([
                f"# FINAL Chip {label}", "", "当日无命中记录。", "",
                "仅为 FINAL Chip 规则研究输出，不构成投资建议。",
            ])
            results.append(_send(title, body))
            return results
        total_batches = (len(row_lines) + batch_size - 1) // batch_size
        for batch_index in range(total_batches):
            chunk = row_lines[batch_index * batch_size: (batch_index + 1) * batch_size]
            title = f"FINAL｜{label} {batch_index + 1}/{total_batches}（共{len(row_lines)}只）"
            body_lines = (
                [f"# FINAL Chip {label} 批次{batch_index + 1}/{total_batches}", "",
                 f"共{len(row_lines)}只，本批{len(chunk)}只", ""]
                + chunk
                + ["", "仅为 FINAL Chip 规则研究输出，不构成投资建议。"]
            )
            results.append(_send(title, "\n".join(body_lines)))
        return results

    notifications: dict[str, list[dict[str, object]]] = {
        "washout": [],
        "first_red_confirmed": [],
        "red_heavy_wide_zone": [],
        "long_shadow": [],
        "bk200": [],
        "low_outside": [],
        "deep_strong": [],
        "ma_strong": [],
    }
    if args.notify:
        notifications["washout"] = push_batches("洗盘专区", washout_lines)
        notifications["first_red_confirmed"] = push_batches("首红确认专区", first_red_lines)
        notifications["red_heavy_wide_zone"] = push_batches(
            "红筹码占优∩宽幅堆积区专区", red_heavy_wide_lines
        )
        notifications["long_shadow"] = push_batches("长影线·变盘共振专区", long_shadow_lines)
        notifications["bk200"] = push_batches("拐点专区", bk200_lines)
        notifications["low_outside"] = push_batches("低位破低反包专区", outside_lines)
        notifications["deep_strong"] = push_batches("深跌巨量强收专区", deep_strong_lines)
        notifications["ma_strong"] = push_batches("均线强多专区", ma_strong_lines)
    else:
        for k in notifications:
            notifications[k] = [{"status": "not_requested"}]

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
        "bk200_count": len(bk200),
        "low_outside_count": len(outside),
        "deep_strong_count": len(deep_strong),
        "ma_strong_count": len(ma_strong),
        "batch_size": args.batch_size,
        "notifications": notifications,
        "disclosure": (
            "FINAL Chip 汇总：洗盘、首红确认、红筹码∩宽幅、长影线·变盘共振(长下影∩拐点|破低反包)、"
            "拐点(near_bk200)、低位破低反包、深跌巨量强收、均线强多；"
            "变盘三类为研究观察推送（非已验证高胜率买点）；通知失败不阻断 artifact。"
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
