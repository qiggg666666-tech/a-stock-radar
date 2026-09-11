#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定海神针独立汇总与单条、非阻断Server酱通知。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """安全地把布尔列转换为bool Series。

    CSV往返之后布尔值可能变成字符串"True"/"False"，也可能被pandas自动推断为
    bool dtype，取决于分片写入时的具体情况；这里统一按字符串比较处理。
    如果某个分片是用旧版核心引擎跑的、还没有这个字段，缺失列一律按False处理，
    不会因为字段不存在而报错或让排序/推送失败。
    """
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.strip().str.lower().isin({"true", "1"})


def observation_mask(frame: pd.DataFrame) -> pd.Series:
    return bool_column(frame, "daily_dinghai_observation")


def send_serverchan(summary: dict[str, object], candidates: pd.DataFrame) -> dict[str, object]:
    sendkey = os.environ.get("SENDKEY", "").strip()
    if not sendkey:
        return {"status": "skipped:sendkey_not_configured"}
    preview = candidates.head(20).copy()
    if not preview.empty:
        preview["div_tag"] = bool_column(preview, "weekly_macd_bottom_div").map(
            {True: "【周线底背离】", False: ""}
        )
    else:
        preview["div_tag"] = pd.Series(dtype=str)
    records = [
        f"{row.code} {row.name}（研究分 {row.dinghai_research_score}）{row.div_tag}".rstrip()
        for row in preview.itertuples(index=False)
    ]
    title = f"定海神针研究｜{summary['status']}｜{summary['candidate_count']}条观察"
    body = "\n".join([
        f"信号日：{summary.get('signal_date', '未记录')}",
        f"有效记录：{summary['valid_record_count']}；错误：{summary['error_count']}；缺片：{summary['missing_shards']}",
        "", "观察记录（最多20条，含周线底背离标记）：", *(records or ["当日无符合透明观察条件的记录。"]), "",
        "仅为基于公开日线的研究观察，不构成买卖建议、收益预测或仓位建议。",
    ])
    return _post_serverchan(sendkey, title, body)


def send_serverchan_universe_unavailable(summary: dict[str, object]) -> dict[str, object]:
    """当天全部分片都因universe不可用(含缓存兜底)而跳过时，单独发一条说明清楚的通知，
    而不是走send_serverchan()走出"0条观察"这种容易被误读成"扫描完成、当天没有
    符合条件的股票"的文案——这两种情况对使用者的含义完全不同，必须分开表达。"""
    sendkey = os.environ.get("SENDKEY", "").strip()
    if not sendkey:
        return {"status": "skipped:sendkey_not_configured"}
    title = f"定海神针研究｜⚠️股票池不可用｜{summary['completed_shard_count']}个分片全部跳过"
    body = "\n".join([
        f"信号日：{summary.get('signal_date', '未记录')}",
        "今日所有分片都因为股票池获取失败(实时源+磁盘缓存都不可用)而跳过，没有执行任何扫描，",
        "不是「当日没有符合条件的股票」，是根本没有跑起来。",
        "", "各分片universe状态：",
        *(f"- 分片{index}：{status}" for index, status in summary.get("shard_universe_statuses", {}).items()),
        "", "建议检查 dinghai_universe.py 的实时数据源和磁盘缓存是否都已失效。",
    ])
    return _post_serverchan(sendkey, title, body)


def _post_serverchan(sendkey: str, title: str, body: str) -> dict[str, object]:
    request = Request(f"https://sctapi.ftqq.com/{sendkey}.send", data=urlencode({"title": title, "desp": body}).encode("utf-8"), method="POST")
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        code = payload.get("code")
        if code == 0:
            return {"status": "sent", "serverchan_code": 0}
        return {"status": f"failed:serverchan_code_{code}", "serverchan_code": code}
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": f"failed:{type(exc).__name__}", "error": str(exc)[:200]}
    except Exception as exc:  # Notification must never fail the research artifact.
        return {"status": f"failed:{type(exc).__name__}", "error": str(exc)[:200]}


def main() -> int:
    parser = argparse.ArgumentParser(description="定海神针独立全市场汇总")
    parser.add_argument("--input-root", type=Path, default=Path("collected"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("dinghai_summary"))
    args = parser.parse_args()
    frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    statuses: dict[str, object] = {}
    missing: list[int] = []
    for index in range(args.shard_count):
        matches = list(args.input_root.rglob(f"shard-{index}/status.json"))
        if not matches:
            missing.append(index); statuses[str(index)] = {"status": "missing"}; continue
        folder = matches[0].parent
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        statuses[str(index)] = status
        raw = safe_csv(folder / "raw_records.csv")
        errors = safe_csv(folder / "errors.csv")
        if not raw.empty:
            frames.append(raw.assign(source_shard=index))
        if not errors.empty:
            error_frames.append(errors.assign(source_shard=index))
    raw_all = pd.concat(frames, ignore_index=True).drop_duplicates("code") if frames else pd.DataFrame()
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=["code", "name", "stage", "error_type", "error_message", "attempts", "source_shard"])
    candidates = raw_all.loc[observation_mask(raw_all)].copy() if not raw_all.empty else pd.DataFrame()
    if not candidates.empty:
        # 排序优先级：周线MACD底背离 > 研究分 > 结构支持数量 > 代码。
        # 布尔列先转换为可靠的bool再排序，避免CSV往返后的字符串/NaN导致排序失真。
        candidates = candidates.assign(_bottom_div=bool_column(candidates, "weekly_macd_bottom_div"))
        candidates = candidates.sort_values(
            ["_bottom_div", "dinghai_research_score", "structure_support_count", "code"],
            ascending=[False, False, False, True],
        ).drop(columns="_bottom_div").head(args.top_n).reset_index(drop=True)
        candidates["research_rank"] = range(1, len(candidates) + 1)
    completed_shards = [index for index, value in statuses.items() if isinstance(value, dict) and value.get("status") not in {None, "missing"}]
    degraded = [index for index, value in statuses.items() if isinstance(value, dict) and value.get("status") not in {"ready", "missing"}]
    # 全部跳过检测：每个分片各自因为universe不可用(含缓存兜底)而提前退出——不是部分
    # 降级，是当天压根没跑起来。跟"degraded"(部分数据+部分错误)要分开表达，不然
    # 用户只看summary的final状态，会把"today完全没数据"误读成"今天数据质量差"。
    skipped_shards = [
        index for index, value in statuses.items()
        if isinstance(value, dict) and str(value.get("status", "")).startswith("skipped:")
    ]
    universe_unavailable_today = bool(completed_shards) and len(skipped_shards) == len(completed_shards)
    shard_universe_statuses = {
        index: value.get("universe_status", value.get("status"))
        for index, value in statuses.items()
        if isinstance(value, dict)
    }
    if not completed_shards:
        final = "unavailable"
    elif missing:
        final = "partial"
    elif universe_unavailable_today:
        final = "unavailable:universe_unavailable"
    elif degraded or not errors_all.empty:
        final = "degraded"
    elif raw_all.empty:
        final = "unavailable"
    else:
        final = "ready"
    signal_dates = sorted({str(value.get("signal_date_requested", "")) for value in statuses.values() if isinstance(value, dict) and value.get("signal_date_requested")})
    summary: dict[str, object] = {
        "schema_version": "dinghai-summary/v2",
        "status": final,
        "signal_date": signal_dates[0] if len(signal_dates) == 1 else None,
        "signal_date_mismatch": signal_dates if len(signal_dates) > 1 else [],
        "missing_shards": missing,
        "degraded_shards": degraded,
        "skipped_shards": skipped_shards,
        "universe_unavailable_today": universe_unavailable_today,
        "shard_universe_statuses": shard_universe_statuses,
        "completed_shard_count": len(completed_shards),
        "valid_record_count": len(raw_all),
        "candidate_count": len(candidates),
        "error_count": len(errors_all),
        "shard_statuses": statuses,
        "notification": {"status": "not_requested"},
        "disclosure": "定海神针独立多周期日线研究观察；不与DistilledQuant核心或520低位首红共享输入输出；不构成买卖建议、收益预测或仓位建议。",
    }
    if args.notify and universe_unavailable_today:
        summary["notification"] = send_serverchan_universe_unavailable(summary)
    elif args.notify and completed_shards:
        summary["notification"] = send_serverchan(summary, candidates)
    elif args.notify:
        summary["notification"] = {"status": "skipped:no_completed_shards"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "dinghai_candidates.csv", index=False, encoding="utf-8-sig")
    errors_all.to_csv(args.output_dir / "dinghai_errors.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "dinghai_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 定海神针独立多周期研究",
        "",
        f"- 状态：`{final}`",
    ]
    if universe_unavailable_today:
        lines.append("- ⚠️ 今日所有分片都因为股票池获取失败(实时源+磁盘缓存都不可用)而跳过，没有执行任何扫描。")
    lines.extend([
        f"- 信号日：`{summary['signal_date'] or '不一致/未记录'}`",
        f"- 观察：{len(candidates)}",
        f"- 有效记录：{len(raw_all)}",
        f"- 错误：{len(errors_all)}",
        f"- 通知：`{summary['notification']['status']}`",
        "",
        "> 仅为公开日线研究观察，不构成买卖建议、收益预测或仓位建议。",
    ])
    (args.output_dir / "dinghai_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
