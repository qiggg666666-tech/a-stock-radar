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


def observation_mask(frame: pd.DataFrame) -> pd.Series:
    if "daily_dinghai_observation" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["daily_dinghai_observation"].astype(str).str.strip().str.lower().isin({"true", "1"})


def send_serverchan(summary: dict[str, object], candidates: pd.DataFrame) -> dict[str, object]:
    sendkey = os.environ.get("SENDKEY", "").strip()
    if not sendkey:
        return {"status": "skipped:sendkey_not_configured"}
    preview = candidates.head(20)
    records = [f"{row.code} {row.name}（研究分 {row.dinghai_research_score}）" for row in preview.itertuples(index=False)]
    title = f"定海神针研究｜{summary['status']}｜{summary['candidate_count']}条观察"
    body = "\n".join([
        f"信号日：{summary.get('signal_date', '未记录')}",
        f"有效记录：{summary['valid_record_count']}；错误：{summary['error_count']}；缺片：{summary['missing_shards']}",
        "", "观察记录（最多20条）：", *(records or ["当日无符合透明观察条件的记录。"]), "",
        "仅为基于公开日线的研究观察，不构成买卖建议、收益预测或仓位建议。",
    ])
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
        candidates = candidates.sort_values(["dinghai_research_score", "structure_support_count", "code"], ascending=[False, False, True]).head(args.top_n).reset_index(drop=True)
        candidates["research_rank"] = range(1, len(candidates) + 1)
    degraded = [index for index, value in statuses.items() if isinstance(value, dict) and value.get("status") != "ready"]
    final = "partial" if missing else "degraded" if degraded or not errors_all.empty else "unavailable" if raw_all.empty else "ready"
    signal_dates = sorted({str(value.get("signal_date_requested", "")) for value in statuses.values() if isinstance(value, dict) and value.get("signal_date_requested")})
    summary: dict[str, object] = {"schema_version": "dinghai-summary/v1", "status": final, "signal_date": signal_dates[0] if len(signal_dates) == 1 else None, "signal_date_mismatch": signal_dates if len(signal_dates) > 1 else [], "missing_shards": missing, "degraded_shards": degraded, "valid_record_count": len(raw_all), "candidate_count": len(candidates), "error_count": len(errors_all), "shard_statuses": statuses, "notification": {"status": "not_requested"}, "disclosure": "定海神针独立多周期日线研究观察；不与DistilledQuant核心或520低位首红共享输入输出；不构成买卖建议、收益预测或仓位建议。"}
    if args.notify:
        summary["notification"] = send_serverchan(summary, candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "dinghai_candidates.csv", index=False, encoding="utf-8-sig")
    errors_all.to_csv(args.output_dir / "dinghai_errors.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "dinghai_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 定海神针独立多周期研究", "", f"- 状态：`{final}`", f"- 信号日：`{summary['signal_date'] or '不一致/未记录'}`", f"- 观察：{len(candidates)}", f"- 有效记录：{len(raw_all)}", f"- 错误：{len(errors_all)}", f"- 通知：`{summary['notification']['status']}`", "", "> 仅为公开日线研究观察，不构成买卖建议、收益预测或仓位建议。"]
    (args.output_dir / "dinghai_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
