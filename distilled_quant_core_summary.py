"""独立DistilledQuant核心全截面汇总器；仅最终汇总可选发送一条通知。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

import quant_distilled_core_research as core


def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def send_serverchan(summary: dict[str, object], candidates: pd.DataFrame) -> dict[str, object]:
    """Send exactly one summary notification; no failure here changes research outputs."""
    sendkey = os.environ.get("SENDKEY", "").strip()
    if not sendkey:
        return {"status": "skipped:sendkey_not_configured"}
    preview = candidates.head(20)
    lines = [f"{row.code} {row.name}（研究分 {row.distilled_research_score:.1f}）" for row in preview.itertuples(index=False)]
    title = f"DistilledQuant核心｜{summary['status']}｜{summary['candidate_count']}条研究排序"
    body = "\n".join([
        f"有效记录：{summary['valid_record_count']}；错误：{summary['error_count']}；缺片：{summary['missing_shards']}",
        "", "前20条研究排序：", *(lines or ["当次没有形成可排序研究记录。"]), "",
        "仅为公开日线横截面研究排序，不构成买卖建议、收益预测或仓位建议。",
    ])
    request = Request(f"https://sctapi.ftqq.com/{sendkey}.send", data=urlencode({"title": title, "desp": body}).encode("utf-8"), method="POST")
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        code = payload.get("code")
        return {"status": "sent", "serverchan_code": 0} if code == 0 else {"status": f"failed:serverchan_code_{code}", "serverchan_code": code}
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": f"failed:{type(exc).__name__}", "error": str(exc)[:200]}
    except Exception as exc:  # Notification must never fail a completed research run.
        return {"status": f"failed:{type(exc).__name__}", "error": str(exc)[:200]}


def resolve_status(
    missing: list[int],
    degraded: list[str],
    errors_all: pd.DataFrame,
    scored: pd.DataFrame,
) -> str:
    """与quant_distilled_core_research.py保持一致的语义：候选为空时必须是unavailable，
    不能因为errors_all非空或某些分片degraded就误标为degraded——那样会让下游通知
    在"这次完全没有产出任何排序"的情况下仍显示degraded，误导为"部分成功"。
    """
    if missing:
        return "partial"
    if scored.empty:
        return "unavailable"
    if degraded or not errors_all.empty:
        return "degraded"
    return "ready"


def main() -> int:
    parser = argparse.ArgumentParser(description="独立DistilledQuant核心汇总")
    parser.add_argument("--input-root", type=Path, default=Path("collected"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("distilled_core_summary"))
    args = parser.parse_args()
    frames, error_frames, statuses, missing = [], [], {}, []
    for index in range(args.shard_count):
        candidates = list(args.input_root.rglob(f"shard-{index}/status.json"))
        if not candidates:
            missing.append(index); statuses[str(index)] = {"status": "missing"}; continue
        folder = candidates[0].parent
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        statuses[str(index)] = status
        raw = safe_csv(folder / "raw_records.csv")
        if not raw.empty:
            frames.append(raw.assign(source_shard=index))
        errors = safe_csv(folder / "errors.csv")
        if not errors.empty:
            error_frames.append(errors.assign(source_shard=index))
    raw_all = pd.concat(frames, ignore_index=True).drop_duplicates("code") if frames else pd.DataFrame()
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=["code", "name", "stage", "error_type", "error_message", "attempts", "source_shard"])
    scored = core.score_cross_section(raw_all).head(args.top_n) if not raw_all.empty else pd.DataFrame()
    # missing分片已经单独用missing_shards表达，这里排除它们，避免同一个分片索引
    # 同时出现在missing_shards和degraded_shards里造成误导性的重复。
    missing_keys = {str(index) for index in missing}
    degraded = [index for index, value in statuses.items() if index not in missing_keys and value.get("status") != "ready"]
    final = resolve_status(missing, degraded, errors_all, scored)
    summary: dict[str, object] = {"schema_version": "distilled-quant-core-summary/v1", "status": final, "missing_shards": missing, "degraded_shards": degraded, "valid_record_count": len(raw_all), "candidate_count": len(scored), "error_count": len(errors_all), "fixed_factor_weights": core.FACTOR_WEIGHTS, "shard_statuses": statuses, "notification": {"status": "not_requested"}, "disclosure": "独立真实OHLCV研究排序；不与520低位首红任务共享输入或输出；不构成买卖建议、收益预测或仓位建议。"}
    if args.notify:
        summary["notification"] = send_serverchan(summary, scored)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "distilled_quant_core_candidates.csv", index=False, encoding="utf-8-sig")
    errors_all.to_csv(args.output_dir / "distilled_quant_core_errors.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "distilled_quant_core_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "distilled_quant_core_summary.md").write_text("\n".join(["# DistilledQuant Core", "", f"- 状态：`{final}`", f"- 候选：{len(scored)}", f"- 错误：{len(errors_all)}", f"- 通知：`{summary['notification']['status']}`", "", "> 基于真实日线的横截面研究排序，不代表未来收益预测、买卖建议或仓位建议。", ""]) , encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
