#!/usr/bin/env python3
"""FINAL Chip专属A–D分片扫描器；只输出研究记录和显式错误。"""
from __future__ import annotations

import argparse
import atexit
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from final_chip_research import (
    analyze,
    fetch_intraday_ma5_turnup,
    fetch_main_fund_flow,
    fetch_ohlcv,
    RB_CFG,
    self_test,
)

# baostock 会话在进程内只登录一次（由研究脚本的取数函数按需登录）；这里保证进程结束时退出登录。
try:
    from final_chip_research import bs_logout as _bs_logout
except ImportError:  # 旧版研究脚本没有会话管理
    def _bs_logout() -> None:  # type: ignore[misc]
        return None


# 长影线·变盘共振专区的补充因子（支撑/MA20溢价/HO/换手/周期/年季/EMA聚类/TA），
# 复用 live_stock_selector 的同口径计算；导入失败则静默跳过（汇总里这些字段会缺失，推送照常）。
try:
    from live_stock_selector import compute_zone_factors as _zone_factors
except Exception as _exc:  # noqa: BLE001
    _zone_factors = None
    print(f"提示：未能加载 live_stock_selector.compute_zone_factors，长影线专区将不带补充因子: {type(_exc).__name__}: {_exc}")


# 压力位 / 目标价（target_price.py）：对各专区候选计算；导入失败则跳过（推送里不带目标价）
try:
    from target_price import target_fields as _target_fields
except Exception as _exc2:  # noqa: BLE001
    _target_fields = None
    print(f"提示：未能加载 target_price.target_fields，推送将不带压力位/目标价: {type(_exc2).__name__}: {_exc2}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-total", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--intraday-timeout-seconds", type=float, default=15, help="首红候选二次确认(盘中MA5拐头)的单次超时")
    parser.add_argument("--intraday-retries", type=int, default=1, help="首红候选二次确认的重试次数（每次重试会在 2 个数据源各试一遍）")
    parser.add_argument("--skip-intraday-confirm", action="store_true", help="跳过首红候选的盘中二次确认，只保留日线 is_first_red_daily")
    parser.add_argument("--intraday-request-interval", type=float, default=2.0, help="连续两次盘中确认调用之间的等待秒数，避免打得太密被接口限流/断连")
    parser.add_argument("--intraday-max-calls", type=int, default=100, help="单个分片盘中二次确认调用次数硬顶，不管命中多少候选，到点就不再对后续候选发起确认，保证单分片耗时有可预测上限")
    parser.add_argument("--fund-flow-timeout-seconds", type=float, default=12.0, help="初筛候选的主力资金二次确认单次超时")
    parser.add_argument("--fund-flow-retries", type=int, default=1, help="主力资金二次确认的重试次数")
    parser.add_argument(
        "--skip-fund-flow-confirm",
        action="store_true",
        help="跳过洗盘/红筹码占优/长影线/拐点/破低反包/深跌巨量候选的主力资金二次确认",
    )
    parser.add_argument("--fund-flow-request-interval", type=float, default=2.0, help="连续两次主力资金确认调用之间的等待秒数，避免打得太密被资金流接口限流")
    parser.add_argument("--fund-flow-max-calls", type=int, default=80, help="单个分片主力资金确认调用次数硬顶，不管命中多少候选，到点就不再对后续候选发起确认，保证单分片耗时有可预测上限")
    parser.add_argument(
        "--rb-level",
        choices=["daily", "medium", "high"],
        default=None,
        help="圆弧底(ATR自适应)最低级别：daily=日线；medium=日线+周线（默认）；high=日线+周线+年/季线",
    )
    parser.add_argument(
        "--skip-targets",
        action="store_true",
        help="不为各专区候选计算压力位/目标价（target_price）",
    )
    parser.add_argument(
        "--skip-zone-factors",
        action="store_true",
        help="不为长影线·变盘共振候选计算补充因子（支撑/MA20溢价/HO/换手/周期/年季/EMA聚类/TA）",
    )
    parser.add_argument(
        "--time-budget-seconds", type=float, default=0.0,
        help="整个分片的处理时间预算（秒，0=不限制）。到点后不再处理剩余股票，把已有结果写出并标记为 completed_partial，"
             "避免被 GitHub 的 job 超时直接取消而丢失全部结果",
    )
    parser.add_argument(
        "--confirm-time-budget-seconds", type=float, default=2400.0,
        help="盘中确认 + 主力资金确认 这两类慢接口调用的总耗时预算（秒）。用完后不再发起确认调用",
    )
    parser.add_argument(
        "--confirm-fail-limit", type=int, default=6,
        help="某一类确认调用连续失败这么多次，就判定该接口当前不可用，本分片后续不再调用（0=不熔断）",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=150,
        help="每处理多少只股票就把当前结果写盘一次（即使 job 之后被取消，artifact 里也有已完成部分；0=关闭）",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="不从输出目录里已有的 completed_partial 结果续跑（默认：如果同一分片的上次结果是 completed_partial，就跳过已处理的股票接着跑）",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    atexit.register(_bs_logout)
    if args.rb_level:
        RB_CFG["min_level"] = args.rb_level

    if args.self_test:
        self_test()
        assert [0, 1, 2, 3][1::2] == [1, 3]
        print("FINAL_CHIP_SHARD_SELF_TEST_OK")
        return 0

    if None in (args.universe, args.shard_index, args.output_dir):
        parser.error("--universe, --shard-index and --output-dir are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    state = universe.get("state")

    if state not in {"ready", "degraded_cache"}:
        status = {
            "schema_version": "final-chip-shard/v4-turning",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": "skipped:universe_unavailable",
            "shard_index": args.shard_index,
            "shard_total": args.shard_total,
            "record_count": 0,
            "error_count": 0,
            "universe_state": state,
        }
        pd.DataFrame().to_csv(args.output_dir / "records.csv", index=False)
        pd.DataFrame(columns=["code", "name", "error_type", "error_message"]).to_csv(
            args.output_dir / "errors.csv", index=False
        )
        (args.output_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(status, ensure_ascii=False))
        return 0

    selected = universe["universe"][args.shard_index :: args.shard_total]
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    intraday_confirm_count = 0
    fund_flow_confirm_count = 0
    t_start = time.time()
    processed_count = 0
    stopped_by_budget = False
    confirm_seconds = 0.0
    intraday_fail_streak = fund_fail_streak = 0
    intraday_disabled = fund_disabled = False
    timings = {"fetch": 0.0, "analyze": 0.0, "confirm": 0.0}

    # ---- 续跑：同一分片的上次结果是 completed_partial 时，跳过已处理的股票 ----
    done_codes: set[str] = set()
    if not args.no_resume:
        try:
            st_path, rec_path, err_path = (args.output_dir / n for n in ("status.json", "records.csv", "errors.csv"))
            if st_path.exists() and rec_path.exists():
                prev = json.loads(st_path.read_text(encoding="utf-8"))
                if (
                    prev.get("state") == "completed_partial"
                    and prev.get("shard_index") == args.shard_index
                    and prev.get("shard_total") == args.shard_total
                    and prev.get("universe_count") == len(selected)
                ):
                    prev_rec = pd.read_csv(rec_path, encoding="utf-8-sig", dtype={"code": str})
                    records = prev_rec.where(prev_rec.notna(), None).to_dict("records")
                    if err_path.exists() and err_path.stat().st_size > 2:
                        errors = pd.read_csv(err_path, encoding="utf-8-sig", dtype=str).fillna("").to_dict("records")
                    done_codes = {str(r["code"]).zfill(6) for r in records} | {str(e["code"]).zfill(6) for e in errors}
                    source_counts = dict(prev.get("data_source_counts") or {})
                    intraday_confirm_count = int(prev.get("intraday_confirm_calls") or 0)
                    fund_flow_confirm_count = int(prev.get("fund_flow_confirm_calls") or 0)
                    print(f"分片 {args.shard_index}: 从上次的 completed_partial 结果续跑，已处理 {len(done_codes)}/{len(selected)}")
        except Exception as exc:  # noqa: BLE001
            print(f"续跑读取失败，改为从头开始: {type(exc).__name__}: {exc}")
            records, errors, done_codes = [], [], set()
            source_counts, intraday_confirm_count, fund_flow_confirm_count = {}, 0, 0

    def write_outputs(state_name: str) -> dict:
        """写出 records/errors/status。中途调用（checkpoint）与结束时调用同一个函数，汇总脚本把 completed_partial 也当作可用结果。"""
        pd.DataFrame(records).to_csv(args.output_dir / "records.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(errors, columns=["code", "name", "error_type", "error_message"]).to_csv(
            args.output_dir / "errors.csv", index=False, encoding="utf-8-sig"
        )
        status = {
            "schema_version": "final-chip-shard/v4-turning",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": state_name,
            "shard_index": args.shard_index,
            "shard_total": args.shard_total,
            "universe_state": state,
            "universe_count": len(selected),
            "processed": processed_count,
            "unprocessed": len(selected) - processed_count,
            "stopped_by_time_budget": stopped_by_budget,
            "record_count": len(records),
            "error_count": len(errors),
            "data_source_counts": source_counts,
            "intraday_confirm_calls": intraday_confirm_count,
            "fund_flow_confirm_calls": fund_flow_confirm_count,
            "intraday_disabled_by_failures": intraday_disabled,
            "fund_flow_disabled_by_failures": fund_disabled,
            "elapsed_seconds": round(time.time() - t_start, 1),
            "timing_seconds": {k: round(v, 1) for k, v in timings.items()},
            "disclosure": (
                "FINAL Chip专属分片；无合成记录、无分片推送。"
                "盘中MA5二次确认：仅 is_first_red_daily（有次数/总时长硬顶，连续失败熔断）。"
                "主力资金二次确认：洗盘 / 红筹码占优 / 长下影 / 拐点near_bk200 / "
                "低位破低反包 / 深跌巨量强收 / 圆弧底（有次数/总时长硬顶，连续失败熔断；不含宽幅alone与均线强多）。"
                "变盘字段由 final_chip_research.analyze 写出，summary 单独专区推送。"
            ),
        }
        (args.output_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    for position, item in enumerate(selected, 1):
        if args.time_budget_seconds > 0 and time.time() - t_start > args.time_budget_seconds:
            stopped_by_budget = True
            print(f"分片 {args.shard_index} 达到时间预算 {args.time_budget_seconds:.0f}s，已处理 {processed_count}/{len(selected)}，停止并写出已有结果")
            break
        code, name = str(item["code"]), str(item.get("name", ""))
        if code.zfill(6) in done_codes:
            processed_count = position
            continue
        try:
            _t0 = time.time()
            frame, source, prior_errors = fetch_ohlcv(
                code, args.timeout_seconds, args.retries
            )
            timings["fetch"] += time.time() - _t0
            _t0 = time.time()
            row = analyze(code, name, frame)
            timings["analyze"] += time.time() - _t0
            row["data_source"] = source
            row["source_errors"] = " | ".join(prior_errors)

            # 长影线·变盘共振候选（长下影 ∩ (拐点 | 低位破低反包)）：补充实战选股同口径因子。
            # 纯量价、无网络请求，只对这个专区的少量候选计算。
            if (
                _zone_factors is not None
                and not args.skip_zone_factors
                and (
                    (bool(row.get("daily_long_lower"))
                     and (bool(row.get("near_bk200")) or bool(row.get("low_bullish_outside"))))
                    or bool(row.get("near_round_bottom"))  # 圆弧底专区同样展示评分/支撑等因子
                )
            ):
                try:
                    # frame 来自 fetch_ohlcv，已经 normalize 过（换手率已折成小数）。
                    # 这里不能再 normalize_frame 一次，否则换手率会再除一次 100。
                    row.update(_zone_factors(frame))
                except Exception as zexc:  # noqa: BLE001
                    row["zone_factors_ok"] = False
                    row["zone_factors_error"] = f"{type(zexc).__name__}:{str(zexc)[:160]}"

            confirm_budget_left = confirm_seconds < args.confirm_time_budget_seconds
            if (
                row.get("is_first_red_daily") and not args.skip_intraday_confirm
                and intraday_confirm_count < args.intraday_max_calls
                and not intraday_disabled and confirm_budget_left
            ):
                _t0 = time.time()
                if intraday_confirm_count > 0:
                    time.sleep(args.intraday_request_interval)
                intraday_failed = False
                try:
                    is_turn_up, slope, note, intraday_errors = fetch_intraday_ma5_turnup(
                        code,
                        timeout_seconds=args.intraday_timeout_seconds,
                        retries=args.intraday_retries,
                    )
                    row["intraday_ma5_turn_up"] = is_turn_up
                    row["intraday_ma5_slope"] = round(slope, 5)
                    row["intraday_ma5_note"] = note
                    row["intraday_errors"] = " | ".join(intraday_errors)
                    row["is_first_red_confirmed"] = bool(is_turn_up)
                    intraday_failed = str(note).startswith("盘中数据获取失败")
                except Exception as e:
                    row["intraday_ma5_turn_up"] = False
                    row["intraday_ma5_slope"] = 0.0
                    row["intraday_ma5_note"] = "盘中确认异常"
                    row["intraday_errors"] = f"shard_wrapper:{type(e).__name__}:{str(e)[:200]}"
                    row["is_first_red_confirmed"] = False
                    intraday_failed = True
                intraday_confirm_count += 1
                intraday_fail_streak = intraday_fail_streak + 1 if intraday_failed else 0
                if args.confirm_fail_limit > 0 and intraday_fail_streak >= args.confirm_fail_limit:
                    intraday_disabled = True
                    print(f"分片 {args.shard_index}: 盘中确认连续失败 {intraday_fail_streak} 次，判定接口不可用，后续不再调用")
                _dt = time.time() - _t0
                confirm_seconds += _dt
                timings["confirm"] += _dt
            else:
                row["intraday_ma5_turn_up"] = False
                row["intraday_ma5_slope"] = 0.0
                row["intraday_ma5_note"] = (
                    "跳过" if args.skip_intraday_confirm
                    else "已达单分片调用上限未调用" if row.get("is_first_red_daily") and intraday_confirm_count >= args.intraday_max_calls
                    else "接口连续失败已停用" if row.get("is_first_red_daily") and intraday_disabled
                    else "确认总时长预算已用完未调用" if row.get("is_first_red_daily") and not confirm_budget_left
                    else "非首红候选未调用"
                )
                row["intraday_errors"] = ""
                row["is_first_red_confirmed"] = False

            # 主力资金二次确认：只对「少见」初筛信号发起昂贵接口。
            # 原：洗盘 / 红筹码占优 / 长下影
            # 增：拐点(near_bk200) / 低位破低反包 / 深跌巨量强收
            # 仍不用 is_wide_zone、ma_signal 强多（命中面太广会打爆分片时限）。
            # 次数硬顶 fund_flow_max_calls 兜底。
            is_fund_flow_candidate = bool(
                row.get("is_washout")
                or row.get("is_red_heavy_chip")
                or row.get("daily_long_lower")
                or row.get("near_bk200")
                or row.get("low_bullish_outside")
                or row.get("deep_huge_strong")
                or row.get("near_round_bottom")
            )
            confirm_budget_left = confirm_seconds < args.confirm_time_budget_seconds
            if (
                is_fund_flow_candidate and not args.skip_fund_flow_confirm
                and fund_flow_confirm_count < args.fund_flow_max_calls
                and not fund_disabled and confirm_budget_left
            ):
                _t0 = time.time()
                if fund_flow_confirm_count > 0:
                    time.sleep(args.fund_flow_request_interval)
                fund_failed = False
                try:
                    fund = fetch_main_fund_flow(
                        code,
                        timeout_seconds=args.fund_flow_timeout_seconds,
                        retries=args.fund_flow_retries,
                    )
                    row["main_fund_ok"] = fund["main_fund_ok"]
                    row["main_fund_today_net"] = fund["main_fund_today_net"]
                    row["main_fund_days3_net"] = fund["main_fund_days3_net"]
                    row["main_fund_today_ratio_pct"] = fund["main_fund_today_ratio_pct"]
                    row["main_fund_errors"] = fund["main_fund_errors"]
                    # 取不到数据时 fetch_main_fund_flow 返回全 0 并带错误信息
                    fund_failed = bool(fund["main_fund_errors"]) and float(fund["main_fund_today_net"]) == 0.0 \
                        and float(fund["main_fund_days3_net"]) == 0.0
                except Exception as e:
                    row["main_fund_ok"] = False
                    row["main_fund_today_net"] = 0.0
                    row["main_fund_days3_net"] = 0.0
                    row["main_fund_today_ratio_pct"] = 0.0
                    row["main_fund_errors"] = f"shard_wrapper:{type(e).__name__}:{str(e)[:200]}"
                    fund_failed = True
                fund_flow_confirm_count += 1
                fund_fail_streak = fund_fail_streak + 1 if fund_failed else 0
                if args.confirm_fail_limit > 0 and fund_fail_streak >= args.confirm_fail_limit:
                    fund_disabled = True
                    print(f"分片 {args.shard_index}: 主力资金确认连续失败 {fund_fail_streak} 次，判定接口不可用，后续不再调用")
                _dt = time.time() - _t0
                confirm_seconds += _dt
                timings["confirm"] += _dt
            else:
                row["main_fund_ok"] = False
                row["main_fund_today_net"] = 0.0
                row["main_fund_days3_net"] = 0.0
                row["main_fund_today_ratio_pct"] = 0.0
                row["main_fund_errors"] = (
                    "" if args.skip_fund_flow_confirm
                    else "已达单分片调用上限未调用" if is_fund_flow_candidate and fund_flow_confirm_count >= args.fund_flow_max_calls
                    else "接口连续失败已停用" if is_fund_flow_candidate and fund_disabled
                    else "确认总时长预算已用完未调用" if is_fund_flow_candidate and not confirm_budget_left
                    else "非初筛候选未调用"
                )

            # 压力位 / 目标价：只对会进入推送专区的候选计算（纯量价，无网络请求）
            if (
                _target_fields is not None
                and not args.skip_targets
                and (
                    bool(row.get("is_washout"))
                    or bool(row.get("is_first_red_confirmed"))
                    or (bool(row.get("is_red_heavy_chip")) and bool(row.get("is_wide_zone")))
                    or bool(row.get("near_bk200"))
                    or bool(row.get("low_bullish_outside"))
                    or bool(row.get("deep_huge_strong"))
                    or bool(row.get("near_round_bottom"))
                    or (bool(row.get("daily_long_lower")) and (bool(row.get("near_bk200")) or bool(row.get("low_bullish_outside"))))
                    or str(row.get("ma_signal") or "").startswith("强多")  # 均线强多专区
                )
            ):
                extra = None
                if row.get("near_round_bottom"):  # 圆弧底：加入“左沿 + 碗深×0.5/1.0”量度目标
                    try:
                        lp, bl = float(row.get("rb_left_peak")), float(row.get("rb_bowl_low"))
                        extra = [(lp + 0.5 * (lp - bl), "BOWL量度0.5x"), (lp + (lp - bl), "BOWL量度1.0x")]
                    except (TypeError, ValueError):
                        extra = None
                row.update(_target_fields(frame, code, name, extra_levels=extra))

            records.append(row)
            source_counts[source] = source_counts.get(source, 0) + 1

        except Exception as exc:
            errors.append(
                {
                    "code": code,
                    "name": name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )

        processed_count = position
        if args.checkpoint_every > 0 and position % args.checkpoint_every == 0:
            write_outputs("completed_partial")   # 中途保存：job 万一被取消，已完成部分仍会被上传
        if position % 50 == 0:
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "processed": position,
                        "total": len(selected),
                        "records": len(records),
                        "errors": len(errors),
                        "intraday_confirm_calls": intraday_confirm_count,
                        "fund_flow_confirm_calls": fund_flow_confirm_count,
                        "elapsed_s": round(time.time() - t_start),
                        "fetch_s": round(timings["fetch"]), "analyze_s": round(timings["analyze"]), "confirm_s": round(timings["confirm"]),
                    },
                    ensure_ascii=False,
                )
            )

    final_state = "completed_partial" if stopped_by_budget else ("completed" if records else "completed_zero_records")
    status = write_outputs(final_state)
    _bs_logout()
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
