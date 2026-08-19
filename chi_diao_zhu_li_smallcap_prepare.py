#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吃掉主力研究：一次性小市值共同股票池准备器。

仅为当日自动研究扫描提供可审计的候选池快照；不用于历史回测，也不推断任何真实账户行为。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from chi_diao_zhu_li_smallcap_scanner import Config, DataSourceError, build_universe, get_spot_hard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备吃掉主力小市值共同股票池")
    parser.add_argument("--output", type=Path, default=Path("output/chi_diao_smallcap_universe.csv"))
    parser.add_argument("--status-output", type=Path, default=Path("output/chi_diao_smallcap_universe_status.json"))
    parser.add_argument("--min-float-cap-yi", type=float, default=20.0)
    parser.add_argument("--max-float-cap-yi", type=float, default=200.0)
    parser.add_argument("--min-amount-yi", type=float, default=0.20)
    parser.add_argument("--ak-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_float_cap_yi <= args.min_float_cap_yi:
        raise SystemExit("max-float-cap-yi必须大于min-float-cap-yi")
    config = Config(
        min_float_cap_yi=float(args.min_float_cap_yi),
        max_float_cap_yi=float(args.max_float_cap_yi),
        min_amount_yi=float(args.min_amount_yi),
        ak_timeout=max(1, int(args.ak_timeout)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        universe = build_universe(get_spot_hard(config), config)
    except DataSourceError as error:
        payload = {
            "state": "failed", "reason": str(error), "generated_at": datetime.now(timezone.utc).isoformat(),
            "universe_count": 0,
        }
        args.status_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(2)
    universe.to_csv(args.output, index=False, encoding="utf-8-sig")
    payload = {
        "state": "completed", "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_as_of": datetime.now().strftime("%Y-%m-%d"), "universe_count": int(len(universe)),
        "filters": {
            "float_cap_yi": [config.min_float_cap_yi, config.max_float_cap_yi],
            "min_amount_yi": config.min_amount_yi,
            "exclude": "ST、退市整理、非沪深A股",
        },
    }
    args.status_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
