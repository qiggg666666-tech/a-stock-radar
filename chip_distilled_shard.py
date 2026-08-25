#!/usr/bin/env python3
"""筹码尖峰蒸馏独立A–D分片扫描器。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from chip_distilled_research import analyze_frame, fetch_ohlcv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', type=Path)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-total', type=int, default=4)
    parser.add_argument('--signal-date', default='')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=35)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert [item['code'] for item in [{'code':'000001'}, {'code':'000002'}, {'code':'000003'}, {'code':'000004'}][1::2]] == ['000002','000004']
        print('SELF_TEST_OK')
        return 0
    if args.universe is None or args.shard_index is None or args.output_dir is None:
        parser.error('--universe, --shard-index and --output-dir are required unless --self-test is used')
    if args.shard_index not in range(args.shard_total):
        raise ValueError('invalid_shard_index')
    payload = json.loads(args.universe.read_text(encoding='utf-8'))
    selected = payload.get('universe', [])[args.shard_index::args.shard_total]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, errors, source_counts = [], [], {}
    for position, item in enumerate(selected, 1):
        code, name = str(item.get('code','')), str(item.get('name',''))
        try:
            frame, source, source_errors = fetch_ohlcv(code, args.signal_date, args.timeout_seconds, args.retries)
            row = analyze_frame(code, name, frame)
            row['data_source'] = source
            row['source_errors'] = ' | '.join(source_errors)
            rows.append(row)
            source_counts[source] = source_counts.get(source, 0) + 1
        except Exception as exc:
            errors.append({'code':code,'name':name,'error_type':type(exc).__name__,'error_message':str(exc)[:500]})
        if position % 50 == 0:
            print(json.dumps({'shard':args.shard_index,'processed':position,'total':len(selected),'records':len(rows),'errors':len(errors)}, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(args.output_dir / 'raw_records.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(errors, columns=['code','name','error_type','error_message']).to_csv(args.output_dir / 'errors.csv', index=False, encoding='utf-8-sig')
    state = 'completed' if rows else ('failed' if errors and not rows else 'completed_zero_records')
    status = {'schema_version':'chip-distilled-shard-status/v1','generated_at':datetime.now().astimezone().isoformat(timespec='seconds'),'shard_index':args.shard_index,'shard_total':args.shard_total,'state':state,'universe_count':len(selected),'record_count':len(rows),'error_count':len(errors),'data_source_counts':source_counts,'signal_date_requested':args.signal_date,'disclosure':'筹码尖峰蒸馏专属分片状态，不与其他任务共享。'}
    (args.output_dir / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
