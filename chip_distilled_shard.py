#!/usr/bin/env python3
"""筹码尖峰蒸馏独立A–D分片扫描器。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from chip_distilled_research import analyze_frame, enrich_candidate, fetch_ohlcv, self_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', type=Path)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-total', type=int, default=4)
    parser.add_argument('--signal-date', default='')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=35)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--candidate-enrichment-limit', type=int, default=40)
    parser.add_argument('--candidate-factor-timeout-seconds', type=float, default=12)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        assert [item['code'] for item in [{'code': '000001'}, {'code': '000002'}, {'code': '000003'}, {'code': '000004'}][1::2]] == ['000002', '000004']
        print('SELF_TEST_OK')
        return 0
    if args.universe is None or args.shard_index is None or args.output_dir is None:
        parser.error('--universe, --shard-index and --output-dir are required unless --self-test is used')
    if args.shard_index not in range(args.shard_total):
        raise ValueError('invalid_shard_index')
    payload = json.loads(args.universe.read_text(encoding='utf-8'))
    selected = payload.get('universe', [])[args.shard_index::args.shard_total]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    source_counts: dict[str, int] = {}
    shortlist: list[dict[str, object]] = []
    for position, item in enumerate(selected, 1):
        code, name = str(item.get('code', '')), str(item.get('name', ''))
        try:
            frame, source, source_errors = fetch_ohlcv(code, args.signal_date, args.timeout_seconds, args.retries)
            row = analyze_frame(code, name, frame)
            row.update({'data_source': source, 'source_errors': ' | '.join(source_errors), 'factor_status': 'not_selected', 'factor_errors': '', 'factor_scope': 'candidate_only'})
            rows.append(row)
            if bool(row.get('is_approaching')):
                shortlist.append({'code': code, 'name': name, 'index': len(rows) - 1, 'is_tradeable': bool(row.get('is_tradeable')), 'score': float(row.get('score', 0)), 'peak_ratio': float(row.get('peak_ratio', 0))})
            source_counts[source] = source_counts.get(source, 0) + 1
        except Exception as exc:
            errors.append({'code': code, 'name': name, 'scope': 'core_scan', 'error_type': type(exc).__name__, 'error_message': str(exc)[:500]})
        if position % 50 == 0:
            print(json.dumps({'shard': args.shard_index, 'processed': position, 'total': len(selected), 'records': len(rows), 'errors': len(errors)}, ensure_ascii=False))
    selected_candidates = sorted(shortlist, key=lambda item: (bool(item['is_tradeable']), float(item['score']), float(item['peak_ratio'])), reverse=True)[:max(args.candidate_enrichment_limit, 0)]
    enrichment_completed = 0
    for candidate in selected_candidates:
        code, name, row_index = str(candidate['code']), str(candidate['name']), int(candidate['index'])
        try:
            frame, source, source_errors = fetch_ohlcv(code, args.signal_date, args.timeout_seconds, args.retries)
            enriched, factor_errors = enrich_candidate(code, name, frame, args.candidate_factor_timeout_seconds)
            enriched.update({'data_source': source, 'source_errors': ' | '.join(source_errors), 'factor_scope': 'candidate_only'})
            rows[row_index] = enriched
            enrichment_completed += 1
            for item in factor_errors:
                errors.append({'code': code, 'name': name, 'scope': 'candidate_factor', 'error_type': 'FactorUnavailable', 'error_message': item})
        except Exception as exc:
            rows[row_index]['factor_status'] = 'failed'
            rows[row_index]['factor_errors'] = f'{type(exc).__name__}:{str(exc)[:500]}'
            errors.append({'code': code, 'name': name, 'scope': 'candidate_factor', 'error_type': type(exc).__name__, 'error_message': str(exc)[:500]})
    pd.DataFrame(rows).to_csv(args.output_dir / 'raw_records.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(errors, columns=['code', 'name', 'scope', 'error_type', 'error_message']).to_csv(args.output_dir / 'errors.csv', index=False, encoding='utf-8-sig')
    state = 'completed' if rows else ('failed' if errors else 'completed_zero_records')
    status = {'schema_version': 'chip-distilled-shard-status/v2', 'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'shard_index': args.shard_index, 'shard_total': args.shard_total, 'state': state, 'universe_count': len(selected), 'record_count': len(rows), 'error_count': len(errors), 'data_source_counts': source_counts, 'candidate_enrichment_requested': len(selected_candidates), 'candidate_enrichment_completed': enrichment_completed, 'signal_date_requested': args.signal_date, 'disclosure': '筹码尖峰蒸馏专属全市场分片；外部股东和资金流只补充候选，并记录错误。'}
    (args.output_dir / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

