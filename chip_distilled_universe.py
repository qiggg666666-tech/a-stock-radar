#!/usr/bin/env python3
"""筹码尖峰蒸馏任务专属共同股票池。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from chip_distilled_research import provider_call


def _normalize(raw) -> list[dict[str, str]]:
    import pandas as pd
    frame = raw[['code', 'name']].copy()
    frame['code'] = frame['code'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    frame['name'] = frame['name'].astype(str)
    frame = frame[frame['code'].str.fullmatch(r'(?:00|30|60|68)\d{4}', na=False)]
    frame = frame[~frame['name'].str.contains(r'ST|\*ST|退|摘', regex=True, case=False, na=False)]
    return [{'code': row.code, 'name': row.name} for row in frame.drop_duplicates('code').sort_values('code').itertuples(index=False)]


def _akshare():
    import akshare as ak
    return ak.stock_info_a_code_name().rename(columns={'代码': 'code', '名称': 'name'})


def _baostock():
    import baostock as bs
    import pandas as pd
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'baostock_login:{login.error_code}:{login.error_msg}')
    try:
        result = bs.query_stock_basic()
        if result.error_code != '0':
            raise RuntimeError(f'baostock_query:{result.error_code}:{result.error_msg}')
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
        if 'status' in raw:
            raw = raw[raw['status'].astype(str) == '1']
        return pd.DataFrame({'code': raw['code'].astype(str).str.split('.').str[-1], 'name': raw['code_name'].astype(str)})
    finally:
        bs.logout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('chip_distilled_universe.json'))
    parser.add_argument('--signal-date', default='')
    parser.add_argument('--timeout-seconds', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert _normalize(__import__('pandas').DataFrame({'code':['000001','830001'], 'name':['样本','退市']})) == [{'code':'000001','name':'样本'}]
        print('SELF_TEST_OK')
        return 0
    errors = []
    source = ''
    raw = None
    for label, function in (('akshare', _akshare), ('baostock', _baostock)):
        for attempt in range(1, args.retries + 1):
            try:
                raw, source = provider_call(f'{label}_universe', args.timeout_seconds, function), label
                break
            except Exception as exc:
                errors.append(f'{label}:{attempt}:{type(exc).__name__}:{str(exc)[:220]}')
        if raw is not None:
            break
    universe = _normalize(raw) if raw is not None else []
    if not universe:
        raise RuntimeError('universe_unavailable:' + ' | '.join(errors))
    payload = {'schema_version':'chip-distilled-universe/v1','generated_at':datetime.now().astimezone().isoformat(timespec='seconds'),'signal_date_requested':args.signal_date,'count':len(universe),'data_source':source,'source_errors':errors,'universe':universe,'disclosure':'仅供筹码尖峰蒸馏独立任务使用的共同股票池快照。'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
