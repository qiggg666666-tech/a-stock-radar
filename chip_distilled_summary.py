#!/usr/bin/env python3
"""筹码尖峰蒸馏独立汇总器；仅在存在完成分片时尝试单条通知。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8-sig') if path.exists() and path.stat().st_size else pd.DataFrame()


def _notify(title: str, body: str) -> dict[str, object]:
    key = os.getenv('SENDKEY', '').strip()
    if not key:
        return {'status': 'skipped', 'reason': 'missing_sendkey'}
    try:
        import requests
        response = requests.post(f'https://sctapi.ftqq.com/{key}.send', data={'title': title, 'desp': body}, timeout=20)
        data = response.json()
        if response.ok and data.get('code') == 0:
            return {'status': 'sent', 'http_status': response.status_code, 'code': data.get('code')}
        return {'status': 'failed', 'http_status': response.status_code, 'code': data.get('code'), 'message': str(data)[:300]}
    except Exception as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}:{str(exc)[:300]}'}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq('true') if series.dtype == object else series.astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--shard-total', type=int, default=4)
    parser.add_argument('--notify', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert _read_csv(Path('/not-present.csv')).empty
        print('SELF_TEST_OK')
        return 0
    if args.input_dir is None or args.output_dir is None:
        parser.error('--input-dir and --output-dir are required unless --self-test is used')
    statuses, frames, error_frames = [], [], []
    for path in sorted(args.input_dir.rglob('status.json')):
        try:
            statuses.append(json.loads(path.read_text(encoding='utf-8')))
            frames.append(_read_csv(path.parent / 'raw_records.csv'))
            error_frames.append(_read_csv(path.parent / 'errors.csv'))
        except Exception as exc:
            statuses.append({'state': 'artifact_read_error', 'error': f'{type(exc).__name__}:{str(exc)[:200]}'})
    completed = [item for item in statuses if item.get('state') in {'completed', 'completed_zero_records'}]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not completed:
        status = {'schema_version': 'chip-distilled-summary/v2', 'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'state': 'skipped:no_completed_shards', 'completed_shards': [], 'missing_shards': list(range(args.shard_total)), 'candidate_count': 0, 'notification': {'status': 'skipped', 'reason': 'no_completed_shards'}, 'disclosure': '无完成分片，不构成研究结果。'}
        (args.output_dir / 'chip_distilled_summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output_dir / 'chip_distilled_report.md').write_text('# 筹码尖峰蒸馏汇总\n\n本次没有完成分片，未生成研究排序，也未发送通知。\n', encoding='utf-8')
        print(json.dumps(status, ensure_ascii=False))
        return 0
    records = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    errors = pd.concat([frame for frame in error_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in error_frames) else pd.DataFrame(columns=['code', 'name', 'scope', 'error_type', 'error_message'])
    if not records.empty:
        for column in ['is_tradeable', 'is_approaching']:
            if column in records:
                records[column] = _as_bool(records[column])
            else:
                records[column] = False
        for column in ['score', 'conc70_90_width_pct', 'dist_to_peak_pct']:
            records[column] = pd.to_numeric(records.get(column, 0), errors='coerce').fillna(0.0)
        records = records.sort_values(['is_tradeable', 'is_approaching', 'score', 'conc70_90_width_pct', 'dist_to_peak_pct'], ascending=[False, False, False, True, False])
    candidates = records.head(100)
    notification_preview = candidates.head(50)
    candidates.to_csv(args.output_dir / 'chip_distilled_candidates.csv', index=False, encoding='utf-8-sig')
    errors.to_csv(args.output_dir / 'chip_distilled_errors.csv', index=False, encoding='utf-8-sig')
    missing = sorted(set(range(args.shard_total)) - {int(item.get('shard_index', -1)) for item in completed})
    report = ['# 筹码尖峰蒸馏汇总', '', f'- 完成分片：{len(completed)}/{args.shard_total}', f'- 有效记录：{len(records)}', f'- 数据错误：{len(errors)}', f'- 排序记录（artifact）：{len(candidates)}', f'- 推送展示：{len(notification_preview)}（最多50条）', '- 集中度：70%/90%成本区间与区间宽度仅用于排序和展示，不是硬过滤。', '', '## 推送预览（最多50条）']
    fields = ['code', 'name', 'stage', 'score', 'conc70', 'conc90', 'conc70_90_width_pct', 'factor_status']
    for index, (_, row) in enumerate(notification_preview.iterrows(), start=1):
        report.append(f"{index:02d}. {row.get('code', '')} {row.get('name', '')}｜{row.get('stage', '')}｜评分{row.get('score', '')}｜70%/90%集中度{row.get('conc70', '')}/{row.get('conc90', '')}｜区间差{row.get('conc70_90_width_pct', '')}｜因子{row.get('factor_status', '')}")
    if notification_preview.empty:
        report.append('当日没有可展示的研究记录；完整状态和错误审计见artifact。')
    report.extend(['', '仅为筹码尖峰规则的研究输出，不构成投资建议。'])
    notification = {'status': 'not_requested'}
    if args.notify:
        notification = _notify(f'筹码尖峰蒸馏 | {len(completed)}/{args.shard_total}分片 | 展示{len(notification_preview)}/排序{len(candidates)}', '\n'.join(report))
    status = {'schema_version': 'chip-distilled-summary/v2', 'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'state': 'ready' if not missing else 'partial', 'completed_shards': sorted(int(item.get('shard_index', -1)) for item in completed), 'missing_shards': missing, 'record_count': len(records), 'error_count': len(errors), 'candidate_count': len(candidates), 'notification_preview_count': len(notification_preview), 'notification': notification, 'disclosure': '筹码尖峰蒸馏专属汇总；70%–90%集中度用于排序和展示，外部因子只补充候选。'}
    (args.output_dir / 'chip_distilled_summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'chip_distilled_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
