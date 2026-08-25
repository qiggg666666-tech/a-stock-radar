#!/usr/bin/env python3
"""筹码尖峰蒸馏独立汇总器，只有存在完成分片时才允许单条通知。"""
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
    key = os.getenv('SENDKEY','').strip()
    if not key:
        return {'status':'skipped','reason':'missing_sendkey'}
    try:
        import requests
        response = requests.post(f'https://sctapi.ftqq.com/{key}.send', data={'title':title, 'desp':body}, timeout=20)
        data = response.json()
        if response.ok and data.get('code') == 0:
            return {'status':'sent','http_status':response.status_code,'code':data.get('code')}
        return {'status':'failed','http_status':response.status_code,'code':data.get('code'),'message':str(data)[:300]}
    except Exception as exc:
        return {'status':'failed','error':f'{type(exc).__name__}:{str(exc)[:300]}'}


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
            statuses.append({'state':'artifact_read_error','error':f'{type(exc).__name__}:{str(exc)[:200]}'})
    completed = [item for item in statuses if item.get('state') in {'completed','completed_zero_records'}]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not completed:
        status = {'schema_version':'chip-distilled-summary/v1','generated_at':datetime.now().astimezone().isoformat(timespec='seconds'),'state':'skipped:no_completed_shards','completed_shards':[],'missing_shards':list(range(args.shard_total)),'candidate_count':0,'notification':{'status':'skipped','reason':'no_completed_shards'},'disclosure':'无完成分片，不构成研究结果。'}
        (args.output_dir / 'chip_distilled_summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output_dir / 'chip_distilled_report.md').write_text('# 筹码尖峰蒸馏汇总\n\n本次没有完成分片，未生成研究排序，也未发送通知。\n', encoding='utf-8')
        print(json.dumps(status, ensure_ascii=False))
        return 0
    records = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    errors = pd.concat([frame for frame in error_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in error_frames) else pd.DataFrame(columns=['code','name','error_type','error_message'])
    if not records.empty:
        records['is_tradeable'] = records['is_tradeable'].astype(str).str.lower().eq('true') if records['is_tradeable'].dtype == object else records['is_tradeable'].astype(bool)
        records['is_approaching'] = records['is_approaching'].astype(str).str.lower().eq('true') if records['is_approaching'].dtype == object else records['is_approaching'].astype(bool)
        records = records.sort_values(['is_tradeable','is_approaching','dist_to_peak_pct'], ascending=[False,False,False])
    candidates = records.head(100)
    candidates.to_csv(args.output_dir / 'chip_distilled_candidates.csv', index=False, encoding='utf-8-sig')
    errors.to_csv(args.output_dir / 'chip_distilled_errors.csv', index=False, encoding='utf-8-sig')
    missing = sorted(set(range(args.shard_total)) - {int(item.get('shard_index',-1)) for item in completed})
    report = ['# 筹码尖峰蒸馏汇总', '', f'- 完成分片：{len(completed)}/{args.shard_total}', f'- 有效记录：{len(records)}', f'- 数据错误：{len(errors)}', f'- 排序记录：{len(candidates)}', '', '仅为既有筹码尖峰规则的研究输出。']
    notification = {'status':'not_requested'}
    if args.notify:
        notification = _notify(f'筹码尖峰蒸馏 | {len(completed)}/{args.shard_total}分片 | {len(candidates)}条排序', '\n'.join(report))
    status = {'schema_version':'chip-distilled-summary/v1','generated_at':datetime.now().astimezone().isoformat(timespec='seconds'),'state':'ready' if not missing else 'partial','completed_shards':sorted(int(item.get('shard_index',-1)) for item in completed),'missing_shards':missing,'record_count':len(records),'error_count':len(errors),'candidate_count':len(candidates),'notification':notification,'disclosure':'筹码尖峰蒸馏专属汇总，不与其他任务共享。'}
    (args.output_dir / 'chip_distilled_summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'chip_distilled_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
