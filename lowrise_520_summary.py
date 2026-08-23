#!/usr/bin/env python3
"""独立520日低位首红汇总器；不读取DistilledQuant核心artifact。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
def read(path:Path)->pd.DataFrame:
    if not path.exists() or path.stat().st_size==0:return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,default=Path("collected"));p.add_argument("--shard-count",type=int,default=4);p.add_argument("--output-dir",type=Path,default=Path("lowrise_summary"));args=p.parse_args()
    raws=[];errs=[];statuses={};missing=[]
    for i in range(args.shard_count):
        matches=list(args.input_root.rglob(f"shard-{i}/status.json"))
        if not matches:missing.append(i);statuses[str(i)]={"status":"missing"};continue
        folder=matches[0].parent; statuses[str(i)]=json.loads((folder/"status.json").read_text(encoding="utf-8")); raw=read(folder/"raw_records.csv"); err=read(folder/"errors.csv")
        if not raw.empty:raws.append(raw.assign(source_shard=i))
        if not err.empty:errs.append(err.assign(source_shard=i))
    raw=pd.concat(raws,ignore_index=True).drop_duplicates("code") if raws else pd.DataFrame(); errors=pd.concat(errs,ignore_index=True) if errs else pd.DataFrame(columns=["code","name","stage","error_type","error_message","attempts","source_shard"])
    observations=raw.loc[raw["lowrise_observation"].fillna(False).astype(bool)].copy() if not raw.empty else pd.DataFrame()
    if not observations.empty:observations=observations.sort_values(["distance_to_520_low_pct","volume_ratio_previous_5"],ascending=[True,False]).reset_index(drop=True);observations["priority_rank"]=range(1,len(observations)+1)
    degraded=[index for index,value in statuses.items() if value.get("status")!="ready"];status="partial" if missing else "degraded" if degraded or not errors.empty else "unavailable" if raw.empty else "ready"
    args.output_dir.mkdir(parents=True,exist_ok=True);observations.to_csv(args.output_dir/"lowrise_520_observations.csv",index=False,encoding="utf-8-sig");errors.to_csv(args.output_dir/"lowrise_520_errors.csv",index=False,encoding="utf-8-sig")
    summary={"schema_version":"lowrise-520-summary/v1","status":status,"missing_shards":missing,"degraded_shards":degraded,"valid_record_count":len(raw),"observation_count":len(observations),"error_count":len(errors),"shard_statuses":statuses,"disclosure":"独立520日低位首红观察任务；不与DistilledQuant核心任务共享输入或输出。"};(args.output_dir/"lowrise_520_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(args.output_dir/"lowrise_520_summary.md").write_text(f"# 520低位首红\n\n- 状态：`{status}`\n- 观察：{len(observations)}\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
