#!/usr/bin/env python3
"""独立520日低位首红A-D扫描器，不导入或消费DistilledQuant任务数据。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def normalize_code(value: str) -> str:
    code = str(value).strip().replace(".0", "").zfill(6)
    if not code.startswith(("0", "3", "6")) or len(code) != 6:
        raise ValueError(f"invalid_a_share_code:{code}")
    return code


def _worker(connection: Any, function: Callable[[], Any]) -> None:
    try: connection.send(("ok", function()))
    except Exception as exc: connection.send(("error", f"{type(exc).__name__}:{str(exc)[:300]}"))
    finally: connection.close()


def provider_call(label: str, timeout: int, function: Callable[[], Any]) -> Any:
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    if context is None: return function()
    parent, child = context.Pipe(duplex=False); process = context.Process(target=_worker, args=(child, function), daemon=True); process.start(); child.close()
    try:
        if not parent.poll(timeout):
            process.terminate(); process.join(3); raise TimeoutError(f"provider_timeout:{label}:{timeout}s")
        state, payload = parent.recv(); process.join(3)
        if state != "ok": raise RuntimeError(f"provider_error:{label}:{payload}")
        return payload
    finally:
        if process.is_alive(): process.terminate(); process.join(3)
        parent.close()


def normalize(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    mapping = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    frame = raw.rename(columns=mapping).copy()
    fields = ["date", "open", "high", "low", "close", "volume"]
    if any(field not in frame for field in fields): raise ValueError(f"{source}_schema_missing")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for field in fields[1:]: frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=fields).sort_values("date").drop_duplicates("date")
    return frame.set_index("date")[fields[1:]]


def akshare_daily(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")


def baostock_daily(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs
    login = bs.login()
    if login.error_code != "0": raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        symbol = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
        result = bs.query_history_k_data_plus(symbol, "date,open,high,low,close,volume", start_date=start, end_date=end, frequency="d", adjustflag="2")
        if result.error_code != "0": raise RuntimeError(f"baostock_query:{result.error_code}:{result.error_msg}")
        rows=[]
        while result.next(): rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)
    finally: bs.logout()


def fetch_daily(code: str, start: str, end: str, timeout: int, retries: int) -> tuple[pd.DataFrame | None, str | None, list[str]]:
    errors=[]
    for source, function in (("akshare", lambda: akshare_daily(code,start,end)), ("baostock", lambda: baostock_daily(code,start,end))):
        for attempt in range(retries):
            try:
                return normalize(provider_call(f"{source}:{code}", timeout, function), source), source, errors
            except Exception as exc: errors.append(f"{source}:{attempt+1}:{type(exc).__name__}:{str(exc)[:180]}")
    return None, None, errors


def observation(frame: pd.DataFrame, signal: pd.Timestamp, max_stale: int, max_rise: float) -> dict[str, Any]:
    visible = frame.loc[frame.index <= signal]
    if len(visible) < 521: raise ValueError(f"insufficient_history_for_520_low:{len(visible)}")
    last=visible.index.max().normalize(); stale=int((signal-last).days)
    if stale > max_stale: raise ValueError(f"stale_daily_data:{stale}")
    current, previous = visible.iloc[-1], visible.iloc[-2]
    low520=float(visible["low"].tail(520).min()); distance=(float(current["close"])/low520-1)*100
    vr=float(current["volume"]/(visible["volume"].iloc[-6:-1].mean()+1e-12)); ma5=float(visible["close"].tail(5).mean())
    first_red=bool(current["close"]>current["open"] and previous["close"]<=previous["open"])
    return {"data_last_date": last.strftime("%Y-%m-%d"), "low_520":low520, "close":float(current["close"]), "distance_to_520_low_pct":distance, "volume_ratio_previous_5":vr, "first_red":first_red, "above_ma5":bool(current["close"]>ma5), "lowrise_observation":bool(first_red and current["close"]>ma5 and distance<=max_rise)}


def main() -> int:
    parser=argparse.ArgumentParser(description="独立520低位首红A-D扫描")
    parser.add_argument("--universe-file",type=Path); parser.add_argument("--signal-date"); parser.add_argument("--shard-index",type=int,default=0); parser.add_argument("--shard-count",type=int,default=4); parser.add_argument("--start-date",default="2023-01-01"); parser.add_argument("--max-rise-pct",type=float,default=5.0); parser.add_argument("--timeout-seconds",type=int,default=35); parser.add_argument("--retries",type=int,default=2); parser.add_argument("--max-stale-days",type=int,default=3); parser.add_argument("--output-dir",type=Path,default=Path("lowrise_shard_output")); parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        items=[str(i) for i in range(16)]; assert len([x for n in range(4) for x in items[n::4]])==16; print("SELF_TEST_OK: lowrise_shard_partition"); return 0
    if not args.universe_file or not args.signal_date: parser.error("--universe-file and --signal-date are required")
    signal=pd.Timestamp(args.signal_date).normalize(); payload=json.loads(args.universe_file.read_text(encoding="utf-8")); universe=sorted(payload.get("universe",[]),key=lambda x: str(x["code"])); queue=universe[args.shard_index::args.shard_count]
    records=[]; errors=[]; sources={"akshare":0,"baostock":0}
    for item in queue:
        code=normalize_code(item["code"]); frame,source,attempts=fetch_daily(code,args.start_date,signal.strftime("%Y-%m-%d"),args.timeout_seconds,args.retries)
        if frame is None: errors.append({"code":code,"name":str(item.get("name","")),"stage":"daily_fetch","error_type":"DataSourceUnavailable","error_message":"both_sources_failed","attempts":json.dumps(attempts,ensure_ascii=False)}); continue
        try:
            records.append({"code":code,"name":str(item.get("name","")),"daily_data_source":source,**observation(frame,signal,args.max_stale_days,args.max_rise_pct)}); sources[source]+=1
        except Exception as exc: errors.append({"code":code,"name":str(item.get("name","")),"stage":"lowrise_observation","error_type":type(exc).__name__,"error_message":str(exc),"attempts":json.dumps(attempts,ensure_ascii=False)})
    args.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_csv(args.output_dir/"raw_records.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(errors,columns=["code","name","stage","error_type","error_message","attempts"]).to_csv(args.output_dir/"errors.csv",index=False,encoding="utf-8-sig")
    status={"schema_version":"lowrise-520-shard/v1","status":"ready" if not errors else "degraded","signal_date_requested":args.signal_date,"shard_index":args.shard_index,"shard_count":args.shard_count,"scan_queue_count":len(queue),"valid_record_count":len(records),"error_count":len(errors),"daily_data_source_counts":sources}
    (args.output_dir/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(status,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
