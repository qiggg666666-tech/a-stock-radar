#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为GitHub Actions分片筛选脚本写入可上传的运行状态契约。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行策略分片并写入状态JSON")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--shard", choices=["a", "b", "c", "d", "e"], required=True)
    parser.add_argument("--script", required=True)
    # 子脚本的参数（例如 --output-dir、--processes、--query-timeout）由运行器
    # 原样转发；不使用shell，避免把参数拼接为字符串后执行。
    args, forwarded_args = parser.parse_known_args()
    args.forwarded_args = forwarded_args
    return args


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    script = Path(args.script)
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    status_path = output_dir / f"strategy_shard_status_{args.strategy}_{args.shard}.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    base = {
        "schema_version": "1.0",
        "strategy": args.strategy,
        "shard": args.shard.upper(),
        "script": args.script,
        "forwarded_args": args.forwarded_args,
        "started_at": started_at,
    }
    if not script.is_file():
        write_status(status_path, {**base, "state": "failed", "exit_code": 127, "reason": "script_missing", "finished_at": datetime.now().isoformat(timespec="seconds")})
        print(f"STRATEGY_SHARD_STATUS failed strategy={args.strategy} shard={args.shard} reason=script_missing", file=sys.stderr)
        return 127
    write_status(status_path, {**base, "state": "running"})
    command = [sys.executable, str(script), *args.forwarded_args]
    print("STRATEGY_SHARD_COMMAND", " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: deque[str] = deque(maxlen=18)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        tail.append(line.rstrip())
    exit_code = process.wait()
    state = "success" if exit_code == 0 else "failed"
    write_status(status_path, {
        **base,
        "state": state,
        "exit_code": exit_code,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "" if exit_code == 0 else "screening_process_nonzero_exit",
        "error_tail": "\n".join(tail)[-1400:] if exit_code != 0 else "",
    })
    print(f"STRATEGY_SHARD_STATUS {state} strategy={args.strategy} shard={args.shard} exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
