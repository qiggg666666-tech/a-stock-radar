#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""涨停基因 + 资金流入监控。

运行口径：近5个交易日出现过涨停，且今日主力净流入不少于3000万元；默认保留Top 20。
密钥只从SERVERCHAN_SENDKEY或SENDKEY环境变量读取，不写入代码和输出文件。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import requests

LOG = logging.getLogger("limit-up-money-monitor")


def normalize_code(value: object) -> str:
    digits = re.search(r"(\d{6})", str(value))
    return digits.group(1) if digits else ""


def serverchan_key() -> str:
    return (os.getenv("SERVERCHAN_SENDKEY") or os.getenv("SENDKEY") or "").strip()


def send_serverchan(title: str, content: str) -> bool:
    key = serverchan_key()
    if not key:
        LOG.warning("未配置SERVERCHAN_SENDKEY/SENDKEY，跳过推送")
        return False
    url = f"https://sctapi.ftqq.com/{key}.send"
    try:
        response = requests.post(url, data={"title": title, "desp": content}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        ok = payload.get("code") == 0
        if not ok:
            LOG.error("Server酱业务返回失败: code=%s message=%s", payload.get("code"), payload.get("message"))
        return bool(ok)
    except Exception as exc:
        LOG.error("Server酱请求失败: %s", str(exc)[:180])
        return False


def get_limit_up_codes(days: int) -> tuple[set[str], list[str]]:
    codes: set[str] = set()
    failures: list[str] = []
    current = date.today()
    checked = 0
    for offset in range(max(1, days * 2 + 3)):
        day = current - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        checked += 1
        if checked > days:
            break
        day_text = day.strftime("%Y%m%d")
        try:
            frame = ak.stock_zt_pool_em(date=day_text)
            if frame is None or frame.empty or "代码" not in frame.columns:
                continue
            day_codes = {normalize_code(x) for x in frame["代码"].tolist()}
            codes.update(code for code in day_codes if code)
            LOG.info("%s涨停池：%d只，累计基因：%d只", day_text, len(day_codes), len(codes))
        except Exception as exc:
            failures.append(f"{day_text}:{type(exc).__name__}:{str(exc)[:120]}")
            LOG.warning("%s涨停池获取失败：%s", day_text, str(exc)[:160])
    return codes, failures


def get_fund_flow() -> pd.DataFrame:
    frame = ak.stock_individual_fund_flow_rank(indicator="今日")
    if frame is None or frame.empty:
        return pd.DataFrame()
    rename: dict[object, str] = {}
    for column in frame.columns:
        text = str(column).replace(" ", "")
        if "代码" in text:
            rename[column] = "代码"
        elif "名称" in text:
            rename[column] = "名称"
        elif "最新价" in text:
            rename[column] = "最新价"
        elif "涨跌幅" in text:
            rename[column] = "涨跌幅"
        elif "主力净流入" in text and ("净额" in text or "净流入" == text):
            rename[column] = "主力净流入净额"
    return frame.rename(columns=rename)


def format_content(result: pd.DataFrame, threshold: float, genes: int, checked_failures: int) -> str:
    if result.empty:
        return (
            f"**近5个交易日涨停基因：{genes}只；今日主力净流入阈值：≥{threshold:.0f}万元**\n\n"
            "当前没有同时满足两项条件的股票。\n"
            f"涨停池读取失败日数：{checked_failures}"
        )
    lines = [
        f"**共筛选出 {len(result)} 只股票**",
        f"近5个交易日涨停基因：{genes}只；主力净流入阈值：≥{threshold:.0f}万元",
        "",
        "| 代码 | 名称 | 最新价 | 涨跌幅 | 主力净流入(万元) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in result.iterrows():
        net = float(row.get("主力净流入净额", 0) or 0)
        lines.append(
            f"| {row.get('代码', '')} | {row.get('名称', '')} | {row.get('最新价', '')} | "
            f"{row.get('涨跌幅', '')} | {net:,.0f} |"
        )
    if checked_failures:
        lines.append(f"\n> 有{checked_failures}个历史交易日涨停池读取失败，本次结果请结合运行日志复核。")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    started = datetime.now().astimezone()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genes, history_failures = get_limit_up_codes(args.lookback_days)
    if not genes:
        content = f"近{args.lookback_days}个交易日未取得涨停基因股票池。失败日数：{len(history_failures)}"
        send_serverchan(f"涨停基因+资金流入监控 {started:%m-%d %H:%M}", content)
        return 2
    try:
        fund = get_fund_flow()
    except Exception as exc:
        LOG.exception("今日资金流获取失败")
        send_serverchan(f"涨停基因+资金流入监控 {started:%m-%d %H:%M}", f"今日资金流获取失败：{type(exc).__name__}")
        return 2
    required = {"代码", "主力净流入净额"}
    if fund.empty or not required.issubset(fund.columns):
        LOG.error("资金流结果为空或缺少关键列：%s", list(fund.columns))
        return 2
    fund = fund.copy()
    fund["代码"] = fund["代码"].map(normalize_code)
    fund["主力净流入净额"] = pd.to_numeric(fund["主力净流入净额"], errors="coerce")
    result = fund[
        fund["代码"].isin(genes) & (fund["主力净流入净额"] >= args.min_net_inflow)
    ].sort_values("主力净流入净额", ascending=False).head(args.top_n)
    columns = ["代码", "名称", "最新价", "涨跌幅", "主力净流入净额"]
    result = result[[column for column in columns if column in result.columns]]
    stamp = started.strftime("%Y%m%d_%H%M%S")
    result.to_csv(output_dir / f"limit_up_money_monitor_{stamp}.csv", index=False, encoding="utf-8-sig")
    payload: dict[str, Any] = {
        "generated_at": started.isoformat(),
        "lookback_days": args.lookback_days,
        "limit_up_gene_count": len(genes),
        "min_net_inflow_wan": args.min_net_inflow,
        "candidate_count": int(len(result)),
        "history_fetch_failures": len(history_failures),
        "status": "ready",
        "candidates": result.to_dict(orient="records"),
    }
    (output_dir / f"limit_up_money_monitor_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    title = f"涨停基因+资金流入监控 {started:%m-%d %H:%M}"
    notified = send_serverchan(title, format_content(result, args.min_net_inflow, len(genes), len(history_failures)))
    LOG.info("完成：候选=%d，通知=%s，输出=%s", len(result), notified, output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="涨停基因+资金流入监控")
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("LOOKBACK_DAYS", "5")))
    parser.add_argument("--min-net-inflow", type=float, default=float(os.getenv("MIN_NET_INFLOW_WAN", "3000")))
    parser.add_argument("--top-n", type=int, default=int(os.getenv("TOP_N", "20")))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "output"))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        LOG.exception("监控程序异常终止")
        raise SystemExit(1)
