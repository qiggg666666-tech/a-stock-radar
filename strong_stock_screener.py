#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strong_stock_screener.py —— 盘中强势股快照筛选（创业板 + 高涨幅 + 高量比 + 排除ST/新股），可推送 Server酱

做什么
  取全市场实时行情快照，按 涨幅、量比、成交额 筛出“今天特别强”的股票，按涨幅排序输出，
  并可通过 Server酱 推送（环境变量 SENDKEY）。结果同时保存成 CSV。

数据源
  主：东方财富 ak.stock_zh_a_spot_em（含量比、换手率）
  备：新浪 ak.stock_zh_a_spot（没有量比/换手率）。主源连续失败时自动切换，此时只能按 涨幅+成交额 筛选，
      推送标题带“·无量比”，正文里说明，请不要把它等同于带量比的结果。

注意
  * 这是“盘中快照”：只有交易时段（北京时间 9:30–11:30、13:00–15:00）的量比才有实时含义；
    其它时间运行得到的是最近一个交易日的收盘快照，脚本会在输出里提示。
  * 在 GitHub Actions 的海外机器上，东方财富接口可能被限流或不可用，所以保留了重试和备用源，
    建议先用 workflow_dispatch 手动触发验证，稳定后再考虑加定时。
  * 涨幅≥10% 的创业板股票大多已被拉起，次日开盘常常买不到好价位（T+1，且可能高开/涨停），仅作研究，非投资建议。
  * 默认排除 ST/退市 和 新股（名称以 N/C 开头：首日/前5日不设涨跌幅，涨幅会很夸张）。

用法
  python strong_stock_screener.py                       # 默认：创业板，涨幅>=10%，量比>=2.5，成交额>=0.5亿
  python strong_stock_screener.py --notify              # 推送 Server酱（需 SENDKEY）
  python strong_stock_screener.py --boards cyb_kcb --min-pct 9 --min-volume-ratio 2
  python strong_stock_screener.py --self-test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BJ = timezone(timedelta(hours=8))
TITLE_MAX = 32                      # Server酱标题长度上限
REQUIRED_EM = ["代码", "名称", "最新价", "涨跌幅", "量比", "换手率", "成交额"]
BOARD_PREFIX = {
    "cyb": ("300", "301"),
    "cyb_kcb": ("300", "301", "688", "689"),
    "all": ("",),
}


def now_bj() -> datetime:
    return datetime.now(BJ)


def session_note(dt: datetime) -> str:
    """交易时段提示（只看星期和时间，不判断节假日）。"""
    hm = dt.hour * 60 + dt.minute
    in_session = dt.weekday() < 5 and ((9 * 60 + 30 <= hm <= 11 * 60 + 30) or (13 * 60 <= hm <= 15 * 60))
    if in_session:
        return "盘中快照（量比有实时含义）"
    return "非交易时段：以下为最近一个交易日的收盘快照，量比不代表当前盘中状态"


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def _retry(fn, attempts: int, label: str):
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  {label} 第{i}/{attempts}次失败: {type(exc).__name__}: {str(exc)[:120]}")
            if i < attempts:
                time.sleep(2.0 * i)
    raise RuntimeError(f"{label} 连续失败: {type(last).__name__}: {str(last)[:160]}")


def fetch_spot(attempts: int = 3) -> tuple[pd.DataFrame, str]:
    """返回 (行情表, 来源)；来源为 'eastmoney' 或 'sina'。"""
    import akshare as ak

    try:
        df = _retry(ak.stock_zh_a_spot_em, attempts, "东方财富行情")
        missing = [c for c in REQUIRED_EM if c not in df.columns]
        if missing:
            raise RuntimeError(f"东方财富行情缺少列 {missing}；实际列: {list(df.columns)[:20]}")
        return df, "eastmoney"
    except Exception as exc:  # noqa: BLE001
        print(f"东方财富不可用，切换新浪备用源（无量比）: {exc}")
    df = _retry(ak.stock_zh_a_spot, 2, "新浪行情")
    need = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"新浪行情缺少列 {missing}；实际列: {list(df.columns)[:20]}")
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
    df["量比"] = float("nan")
    df["换手率"] = float("nan")
    return df, "sina"


# ----------------------------------------------------------------------------
# 筛选
# ----------------------------------------------------------------------------
def screen(df: pd.DataFrame, source: str, args: argparse.Namespace) -> pd.DataFrame:
    d = df.copy()
    d["代码"] = d["代码"].astype(str).str.zfill(6)
    for col in ("最新价", "涨跌幅", "量比", "换手率", "成交额"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    name = d["名称"].astype(str)
    if not args.include_st:
        d = d[~name.str.upper().str.contains("ST|退", na=False)]
        name = d["名称"].astype(str)
    if not args.include_new:
        d = d[~name.str.upper().str.startswith(("N", "C"))]
    d = d[d["代码"].str.startswith(BOARD_PREFIX[args.boards])]
    cond = (d["涨跌幅"] >= args.min_pct) & (d["成交额"] >= args.min_amount_yi * 1e8)
    if source == "eastmoney":
        cond &= d["量比"] >= args.min_volume_ratio        # 量比为空(NaN)的一律不通过
    out = d[cond].sort_values("涨跌幅", ascending=False).head(args.top).reset_index(drop=True)
    return out


def format_lines(result: pd.DataFrame, source: str) -> list[str]:
    lines = []
    for i, r in result.iterrows():
        vr = "-" if pd.isna(r["量比"]) else f"{r['量比']:.2f}"
        tr = "-" if pd.isna(r["换手率"]) else f"{r['换手率']:.1f}%"
        lines.append(
            f"{i + 1:03d}. {r['代码']} {r['名称']}｜{r['最新价']:.2f}｜涨{r['涨跌幅']:.2f}%"
            f"｜量比{vr}｜换手{tr}｜成交{r['成交额'] / 1e8:.2f}亿"
        )
    return lines


# ----------------------------------------------------------------------------
# 推送
# ----------------------------------------------------------------------------
def notify(title: str, body: str, retries: int = 2) -> dict:
    key = os.getenv("SENDKEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "missing_sendkey"}
    import requests

    title = title[:TITLE_MAX]
    last: dict = {}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": body}, timeout=20)
            data = resp.json()
            if resp.ok and data.get("code") == 0:
                return {"status": "sent"}
            last = {"status": "failed", "http": resp.status_code, "message": str(data.get("message", ""))[:100]}
        except Exception as exc:  # noqa: BLE001
            last = {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:160]}"}
        if attempt < retries:
            time.sleep(2)
    return last


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="盘中强势股快照筛选（创业板 + 高涨幅 + 高量比 + 排除ST/新股）")
    p.add_argument("--min-pct", type=float, default=10.0, help="最低涨幅（百分数，10 即 10%%）")
    p.add_argument("--min-volume-ratio", type=float, default=2.5, help="最低量比（仅东方财富源有量比）")
    p.add_argument("--min-amount-yi", type=float, default=0.5, help="最低成交额（亿元，0.5 即 5000 万）")
    p.add_argument("--boards", choices=list(BOARD_PREFIX), default="cyb", help="cyb=创业板；cyb_kcb=创业板+科创板；all=全部")
    p.add_argument("--include-st", action="store_true", help="包含 ST/退市（默认排除）")
    p.add_argument("--include-new", action="store_true", help="包含新股（名称以 N/C 开头，默认排除）")
    p.add_argument("--top", type=int, default=30, help="最多输出多少只")
    p.add_argument("--notify", action="store_true", help="推送 Server酱（需要环境变量 SENDKEY）")
    p.add_argument("--no-notify-empty", action="store_true", help="没有命中时不推送")
    p.add_argument("--out-dir", default=".", help="CSV 输出目录")
    p.add_argument("--retries", type=int, default=3, help="东方财富行情的重试次数")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        r = screen(
            pd.DataFrame({"代码": ["300001", "600002"], "名称": ["甲", "乙"], "最新价": [10.0, 10.0], "涨跌幅": [12.0, 12.0],
                          "量比": [3.0, 3.0], "换手率": [5.0, 5.0], "成交额": [9e8, 9e8]}), "eastmoney", args)
        assert r["代码"].tolist() == ["300001"], r
        assert len(f"强势股·无量比1030|{99}只") <= TITLE_MAX
        print("STRONG_STOCK_SELF_TEST_OK")
        return 0

    dt = now_bj()
    stamp = dt.strftime("%H%M")
    print("=" * 70)
    print(f"强势股筛选 | {dt.strftime('%Y-%m-%d %H:%M:%S')} 北京时间 | {session_note(dt)}")
    print(f"条件：板块={args.boards} 涨幅≥{args.min_pct}% 量比≥{args.min_volume_ratio} 成交额≥{args.min_amount_yi}亿 "
          f"{'含ST' if args.include_st else '排除ST'} {'含新股' if args.include_new else '排除新股'}")
    print("=" * 70)

    try:
        df, source = fetch_spot(args.retries)
    except Exception as exc:  # noqa: BLE001
        msg = f"行情接口不可用：{exc}"
        print(msg)
        if args.notify:
            print("Server酱:", notify(f"强势股{stamp}|取数失败", msg + "\n\n东方财富和新浪两个数据源都失败，请稍后重试或检查网络。"))
        return 1

    result = screen(df, source, args)
    note = session_note(dt)
    if source == "sina":
        note += "；⚠ 东方财富不可用，已改用新浪备用源，没有量比/换手率，仅按涨幅+成交额筛选"
    csv_path = Path(args.out_dir) / f"strong_stocks_{dt.strftime('%Y%m%d_%H%M')}.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tag = "·无量比" if source == "sina" else ""

    if result.empty:
        print("\n未找到符合条件的股票。可尝试把 --min-volume-ratio 降到 2.0，或 --min-pct 降到 9。")
        if args.notify and not args.no_notify_empty:
            body = f"{note}\n\n未找到符合条件的股票（涨幅≥{args.min_pct}%、量比≥{args.min_volume_ratio}、成交额≥{args.min_amount_yi}亿）。"
            print("Server酱:", notify(f"强势股{tag}{stamp}|0只", body))
        return 0

    lines = format_lines(result, source)
    print(f"\n共 {len(result)} 只（按涨幅排序），已保存 {csv_path}\n")
    for line in lines:
        print(line)
    print("\n仅为研究输出，不构成投资建议。")
    if args.notify:
        body = "\n\n".join([note] + lines) + "\n\n仅为研究输出，不构成投资建议。"
        print("Server酱:", notify(f"强势股{tag}{stamp}|{len(result)}只", body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
