#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strong_stock_screener.py —— 盘中强势股快照筛选（创业板 + 高涨幅 + 高量比 + 排除ST/新股），可推送 Server酱

做什么
  按涨幅从高到低只取“前几页”行情（不再拉全市场几十页），再按 涨幅、量比、成交额 筛选，
  按涨幅排序输出并可通过 Server酱 推送（环境变量 SENDKEY），结果同时保存成 CSV。

为什么改成“只取前几页”
  GitHub Actions 的海外机器上，akshare 的全市场行情接口要翻几十页（东财、新浪都是），很容易被限流或超时
  （实测新浪在第 29/70 页超时）。强势股只需要涨幅最高的那几十只，按涨幅降序取第 1 页就够了，请求量小两个数量级。

数据源（按顺序尝试，第一个成功的就用）
  1) eastmoney_light  东方财富 clist 接口，按涨幅降序，含量比、换手率
  2) sina_light       新浪 getHQNodeData，按涨幅降序，没有量比（推送标题带“·无量比”）
  3) akshare 全市场（仅加 --try-full-akshare 才用，默认关闭，因为在 Actions 上经常超时）
  每次请求都会把状态码/耗时/错误打印出来，全部失败时这些诊断会一并推送，方便定位是哪一环不通。
  --probe：依次探测每个数据源并打印结果（不筛选），用来判断哪些源在当前网络可用。

注意
  * 接口参数与字段来自对这两个公开接口的了解，没有在 Actions 的网络里验证过——请先手动触发一次，看日志里的诊断。
  * “盘中快照”只有交易时段（北京时间 9:30–11:30、13:00–15:00）的量比才有实时含义，其它时间得到的是最近一个交易日的收盘快照。
  * 涨幅≥10% 的创业板股票大多已被拉起，次日开盘常常买不到好价位（T+1），仅作研究，非投资建议。
  * 默认排除 ST/退市 和 新股（名称以 N/C 开头：首日/前5日不设涨跌幅，涨幅会很夸张）。

用法
  python strong_stock_screener.py                       # 默认：创业板，涨幅>=10%，量比>=2.5，成交额>=0.5亿
  python strong_stock_screener.py --notify              # 推送 Server酱（需 SENDKEY）
  python strong_stock_screener.py --probe --notify      # 探测各数据源是否可用，并把结果推送出来
  python strong_stock_screener.py --boards cyb_kcb --min-pct 9 --min-volume-ratio 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BJ = timezone(timedelta(hours=8))
TITLE_MAX = 32                      # Server酱标题长度上限
COLS = ["代码", "名称", "最新价", "涨跌幅", "量比", "换手率", "成交额"]
BOARD_PREFIX = {"cyb": ("300", "301"), "cyb_kcb": ("300", "301", "688", "689"), "all": ("",)}
# 东方财富 clist 的 fs 参数：m:0+t:80=深市创业板，m:1+t:23=沪市科创板
EM_FS = {
    "cyb": "m:0+t:80",
    "cyb_kcb": "m:0+t:80,m:1+t:23",
    "all": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
}
EM_HOSTS = ["https://82.push2.eastmoney.com", "https://push2.eastmoney.com"]
EM_PATH = "/api/qt/clist/get"
SINA_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
SINA_NODES = {"cyb": ["cyb"], "cyb_kcb": ["cyb", "kcb"], "all": ["hs_a"]}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


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
# HTTP 与诊断
# ----------------------------------------------------------------------------
class Diag:
    """记录每次请求的结果，全部失败时一并推送。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, msg: str) -> None:
        self.lines.append(msg)
        print("  " + msg)


def http_get(url: str, params: dict[str, Any], timeout: float, label: str, diag: Diag, headers: dict[str, str] | None = None) -> str:
    import requests

    t0 = time.time()
    try:
        resp = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
        ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            diag.add(f"{label}: HTTP {resp.status_code} {ms}ms")
            raise RuntimeError(f"HTTP {resp.status_code}")
        diag.add(f"{label}: HTTP 200 {ms}ms {len(resp.text)}字节")
        return resp.text
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        ms = int((time.time() - t0) * 1000)
        diag.add(f"{label}: {type(exc).__name__} {ms}ms {str(exc)[:80]}")
        raise


def parse_js_list(text: str) -> list[dict[str, Any]]:
    """新浪返回的有时是键不带引号的 JS 对象列表：先按 JSON 解析，不行再给键补引号。"""
    try:
        data = json.loads(text)
    except ValueError:
        fixed = re.sub(r'(?<=[{,])\s*([A-Za-z_]\w*)\s*:', r'"\1":', text)
        data = json.loads(fixed)
    if not isinstance(data, list):
        raise ValueError(f"返回不是列表: {str(data)[:80]}")
    return data


# ----------------------------------------------------------------------------
# 数据源
# ----------------------------------------------------------------------------
def fetch_em_light(args: argparse.Namespace, diag: Diag) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page_size = 100
    for page in range(1, args.max_pages + 1):
        text = None
        last_exc: Exception | None = None
        for attempt in range(1, args.attempts + 1):
            for host in EM_HOSTS:
                try:
                    text = http_get(
                        host + EM_PATH,
                        {"pn": page, "pz": page_size, "po": "1", "np": "1", "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                         "fltt": "2", "invt": "2", "fid": "f3", "fs": EM_FS[args.boards], "fields": "f2,f3,f5,f6,f8,f10,f12,f14"},
                        args.http_timeout, f"东财 第{page}页 {host.split('//')[1].split('.')[0]} 尝试{attempt}", diag,
                        {"Referer": "https://quote.eastmoney.com/"},
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
            if text is not None:
                break
            if attempt < args.attempts:
                time.sleep(1.5 * attempt)
        if text is None:
            raise RuntimeError(f"东方财富请求失败: {type(last_exc).__name__}: {str(last_exc)[:100]}")
        data = (json.loads(text).get("data") or {})
        diff = data.get("diff") or []
        if isinstance(diff, dict):
            diff = list(diff.values())
        for it in diff:
            rows.append({"代码": it.get("f12"), "名称": it.get("f14"), "最新价": it.get("f2"), "涨跌幅": it.get("f3"),
                         "量比": it.get("f10"), "换手率": it.get("f8"), "成交额": it.get("f6")})
        pct = pd.to_numeric(pd.Series([it.get("f3") for it in diff]), errors="coerce")
        if len(diff) < page_size or (len(pct) and pct.min() < args.min_pct):
            break                                          # 已按涨幅降序：这一页里已经出现低于阈值的，后面不用再取
    if not rows:
        raise RuntimeError("东方财富返回 0 行")
    return pd.DataFrame(rows, columns=COLS)


def fetch_sina_light(args: argparse.Namespace, diag: Diag) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node in SINA_NODES[args.boards]:
        for page in range(1, args.max_pages + 1):
            text = None
            last_exc: Exception | None = None
            for attempt in range(1, args.attempts + 1):
                try:
                    text = http_get(SINA_URL, {"page": page, "num": 80, "sort": "changepercent", "asc": "0", "node": node,
                                               "symbol": "", "_s_r_a": "page"},
                                    args.http_timeout, f"新浪 {node} 第{page}页 尝试{attempt}", diag)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < args.attempts:
                        time.sleep(1.5 * attempt)
            if text is None:
                raise RuntimeError(f"新浪请求失败: {type(last_exc).__name__}: {str(last_exc)[:100]}")
            items = parse_js_list(text)
            for it in items:
                rows.append({"代码": it.get("code"), "名称": it.get("name"), "最新价": it.get("trade"),
                             "涨跌幅": it.get("changepercent"), "量比": float("nan"),
                             "换手率": it.get("turnoverratio"), "成交额": it.get("amount")})
            pct = pd.to_numeric(pd.Series([it.get("changepercent") for it in items]), errors="coerce")
            if len(items) < 80 or (len(pct) and pct.min() < args.min_pct):
                break
    if not rows:
        raise RuntimeError("新浪返回 0 行")
    return pd.DataFrame(rows, columns=COLS)


def fetch_akshare_full(args: argparse.Namespace, diag: Diag) -> tuple[pd.DataFrame, str]:
    """全市场接口（慢，Actions 上常超时）：先东财后新浪。"""
    import akshare as ak

    try:
        df = ak.stock_zh_a_spot_em()
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"缺少列 {missing}")
        diag.add("akshare 全市场(东财): 成功")
        return df[COLS], "eastmoney_full"
    except Exception as exc:  # noqa: BLE001
        diag.add(f"akshare 全市场(东财): {type(exc).__name__} {str(exc)[:80]}")
    df = ak.stock_zh_a_spot()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
    df["量比"] = float("nan")
    df["换手率"] = float("nan")
    diag.add("akshare 全市场(新浪): 成功")
    return df[COLS], "sina_full"


SOURCES = [("eastmoney_light", fetch_em_light), ("sina_light", fetch_sina_light)]


def fetch_spot(args: argparse.Namespace, diag: Diag) -> tuple[pd.DataFrame, str]:
    for name, fn in SOURCES:
        print(f"尝试数据源 {name} ...")
        try:
            return fn(args, diag), name
        except Exception as exc:  # noqa: BLE001
            diag.add(f"[{name}] 失败: {str(exc)[:140]}")
    if args.try_full_akshare:
        print("尝试 akshare 全市场接口（较慢）...")
        try:
            return fetch_akshare_full(args, diag)
        except Exception as exc:  # noqa: BLE001
            diag.add(f"[akshare_full] 失败: {type(exc).__name__}: {str(exc)[:140]}")
    raise RuntimeError("所有数据源都失败：" + " | ".join(l for l in diag.lines if l.startswith("[")))


def probe(args: argparse.Namespace) -> tuple[list[str], int]:
    """依次探测每个数据源（每个只试一轮），返回汇总文字和可用数量。"""
    rows, ok_n = [], 0
    probe_args = argparse.Namespace(**{**vars(args), "attempts": 1, "max_pages": 1})
    candidates = list(SOURCES) + ([("akshare_full", lambda a, d: fetch_akshare_full(a, d)[0])] if args.try_full_akshare else [])
    for name, fn in candidates:
        d = Diag()
        t0 = time.time()
        try:
            df = fn(probe_args, d)
            ok_n += 1
            has_vr = bool(pd.to_numeric(df["量比"], errors="coerce").notna().any())
            rows.append(f"✅ {name}: {len(df)}行 {time.time() - t0:.1f}s 量比{'有' if has_vr else '无'} 首行={df.iloc[0]['代码']} {df.iloc[0]['名称']} 涨{df.iloc[0]['涨跌幅']}%")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"❌ {name}: {time.time() - t0:.1f}s {str(exc)[:110]}")
    return rows, ok_n


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
    if source.startswith("eastmoney"):
        cond &= d["量比"] >= args.min_volume_ratio        # 量比为空(NaN)的一律不通过
    return d[cond].sort_values("涨跌幅", ascending=False).head(args.top).reset_index(drop=True)


def format_lines(result: pd.DataFrame) -> list[str]:
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
    p.add_argument("--http-timeout", type=float, default=12.0, help="单次 HTTP 请求超时（秒）")
    p.add_argument("--attempts", type=int, default=2, help="每个数据源每页的重试次数")
    p.add_argument("--max-pages", type=int, default=4, help="每个数据源最多取几页（按涨幅降序，通常 1 页就够）")
    p.add_argument("--try-full-akshare", action="store_true", help="轻量源都失败后，再尝试 akshare 全市场接口（慢，Actions 上常超时）")
    p.add_argument("--probe", action="store_true", help="探测各数据源是否可用并打印结果，不做筛选")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        r = screen(
            pd.DataFrame({"代码": ["300001", "600002"], "名称": ["甲", "乙"], "最新价": [10.0, 10.0], "涨跌幅": [12.0, 12.0],
                          "量比": [3.0, 3.0], "换手率": [5.0, 5.0], "成交额": [9e8, 9e8]}), "eastmoney_light", args)
        assert r["代码"].tolist() == ["300001"], r
        assert len(f"强势股·无量比1030|{99}只") <= TITLE_MAX
        assert parse_js_list('[{symbol:"sz300001",code:"300001",name:"甲",changepercent:12.3}]')[0]["code"] == "300001"
        print("STRONG_STOCK_SELF_TEST_OK")
        return 0

    dt = now_bj()
    stamp = dt.strftime("%H%M")
    print("=" * 70)
    print(f"强势股筛选 | {dt.strftime('%Y-%m-%d %H:%M:%S')} 北京时间 | {session_note(dt)}")
    print(f"条件：板块={args.boards} 涨幅≥{args.min_pct}% 量比≥{args.min_volume_ratio} 成交额≥{args.min_amount_yi}亿 "
          f"{'含ST' if args.include_st else '排除ST'} {'含新股' if args.include_new else '排除新股'}")
    print("=" * 70)

    if args.probe:
        rows, ok_n = probe(args)
        print("\n数据源探测结果：")
        for r in rows:
            print("  " + r)
        if args.notify:
            print("Server酱:", notify(f"行情源探测{stamp}|{ok_n}/{len(rows)}可用", f"{session_note(dt)}\n\n" + "\n\n".join(rows)))
        return 0 if ok_n else 1

    diag = Diag()
    try:
        df, source = fetch_spot(args, diag)
    except Exception as exc:  # noqa: BLE001
        msg = f"行情接口不可用：{str(exc)[:300]}"
        print(msg)
        if args.notify:
            body = msg + "\n\n诊断（最近 10 条）：\n" + "\n".join(diag.lines[-10:]) + "\n\n可用 --probe 查看各数据源状态。"
            print("Server酱:", notify(f"强势股{stamp}|取数失败", body))
        return 1

    result = screen(df, source, args)
    note = session_note(dt)
    has_vr = source.startswith("eastmoney")
    if not has_vr:
        note += f"；⚠ 当前数据源({source})没有量比，仅按涨幅+成交额筛选"
    csv_path = Path(args.out_dir) / f"strong_stocks_{dt.strftime('%Y%m%d_%H%M')}.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tag = "" if has_vr else "·无量比"
    print(f"\n数据源: {source}（取回 {len(df)} 行，其中涨幅≥{args.min_pct}% 的才进入筛选）")

    if result.empty:
        print("\n未找到符合条件的股票。可尝试把 --min-volume-ratio 降到 2.0，或 --min-pct 降到 9。")
        if args.notify and not args.no_notify_empty:
            body = f"{note}\n\n未找到符合条件的股票（涨幅≥{args.min_pct}%、量比≥{args.min_volume_ratio}、成交额≥{args.min_amount_yi}亿）。"
            print("Server酱:", notify(f"强势股{tag}{stamp}|0只", body))
        return 0

    lines = format_lines(result)
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
