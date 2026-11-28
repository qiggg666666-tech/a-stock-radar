# -*- coding: utf-8 -*-
"""
predict_today_scan.py (分片版)
MODE=prepare : 板块打分 + 候选池 + 行情快照 -> prep/
MODE=analyze : 读 prep/, 按 SHARD_INDEX/SHARD_TOTAL 分析个股 -> results/result_N.csv
MODE=merge   : 合并 results/, 输出榜单 + Server酱推送 -> output/
MODE=all     : 本地一把跑完(不分片), 方便本地测试

环境变量: SENDKEY, MIN_SCORE, TOP_SECTORS, TOP_CONCEPTS, ALLOW_ZT,
         FORCE(1=非交易日也运行), PER_SHARD(每片股票数), MAX_SHARDS
"""
import os
import sys
import json
import math
import glob
import time
import datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import akshare as ak
import requests

MODE = os.getenv("MODE", "all")
MIN_SCORE = int(os.getenv("MIN_SCORE", "6") or 6)
TOP_SECTORS = int(os.getenv("TOP_SECTORS", 6))
TOP_CONCEPTS = int(os.getenv("TOP_CONCEPTS", 10))
ALLOW_ZT = os.getenv("ALLOW_ZT", "0") == "1"
FORCE = os.getenv("FORCE", "0") == "1"
PER_SHARD = int(os.getenv("PER_SHARD", 80))
MAX_SHARDS = int(os.getenv("MAX_SHARDS", 8))
SHARD_INDEX = int(os.getenv("SHARD_INDEX", 0) or 0)
SHARD_TOTAL = int(os.getenv("SHARD_TOTAL", 1) or 1)
SENDKEY = os.getenv("SENDKEY", "")

PREP_DIR, RES_DIR, OUT_DIR = "prep", "results", "output"

IND_COLS = ["板块", "类型", "昨日涨幅%", "5日涨幅%", "额比", "涨停家数", "得分"]

TODAY = dt.datetime.now(ZoneInfo("Asia/Shanghai")).date()
START = (TODAY - dt.timedelta(days=70)).strftime("%Y%m%d")
END = TODAY.strftime("%Y%m%d")

COLS = [
    "代码", "名称", "昨日涨幅%", "5日涨幅%", "距五日线%", "昨日量比", "收盘位置",
    "流通市值(亿)", "竞价高开%", "昨日涨停", "板块加分", "得分",
]


def limit_ratio(code: str) -> float:
    return 20.0 if code.startswith(("300", "301", "688")) else 10.0


_WARN_COUNT = {}


def warn(tag: str, msg: str, limit: int = 3):
    """同一类错误只打印前几次, 避免刷屏"""
    c = _WARN_COUNT.get(tag, 0) + 1
    _WARN_COUNT[tag] = c
    if c <= limit:
        print(f"[warn] {tag}: {msg}")


def retry(fn, *args, n=3, **kw):
    name = getattr(fn, "__name__", str(fn))
    for _ in range(n):
        try:
            r = fn(*args, **kw)
            if r is not None and len(r) > 0:
                return r
            warn(name + ":empty", "返回空数据")
        except Exception as e:
            warn(name, f"{type(e).__name__}: {str(e)[:200]}")
        time.sleep(1)
    return None


def safe_read(path: str, **kw) -> pd.DataFrame:
    """读取 CSV, 文件为空或不存在时返回空表, 不崩溃"""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"[warn] 读取 {path} 失败: {type(e).__name__}")
        return pd.DataFrame()


def gh_output(key: str, value: str):
    print(f"[output] {key}={value}")
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def trade_days() -> list:
    """交易日历, 兼容 akshare 新旧函数名; 都失败则退化为工作日判断"""
    cal = None
    for name in ("calendar_trade_date_hist_sina", "tool_trade_date_hist_sina"):
        fn = getattr(ak, name, None)
        if fn is None:
            continue
        try:
            cal = fn()
            break
        except Exception as e:
            print(f"{name} 失败:", e)
    if cal is not None and len(cal) > 0:
        col = "trade_date" if "trade_date" in cal.columns else cal.columns[0]
        return list(pd.to_datetime(cal[col]).dt.date)
    print("交易日历获取失败, 退化为工作日判断")
    d0 = dt.date(TODAY.year - 1, 1, 1)
    days = [d0 + dt.timedelta(days=i) for i in range((TODAY - d0).days + 1)]
    return [d for d in days if d.weekday() < 5]


def cut_today(df: pd.DataFrame) -> pd.DataFrame:
    """去掉今天的数据, 只保留截至昨天的日线"""
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.date
    return df[df["日期"] < TODAY].reset_index(drop=True)


EM_FAILS = 0
EM_OFF = False


def sina_symbol(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def fetch_daily(code: str):
    """个股日线(前复权): 先东财, 连续失败则整体切换到新浪
    (GitHub 海外机房常被东财历史行情接口限制)"""
    global EM_FAILS, EM_OFF
    if not EM_OFF:
        h = retry(
            ak.stock_zh_a_hist, n=1,
            symbol=code, period="daily", start_date=START, end_date=END, adjust="qfq",
        )
        if h is not None:
            return cut_today(h)
        EM_FAILS += 1
        if EM_FAILS >= 3:
            EM_OFF = True
            print("[warn] 东财日线连续失败, 切换到新浪数据源")
    h = retry(
        ak.stock_zh_a_daily, n=2,
        symbol=sina_symbol(code), start_date=START, end_date=END, adjust="qfq",
    )
    if h is None:
        return None
    h = h.rename(columns={
        "date": "日期", "open": "开盘", "close": "收盘", "high": "最高",
        "low": "最低", "volume": "成交量", "amount": "成交额",
    })
    h["涨跌幅"] = h["收盘"].pct_change() * 100
    time.sleep(0.3)
    return cut_today(h)


# ---------------------------------------------------------------- 板块 ----
def zt_count_by_industry(day: dt.date) -> dict:
    d = retry(ak.stock_zt_pool_em, date=day.strftime("%Y%m%d"))
    if d is None or "所属行业" not in d.columns:
        return {}
    return d["所属行业"].value_counts().to_dict()


def score_industry(name: str, zt_cnt: dict):
    h = retry(
        ak.stock_board_industry_hist_em,
        symbol=name, start_date=START, end_date=END, period="日k", adjust="",
    )
    if h is None:
        return None
    h = cut_today(h)
    if len(h) < 8:
        return None
    c, amt = h["收盘"], h["成交额"]
    y_chg = float(h["涨跌幅"].iloc[-1])
    chg5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100
    ma5 = c.tail(5).mean()
    base = amt.iloc[-6:-1].mean()
    amt_ratio = amt.iloc[-1] / base if base > 0 else 0
    zt = int(zt_cnt.get(name, 0))

    s = 0
    s += 1 if 0.5 <= y_chg <= 4 else 0
    s += 1 if chg5 <= 3 else 0
    s += 1 if amt_ratio >= 1.2 else 0
    s += 1 if c.iloc[-1] >= ma5 else 0
    s += 1 if (h["涨跌幅"].tail(2) > 0).all() else 0
    s += 2 if zt >= 6 else (1 if zt >= 3 else 0)
    return {
        "板块": name, "类型": "行业", "昨日涨幅%": round(y_chg, 2),
        "5日涨幅%": round(chg5, 2), "额比": round(amt_ratio, 2),
        "涨停家数": zt, "得分": s,
    }


def pick_sectors(prev_day: dt.date):
    zt_cnt = zt_count_by_industry(prev_day)
    names = retry(ak.stock_board_industry_name_em)
    rows = []
    if names is not None:
        plist = names["板块名称"].tolist()
        # 先探测历史接口是否可用, 不可用就跳过整轮请求(省十几分钟)
        hist_ok = False
        for n in plist[:2]:
            r = score_industry(n, zt_cnt)
            if r:
                rows.append(r)
                hist_ok = True
                break
        if hist_ok:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(score_industry, n, zt_cnt) for n in plist[1:]]
                for f in futs:
                    try:
                        r = f.result()
                        if r:
                            rows.append(r)
                    except Exception:
                        pass
        else:
            print("[warn] 行业历史接口不可用, 改用当日涨幅 + 涨停家数排名")
            nm = names.copy()
            tot = (nm["上涨家数"] + nm["下跌家数"]).replace(0, np.nan)
            nm["上涨占比"] = nm["上涨家数"] / tot
            nm = nm[(nm["涨跌幅"] > 0.5) & (nm["上涨占比"] >= 0.6)]
            for _, r in nm.iterrows():
                zt = int(zt_cnt.get(r["板块名称"], 0))
                rows.append({
                    "板块": r["板块名称"], "类型": "行业(无历史)",
                    "昨日涨幅%": round(float(r["涨跌幅"]), 2),
                    "5日涨幅%": None, "额比": None, "涨停家数": zt,
                    "得分": 1 + (2 if zt >= 6 else (1 if zt >= 3 else 0)),
                })
    ind = pd.DataFrame(rows, columns=IND_COLS)
    if not ind.empty:
        ind = ind.drop_duplicates("板块").sort_values(
            ["得分", "昨日涨幅%"], ascending=False
        ).head(TOP_SECTORS)

    con = retry(ak.stock_board_concept_name_em)
    con_top = pd.DataFrame()
    if con is not None:
        con = con.copy()
        tot = (con["上涨家数"] + con["下跌家数"]).replace(0, np.nan)
        con["上涨占比"] = (con["上涨家数"] / tot).round(2)
        con = con[(con["涨跌幅"] > 0.5) & (con["上涨占比"] >= 0.6)]
        con_top = con.sort_values("涨跌幅", ascending=False).head(TOP_CONCEPTS)
    return ind.reset_index(drop=True), con_top.reset_index(drop=True)


def get_cons(kind: str, name: str) -> list:
    fn = ak.stock_board_industry_cons_em if kind == "行业" else ak.stock_board_concept_cons_em
    d = retry(fn, symbol=name)
    if d is None:
        return []
    return d["代码"].astype(str).tolist()


# ---------------------------------------------------------------- 个股 ----
def analyze(code: str, name: str, sector_bonus: int, spot_row: dict):
    h = fetch_daily(code)
    if h is None or len(h) < 12:
        return None

    c, v = h["收盘"], h["成交量"]
    lim = limit_ratio(code)
    last = h.iloc[-1]
    y_chg = float(last["涨跌幅"])
    yday_zt = y_chg >= lim - 0.3
    if yday_zt and not ALLOW_ZT:
        return None

    ma5 = c.tail(5).mean()
    chg5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100
    dist5 = (c.iloc[-1] / ma5 - 1) * 100
    vol_mean5 = v.iloc[-6:-1].mean()
    vol_ratio = v.iloc[-1] / vol_mean5 if vol_mean5 > 0 else 0
    hl = last["最高"] - last["最低"]
    close_pos = (last["收盘"] - last["最低"]) / hl if hl > 0 else 0

    weak = chg5 <= 3
    above_ma5 = 0 <= dist5 <= 4
    mild_vol = 1.3 <= vol_ratio <= 4
    shrink = v.iloc[-6:-1].min() <= v.iloc[-11:-1].mean() * 0.6
    strong_close = close_pos >= 0.7 and 0 <= y_chg <= lim * 0.7
    had_zt = bool((h["涨跌幅"].tail(20) >= lim - 0.3).any())

    cap = spot_row.get("流通市值")
    cap = float(cap) if pd.notna(cap) else None
    cap_ok = cap is not None and 20e8 <= cap <= 300e8

    gap, auction = None, False
    op, pc, lb = spot_row.get("今开"), spot_row.get("昨收"), spot_row.get("量比")
    if pd.notna(op) and pd.notna(pc) and op > 0 and pc > 0:
        gap = (op / pc - 1) * 100
        auction = 0.5 <= gap <= 5 and pd.notna(lb) and lb >= 1.5

    score = sum([weak, above_ma5, mild_vol, shrink, strong_close, had_zt, cap_ok, auction]) + sector_bonus
    return {
        "代码": code, "名称": name, "昨日涨幅%": round(y_chg, 2),
        "5日涨幅%": round(chg5, 2), "距五日线%": round(dist5, 2),
        "昨日量比": round(vol_ratio, 2), "收盘位置": round(close_pos, 2),
        "流通市值(亿)": round(cap / 1e8, 1) if cap else None,
        "竞价高开%": round(gap, 2) if gap is not None else None,
        "昨日涨停": yday_zt, "板块加分": sector_bonus, "得分": score,
    }


def push(title: str, text: str):
    if not SENDKEY:
        print("未设置 SENDKEY, 跳过推送")
        return
    try:
        r = requests.post(
            f"https://sctapi.ftqq.com/{SENDKEY}.send",
            data={"title": title[:90], "desp": text}, timeout=15,
        )
        print("Server酱返回:", r.status_code, r.text[:200])
    except Exception as e:
        print("推送失败:", e)


# ---------------------------------------------------------------- 阶段 ----
def do_prepare():
    os.makedirs(PREP_DIR, exist_ok=True)
    days = trade_days()
    if TODAY not in days and not FORCE:
        print(f"{TODAY} 非交易日, 跳过")
        gh_output("trade", "false")
        return
    prev_day = [d for d in days if d < TODAY][-1]
    print(f"信号日(昨日): {prev_day}  预判日: {TODAY}")

    ind, con = pick_sectors(prev_day)
    ind.to_csv(f"{PREP_DIR}/sectors_ind.csv", index=False, encoding="utf-8-sig")
    con.to_csv(f"{PREP_DIR}/sectors_con.csv", index=False, encoding="utf-8-sig")

    bonus = {}
    for _, r in ind.iterrows():
        for code in get_cons("行业", r["板块"]):
            bonus[code] = max(bonus.get(code, 0), 2 if r["得分"] >= 5 else 1)
    for _, r in con.iterrows():
        for code in get_cons("概念", r["板块名称"]):
            bonus[code] = max(bonus.get(code, 0), 1)
    print(f"强势板块成分股: {len(bonus)} 只")

    spot = retry(ak.stock_zh_a_spot_em)
    if spot is None or not bonus:
        print("行情快照或候选为空, 终止")
        gh_output("trade", "false")
        return
    spot = spot[~spot["名称"].str.contains("ST|退", na=False)]
    spot = spot[~spot["代码"].str.startswith(("8", "4", "9"))]
    spot = spot[spot["代码"].isin(bonus.keys())].copy()
    for col in ["今开", "昨收", "量比", "流通市值"]:
        spot[col] = pd.to_numeric(spot[col], errors="coerce")
    spot["板块加分"] = spot["代码"].map(bonus)
    spot = spot[["代码", "名称", "板块加分", "今开", "昨收", "量比", "流通市值"]]
    spot.to_csv(f"{PREP_DIR}/candidates.csv", index=False, encoding="utf-8-sig")

    n = len(spot)
    total = min(MAX_SHARDS, max(1, math.ceil(n / PER_SHARD)))
    print(f"候选 {n} 只, 分 {total} 片")
    gh_output("trade", "true")
    gh_output("total", str(total))
    gh_output("shards", json.dumps(list(range(total))))


def do_analyze(index: int, total: int):
    os.makedirs(RES_DIR, exist_ok=True)
    cands = pd.read_csv(f"{PREP_DIR}/candidates.csv", dtype={"代码": str})
    mine = cands.iloc[index::total]
    print(f"分片 {index}/{total}: {len(mine)} 只")

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(analyze, r["代码"], r["名称"], int(r["板块加分"]), r.to_dict())
            for _, r in mine.iterrows()
        ]
        for f in futs:
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                print("分析出错:", e)
    out = pd.DataFrame(rows, columns=COLS)
    out.to_csv(f"{RES_DIR}/result_{index}.csv", index=False, encoding="utf-8-sig")
    print(f"分片 {index} 完成, 有效 {len(out)} 只")


def do_merge():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{RES_DIR}/result_*.csv"))
    print(f"收到分片结果 {len(files)} / {SHARD_TOTAL}")
    frames = [safe_read(f, dtype={"代码": str}) for f in files]
    frames = [x for x in frames if not x.empty]
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    if "代码" not in allr.columns:
        allr = pd.DataFrame(columns=COLS)
    allr = allr.drop_duplicates("代码")

    ind = safe_read(f"{PREP_DIR}/sectors_ind.csv")
    con = safe_read(f"{PREP_DIR}/sectors_con.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n===== 行业板块 预判榜 =====")
    print(ind.to_string(index=False) if not ind.empty else "无")
    print("\n===== 概念板块 强势榜(昨日涨幅) =====")
    if not con.empty:
        print(con[["板块名称", "涨跌幅", "上涨占比", "领涨股票"]].to_string(index=False))

    out = allr[allr["得分"] >= MIN_SCORE].sort_values(["得分", "昨日量比"], ascending=False)
    print("\n===== 个股预判榜 =====")
    print(out.to_string(index=False) if not out.empty else f"没有得分>={MIN_SCORE}的股票")
    out.to_csv(f"{OUT_DIR}/predict_today_{TODAY.strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")

    lines = [f"分片 {len(files)}/{SHARD_TOTAL}, 分析 {len(allr)} 只, 入选 {len(out)} 只", ""]
    if not ind.empty:
        lines.append("**行业**: " + "、".join(ind["板块"].tolist()))
    if not con.empty:
        lines.append("")
        lines.append("**概念**: " + "、".join(con["板块名称"].tolist()))
    lines.append("")
    for _, r in out.head(20).iterrows():
        gap = f" 竞价{r['竞价高开%']}%" if pd.notna(r["竞价高开%"]) else ""
        lines.append(
            f"- {r['名称']}({r['代码']}) 得分{r['得分']} 昨日{r['昨日涨幅%']}% "
            f"量比{r['昨日量比']} 距MA5 {r['距五日线%']}%{gap}"
        )
    push(f"{TODAY} 弱转强预判 入选{len(out)}只", "\n\n".join(lines))


def main():
    if MODE == "prepare":
        do_prepare()
    elif MODE == "analyze":
        do_analyze(SHARD_INDEX, SHARD_TOTAL)
    elif MODE == "merge":
        do_merge()
    elif MODE == "all":
        do_prepare()
        if os.path.exists(f"{PREP_DIR}/candidates.csv"):
            do_analyze(0, 1)
            do_merge()
    else:
        print("未知 MODE:", MODE)
        sys.exit(1)


if __name__ == "__main__":
    main()
