# -*- coding: utf-8 -*-
"""
target_price.py —— 压力位 / 目标价计算模块（只做估算，不是预测最高价）

用法:
    from target_price import calc_targets, format_targets
    res = calc_targets(df, code="600536", name="中国软件")
    print(format_targets(res))

    # 给选股脚本用：返回可直接写进 CSV 的扁平字段 + 展示文本
    from target_price import target_fields, target_tag
    row.update(target_fields(df, code, name))
    text = target_tag(row)

df: 日K DataFrame，至少含 开高低收（成交量缺失时跳过量价密集区）。列名支持英文
    (date/open/high/low/close/volume) 或 akshare 中文
    (日期/开盘/最高/最低/收盘/成交量)。按时间升序，最后一行 = 最新交易日。

五种方法，多个位置重合 = 强压力区:
  FIB     斐波那契反弹位 (0.618 / 0.786)
  BODY    前高K线实体顶部 / 窗口内最高收盘价
  LIMIT   涨停阶梯 (1板/2板/3板)——从「最新收盘价」起算（收盘后看的是下一个交易日）
  MA      日K MA20/MA60、周K MA20
  CHIP    成交量堆积的上方密集区 (用日K量价近似筹码)
  BOWL    （可选）圆弧底量度目标：左沿 + 碗深 × 0.5 / 1.0，由调用方通过 extra_levels 传入

相对原版的改动：
  1) 涨停阶梯改为从最新收盘价起算（limit_base="last"）。原版用昨收，收盘后使用时 1 板价其实是“今天的涨停价”，
     整体错位一板；盘前使用想保持原行为可传 limit_base="prev"。
  2) ST 判断改为名称开头匹配（原版 "ST" in name 会误伤名称里碰巧含 st 的股票）。
  3) 缺少 volume 列不再报错（量价密集区自动跳过）。
  4) 新增 extra_levels（圆弧底量度目标等）、target_fields / target_tag（选股脚本用）。
注：涨跌停幅度 ST=5% 请按当前交易所规则核对（可用 st_limit 参数覆盖）。
"""
import re

import numpy as np
import pandas as pd

_RENAME = {"日期": "date", "开盘": "open", "最高": "high",
           "最低": "low", "收盘": "close", "成交量": "volume"}


def _prep(df):
    d = df.rename(columns=_RENAME).copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date")
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    if "volume" not in d.columns:
        d["volume"] = np.nan
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["open", "high", "low", "close"])


def limit_pct(code, name="", st_limit=0.05):
    """涨跌停幅度: ST 5%（st_limit 可改），创业板/科创板 20%，北交所 30%，其余 10%"""
    code = str(code)
    if re.match(r"^\s*(\*?ST|S\*ST|SST)", str(name).upper().replace(" ", "")):
        return st_limit
    if code.startswith(("688", "689", "300", "301")):
        return 0.20
    if code.startswith(("8", "4", "920")):
        return 0.30
    return 0.10


def _method(label):
    for m in ("FIB", "BODY", "LIMIT", "MA", "CHIP", "BOWL"):
        if label.startswith(m):
            return m
    return label


def _fib_levels(d, lookback):
    w = d.tail(lookback)
    hi_idx = w["high"].idxmax()
    hi = float(w.loc[hi_idx, "high"])
    after = w.loc[hi_idx:]
    lo = float(after["low"].min())
    if hi <= lo:
        return {}, hi, lo, hi_idx
    rng = hi - lo
    lv = {f"FIB{r}": round(lo + r * rng, 2) for r in (0.618, 0.786)}
    return lv, hi, lo, hi_idx


def _volume_profile_peaks(d, lookback, bins=60, top=2):
    w = d.tail(lookback)
    if "volume" not in w or not np.isfinite(w["volume"]).any():
        return []
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if hi <= lo:
        return []
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for _, r in w.iterrows():
        v = r["volume"]
        if not np.isfinite(v) or v <= 0:
            continue
        mask = (edges[1:] > r["low"]) & (edges[:-1] < r["high"])
        n = mask.sum()
        if n:
            vol[mask] += v / n
    if vol.sum() <= 0:
        return []
    centers = (edges[1:] + edges[:-1]) / 2
    last = float(d["close"].iloc[-1])
    cand = []
    for i in range(1, bins - 1):
        if centers[i] > last * 1.003 and vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1] \
                and vol[i] > vol.mean() * 1.3:
            cand.append((vol[i], round(float(centers[i]), 2)))
    cand.sort(reverse=True)
    return [p for _, p in cand[:top]]


def calc_targets(df, code="", name="", lookback=80, chip_lookback=120,
                 merge_pct=0.02, limit_base="last", st_limit=0.05,
                 extra_levels=None):
    """
    extra_levels: [(price, label), ...]，label 以 BOWL 等方法名开头（用于“重合项数”统计）。
    limit_base: "last"=以最新收盘起算（收盘后用，默认）；"prev"=以昨收起算（原版行为，盘前用）。
    """
    d = _prep(df)
    if len(d) < 30:
        return {"ok": False, "msg": "K线不足30根"}
    last = float(d["close"].iloc[-1])
    prev = float(d["close"].iloc[-2])
    cands = []  # (price, label)

    # 1) 斐波那契
    fib, hi, lo, hi_idx = _fib_levels(d, lookback)
    for k, v in fib.items():
        cands.append((v, k))

    # 2) 前高实体顶部 + 窗口最高收盘
    if fib:
        body_top = float(max(d.loc[hi_idx, "open"], d.loc[hi_idx, "close"]))
        cands.append((round(body_top, 2), "BODY前高实体顶"))
    cands.append((round(float(d.tail(lookback)["close"].max()), 2), "BODY最高收盘"))

    # 3) 涨停阶梯
    pct = limit_pct(code, name, st_limit)
    p = last if limit_base == "last" else prev
    for n in (1, 2, 3):
        p = round(p * (1 + pct), 2)
        cands.append((p, f"LIMIT{n}板"))

    # 4) 均线
    c = d["close"]
    for n in (20, 60):
        cands.append((round(float(c.rolling(n).mean().iloc[-1]), 2), f"MA日{n}"))
    wk = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()
    if len(wk) >= 20:
        cands.append((round(float(wk["close"].rolling(20).mean().iloc[-1]), 2), "MA周20"))

    # 5) 量价密集区
    for pk in _volume_profile_peaks(d, chip_lookback):
        cands.append((pk, "CHIP密集区"))

    # 6) 调用方给的额外位置（如圆弧底量度目标）
    for price, label in (extra_levels or []):
        try:
            cands.append((round(float(price), 2), str(label)))
        except (TypeError, ValueError):
            continue

    above = sorted([(p, l) for p, l in cands
                    if np.isfinite(p) and p > last * 1.003])
    below = sorted([(p, l) for p, l in cands
                    if np.isfinite(p) and p <= last * 1.003], reverse=True)

    # 聚类：相邻价格差在 merge_pct 内归为一个压力区
    zones, cur = [], []
    for p, l in above:
        if cur and p > cur[0][0] * (1 + merge_pct):
            zones.append(cur)
            cur = []
        cur.append((p, l))
    if cur:
        zones.append(cur)
    out_zones = []
    for z in zones:
        prices = [x[0] for x in z]
        out_zones.append({
            "lo": min(prices), "hi": max(prices),
            "mid": round(float(np.mean(prices)), 2),
            "score": len({_method(x[1]) for x in z}),
            "items": [f"{l}:{p}" for p, l in z],
        })

    primary = None
    if out_zones:
        best = max(z["score"] for z in out_zones)
        primary = next(z for z in out_zones if z["score"] == best)  # 同分取最近

    return {
        "ok": True, "code": code, "name": name, "last": last,
        "swing_high": round(hi, 2), "swing_low": round(lo, 2),
        "limit_pct": pct, "zones": out_zones, "primary": primary,
        "supports": [(p, l) for p, l in below[:3]],
    }


def format_targets(r):
    if not r.get("ok"):
        return f"压力位计算失败: {r.get('msg')}"
    lines = [f"{r['name']}({r['code']}) 现价 {r['last']:.2f}",
             f"波段 高{r['swing_high']} / 低{r['swing_low']}"]
    for z in r["zones"][:4]:
        lines.append(f"压力区 {z['lo']}~{z['hi']} 重合{z['score']}项 | "
                     + " ".join(z["items"]))
    pz = r["primary"]
    if pz:
        lines.append(f"▶ 重点目标区 {pz['lo']}~{pz['hi']}（重合{pz['score']}项）")
    if r["supports"]:
        lines.append("下方参考 " + " ".join(f"{l}:{p}" for p, l in r["supports"]))
    lines.append("注：压力位≠最高价，仅作止盈参考")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 给选股脚本用的扁平字段 / 展示文本
# ----------------------------------------------------------------------------
def target_fields(df, code="", name="", extra_levels=None, **kw):
    """返回可直接写入 CSV 的 tp_* 字段；数据不足/出错时返回 {"tp_ok": False}，不抛异常。"""
    try:
        r = calc_targets(df, code, name, extra_levels=extra_levels, **kw)
    except Exception:
        return {"tp_ok": False}
    if not r.get("ok"):
        return {"tp_ok": False}
    last = r["last"]
    zones = r["zones"]
    pz = r["primary"]
    out = {
        "tp_ok": True,
        "tp_swing_high": r["swing_high"],
        "tp_swing_low": r["swing_low"],
        "tp_limit_pct": r["limit_pct"],
        "tp_primary_lo": None, "tp_primary_hi": None, "tp_primary_mid": None,
        "tp_primary_score": 0, "tp_primary_items": "", "tp_upside_pct": None,
        "tp_next_zone": "",
        "tp_zones_str": "; ".join(f"{z['lo']}~{z['hi']}({z['score']}项)" for z in zones[:4]),
        "tp_support_str": " ".join(f"{l}:{p}" for p, l in r["supports"][:3]),
    }
    if pz:
        out.update(
            tp_primary_lo=pz["lo"], tp_primary_hi=pz["hi"], tp_primary_mid=pz["mid"],
            tp_primary_score=pz["score"], tp_primary_items=" ".join(pz["items"]),
            tp_upside_pct=round(pz["mid"] / last - 1.0, 4) if last > 0 else None,
        )
        later = [z for z in zones if z["lo"] > pz["hi"]]
        if later:
            n = later[0]
            out["tp_next_zone"] = f"{n['lo']}~{n['hi']}({n['score']}项)"
    return out


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def target_tag(row, sep=" | "):
    """
    把 tp_* 字段拼成一小段推送文本；没有数据时返回空串。
    例：" | 目标区13.2~13.5(重合3项,+8.1%) | 次压力14.4~14.6(2项) | 下方参考 MA日20:12.3 MA日60:11.8"
    """
    lo, hi = _num(row.get("tp_primary_lo")), _num(row.get("tp_primary_hi"))
    if lo is None or hi is None:
        return ""
    score = _num(row.get("tp_primary_score")) or 0
    up = _num(row.get("tp_upside_pct"))
    s = f"目标区{lo:g}~{hi:g}(重合{int(score)}项" + (f",{up * 100:+.1f}%" if up is not None else "") + ")"
    parts = [s]
    nxt = str(row.get("tp_next_zone") or "")
    if nxt and nxt != "nan":
        parts.append(f"次压力{nxt}")
    sup = str(row.get("tp_support_str") or "")
    if sup and sup != "nan":
        parts.append("下方参考 " + " ".join(sup.split()[:2]))
    return sep + sep.join(parts)


if __name__ == "__main__":
    # 自测：需要 akshare（pip install akshare），可换成你自己的数据源
    import akshare as ak
    code = "600536"
    df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq").tail(250)
    print(format_targets(calc_targets(df, code, "中国软件")))
