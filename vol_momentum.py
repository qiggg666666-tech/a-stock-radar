# -*- coding: utf-8 -*-
"""
vol_momentum.py —— 三项“以 ATR 为尺度”的动能标签（只标注，不参与打分）

借鉴自 harryguiacorn/TradingView-Proprietary-Indicators（Pine Script）的思路，按 A 股日线重写：
  1) MACD-V   ATR 标准化的 MACD：macdv = MACD柱 / ATR(26) * 100。
              不同价位的股票之间可比，可设统一阈值。阈值沿用该仓库的 ±40（柱口径），
              只是参考值，A 股请用回测校准（signal_backtest.py 可按 macdv_tag 分组）。
  2) ADX      Wilder ADX(14)：ADX>=20 视为有趋势，<20 视为震荡；
              “趋势启动”= 最近 5 日内从 20 以下升到 20 以上且仍在上行。
  3) 振幅/ATR 当日真实波幅 TR 与「此前」ATR(14) 的比值；>=2 为“透支”，>=1.5 为“偏大”，
              用来提醒追突破时动能可能已经用掉一部分、止盈可以收紧。

用法:
    from vol_momentum import calc_vol_momentum, vm_tag
    fields = calc_vol_momentum(df)      # df 至少含 open/high/low/close，按时间升序，最后一行=最新交易日
    text = vm_tag(row)                  # 把字段拼成一小段推送文本

字段（均可直接写入 CSV）:
    vm_ok, macdv, macdv_tag(超买/超卖/空串), adx, adx_prev, plus_di, minus_di,
    adx_tag(趋势启动·偏多/趋势启动·偏空/趋势中·偏多/趋势中·偏空/震荡), tr_atr, atr_tag(透支/偏大/空串)
  4) 突破日K线质量（借鉴 BodyMass 思路 + 另一份圆弧底脚本的K线过滤，按阳线重写）
              今日必须是阳线，且 实体占比>=52%、上影<=42%、下影<=38%、阳线实体进入近 26 日阳线实体前 5 名，
              才标“强阳线”；否则给出第一个不满足的原因（阴线/小实体/长上影/长下影/实体不够大）。
  5) QQC-lite 近 9 日多头力量：阳线占比>=55% 且 “带符号实体×成交量”之和>0 → 多头主导；
              阳线占比<=45% 且和<0 → 空头主导；其余为分歧。（只用符号，避免量纲问题）

字段新增: body_ratio, upper_wick_ratio, lower_wick_ratio, body_rank, candle_tag(强阳线/阴线/小实体/长上影/长下影/实体不够大/无振幅),
          qqc_green, qqc_tag(多头主导/空头主导/分歧)
注：这些是通用技术指标，没有 A 股验证；建议先作为标签观察，用回测确认有用再考虑加分。
"""
import numpy as np
import pandas as pd

MACDV_FAST, MACDV_SLOW, MACDV_SIGNAL, MACDV_ATR = 12, 26, 9, 26
MACDV_OVERBOUGHT, MACDV_OVERSOLD = 40.0, -40.0
ADX_LEN = 14
ADX_TREND = 20.0
ATR_LEN = 14
TR_ATR_BIG, TR_ATR_OVER = 1.5, 2.0
CANDLE_BODY_MIN, CANDLE_UPPER_MAX, CANDLE_LOWER_MAX = 0.52, 0.42, 0.38
CANDLE_RANK_LOOKBACK, CANDLE_RANK_TOP = 26, 5
QQC_LOOKBACK, QQC_GREEN_MIN, QQC_GREEN_MAX = 9, 0.55, 0.45
_RENAME = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}


def _rma(s, n):
    """Wilder 平滑（TradingView 的 ta.rma / ta.atr 同款）：alpha = 1/n，从第 n 个有效值起输出。"""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _adx(h, l, c, n=ADX_LEN):
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=h.index).where(up.notna())
    minus_dm = pd.Series(minus_dm, index=h.index).where(down.notna())
    tr_n = _rma(_true_range(h, l, c), n)
    plus = 100.0 * _rma(plus_dm, n) / tr_n
    minus = 100.0 * _rma(minus_dm, n) / tr_n
    total = (plus + minus).replace(0, np.nan)
    dx = (plus - minus).abs() / total
    adx = 100.0 * _rma(dx, n)
    return adx, plus, minus


def calc_vol_momentum(df):
    """数据不足（<60 根）或出错时返回 {"vm_ok": False}，不抛异常。"""
    out = {"vm_ok": False}
    try:
        d = df.rename(columns=_RENAME).copy()
        if "date" in d.columns:
            d = d.sort_values("date")
        for col in ("open", "high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
        if len(d) < 60:
            return out
        h, l, c = d["high"], d["low"], d["close"]

        # ---- 1) MACD-V ----
        ema_f = c.ewm(span=MACDV_FAST, adjust=False).mean()
        ema_s = c.ewm(span=MACDV_SLOW, adjust=False).mean()
        dif = ema_f - ema_s
        dea = dif.ewm(span=MACDV_SIGNAL, adjust=False).mean()
        hist = dif - dea                                   # TradingView 口径：柱 = DIF - DEA
        atr26 = _rma(_true_range(h, l, c), MACDV_ATR)
        macdv = float(hist.iloc[-1] / atr26.iloc[-1] * 100.0) if atr26.iloc[-1] > 0 else np.nan
        macdv_tag = ""
        if np.isfinite(macdv):
            macdv_tag = "超买" if macdv > MACDV_OVERBOUGHT else "超卖" if macdv < MACDV_OVERSOLD else ""

        # ---- 2) ADX ----
        adx, plus, minus = _adx(h, l, c)
        a0, a1 = float(adx.iloc[-1]), float(adx.iloc[-2])
        recent_min = float(adx.iloc[-6:-1].min())
        bull = bool(plus.iloc[-1] > minus.iloc[-1])
        side = "偏多" if bull else "偏空"
        if not np.isfinite(a0):
            adx_tag = ""
        elif a0 >= ADX_TREND and recent_min < ADX_TREND and a0 > a1:
            adx_tag = f"趋势启动·{side}"
        elif a0 >= ADX_TREND:
            adx_tag = f"趋势中·{side}"
        else:
            adx_tag = "震荡"

        # ---- 3) 振幅 / ATR（TR 含跳空；分母用“此前”的 ATR(14)，不含当日，避免当日大振幅把自己的分母拉大）----
        tr = _true_range(h, l, c)
        atr14_prev = float(_rma(tr, ATR_LEN).iloc[-2])
        tr_atr = float(tr.iloc[-1] / atr14_prev) if atr14_prev > 0 else np.nan
        atr_tag = ""
        if np.isfinite(tr_atr):
            atr_tag = "透支" if tr_atr >= TR_ATR_OVER else "偏大" if tr_atr >= TR_ATR_BIG else ""

        # ---- 4) 突破日K线质量 ----
        o = pd.to_numeric(d["open"], errors="coerce")
        o0, h0, l0, c0 = float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), float(c.iloc[-1])
        rng = h0 - l0
        body_ratio = upper_r = lower_r = None
        body_rank = None
        if rng <= 0 or not np.isfinite(o0):
            candle_tag = "无振幅"
        else:
            body = abs(c0 - o0)
            body_ratio = body / rng
            upper_r = (h0 - max(o0, c0)) / rng
            lower_r = (min(o0, c0) - l0) / rng
            if c0 > o0:
                recent = d.iloc[-CANDLE_RANK_LOOKBACK:]
                bull = np.where(recent["close"] > recent["open"], (recent["close"] - recent["open"]).abs(), 0.0)
                body_rank = int((bull > body).sum() + 1)
                # 影线问题优先报告（长影线K线实体必然偏小，报“长上影/长下影”比“小实体”信息量更大）
                if upper_r > CANDLE_UPPER_MAX:
                    candle_tag = "长上影"
                elif lower_r > CANDLE_LOWER_MAX:
                    candle_tag = "长下影"
                elif body_ratio < CANDLE_BODY_MIN:
                    candle_tag = "小实体"
                elif body_rank > CANDLE_RANK_TOP:
                    candle_tag = "实体不够大"
                else:
                    candle_tag = "强阳线"
            else:
                candle_tag = "阴线"

        # ---- 5) QQC-lite：近 9 日多头力量 ----
        rec = d.iloc[-QQC_LOOKBACK:]
        green = int((rec["close"] > rec["open"]).sum())
        green_ratio = green / QQC_LOOKBACK
        commit = np.nan
        if "volume" in rec.columns:
            vol = pd.to_numeric(rec["volume"], errors="coerce")
            sb = np.where(rec["close"] > rec["open"], 1.0, np.where(rec["close"] < rec["open"], -1.0, 0.0)) * (rec["close"] - rec["open"]).abs()
            commit = float((sb * vol).sum())
        if np.isfinite(commit) and green_ratio >= QQC_GREEN_MIN and commit > 0:
            qqc_tag = "多头主导"
        elif np.isfinite(commit) and green_ratio <= QQC_GREEN_MAX and commit < 0:
            qqc_tag = "空头主导"
        else:
            qqc_tag = "分歧"

        out.update(
            body_ratio=round(body_ratio, 3) if body_ratio is not None else None,
            upper_wick_ratio=round(upper_r, 3) if upper_r is not None else None,
            lower_wick_ratio=round(lower_r, 3) if lower_r is not None else None,
            body_rank=body_rank,
            candle_tag=candle_tag,
            qqc_green=green,
            qqc_tag=qqc_tag,
        )
        out.update(
            vm_ok=True,
            macdv=round(macdv, 1) if np.isfinite(macdv) else None,
            macdv_tag=macdv_tag,
            adx=round(a0, 1) if np.isfinite(a0) else None,
            adx_prev=round(a1, 1) if np.isfinite(a1) else None,
            plus_di=round(float(plus.iloc[-1]), 1) if np.isfinite(plus.iloc[-1]) else None,
            minus_di=round(float(minus.iloc[-1]), 1) if np.isfinite(minus.iloc[-1]) else None,
            adx_tag=adx_tag,
            tr_atr=round(tr_atr, 2) if np.isfinite(tr_atr) else None,
            atr_tag=atr_tag,
        )
    except Exception:
        return {"vm_ok": False}
    return out


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _txt(v):
    s = "" if v is None else str(v)
    return "" if s == "nan" or s == "None" else s


def vm_tag(row, sep=" | "):
    """
    例：" | MACD-V+23 | ADX27趋势启动·偏多 | 振幅/ATR2.3⚠透支 | K线:强阳线(阳线实体第2名) | QQC多头主导(阳6/9)"
    MACD-V、ADX、K线、QQC 有数据就显示；振幅/ATR 只在 >=1.5（偏大/透支）时显示，避免刷屏。没有数据返回空串。
    """
    parts = []
    mv = _num(row.get("macdv"))
    if mv is not None:
        tag = _txt(row.get("macdv_tag"))
        parts.append(f"MACD-V{mv:+.0f}" + (f"({tag})" if tag else ""))
    adx = _num(row.get("adx"))
    if adx is not None:
        parts.append(f"ADX{adx:.0f}{_txt(row.get('adx_tag'))}")
    tr = _num(row.get("tr_atr"))
    if tr is not None and tr >= TR_ATR_BIG:
        warn = _txt(row.get("atr_tag"))
        parts.append(f"振幅/ATR{tr:.1f}" + ("⚠透支" if warn == "透支" else "偏大"))
    ct = _txt(row.get("candle_tag"))
    if ct:
        rk = _num(row.get("body_rank"))
        parts.append("K线:" + ct + (f"(阳线实体第{int(rk)}名)" if ct == "强阳线" and rk is not None else ""))
    qt = _txt(row.get("qqc_tag"))
    if qt:
        g = _num(row.get("qqc_green"))
        parts.append(f"QQC{qt}" + (f"(阳{int(g)}/{QQC_LOOKBACK})" if g is not None else ""))
    return (sep + sep.join(parts)) if parts else ""
