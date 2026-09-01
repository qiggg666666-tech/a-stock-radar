#!/usr/bin/env python3
"""FINAL Chip的日线筹码峰、集中度、突破和五维评分研究内核。"""
from __future__ import annotations

import math
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

DECAY, STEP, LOOKBACK = 0.5, 0.01, 100
# DECAY=0.5 实测(600714真实数据)不够：below_peak/wide_zone 跟 DECAY=1.0 时几乎没变化，
# 说明放量拉升阶段的量能差距不是靠一个全局衰减乘数能压住的——DECAY 砍太小又会让所有
# 股票的近期筹码都变迟钝，误伤真正该看见的"最近放量突破"信号。
# 改用"衰减下限"：只保护远期筹码，近期窗口维持正常敏感度。
OLD_CHIP_AGE_DAYS = 30       # 最近这么多个交易日算"近期窗口"，正常衰减
OLD_CHIP_DECAY_CAP = 0.02    # 超过近期窗口的"远期筹码"，每天最多衰减这个比例，不管当天换手率多高
# DECAY 从 1.0 降到 0.5（先试这个值）：DECAY 直接乘在换手率上决定当日筹码"清空"比例
# (moved = turnover * DECAY)，降到0.5相当于把每天的筹码更新速度砍半，老筹码保留更久，
# 让11.3~15这类几个月前的堆积区更不容易被后续的高换手行情衰减掉，能重新进入
# below_peak / wide_zone 的候选范围。如果 0.5 让老堆积区权重压过近期筹码太多(反而找
# 不到最近的)，或者还是不够(老堆积区还是被冲掉)，再往 0.3~0.8 之间调。
# 窄幅但非零区间(涨停/一字板/单日巨量窄幅拉升)：区间宽度相对现价趋近于零时，
# 三角分布公式里的分母(high-low)理论上该让密度趋于无穷、图形收成一根尖峰，
# 但原来只有 high==low 严格相等才会走"合并成单一价位"这条路径，差一点点
# (哪怕零点几分钱)就会被摊到旁边几个 STEP 价格步长里，尖峰被削平。
# 加一个相对阈值：区间宽度小于 现价×NARROW_DAY_REL_TOL 就直接当分母趋近零处理。
NARROW_DAY_REL_TOL = 0.003
# 现价下方长红柱：below_band_ratio(窄不窄) 和 below_peak_gap(孤不孤立) 都够高才算。
# 起始值跟之前 github_chip_scan.py 里 SPIKE_BAND_MIN/SPIKE_GAP_MIN 一致，具体够不够
# 严格要等换手率+窄幅尖峰两处修复后拿全市场真实数据重新核一遍再调。
BELOW_SPIKE_BAND_MIN, BELOW_SPIKE_GAP_MIN = 0.12, 3.0
# 占比阈值：这根"下方峰"至少要占现价下方总筹码的多少比例，才算"长"，不是随便一根
# 小尖刺。0.008 沿用参照脚本里验证过思路的量级(1%以内)，同样未经真实数据校准。
BELOW_SPIKE_RATIO_MIN = 0.008

# 宽幅堆积区：不要求单点突出，要求"一段连续区间"整体圈住够多筹码、且区间够窄(不是
# 现价下方全部价格都算)。WIDE_ZONE_MASS_TARGET=圈住现价下方多少比例的筹码；
# WIDE_ZONE_MAX_WIDTH_PCT/MIN_WIDTH_PCT=这段区间宽度相对现价的上下限——宽度上限防止
# "现价下方全部价格"都被当成一个大区间(没有意义)；宽度下限是为了跟 is_below_spike
# (单点尖峰)错开，避免同一个窄区间被两条信号重复报。三个数字都未经真实数据校准。
WIDE_ZONE_MASS_TARGET = 0.55
WIDE_ZONE_MIN_WIDTH_PCT = 0.05
WIDE_ZONE_MAX_WIDTH_PCT = 0.35
APPROACH_RATIO, BREAK_RATIO, VOL_MULTIPLIER, CONC_THRESHOLD = 0.97, 1.01, 1.5, 0.20
WEIGHTS = {"line": 0.15, "conc": 0.15, "peak": 0.10, "break": 0.45, "profit": 0.15}


def _call(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send((True, function()))
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}:{str(exc)[:300]}"))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, function: Callable[[], Any]) -> Any:
    if "fork" not in mp.get_all_start_methods():
        return function()
    parent, child = mp.get_context("fork").Pipe(duplex=False)
    process = mp.get_context("fork").Process(target=_call, args=(child, function), daemon=True)
    process.start(); child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate(); process.join(timeout=2)
            raise TimeoutError(f"provider_timeout:{label}:{timeout_seconds:.0f}s")
        ok, value = parent.recv(); process.join(timeout=2)
        if not ok:
            raise RuntimeError(f"provider_error:{label}:{value}")
        return value
    finally:
        if process.is_alive():
            process.terminate(); process.join(timeout=2)
        parent.close()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover", "turn": "turnover"}
    data = frame.rename(columns={key: value for key, value in columns.items() if key in frame.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    if data.empty or any(item not in data.columns for item in required):
        raise ValueError("invalid_ohlcv_schema")
    for item in required[1:] + ["amount", "turnover"]:
        if item not in data:
            data[item] = 0.0
        data[item] = pd.to_numeric(data[item], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=required).sort_values("date").drop_duplicates("date")
    # 修复：baostock/akshare 的换手率本来就是百分比数字（如 1.85 代表1.85%），
    # 统一 /100 转成 0~1 小数。旧逻辑 t>1.5 才转换，导致绝大多数正常换手率
    # （数字本身 <1.5，即真实换手率<1.5%这个常见情况）被当成小数直接存了
    # 0.5~5 这种量级，冲垮了 update_chip 里的当日筹码衰减权重（几乎每天清零）。
    turnover = data["turnover"].fillna(0.0).astype(float)
    data["turnover"] = turnover / 100.0
    data = data[(data["close"] > 0) & (data["high"] >= data["low"]) & (data["volume"] > 0)]
    if len(data) < 40:
        raise ValueError(f"insufficient_history:{len(data)}")
    return data.tail(LOOKBACK).reset_index(drop=True)


def _ak_history(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    return normalize_frame(ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq"))


def _bs_history(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_code}:{login.error_msg}")
    try:
        exchange = "sh" if code.startswith(("60", "68")) else "sz"
        result = bs.query_history_k_data_plus(f"{exchange}.{code}", "date,open,high,low,close,volume,amount,turn", start_date=start, end_date=end, frequency="d", adjustflag="2")
        if result.error_code != "0":
            raise RuntimeError(f"baostock_history:{result.error_code}:{result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return normalize_frame(pd.DataFrame(rows, columns=result.fields))
    finally:
        bs.logout()


def fetch_ohlcv(code: str, timeout_seconds: float = 35, retries: int = 2) -> tuple[pd.DataFrame, str, list[str]]:
    end, start = datetime.now().date(), datetime.now().date() - timedelta(days=LOOKBACK + 110)
    errors: list[str] = []
    for label, request in (("akshare", _ak_history), ("baostock", _bs_history)):
        for attempt in range(1, max(retries, 1) + 1):
            try:
                return provider_call(f"{label}:{code}", timeout_seconds, lambda fn=request: fn(code, start.isoformat(), end.isoformat())), label, errors
            except Exception as exc:
                errors.append(f"{label}:{attempt}:{type(exc).__name__}:{str(exc)[:220]}")
    raise RuntimeError("ohlcv_unavailable:" + " | ".join(errors))


def average_price(row: pd.Series) -> float:
    if float(row["amount"]) > 0 and float(row["volume"]) > 0:
        return float(row["amount"]) / float(row["volume"])
    return float((row["open"] + row["high"] + row["low"] + row["close"]) / 4)


def daily_distribution(low: float, high: float, average: float, volume: float) -> dict[float, float]:
    if high < low or volume <= 0:
        return {}
    if math.isclose(high, low) or (high - low) <= max(average, 1e-8) * NARROW_DAY_REL_TOL:
        return {round(average, 2): volume}
    prices = np.unique(np.round(np.linspace(low, high, max(int(round((high - low) / STEP)) + 1, 2)), 2))
    values = np.array([max((price - low) / max(average - low, 1e-8), 0.0) if price <= average else max((high - price) / max(high - average, 1e-8), 0.0) for price in prices])
    values = values / values.sum() if values.sum() > 0 else np.ones(len(prices)) / len(prices)
    return {float(price): float(weight * volume) for price, weight in zip(prices, values)}


def update_chip(chip: dict[float, float], row: pd.Series, decay_cap: float = 1.0) -> dict[float, float]:
    """decay_cap 只限制"已有存量筹码"当天被衰减掉多少比例；当天新增筹码永远按
    当天真实换手率全额吸收，不受 decay_cap 限制——否则远期窗口(老堆积区自己形成
    的那段时期)新增筹码也会被一起砍掉，反而把堆积区自身的形成过程削弱了。"""
    absorb = min(max(float(row["turnover"]) * DECAY, 0.0), 1.0)
    erode = min(absorb, decay_cap)
    result = {price: weight * (1 - erode) for price, weight in chip.items() if weight * (1 - erode) > 1e-8}
    for price, weight in daily_distribution(float(row["low"]), float(row["high"]), average_price(row), float(row["volume"])).items():
        result[price] = result.get(price, 0.0) + weight * absorb
    return result


def find_wide_zone(prices_b: np.ndarray, weights_b: np.ndarray, below_total: float, mass_target: float) -> tuple[float, float, float]:
    """在现价下方(prices_b已按价格升序)用双指针滑窗，找覆盖至少 mass_target 比例
    筹码所需要的最窄价格区间。返回 (区间下界, 区间上界, 实际圈住的占比)。
    跟 below_peak 那套"单点尖峰"逻辑完全不同：这里不要求任何单点突出，只看
    "一段连续区间"整体权重够不够高、区间够不够窄。"""
    n = len(prices_b)
    if n == 0 or below_total <= 1e-8:
        return 0.0, 0.0, 0.0
    target = below_total * mass_target
    left = 0
    window_sum = 0.0
    best_width = float("inf")
    best_low, best_high, best_ratio = float(prices_b[0]), float(prices_b[-1]), 0.0
    for right in range(n):
        window_sum += float(weights_b[right])
        while window_sum - float(weights_b[left]) >= target and left < right:
            window_sum -= float(weights_b[left])
            left += 1
        if window_sum >= target:
            width = float(prices_b[right] - prices_b[left])
            if width < best_width:
                best_width = width
                best_low, best_high = float(prices_b[left]), float(prices_b[right])
                best_ratio = window_sum / below_total
    return best_low, best_high, best_ratio


def features(chip: dict[float, float], close: float) -> dict[str, float]:
    items = sorted(chip.items())
    if not items:
        raise ValueError("empty_chip")
    prices = np.array([item[0] for item in items]); weights = np.array([item[1] for item in items]); total = weights.sum()
    peak_index = int(np.argmax(weights)); main_peak, main_weight = float(prices[peak_index]), float(weights[peak_index])
    local = [index for index in range(len(weights)) if (index == 0 or weights[index] >= weights[index - 1]) and (index == len(weights) - 1 or weights[index] >= weights[index + 1])]
    second = float(sorted((weights[index] for index in local), reverse=True)[1]) if len(local) > 1 else main_weight * 0.01
    cumulative = np.cumsum(weights) / total
    p5, p95 = float(prices[min(np.searchsorted(cumulative, .05), len(prices) - 1)]), float(prices[min(np.searchsorted(cumulative, .95), len(prices) - 1)])
    band = max(close * 0.01, STEP * 5)
    # 现价下方的长红柱(用户口径：红=现价下方，不是全局最高峰，也不是"套牢区")。
    # 在全部局部峰(local)里只看价格<现价的那些，挑权重最大的一个，再单独算它
    # 自己的 band_ratio(够不够窄)、peak_gap(比"下方其它峰"高多少，只跟下方比，
    # 不跟上方峰比，否则上方随便一个大峰会不公平地拉低下方峰的锐度分)、
    # peak_ratio(占下方总筹码的比例，够不够"长"/占地方)。三个都要达标才算显著。
    below_indices = [index for index in local if prices[index] < close]
    below_mask = prices < close
    below_total = float(weights[below_mask].sum()) if below_mask.any() else 0.0
    if below_indices and below_total > 1e-8:
        below_index = max(below_indices, key=lambda index: weights[index])
        below_peak, below_weight = float(prices[below_index]), float(weights[below_index])
        other_below_peaks = sorted((weights[index] for index in below_indices if index != below_index), reverse=True)
        below_second = float(other_below_peaks[0]) if other_below_peaks else below_weight * 0.01
        below_gap = below_weight / max(below_second, 1e-8)
        below_band_ratio = float(weights[below_mask & (prices >= below_peak - band) & (prices <= below_peak + band)].sum() / below_total)
        below_peak_ratio = below_weight / below_total
        wide_low, wide_high, wide_ratio = find_wide_zone(prices[below_mask], weights[below_mask], below_total, WIDE_ZONE_MASS_TARGET)
    else:
        below_peak, below_gap, below_band_ratio, below_peak_ratio = None, 1.0, 0.0, 0.0
        wide_low, wide_high, wide_ratio = 0.0, 0.0, 0.0
    return {"main_peak": main_peak, "peak_ratio": main_weight / total, "peak_gap": main_weight / max(second, 1e-8), "band_ratio": float(weights[(prices >= main_peak - band) & (prices <= main_peak + band)].sum() / total), "conc90": (p95 - p5) / (p95 + p5) if p95 + p5 > 0 else 1.0, "p5": p5, "p95": p95, "avg_cost": float((prices * weights).sum() / total), "profit": float(weights[prices <= close].sum() / total), "below_peak": below_peak, "below_peak_gap": below_gap, "below_band_ratio": below_band_ratio, "below_peak_ratio": below_peak_ratio, "wide_zone_low": wide_low, "wide_zone_high": wide_high, "wide_zone_ratio": wide_ratio}


def analyze(code: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    data = normalize_frame(frame); chip: dict[float, float] = {}
    n = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        cap = 1.0 if i >= n - OLD_CHIP_AGE_DAYS else OLD_CHIP_DECAY_CAP
        chip = update_chip(chip, row, decay_cap=cap)
    last, close, feat = data.iloc[-1], float(data.iloc[-1]["close"]), features(chip, float(data.iloc[-1]["close"]))
    ma5, ma10 = data["close"].tail(5).mean(), data["close"].tail(10).mean(); trend = bool(close > ma5 and ma5 > ma10)
    volume_ma = data["volume"].iloc[-6:-1].mean(); volume_ratio = float(last["volume"] / volume_ma) if volume_ma > 0 else 0.0
    in_zone = feat["main_peak"] * APPROACH_RATIO <= close < feat["main_peak"] * BREAK_RATIO
    confirmed = volume_ratio >= VOL_MULTIPLIER and feat["conc90"] <= CONC_THRESHOLD
    tradeable = bool(in_zone and trend and confirmed)
    amplitude = (float(last["high"]) - float(last["low"])) / close if close else 1.0
    line = 100 * (0.30 * np.clip(1 - amplitude / .05, 0, 1) + .25 * np.clip(float(last["turnover"]) / .03, 0, 1) + .30 * np.clip(feat["band_ratio"] / .25, 0, 1) + .15 * (1 if amplitude < .012 else .5 if amplitude < .025 else 0))
    conc = 100 * np.clip(1 - (feat["conc90"] - .05) / .35, 0, 1); peak = 100 * (.55 * np.clip(feat["band_ratio"] / .30, 0, 1) + .25 * np.clip(np.log1p(feat["peak_gap"]) / 4, 0, 1) + .20 * np.clip(feat["peak_ratio"] / .02, 0, 1))
    position = np.clip((close - feat["main_peak"] * APPROACH_RATIO) / max(feat["main_peak"] * (BREAK_RATIO - APPROACH_RATIO), 1e-8), 0, 1) if in_zone else (0.35 if close >= feat["main_peak"] * BREAK_RATIO else max(0, 1 + (close - feat["main_peak"]) / feat["main_peak"] / .15))
    breakout = 100 * (.35 * position + .25 * (1 if trend else .25) + .20 * np.clip(volume_ratio / VOL_MULTIPLIER, 0, 1.2) / 1.2 + .10 * (1 if confirmed else .4) + .10 * (1 if tradeable else .45))
    profit_pct = feat["profit"] * 100; profit = 100 if 20 <= profit_pct <= 55 else max(5, 100 - profit_pct) if profit_pct > 70 else max(10, profit_pct * 2) if profit_pct < 10 else 60
    total = WEIGHTS["line"] * line + WEIGHTS["conc"] * conc + WEIGHTS["peak"] * peak + WEIGHTS["break"] * breakout + WEIGHTS["profit"] * profit
    is_below_spike = bool(feat["below_peak"] is not None and feat["below_band_ratio"] >= BELOW_SPIKE_BAND_MIN and feat["below_peak_gap"] >= BELOW_SPIKE_GAP_MIN and feat["below_peak_ratio"] >= BELOW_SPIKE_RATIO_MIN)

    # ====== 宽幅堆积区：判定 + 评分 + 洗盘/买入状态 ======
    wide_width_pct = 0.0
    is_wide_zone = False
    wide_score = 0.0
    wide_state = "无"
    wide_dist_pct = None
    if feat["wide_zone_high"] > 0 and close > 0:
        wide_width_pct = (feat["wide_zone_high"] - feat["wide_zone_low"]) / close * 100
        is_wide_zone = bool(
            feat["wide_zone_ratio"] >= WIDE_ZONE_MASS_TARGET
            and WIDE_ZONE_MIN_WIDTH_PCT * 100 <= wide_width_pct <= WIDE_ZONE_MAX_WIDTH_PCT * 100
        )
        if is_wide_zone:
            wz_top = feat["wide_zone_high"]
            wide_dist_pct = (close - wz_top) / wz_top * 100
            wide_in_zone = wz_top * APPROACH_RATIO <= close < wz_top * BREAK_RATIO
            wide_confirmed = bool(volume_ratio >= VOL_MULTIPLIER and trend)
            s_ratio = np.clip((feat["wide_zone_ratio"] - WIDE_ZONE_MASS_TARGET) / (1 - WIDE_ZONE_MASS_TARGET), 0, 1)
            s_narrow = np.clip(1 - wide_width_pct / (WIDE_ZONE_MAX_WIDTH_PCT * 100), 0, 1)
            s_pos = 1.0 if wide_in_zone else float(np.clip(1 - abs(wide_dist_pct) / 30, 0, 1))
            s_confirm = 1.0 if wide_confirmed else (0.6 if trend else 0.3)
            wide_score = float(np.clip(100 * (0.30 * s_ratio + 0.25 * s_narrow + 0.25 * s_pos + 0.20 * s_confirm), 0, 100))
            if wide_in_zone and wide_confirmed:
                wide_state = "买入·贴近宽幅堆积区上沿+量能趋势确认"
            elif wide_in_zone:
                wide_state = "洗盘·贴近宽幅堆积区上沿未确认量能趋势"
            elif close < wz_top * BREAK_RATIO:
                wide_state = "洗盘·宽幅堆积区蓄势中"
            else:
                wide_state = "观察·已远离宽幅堆积区"

    signal = "可交易·接近尖峰+趋势确认" if tradeable else "尖峰关注·现价下方长红柱" if is_below_spike else "观察·接近尖峰未确认" if in_zone else "无"

    # ====== 现价下方长红柱专属：评分 + 洗盘/买入状态 ======
    # 复用跟 main_peak 一样的接近/突破比例(APPROACH_RATIO/BREAK_RATIO)，但锚点换成 below_peak。
    below_score = 0.0
    below_state = "无"
    below_dist_pct = None
    if feat["below_peak"] is not None and feat["below_peak"] > 0:
        bp = feat["below_peak"]
        below_dist_pct = (close - bp) / bp * 100
        below_in_zone = bp * APPROACH_RATIO <= close < bp * BREAK_RATIO
        below_confirmed = bool(volume_ratio >= VOL_MULTIPLIER and trend)
        s_band = np.clip(feat["below_band_ratio"] / 0.95, 0, 1)
        s_gap = np.clip(np.log1p(feat["below_peak_gap"]) / np.log1p(50), 0, 1)
        s_ratio = np.clip(feat["below_peak_ratio"] / 0.05, 0, 1)
        s_pos = 1.0 if below_in_zone else float(np.clip(1 - abs(below_dist_pct) / 30, 0, 1))
        s_confirm = 1.0 if below_confirmed else (0.6 if trend else 0.3)
        below_score = float(np.clip(100 * (0.30 * s_band + 0.15 * s_gap + 0.15 * s_ratio + 0.25 * s_pos + 0.15 * s_confirm), 0, 100))
        if not is_below_spike:
            below_state = "无"
        elif below_in_zone and below_confirmed:
            below_state = "买入·贴近下方长红柱+量能趋势确认"
        elif below_in_zone:
            below_state = "洗盘·贴近下方长红柱未确认量能趋势"
        elif close < bp * BREAK_RATIO:
            below_state = "洗盘·下方长红柱蓄势中"
        else:
            below_state = "观察·已远离下方长红柱"
    return {"code": str(code).zfill(6), "name": name, "date": str(last["date"].date()), "close": round(close, 2), "main_peak": round(feat["main_peak"], 2), "avg_cost": round(feat["avg_cost"], 2), "dist_to_peak_pct": round((close - feat["main_peak"]) / feat["main_peak"] * 100, 2), "band_ratio_pct": round(feat["band_ratio"] * 100, 2), "conc90_pct": round(feat["conc90"] * 100, 2), "profit_pct": round(profit_pct, 2), "p5": round(feat["p5"], 2), "p95": round(feat["p95"], 2), "turnover_pct": round(float(last["turnover"]) * 100, 2), "volume_ratio": round(volume_ratio, 2), "line_score": round(float(line), 1), "conc_score": round(float(conc), 1), "peak_score": round(float(peak), 1), "break_score": round(float(breakout), 1), "profit_score": round(float(profit), 1), "total_score": round(float(total), 1), "is_approaching": in_zone, "is_tradeable": tradeable, "below_peak": round(feat["below_peak"], 2) if feat["below_peak"] is not None else None, "below_dist_pct": round(below_dist_pct, 2) if below_dist_pct is not None else None, "below_band_ratio_pct": round(feat["below_band_ratio"] * 100, 2), "below_peak_ratio_pct": round(feat["below_peak_ratio"] * 100, 2), "below_peak_gap": round(feat["below_peak_gap"], 2), "below_score": round(below_score, 1), "below_state": below_state, "is_below_spike": is_below_spike, "wide_zone_low": round(feat["wide_zone_low"], 2) if feat["wide_zone_high"] > 0 else None, "wide_zone_high": round(feat["wide_zone_high"], 2) if feat["wide_zone_high"] > 0 else None, "wide_zone_ratio_pct": round(feat["wide_zone_ratio"] * 100, 2), "wide_width_pct": round(wide_width_pct, 2), "wide_dist_pct": round(wide_dist_pct, 2) if wide_dist_pct is not None else None, "wide_score": round(wide_score, 1), "wide_state": wide_state, "is_wide_zone": is_wide_zone, "signal": signal, "profile": "winrate", "confirm_mode": "and"}


def self_test() -> None:
    dates = pd.date_range("2025-01-01", periods=110, freq="B"); close = np.linspace(10, 12, 110)
    frame = pd.DataFrame({"date": dates, "open": close-.1, "high": close+.2, "low": close-.2, "close": close, "volume": np.linspace(1000, 2000, 110), "amount": close*np.linspace(1000, 2000, 110), "turnover": [.02]*110})
    result = analyze("000001", "样本", frame)
    assert result["code"] == "000001" and 0 <= result["total_score"] <= 100 and "conc90_pct" in result
