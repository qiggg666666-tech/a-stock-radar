#!/usr/bin/env python3
"""筹码尖峰蒸馏研究引擎。

全市场初筛仅使用日线和筹码计算；十大流通股东及主力资金流只对已经
入围的候选补充。所有外部调用均有硬超时并返回显式错误，不能把降级
数据伪装为实时因子。
"""
from __future__ import annotations

import math
import multiprocessing as mp
import re
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

STEP = 0.01
LOOKBACK = 250
WARMUP = 60
PEAK_MIN_RATIO = 0.05
BREAK_RATIO = 1.01
APPROACH_RATIO = 0.97
VOL_MULTIPLIER = 1.5
CONC_THRESHOLD = 0.20
ACTIVE_RATIO = 0.20
DEFAULT_DECAY = 1.0
DISPOSE_K = 0.8


def normalize_code(value: object) -> str:
    code = str(value).strip().replace('.0', '').zfill(6)
    if not re.fullmatch(r'(?:00|30|60|68)\d{4}', code):
        raise ValueError(f'unsupported_a_share_code:{value}')
    return code


def _worker(connection: Any, function: Callable[[], Any]) -> None:
    try:
        connection.send(('ok', function()))
    except Exception as exc:
        connection.send(('error', f'{type(exc).__name__}:{str(exc)[:300]}'))
    finally:
        connection.close()


def provider_call(label: str, timeout_seconds: float, function: Callable[[], Any]) -> Any:
    context = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else None
    if context is None:
        return function()
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, function), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=3)
            raise TimeoutError(f'provider_timeout:{label}:{timeout_seconds:.0f}s')
        state, payload = parent.recv()
        process.join(timeout=3)
        if state != 'ok':
            raise RuntimeError(f'provider_error:{label}:{payload}')
        return payload
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        parent.close()


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if frame is None or frame.empty or any(column not in frame.columns for column in required):
        raise ValueError('invalid_ohlcv_schema')
    output = frame.copy()
    if 'turnover' not in output.columns:
        raise ValueError('missing_turnover_for_chip_formula')
    output['date'] = pd.to_datetime(output['date'], errors='coerce')
    for column in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']:
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors='coerce')
    output = output.dropna(subset=required + ['turnover']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    output = output[(output['close'] > 0) & (output['high'] >= output['low']) & (output['turnover'] >= 0)]
    if len(output) < WARMUP + 10:
        raise ValueError(f'insufficient_history:{len(output)}')
    if float(output['turnover'].sum()) <= 0:
        raise ValueError('nonpositive_turnover_for_chip_formula')
    return output.tail(LOOKBACK + WARMUP).reset_index(drop=True)


def _akshare_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start.replace('-', ''), end_date=end.replace('-', ''), adjust='qfq')
    return _canonicalize(raw.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover'}).assign(turnover=lambda x: pd.to_numeric(x.get('turnover', 0), errors='coerce').fillna(0) / 100.0))


def _baostock_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    exchange = 'sh' if code.startswith(('60', '68')) else 'sz'
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'baostock_login:{login.error_code}:{login.error_msg}')
    try:
        result = bs.query_history_k_data_plus(f'{exchange}.{code}', 'date,open,high,low,close,volume,amount,turn', start_date=start, end_date=end, frequency='d', adjustflag='2')
        if result.error_code != '0':
            raise RuntimeError(f'baostock_history:{result.error_code}:{result.error_msg}')
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
        return _canonicalize(raw.rename(columns={'turn': 'turnover'}).assign(turnover=lambda x: pd.to_numeric(x.get('turnover', 0), errors='coerce').fillna(0) / 100.0))
    finally:
        bs.logout()


def fetch_ohlcv(code: str, signal_date: str = '', timeout_seconds: float = 35.0, retries: int = 2) -> tuple[pd.DataFrame, str, list[str]]:
    code = normalize_code(code)
    end = pd.Timestamp(signal_date).date() if signal_date else datetime.now().date()
    start = end - timedelta(days=LOOKBACK + WARMUP + 100)
    errors: list[str] = []
    for source, function in (('akshare', _akshare_ohlcv), ('baostock', _baostock_ohlcv)):
        for attempt in range(1, max(1, retries) + 1):
            try:
                frame = provider_call(f'{source}:{code}', timeout_seconds, lambda f=function: f(code, start.isoformat(), end.isoformat()))
                return frame[frame['date'].dt.date <= end].reset_index(drop=True), source, errors
            except Exception as exc:
                errors.append(f'{source}:{attempt}:{type(exc).__name__}:{str(exc)[:220]}')
    raise RuntimeError('ohlcv_unavailable:' + ' | '.join(errors))


def _avg_price(row: pd.Series) -> float:
    if float(row.get('amount', 0)) > 0 and float(row.get('volume', 0)) > 0:
        return float(row['amount']) / float(row['volume'])
    return float((row['open'] + row['high'] + row['low'] + row['close']) / 4)


def effective_decay(top10_ratio: float, active_ratio: float = ACTIVE_RATIO, base: float = DEFAULT_DECAY) -> float:
    ratio = min(max(float(top10_ratio or 0.0), 0.0), 0.95)
    return float(base / max(1e-6, 1.0 - ratio * (1.0 - active_ratio)))


def _daily_chip(low: float, high: float, average: float, volume: float) -> dict[float, float]:
    if high < low or volume <= 0:
        return {}
    if math.isclose(high, low):
        return {round(float(high), 2): float(volume)}
    prices = np.unique(np.round(np.linspace(low, high, max(int(round((high - low) / STEP)) + 1, 2)), 2))
    height = 2.0 / max(high - low, 1e-8)
    weights = np.array([max(height / max(average - low, 1e-6) * (price - low), 0) if price <= average else max(height / max(high - average, 1e-6) * (high - price), 0) for price in prices])
    total = float(weights.sum())
    if total <= 0:
        return {float(price): float(volume / len(prices)) for price in prices}
    return {float(price): float(weight) for price, weight in zip(prices, weights / total * volume)}


def _weighted_move_out(chip: dict[float, float], close: float, moved: float, k_dispose: float = DISPOSE_K) -> dict[float, float]:
    total = float(sum(chip.values()))
    if total <= 0 or moved <= 0:
        return dict(chip)
    moved = min(float(moved), 1.0)
    if k_dispose <= 0:
        return {price: weight * (1 - moved) for price, weight in chip.items() if weight * (1 - moved) > 1e-8}
    scores = {price: 1.0 + k_dispose * max((close - price) / max(price, 1e-8), 0.0) for price in chip}
    denominator = sum(scores[price] * weight for price, weight in chip.items())
    if denominator <= 0:
        return {price: weight * (1 - moved) for price, weight in chip.items() if weight * (1 - moved) > 1e-8}
    result: dict[float, float] = {}
    for price, weight in chip.items():
        disposed = total * moved * (scores[price] * weight / denominator)
        remaining = max(weight - disposed, 0.0)
        if remaining > 1e-8:
            result[price] = remaining
    return result


def _update_chip(chip: dict[float, float], row: pd.Series, top10_ratio: float = 0.0) -> dict[float, float]:
    moved = min(float(row.get('turnover', 0)) * effective_decay(top10_ratio), 1.0)
    updated = _weighted_move_out(chip, float(row['close']), moved)
    for price, weight in _daily_chip(float(row['low']), float(row['high']), _avg_price(row), float(row['volume'])).items():
        updated[price] = updated.get(price, 0.0) + weight * moved
    return updated


def _find_peaks(chip: dict[float, float]) -> list[tuple[float, float]]:
    if not chip:
        return []
    items = sorted(chip.items())
    total = sum(weight for _, weight in items)
    maximum = max(weight for _, weight in items)
    threshold = min(PEAK_MIN_RATIO, max(maximum / total * 0.5, 0.005))
    peaks = [(price, weight) for index, (price, weight) in enumerate(items) if ((index == 0 or weight >= items[index - 1][1]) and (index == len(items) - 1 or weight >= items[index + 1][1]) and weight / total >= threshold)]
    return sorted(peaks or [max(items, key=lambda item: item[1])], key=lambda item: item[1], reverse=True)


def concentration_band(chip: dict[float, float], pct: float) -> tuple[float, float, float]:
    items, total = sorted(chip.items()), sum(chip.values())
    if not items or total <= 0:
        return 100.0, 0.0, 0.0
    lower_target, upper_target = total * (1 - pct) / 2, total * (1 + pct) / 2
    cumulative, lower, upper = 0.0, items[0][0], items[-1][0]
    for price, weight in items:
        cumulative += weight
        if cumulative >= lower_target:
            lower = price
            break
    cumulative = 0.0
    for price, weight in items:
        cumulative += weight
        if cumulative >= upper_target:
            upper = price
            break
    width = (upper - lower) / (upper + lower) * 100 if upper + lower > 0 else 100.0
    return float(width), float(lower), float(upper)


def _avg_cost(chip: dict[float, float]) -> float:
    total = sum(chip.values())
    return float(sum(price * weight for price, weight in chip.items()) / total) if total > 0 else 0.0


def _winner(chip: dict[float, float], price: float) -> float:
    total = sum(chip.values())
    return float(sum(weight for chip_price, weight in chip.items() if chip_price < price) / total) if total > 0 else 0.0


def _cross_metrics(previous: dict[float, float], open_price: float, high: float, low: float, close: float, turnover: float) -> tuple[float, float, float, float]:
    total = sum(previous.values())
    if total <= 0 or close <= open_price:
        return 0.0, 0.0, 0.0, 0.0
    body_low, body_high = min(open_price, close), max(open_price, close)
    bar_low, bar_high = min(low, open_price, close), max(high, open_price, close)
    cross = sum(weight for price, weight in previous.items() if body_low <= price <= body_high) / total
    profit = sum(weight for price, weight in previous.items() if bar_low <= price <= bar_high and price < close) / total
    locked = sum(weight for price, weight in previous.items() if bar_low <= price <= bar_high and price > close) / total
    return float(cross), float(profit), float(locked), float(cross / turnover if turnover > 1e-8 else 0.0)


def classify_stage(close: float, average_cost: float, winner: float, conc90: float, recent_high: float, recent_low: float, cross_ratio: float) -> tuple[str, str]:
    if recent_high <= recent_low or average_cost <= 0:
        return '震荡', '数据不足'
    position = (close - recent_low) / (recent_high - recent_low + 1e-8)
    premium = (close - average_cost) / average_cost
    tight, very_tight = conc90 < 15.0, conc90 < 10.0
    if position < 0.35 and winner < 0.45 and tight and abs(premium) < 0.12:
        return '吸筹', '低位较集中，获利盘少'
    if 0.25 < position < 0.55 and 0.35 < winner < 0.65 and conc90 < 22.0 and cross_ratio > 0.08:
        return '洗盘', '中低位洗盘后穿透'
    if position > 0.55 and winner > 0.55 and 0.08 < premium < 0.35:
        return '拉升', '脱离成本区，顺势'
    if position > 0.75 and winner > 0.70 and (very_tight or premium > 0.25):
        return '出货', '高位高获利，警惕'
    if position > 0.80 and winner > 0.75:
        return '出货', '高位风险偏大'
    return '震荡', '筹码分散或多空平衡'


def calc_score(cross_ratio: float, profit_cross: float, locked_cross: float, penetrate: float, winner: float, pct_change: float) -> float:
    cross_score = min(100.0, max(0.0, (cross_ratio - 0.03) / 0.17 * 100))
    side_min, side_sum = min(profit_cross, locked_cross), profit_cross + locked_cross + 1e-8
    balance_score = min(100.0, side_min / 0.03 * 50 + (1 - abs(profit_cross - locked_cross) / side_sum) * 50)
    penetration_score = min(100.0, penetrate / 3.0 * 100)
    winner_score = 100.0 if 0.4 <= winner <= 0.7 else (winner / 0.4 * 80 if winner < 0.4 else max(0.0, 100 - (winner - 0.7) / 0.3 * 100))
    pct_score = 100.0 if 0.02 <= pct_change <= 0.07 else (pct_change / 0.02 * 60 if pct_change < 0.02 else max(0.0, 100 - (pct_change - 0.07) / 0.08 * 80))
    return round(float(cross_score * 0.35 + balance_score * 0.25 + penetration_score * 0.20 + winner_score * 0.15 + pct_score * 0.05), 1)


def flow_score_bonus(flow: dict[str, object], stage: str) -> float:
    score, net, bias, days = 0.0, float(flow.get('flow_net_sum', 0.0)), str(flow.get('flow_bias', '平')), int(flow.get('flow_in_days', 0))
    if stage in {'吸筹', '洗盘', '拉升'} and bias == '流入':
        score += 6 + (3 if days >= 3 else 0)
    if stage in {'吸筹', '洗盘'} and bias == '流出':
        score -= 4
    if stage == '出货' and bias == '流出':
        score += 2
    if stage == '出货' and bias == '流入':
        score -= 5
    if stage == '拉升' and bias == '流出':
        score -= 6
    if abs(net) > 500:
        score += 1 if net > 0 else -1
    return float(max(-10.0, min(10.0, score)))


def _default_flow() -> dict[str, object]:
    return {'flow_net_sum': 0.0, 'flow_net_mean': 0.0, 'flow_net_last': 0.0, 'flow_in_days': 0, 'flow_bias': '平'}


def analyze_frame(code: str, name: str, frame: pd.DataFrame, top10_ratio: float = 0.0, flow: dict[str, object] | None = None) -> dict[str, object]:
    data, flow_data = _canonicalize(frame), flow or _default_flow()
    chip: dict[float, float] = {}
    for _, row in data.head(WARMUP).iterrows():
        chip = _update_chip(chip, row, top10_ratio)
    last: dict[str, object] | None = None
    for index, row in data.iloc[WARMUP:].iterrows():
        previous = dict(chip)
        chip = _update_chip(chip, row, top10_ratio)
        peaks = _find_peaks(chip)
        if not peaks:
            continue
        close, (main_peak, main_weight), total = float(row['close']), peaks[0], sum(chip.values()) or 1.0
        volume_ma5 = float(data.loc[:index, 'volume'].iloc[-6:-1].mean())
        volume_ratio = float(row['volume']) / volume_ma5 if volume_ma5 > 0 else 0.0
        ma5, ma10 = data.loc[:index, 'close'].tail(5).mean(), data.loc[:index, 'close'].tail(10).mean()
        trend_up = bool(close > ma5 and ma5 > ma10)
        conc70, low70, high70 = concentration_band(chip, 0.70)
        conc90, low90, high90 = concentration_band(chip, 0.90)
        approaching = bool(main_peak * APPROACH_RATIO <= close < main_peak * BREAK_RATIO)
        tradeable, signal = False, '⬇️ 未接近'
        if approaching:
            if trend_up and volume_ratio >= VOL_MULTIPLIER and conc90 <= CONC_THRESHOLD * 100:
                signal, tradeable = '🔥 强烈预警（趋势+放量+集中）', True
            elif trend_up and volume_ratio >= VOL_MULTIPLIER:
                signal, tradeable = '⚡ 预警（趋势+放量）', True
            elif trend_up and conc90 <= CONC_THRESHOLD * 100:
                signal, tradeable = '📌 预警（趋势+集中）', True
            elif trend_up:
                signal = '📍 接近尖峰（仅趋势）'
            else:
                signal = '📍 接近尖峰（弱势）'
        elif close >= main_peak * BREAK_RATIO:
            signal = '✅ 已突破尖峰'
        cross, profit_cross, locked_cross, penetrate = _cross_metrics(previous, float(row['open']), float(row['high']), float(row['low']), close, float(row['turnover']))
        winner, average_cost = _winner(chip, close), _avg_cost(chip)
        look = data.iloc[max(0, index - 59):index + 1]
        stage, stage_note = classify_stage(close, average_cost, winner, conc90, float(look['high'].max()), float(look['low'].min()), cross)
        prior_close = float(data.iloc[index - 1]['close']) if index > 0 else float(row['open'])
        pct_change = (close - prior_close) / prior_close if prior_close > 0 else 0.0
        chip_score = calc_score(cross, profit_cross, locked_cross, penetrate, winner, pct_change)
        bonus = flow_score_bonus(flow_data, stage)
        last = {
            'code': normalize_code(code), 'name': name, 'signal_date': str(pd.Timestamp(row['date']).date()), 'close': round(close, 2),
            'main_peak': round(main_peak, 2), 'peak_ratio': round(main_weight / total * 100, 2), 'dist_to_peak_pct': round((close - main_peak) / main_peak * 100, 2),
            'conc70': round(conc70, 2), 'cost70_low': round(low70, 2), 'cost70_high': round(high70, 2),
            'conc90': round(conc90, 2), 'cost90_low': round(low90, 2), 'cost90_high': round(high90, 2),
            'conc70_90_width_pct': round(max(conc90 - conc70, 0.0), 2), 'profit_ratio': round(winner * 100, 2), 'winner': round(winner * 100, 2),
            'cross_ratio': round(cross * 100, 2), 'profit_cross': round(profit_cross * 100, 2), 'locked_cross': round(locked_cross * 100, 2), 'penetrate': round(penetrate, 2),
            'avg_cost': round(average_cost, 2), 'pct_chg': round(pct_change * 100, 2), 'stage': stage, 'stage_note': stage_note,
            'score_chip': chip_score, 'score_flow_bonus': bonus, 'score': round(min(100.0, max(0.0, chip_score + bonus)), 1),
            'flow_bias': str(flow_data['flow_bias']), 'flow_net_sum': round(float(flow_data['flow_net_sum']), 1), 'flow_in_days': int(flow_data['flow_in_days']),
            'top10_float_ratio': round(float(top10_ratio) * 100, 2), 'effective_decay': round(effective_decay(top10_ratio), 3),
            'trend_up': trend_up, 'is_approaching': approaching, 'is_tradeable': tradeable, 'signal': signal,
        }
    if last is None:
        raise ValueError('no_signal_history')
    return last


def _parse_percent_series(values: pd.Series) -> float:
    numeric = pd.to_numeric(values.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False), errors='coerce').dropna()
    return min(max(float(numeric.sum()) / 100.0, 0.0), 0.95) if not numeric.empty else 0.0


def fetch_candidate_factors(code: str, timeout_seconds: float = 12.0) -> tuple[float, dict[str, object], list[str]]:
    errors: list[str] = []
    top10_ratio, flow = 0.0, _default_flow()
    try:
        def top10_request() -> float:
            import akshare as ak
            raw = ak.stock_gdfx_free_top_10_em(symbol=code)
            if raw is None or raw.empty:
                raise ValueError('top10_empty')
            column = next((item for item in raw.columns if '比例' in str(item) or '占比' in str(item)), None)
            if column is None:
                raise ValueError('top10_ratio_column_missing')
            return _parse_percent_series(raw[column])
        top10_ratio = float(provider_call(f'top10:{code}', timeout_seconds, top10_request))
    except Exception as exc:
        errors.append(f'top10:{type(exc).__name__}:{str(exc)[:220]}')
    try:
        def flow_request() -> dict[str, object]:
            import akshare as ak
            raw = ak.stock_individual_fund_flow(stock=code, market='sh' if code.startswith('6') else 'sz')
            if raw is None or raw.empty:
                raise ValueError('fund_flow_empty')
            date_column = next((item for item in raw.columns if '日期' in str(item) or 'date' in str(item).lower()), raw.columns[0])
            net_column = next((item for item in raw.columns if '主力净流入' in str(item) or ('净流入' in str(item) and '超大' not in str(item) and '大单' not in str(item))), None)
            if net_column is None:
                raise ValueError('fund_flow_column_missing')
            values = pd.to_numeric(raw.sort_values(date_column).tail(5)[net_column], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            if len(values) and float(np.nanmax(np.abs(values))) > 1e6:
                values = values / 1e4
            net_sum = float(np.sum(values))
            return {'flow_net_sum': net_sum, 'flow_net_mean': float(np.mean(values)) if len(values) else 0.0, 'flow_net_last': float(values[-1]) if len(values) else 0.0, 'flow_in_days': int(np.sum(values > 0)), 'flow_bias': '流入' if net_sum > 50 else ('流出' if net_sum < -50 else '平')}
        flow = dict(provider_call(f'fund_flow:{code}', timeout_seconds, flow_request))
    except Exception as exc:
        errors.append(f'fund_flow:{type(exc).__name__}:{str(exc)[:220]}')
    return top10_ratio, flow, errors


def enrich_candidate(code: str, name: str, frame: pd.DataFrame, timeout_seconds: float = 12.0) -> tuple[dict[str, object], list[str]]:
    top10_ratio, flow, errors = fetch_candidate_factors(code, timeout_seconds)
    row = analyze_frame(code, name, frame, top10_ratio=top10_ratio, flow=flow)
    row['factor_status'] = 'completed' if not errors else 'partial'
    row['factor_errors'] = ' | '.join(errors)
    row['factor_scope'] = 'candidate_only'
    return row, errors


def self_test() -> None:
    dates = pd.date_range('2025-01-01', periods=120, freq='B')
    frame = pd.DataFrame({'date': dates, 'open': np.linspace(10, 12, 120), 'high': np.linspace(10.2, 12.3, 120), 'low': np.linspace(9.8, 11.7, 120), 'close': np.linspace(10, 12.1, 120), 'volume': np.linspace(1000, 2000, 120), 'amount': np.linspace(10000, 24000, 120), 'turnover': [0.02] * 120})
    result = analyze_frame('000001', '样本', frame)
    assert result['code'] == '000001'
    assert {'conc70', 'conc90', 'cost70_low', 'cost90_high', 'conc70_90_width_pct', 'score'}.issubset(result)
    assert effective_decay(0.5) > effective_decay(0.0)
    assert result['conc90'] >= result['conc70']

