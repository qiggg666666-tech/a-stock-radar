#!/usr/bin/env python3
"""筹码尖峰蒸馏研究引擎。

保留原始筹码分布、尖峰、集中度、量比和趋势信号公式；本模块只负责
信号日及以前的日线研究计算，并提供双源、硬超时与显式错误接口。
"""
from __future__ import annotations

import math
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

STEP = 0.01
DECAY = 1.0
LOOKBACK = 250
WARMUP = 60
PEAK_MIN_RATIO = 0.05
BREAK_RATIO = 1.01
APPROACH_RATIO = 0.97
VOL_MULTIPLIER = 1.5
CONC_THRESHOLD = 0.20
HOLD_DAYS = 5


def normalize_code(value: object) -> str:
    code = str(value).strip().replace('.0', '').zfill(6)
    if not __import__('re').fullmatch(r'(?:00|30|60|68)\d{4}', code):
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
        raise ValueError('missing_turnover_for_original_chip_formula')
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
        raise ValueError('nonpositive_turnover_for_original_chip_formula')
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


def _daily_chip(low: float, high: float, avg: float, volume: float) -> dict[float, float]:
    if high < low or volume <= 0:
        return {}
    if math.isclose(high, low):
        return {float(high): float(volume)}
    n_steps = max(int(round((high - low) / STEP)) + 1, 2)
    prices = np.unique(np.round(np.linspace(low, high, n_steps), 2))
    height = 2.0 / (high - low)
    weights = np.array([max(height / max(avg - low, 1e-6) * (price - low), 0) if price <= avg else max(height / max(high - avg, 1e-6) * (high - price), 0) for price in prices])
    total = float(weights.sum())
    if total <= 0:
        return {float(price): float(volume / len(prices)) for price in prices}
    return {float(price): float(weight) for price, weight in zip(prices, weights / total * volume)}


def _update_chip(chip: dict[float, float], row: pd.Series) -> dict[float, float]:
    moved = min(float(row.get('turnover', 0)) * DECAY, 1.0)
    updated = {price: weight * (1.0 - moved) for price, weight in chip.items() if weight * (1.0 - moved) > 1e-8}
    daily = _daily_chip(float(row['low']), float(row['high']), _avg_price(row), float(row['volume']))
    for price, weight in daily.items():
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
    if not peaks:
        peaks = [max(items, key=lambda item: item[1])]
    return sorted(peaks, key=lambda item: item[1], reverse=True)


def _concentration(chip: dict[float, float], pct: float = 0.9) -> float:
    items, total = sorted(chip.items()), sum(chip.values())
    if not items or total <= 0:
        return 1.0
    target, cumulative, low = total * (1 - pct) / 2, 0.0, items[0][0]
    for price, weight in items:
        cumulative += weight
        if cumulative >= target:
            low = price
            break
    cumulative, high = 0.0, items[-1][0]
    for price, weight in reversed(items):
        cumulative += weight
        if cumulative >= target:
            high = price
            break
    return (high - low) / (high + low) if high + low > 0 else 1.0


def analyze_frame(code: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    data = _canonicalize(frame)
    chip: dict[float, float] = {}
    for _, row in data.head(WARMUP).iterrows():
        chip = _update_chip(chip, row)
    last: dict[str, object] | None = None
    for index, row in data.iloc[WARMUP:].iterrows():
        chip = _update_chip(chip, row)
        peaks = _find_peaks(chip)
        if not peaks:
            continue
        close, (main_peak, main_weight), total = float(row['close']), peaks[0], sum(chip.values()) or 1.0
        volume_ma5 = float(data.loc[:index, 'volume'].iloc[-6:-1].mean())
        volume_ratio = float(row['volume']) / volume_ma5 if volume_ma5 > 0 else 0.0
        ma5, ma10 = data.loc[:index, 'close'].tail(5).mean(), data.loc[:index, 'close'].tail(10).mean()
        trend_up, concentration = close > ma5 and ma5 > ma10, _concentration(chip, 0.9)
        approaching = main_peak * APPROACH_RATIO <= close < main_peak * BREAK_RATIO
        tradeable, signal = False, '⬇️ 未接近'
        if approaching:
            if trend_up and volume_ratio >= VOL_MULTIPLIER and concentration <= CONC_THRESHOLD:
                signal, tradeable = '🔥 强烈预警（趋势+放量+集中）', True
            elif trend_up and volume_ratio >= VOL_MULTIPLIER:
                signal, tradeable = '⚡ 预警（趋势+放量）', True
            elif trend_up and concentration <= CONC_THRESHOLD:
                signal, tradeable = '📌 预警（趋势+集中）', True
            elif trend_up:
                signal = '📍 接近尖峰（仅趋势）'
            else:
                signal = '📍 接近尖峰（弱势）'
        elif close >= main_peak * BREAK_RATIO:
            signal = '✅ 已突破尖峰'
        last = {'code': normalize_code(code), 'name': name, 'signal_date': str(pd.Timestamp(row['date']).date()), 'close': round(close, 2), 'main_peak': round(main_peak, 2), 'peak_ratio': round(main_weight / total * 100, 2), 'dist_to_peak_pct': round((close - main_peak) / main_peak * 100, 2), 'conc90': round(concentration * 100, 2), 'profit_ratio': round(sum(weight for price, weight in chip.items() if price <= close) / total * 100, 2), 'vol_ratio': round(volume_ratio, 2), 'trend_up': bool(trend_up), 'is_approaching': bool(approaching), 'is_tradeable': bool(tradeable), 'signal': signal, 'hold_days_reference': HOLD_DAYS}
    if last is None:
        raise ValueError('no_signal_history')
    return last


def self_test() -> None:
    dates = pd.date_range('2025-01-01', periods=100, freq='B')
    frame = pd.DataFrame({'date': dates, 'open': np.linspace(10, 12, 100), 'high': np.linspace(10.2, 12.3, 100), 'low': np.linspace(9.8, 11.7, 100), 'close': np.linspace(10, 12.1, 100), 'volume': np.linspace(1000, 2000, 100), 'amount': np.linspace(10000, 24000, 100), 'turnover': [0.02] * 100})
    assert normalize_code('302132') == '302132'
    assert analyze_frame('000001', '样本', frame)['code'] == '000001'
    try:
        _canonicalize(frame.drop(columns=['turnover']))
    except ValueError as exc:
        assert 'missing_turnover_for_original_chip_formula' in str(exc)
    else:
        raise AssertionError('missing_turnover_guard_not_triggered')
