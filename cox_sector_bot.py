#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COX 板块量化分析系统 - 通信/电子/光学细分领域

板块成分股:
- 通信: 中兴通讯(000063)、烽火通信(600498)、亨通光电(600487)
- 电子: 立讯精密(002475)、京东方A(000725)、歌尔股份(002241)
- 光学: 欧菲光(002456)、舜宇光学(2382.HK)、水晶光电(002273)

输出: 个股 + 板块综合评分、买入/卖出信号、支撑/阻力线
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import traceback

import numpy as np
import pandas as pd
import akshare as ak
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
class Config:
    """COX 板块配置"""

    # COX 板块成分股 (代码 -> 名称)
    COX_STOCKS = {
        # 通信
        "000063": "中兴通讯",
        "600498": "烽火通信",
        "600487": "亨通光电",
        # 电子
        "002475": "立讯精密",
        "000725": "京东方A",
        "002241": "歌尔股份",
        # 光学
        "002456": "欧菲光",
        "002273": "水晶光电",
        # 港股光学
        # "02382": "舜宇光学",  # 港股需特殊处理
    }

    # Server酱推送
    SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY', '')

    # 输出目录
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

    # 信号阈值
    RSI_BUY = 35
    RSI_SELL = 70
    BIAS_BUY = -8.0      # 个股更宽松
    BIAS_SELL = 8.0
    VOL_CONFIRM = 1.2
    ATR_STOP_MULT = 2.5  # 个股波动大，止损更宽

    # 单次数据查询硬超时（秒），防止网络卡顿导致整个job假死
    QUERY_TIMEOUT_SEC = 20


# ==================== 数据模型 ====================
class SignalType(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"
    NO_DATA = "数据不足"


class PositionSize(Enum):
    FULL = 1.0; HEAVY = 0.7; MEDIUM = 0.5; LIGHT = 0.3; EMPTY = 0.0


@dataclass
class StockSignal:
    date: str
    symbol: str
    name: str
    close: float
    signal: str
    action: str
    position_size: str
    signal_score: int
    buy_score: int
    sell_score: int
    confidence: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    atr: float
    atr_pct: float
    indicators: Dict
    reasons: List[str]
    action_reasons: List[str]
    support_levels: List[Dict]
    resistance_levels: List[Dict]
    sector: str  # 所属细分: 通信/电子/光学

    def to_dict(self):
        return asdict(self)


@dataclass
class SectorSummary:
    """板块汇总"""
    sector: str
    avg_score: float
    buy_count: int
    sell_count: int
    hold_count: int
    top_pick: str
    top_score: int
    risk_level: str

    def to_dict(self):
        return asdict(self)


# ==================== 技术指标 ====================
class TechnicalIndicators:
    @staticmethod
    def ema(s, span): return s.ewm(span=span, adjust=False).mean()
    @staticmethod
    def sma(s, window): return s.rolling(window=window).mean()
    @staticmethod
    def rsi(close, n=14):
        d = close.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        ag = g.ewm(alpha=1/n, adjust=False).mean(); al = l.ewm(alpha=1/n, adjust=False).mean()
        return 100 - (100 / (1 + ag/al))
    @staticmethod
    def atr(df, n=14):
        hl = df["high"] - df["low"]; hc = (df["high"] - df["close"].shift()).abs(); lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()
    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ef = TechnicalIndicators.ema(close, fast); es = TechnicalIndicators.ema(close, slow)
        ml = ef - es; sl = TechnicalIndicators.ema(ml, signal)
        return ml, sl, ml - sl
    @staticmethod
    def bollinger_bands(close, window=20, num_std=2.0):
        ma = close.rolling(window=window).mean(); std = close.rolling(window=window).std()
        return ma + num_std*std, ma, ma - num_std*std
    @staticmethod
    def kdj(high, low, close, n=9, m1=3, m2=3):
        ll = low.rolling(window=n).min(); hh = high.rolling(window=n).max()
        rsv = (close - ll) / (hh - ll) * 100; k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean(); return k, d, 3*k - 2*d


# ==================== 成交量分布 ====================
class VolumeProfile:
    def __init__(self, df, price_col="close", volume_col="volume", bucket_size=1):
        self.df = df; self.price_col = price_col; self.volume_col = volume_col
        self.bucket_size = bucket_size; self.profile = self._calc()
    def _calc(self):
        d = self.df[[self.price_col, self.volume_col]].dropna().copy()
        d["bucket"] = (d[self.price_col] / self.bucket_size).round() * self.bucket_size
        prof = d.groupby("bucket")[self.volume_col].sum().reset_index()
        prof = prof.sort_values(self.volume_col, ascending=False).reset_index(drop=True)
        prof["rank"] = prof.index + 1
        return prof.sort_values("bucket").reset_index(drop=True)
    def get_poc(self): return self.profile.sort_values(self.volume_col, ascending=False).iloc[0]['bucket']
    def get_value_area(self, volume_pct=0.7):
        tv = self.profile[self.volume_col].sum(); tv = tv * volume_pct
        sp = self.profile.sort_values(self.volume_col, ascending=False)
        cs = 0; vp = []
        for _, row in sp.iterrows():
            cs += row[self.volume_col]; vp.append(row['bucket'])
            if cs >= tv: break
        return min(vp), max(vp)
    def get_top_levels(self, n=5):
        return sorted(self.profile.sort_values(self.volume_col, ascending=False).head(n)['bucket'].tolist())


# ==================== 数据获取 ====================
class DataFetcher:
    """个股数据获取"""

    def _query_with_timeout(self, fetch_fn, timeout=Config.QUERY_TIMEOUT_SEC):
        """给单次akshare查询包一层硬超时，防止网络卡顿导致整个job假死"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_fn)
            return future.result(timeout=timeout)

    def get_stock_data(self, symbol, start_date=None, end_date=None, period="daily"):
        if end_date is None: end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None: start_date = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")

        logger.info(f"  获取 {symbol} 数据...")

        try:
            # akshare 个股历史数据
            df = self._query_with_timeout(
                lambda: ak.stock_zh_a_hist(
                    symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq"
                )
            )
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "close", "high", "low", "volume", "amount"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("date").reset_index(drop=True)
        except FutureTimeoutError:
            logger.error(f"  获取 {symbol} 超时")
            return None
        except Exception as e:
            logger.error(f"  获取 {symbol} 失败: {e}")
            return None


# ==================== 支撑线计算 ====================
class SupportCalculator:
    @staticmethod
    def calculate_all(df, symbol, name):
        indicators = TechnicalIndicators()
        out = df.copy()

        out["ma5"] = indicators.sma(out["close"], 5); out["ma10"] = indicators.sma(out["close"], 10)
        out["ma20"] = indicators.sma(out["close"], 20); out["ma60"] = indicators.sma(out["close"], 60)
        out["ma120"] = indicators.sma(out["close"], 120); out["ma250"] = indicators.sma(out["close"], 250)
        out["high5"] = out["high"].rolling(5).max(); out["low5"] = out["low"].rolling(5).min()
        out["high20"] = out["high"].rolling(20).max(); out["low20"] = out["low"].rolling(20).min()
        out["high60"] = out["high"].rolling(60).max(); out["low60"] = out["low"].rolling(60).min()

        ma20 = indicators.sma(out["close"], 20); std20 = out["close"].rolling(20).std()
        out["bb_upper"] = ma20 + 2 * std20; out["bb_lower"] = ma20 - 2 * std20
        out["bb_upper_3"] = ma20 + 3 * std20; out["bb_lower_3"] = ma20 - 3 * std20

        atr14 = indicators.atr(out, 14); out["atr14"] = atr14
        latest = out.iloc[-1]; current = latest["close"]

        vp = VolumeProfile(out.tail(120), bucket_size=max(1, current * 0.01))
        poc = vp.get_poc(); va_low, va_high = vp.get_value_area(0.7)
        high_120 = out["high"].rolling(120).max().iloc[-1]; low_120 = out["low"].rolling(120).min().iloc[-1]
        fib_range = high_120 - low_120

        all_levels = []
        # 阻力
        all_levels.append(("3倍ATR阻力", current + 3*atr14.iloc[-1], "resist"))
        all_levels.append(("2倍ATR阻力", current + 2*atr14.iloc[-1], "resist"))
        all_levels.append(("1倍ATR阻力", current + 1*atr14.iloc[-1], "resist"))
        all_levels.append(("布林上轨(3σ)", latest["bb_upper_3"], "resist"))
        all_levels.append(("布林上轨(2σ)", latest["bb_upper"], "resist"))
        all_levels.append(("近期高点(5日)", latest["high5"], "resist"))
        all_levels.append(("近期高点(20日)", latest["high20"], "resist"))
        all_levels.append(("斐波那契0%", high_120, "resist"))
        all_levels.append(("斐波那契23.6%", high_120 - 0.236*fib_range, "resist"))
        # 当前价
        all_levels.append(("当前收盘价", current, "current"))
        # 支撑
        all_levels.append(("MA5", latest["ma5"], "support"))
        all_levels.append(("MA10", latest["ma10"], "support"))
        all_levels.append(("MA20", latest["ma20"], "support"))
        all_levels.append(("MA60", latest["ma60"], "support"))
        all_levels.append(("MA120", latest["ma120"], "support"))
        all_levels.append(("MA250", latest["ma250"], "support"))
        all_levels.append(("近期低点(5日)", latest["low5"], "support"))
        all_levels.append(("近期低点(20日)", latest["low20"], "support"))
        all_levels.append(("近期低点(60日)", latest["low60"], "support"))
        all_levels.append(("1倍ATR支撑", current - 1*atr14.iloc[-1], "support"))
        all_levels.append(("2倍ATR支撑", current - 2*atr14.iloc[-1], "support"))
        all_levels.append(("3倍ATR支撑", current - 3*atr14.iloc[-1], "support"))
        all_levels.append(("布林下轨(2σ)", latest["bb_lower"], "support"))
        all_levels.append(("布林下轨(3σ)", latest["bb_lower_3"], "support"))
        all_levels.append(("斐波那契38.2%", high_120 - 0.382*fib_range, "support"))
        all_levels.append(("斐波那契50%", high_120 - 0.500*fib_range, "support"))
        all_levels.append(("斐波那契61.8%", high_120 - 0.618*fib_range, "support"))
        all_levels.append(("斐波那契78.6%", high_120 - 0.786*fib_range, "support"))
        all_levels.append(("斐波那契100%", low_120, "support"))
        all_levels.append(("成交量POC", poc, "support"))
        all_levels.append(("价值区域上沿", va_high, "support"))
        all_levels.append(("价值区域下沿", va_low, "support"))

        seen = set(); unique_levels = []
        for label, price, level_type in sorted(all_levels, key=lambda x: x[1], reverse=True):
            pr = round(price, 2)
            if pr not in seen and pr > 0:
                seen.add(pr)
                unique_levels.append({"name": label, "price": pr, "type": level_type,
                                      "distance_pct": round((pr - current) / current * 100, 2)})

        return {
            "current": round(current, 2), "date": str(out["date"].iloc[-1].date()),
            "resistance_levels": [l for l in unique_levels if l["type"] == "resist"],
            "support_levels": [l for l in unique_levels if l["type"] == "support"],
            "summary": {"atr_14": round(atr14.iloc[-1], 2), "atr_pct": round(atr14.iloc[-1]/current*100, 2),
                        "high_120": round(high_120, 2), "low_120": round(low_120, 2),
                        "poc": round(poc, 2), "va_low": round(va_low, 2), "va_high": round(va_high, 2)}
        }


# ==================== 信号引擎 ====================
class SignalEngine:
    def __init__(self): self.indicators = TechnicalIndicators(); self.cfg = Config()

    def calculate_all(self, df):
        out = df.copy()
        out["ma5"] = self.indicators.sma(out["close"], 5); out["ma10"] = self.indicators.sma(out["close"], 10)
        out["ma20"] = self.indicators.sma(out["close"], 20); out["ma60"] = self.indicators.sma(out["close"], 60)
        out["ma120"] = self.indicators.sma(out["close"], 120); out["ma250"] = self.indicators.sma(out["close"], 250)
        out["bias120"] = (out["close"] - out["ma120"]) / out["ma120"] * 100
        out["bias250"] = (out["close"] - out["ma250"]) / out["ma250"] * 100
        out["rsi6"] = self.indicators.rsi(out["close"], 6); out["rsi14"] = self.indicators.rsi(out["close"], 14)
        out["rsi24"] = self.indicators.rsi(out["close"], 24)
        out["macd"], out["macd_signal"], out["macd_hist"] = self.indicators.macd(out["close"])
        out["macd_hist_prev"] = out["macd_hist"].shift(1)
        out["bb_upper"], out["bb_mid"], out["bb_lower"] = self.indicators.bollinger_bands(out["close"])
        out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
        out["k"], out["d"], out["j"] = self.indicators.kdj(out["high"], out["low"], out["close"])
        out["vol_ma5"] = out["volume"].rolling(5).mean(); out["vol_ma20"] = out["volume"].rolling(20).mean()
        out["vol_ratio"] = out["volume"] / out["vol_ma20"]
        out["atr14"] = self.indicators.atr(out, 14); out["atr_pct"] = out["atr14"] / out["close"] * 100
        out["high20"] = out["high"].rolling(20).max(); out["low20"] = out["low"].rolling(20).min()
        out["high60"] = out["high"].rolling(60).max(); out["low60"] = out["low"].rolling(60).min()
        return out

    def generate_signal(self, df, symbol, name, support_data, sector):
        out = self.calculate_all(df)
        vp = VolumeProfile(out.tail(120), bucket_size=max(1, out["close"].iloc[-1] * 0.01))
        poc = vp.get_poc(); va_low, va_high = vp.get_value_area(0.7)
        latest = out.iloc[-1].copy(); date = latest['date']; current = latest['close']

        required_cols = ["ma250", "rsi14", "macd_hist", "vol_ratio", "atr14"]
        if any(pd.isna(latest[col]) for col in required_cols):
            return StockSignal(str(date.date()), symbol, name, latest['close'], SignalType.NO_DATA.value,
                               "HOLD", PositionSize.EMPTY.name, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, ["数据不足"], [],
                               support_data["support_levels"], support_data["resistance_levels"], sector)

        buy_score = 0; sell_score = 0; reasons = []; action_reasons = []

        # 1. 年线趋势
        if latest["close"] > latest["ma250"]:
            buy_score += 1; reasons.append("✅ 价格在年线之上，长期趋势向上")
        else:
            sell_score += 1; reasons.append("❌ 价格在年线之下，长期趋势向下")

        # 2. 均线排列
        if latest["ma5"] > latest["ma20"] > latest["ma60"]:
            buy_score += 2; reasons.append("✅ 均线多头排列(MA5>MA20>MA60)")
        elif latest["ma5"] < latest["ma20"] < latest["ma60"]:
            sell_score += 2; reasons.append("❌ 均线空头排列(MA5<MA20<MA60)")
        elif latest["ma5"] > latest["ma20"]:
            buy_score += 1; reasons.append("✅ 短期均线金叉中期均线")
        elif latest["ma5"] < latest["ma20"]:
            sell_score += 1; reasons.append("❌ 短期均线死叉中期均线")

        # 3. 乖离率 (个股阈值更宽)
        if latest["bias120"] <= self.cfg.BIAS_BUY:
            buy_score += 2; reasons.append(f"✅ BIAS120超卖({latest['bias120']:.2f}%)")
            action_reasons.append(f"乖离率超卖 {latest['bias120']:.2f}%，适合抄底")
        elif latest["bias120"] >= self.cfg.BIAS_SELL:
            sell_score += 2; reasons.append(f"❌ BIAS120超买({latest['bias120']:.2f}%)")
            action_reasons.append(f"乖离率超买 {latest['bias120']:.2f}%，建议减仓")

        if latest["bias250"] <= -10.0:
            buy_score += 1; reasons.append(f"✅ BIAS250深度超卖({latest['bias250']:.2f}%)")
        elif latest["bias250"] >= 10.0:
            sell_score += 1; reasons.append(f"❌ BIAS250深度超买({latest['bias250']:.2f}%)")

        # 4. RSI
        if latest["rsi14"] < self.cfg.RSI_BUY:
            buy_score += 2; reasons.append(f"✅ RSI14超卖({latest['rsi14']:.2f})")
            action_reasons.append(f"RSI14 仅 {latest['rsi14']:.2f}，严重超卖，建议买入")
        elif latest["rsi14"] < 40:
            buy_score += 1; reasons.append(f"✅ RSI14偏低({latest['rsi14']:.2f})")
        elif latest["rsi14"] > self.cfg.RSI_SELL:
            sell_score += 2; reasons.append(f"❌ RSI14超买({latest['rsi14']:.2f})")
            action_reasons.append(f"RSI14 高达 {latest['rsi14']:.2f}，严重超买，建议卖出")
        elif latest["rsi14"] > 60:
            sell_score += 1; reasons.append(f"❌ RSI14偏高({latest['rsi14']:.2f})")

        # 5. MACD
        macd_bullish = latest["macd_hist"] > 0 and latest["macd_hist"] > latest["macd_hist_prev"]
        macd_bearish = latest["macd_hist"] < 0 and latest["macd_hist"] < latest["macd_hist_prev"]
        if macd_bullish:
            buy_score += 2; reasons.append("✅ MACD柱状图转正且扩大，上涨动能增强")
            action_reasons.append("MACD 由负转正并扩大，趋势转多，建议买入")
        elif latest["macd_hist"] > 0:
            buy_score += 1; reasons.append("✅ MACD柱状图为正")
        if macd_bearish:
            sell_score += 2; reasons.append("❌ MACD柱状图转负且扩大，下跌动能增强")
            action_reasons.append("MACD 由正转负并扩大，趋势转空，建议卖出")
        elif latest["macd_hist"] < 0:
            sell_score += 1; reasons.append("❌ MACD柱状图为负")

        # 6. 成交量
        if latest["vol_ratio"] >= self.cfg.VOL_CONFIRM:
            if buy_score > sell_score:
                buy_score += 1; reasons.append(f"✅ 成交量放大({latest['vol_ratio']:.2f}倍)，买盘确认")
                action_reasons.append(f"成交量放大 {latest['vol_ratio']:.2f} 倍，买盘活跃")
            elif sell_score > buy_score:
                sell_score += 1; reasons.append(f"❌ 成交量放大({latest['vol_ratio']:.2f}倍)，卖盘确认")
                action_reasons.append(f"成交量放大 {latest['vol_ratio']:.2f} 倍，卖盘活跃")

        # 7. 价格位置
        near_low20 = pd.notna(latest["low20"]) and latest["close"] <= latest["low20"] * 1.05
        near_low60 = pd.notna(latest["low60"]) and latest["close"] <= latest["low60"] * 1.08
        near_high20 = pd.notna(latest["high20"]) and latest["close"] >= latest["high20"] * 0.95

        if near_low20 or near_low60:
            buy_score += 1; reasons.append("✅ 价格接近近期低点，支撑区域")
            action_reasons.append("价格接近近期低点，支撑区域，适合逢低买入")
        if near_high20:
            sell_score += 1; reasons.append("❌ 价格接近近期高点，阻力区域")
            action_reasons.append("价格接近近期高点，阻力区域，建议逢高减仓")

        # 8. 布林带
        if latest["bb_position"] < 0.1:
            buy_score += 1; reasons.append("✅ 价格触及布林带下轨")
        elif latest["bb_position"] > 0.9:
            sell_score += 1; reasons.append("❌ 价格触及布林带上轨")

        # 9. KDJ
        if latest["j"] < 20 and latest["k"] < 30:
            buy_score += 1; reasons.append(f"✅ KDJ超卖区(J={latest['j']:.2f})")
        elif latest["j"] > 80 and latest["k"] > 70:
            sell_score += 1; reasons.append(f"❌ KDJ超买区(J={latest['j']:.2f})")

        # 信号判定
        signal_score = buy_score - sell_score

        if buy_score >= 6 and signal_score >= 4:
            signal = SignalType.STRONG_BUY; position = PositionSize.HEAVY; action = "BUY"
        elif buy_score >= 5 and signal_score >= 2:
            signal = SignalType.BUY; position = PositionSize.MEDIUM; action = "BUY"
        elif sell_score >= 6 and signal_score <= -4:
            signal = SignalType.STRONG_SELL; position = PositionSize.EMPTY; action = "SELL"
        elif sell_score >= 5 and signal_score <= -2:
            signal = SignalType.SELL; position = PositionSize.LIGHT; action = "SELL"
        else:
            signal = SignalType.HOLD
            if signal_score > 0: position = PositionSize.MEDIUM
            elif signal_score < 0: position = PositionSize.LIGHT
            else: position = PositionSize.MEDIUM
            action = "HOLD"

        # 止损止盈 (个股更宽)
        atr = latest["atr14"]
        if action == "BUY":
            stop_loss = current - self.cfg.ATR_STOP_MULT * atr
            take_profit = current + 2.5 * self.cfg.ATR_STOP_MULT * atr
        elif action == "SELL":
            stop_loss = current + self.cfg.ATR_STOP_MULT * atr
            take_profit = current - 2.5 * self.cfg.ATR_STOP_MULT * atr
        else:
            stop_loss = current - self.cfg.ATR_STOP_MULT * atr
            take_profit = current + self.cfg.ATR_STOP_MULT * atr

        risk = abs(current - stop_loss); reward = abs(take_profit - current)
        rr_ratio = reward / risk if risk > 0 else 0
        confidence = min(abs(signal_score) / 10 * 100, 95)

        indicators = {
            "ma20": round(latest["ma20"], 2), "ma60": round(latest["ma60"], 2),
            "ma120": round(latest["ma120"], 2), "ma250": round(latest["ma250"], 2),
            "bias120": round(latest["bias120"], 2), "bias250": round(latest["bias250"], 2),
            "rsi14": round(latest["rsi14"], 2), "macd_hist": round(latest["macd_hist"], 4),
            "vol_ratio": round(latest["vol_ratio"], 2), "atr_pct": round(latest["atr_pct"], 2),
            "bb_position": round(latest["bb_position"], 2), "k": round(latest["k"], 2),
            "d": round(latest["d"], 2), "j": round(latest["j"], 2),
            "poc": round(poc, 2), "va_low": round(va_low, 2), "va_high": round(va_high, 2)
        }

        return StockSignal(str(date.date()), symbol, name, round(current, 2), signal.value,
                           action, position.name, signal_score, buy_score, sell_score,
                           round(confidence, 1), round(stop_loss, 2), round(take_profit, 2),
                           round(rr_ratio, 2), round(atr, 2), round(atr/current*100, 2),
                           indicators, reasons, action_reasons,
                           support_data["support_levels"], support_data["resistance_levels"], sector)


# ==================== 通知推送 ====================
class Notifier:
    def __init__(self): self.key = Config.SERVERCHAN_KEY
    def send(self, title, content):
        if not self.key: logger.info("未配置 ServerChan Key，跳过推送"); return False
        try:
            resp = requests.post(f"https://sctapi.ftqq.com/{self.key}.send",
                                data={"title": title, "desp": content}, timeout=10)
            if resp.status_code == 200: logger.info("推送成功"); return True
            else: logger.error(f"推送失败: {resp.status_code}"); return False
        except Exception as e: logger.error(f"推送异常: {e}"); return False


# ==================== 板块汇总 ====================
class SectorAnalyzer:
    """板块分析器"""

    SECTOR_MAP = {
        "000063": "通信", "600498": "通信", "600487": "通信",
        "002475": "电子", "000725": "电子", "002241": "电子",
        "002456": "光学", "002273": "光学",
    }

    @staticmethod
    def analyze_sector(results):
        sectors = {}
        for r in results:
            sector = r.sector
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(r)

        summaries = []
        for sector, stocks in sectors.items():
            scores = [s.signal_score for s in stocks]
            avg_score = sum(scores) / len(scores) if scores else 0
            buy_count = sum(1 for s in stocks if s.action == "BUY")
            sell_count = sum(1 for s in stocks if s.action == "SELL")
            hold_count = len(stocks) - buy_count - sell_count
            top = max(stocks, key=lambda x: x.signal_score)

            risk = "高" if any(s.atr_pct > 5 for s in stocks) else "中" if any(s.atr_pct > 3 for s in stocks) else "低"

            summaries.append(SectorSummary(
                sector=sector, avg_score=round(avg_score, 2),
                buy_count=buy_count, sell_count=sell_count, hold_count=hold_count,
                top_pick=f"{top.name}({top.symbol})", top_score=top.signal_score,
                risk_level=risk
            ))

        return summaries


# ==================== 主程序 ====================
class COXBot:
    """COX 板块量化机器人"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.engine = SignalEngine()
        self.support_calc = SupportCalculator()
        self.notifier = Notifier()
        self.sector_analyzer = SectorAnalyzer()
        self.results = []

    def run(self):
        logger.info("="*70)
        logger.info("COX 板块量化分析系统启动")
        logger.info("="*70)

        for symbol, name in Config.COX_STOCKS.items():
            try:
                sector = self.sector_analyzer.SECTOR_MAP.get(symbol, "其他")
                logger.info(f"\n分析: {name} ({symbol}) [{sector}]")

                df = self.fetcher.get_stock_data(symbol)
                if df is None or len(df) < 60:
                    logger.warning(f"  {symbol} 数据不足，跳过")
                    continue

                support_data = self.support_calc.calculate_all(df, symbol, name)
                result = self.engine.generate_signal(df, symbol, name, support_data, sector)
                self.results.append(result)

                logger.info(f"  信号: {result.signal} | 评分: {result.signal_score:+d} | 置信度: {result.confidence:.1f}%")

            except Exception as e:
                logger.error(f"分析 {symbol} 失败: {e}")
                logger.error(traceback.format_exc())

        self._save_results()
        self._send_notification()
        self._print_summary()

        return self.results

    def _save_results(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # JSON
        json_path = os.path.join(Config.OUTPUT_DIR, "cox_signal_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存: {json_path}")

        # Markdown
        md_path = os.path.join(Config.OUTPUT_DIR, "cox_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown())
        logger.info(f"报告已保存: {md_path}")

    def _generate_markdown(self):
        lines = [f"# COX 板块量化分析报告", f"", f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f""]

        # 板块汇总
        sector_summaries = self.sector_analyzer.analyze_sector(self.results)
        lines.append(f"## 板块汇总")
        lines.append(f"")
        lines.append(f"| 板块 | 平均评分 | 买入数 | 卖出数 | 观望数 | 首选标的 | 风险等级 |")
        lines.append(f"|------|----------|--------|--------|--------|----------|----------|")
        for s in sector_summaries:
            lines.append(f"| {s.sector} | {s.avg_score:+.2f} | {s.buy_count} | {s.sell_count} | {s.hold_count} | {s.top_pick} | {s.risk_level} |")
        lines.append(f"")

        # 个股详细
        lines.append(f"## 个股详细分析")
        lines.append(f"")

        for r in self.results:
            action_emoji = "🟢" if r.action == "BUY" else "🔴" if r.action == "SELL" else "⚪"
            lines.append(f"### {action_emoji} {r.name} ({r.symbol}) [{r.sector}]")
            lines.append(f"")
            lines.append(f"- **当前价**: {r.close}")
            lines.append(f"- **交易建议**: {r.signal}")
            lines.append(f"- **建议仓位**: {r.position_size}")
            lines.append(f"- **止损位**: {r.stop_loss} | **止盈位**: {r.take_profit}")
            lines.append(f"- **盈亏比**: {r.risk_reward_ratio}")
            lines.append(f"- **置信度**: {r.confidence:.1f}%")
            lines.append(f"- **ATR**: {r.atr} ({r.atr_pct}%)")
            lines.append(f"")

            if r.action_reasons:
                lines.append(f"**核心交易理由**:")
                for i, reason in enumerate(r.action_reasons, 1):
                    lines.append(f"{i}. {reason}")
                lines.append(f"")

            lines.append(f"**阻力线（从高到低）**:")
            lines.append(f"| 排名 | 名称 | 价格 | 距现价 |")
            lines.append(f"|------|------|------|--------|")
            for i, l in enumerate(r.resistance_levels[:8], 1):
                lines.append(f"| {i} | {l['name']} | {l['price']:.2f} | {l['distance_pct']:+.2f}% |")
            lines.append(f"")

            lines.append(f"**支撑线（从高到低）**:")
            lines.append(f"| 排名 | 名称 | 价格 | 距现价 |")
            lines.append(f"|------|------|------|--------|")
            for i, l in enumerate(r.support_levels[:10], 1):
                lines.append(f"| {i} | {l['name']} | {l['price']:.2f} | {l['distance_pct']:+.2f}% |")
            lines.append(f"")

            lines.append(f"**关键指标**:")
            for k, v in r.indicators.items():
                lines.append(f"- {k}: {v}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        lines.append(f"## 风险提示")
        lines.append(f"")
        lines.append(f"1. 以上分析基于技术指标，不构成投资建议")
        lines.append(f"2. 个股波动较大，建议控制仓位不超过总资金的10%")
        lines.append(f"3. 严格设置止损，个股止损建议放宽至-7%~-10%")
        lines.append(f"4. 关注板块轮动，COX板块受政策影响较大")

        return "\n".join(lines)

    def _send_notification(self):
        if not self.results: return
        strong = [r for r in self.results if "强烈" in r.signal]

        # 修复点：原代码只在有强烈信号时才推送，平时无从得知脚本是否正常运行——
        # 现在改成每次都推送板块汇总，强烈信号单独突出显示
        title = f"【COX板块】{len(strong)}个强烈信号" if strong else "【COX板块】今日无强烈信号"
        content_lines = [f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

        if strong:
            content_lines.append("### ⚠️ 强烈信号")
            for r in strong:
                content_lines.append(f"## {r.name} ({r.symbol}) [{r.sector}]")
                content_lines.append(f"信号: {r.signal}")
                content_lines.append(f"收盘价: {r.close} | 评分: {r.signal_score:+d}")
                content_lines.append(f"建议: {r.position_size} | 止损: {r.stop_loss} | 止盈: {r.take_profit}")
                content_lines.append("")

        content_lines.append("### 板块汇总")
        sector_summaries = self.sector_analyzer.analyze_sector(self.results)
        for s in sector_summaries:
            content_lines.append(
                f"- {s.sector}：平均评分{s.avg_score:+.2f} | 买{s.buy_count}/卖{s.sell_count}/观望{s.hold_count} "
                f"| 首选{s.top_pick}"
            )

        content_lines.append("")
        content_lines.append("### 全部个股概览")
        for r in self.results:
            content_lines.append(f"- {r.name}({r.symbol})[{r.sector}]：{r.close} | {r.signal} | 评分{r.signal_score:+d}")

        self.notifier.send(title, "\n\n".join(content_lines))

    def _print_summary(self):
        print("\n" + "="*85)
        print("📊 COX 板块交易建议报告".center(85))
        print(f"📅 分析日期: {self.results[0].date if self.results else 'N/A'}".center(85))
        print("="*85)

        # 板块汇总
        sector_summaries = self.sector_analyzer.analyze_sector(self.results)
        print(f"\n{'─'*85}")
        print("  板块汇总")
        print(f"{'─'*85}")
        for s in sector_summaries:
            print(f"  📁 {s.sector}: 平均评分 {s.avg_score:+.2f} | 买入{s.buy_count} 卖出{s.sell_count} 观望{s.hold_count}")
            print(f"     🏆 首选: {s.top_pick} (评分: {s.top_score:+d}) | 风险: {s.risk_level}")

        # 个股详细
        for r in self.results:
            print(f"\n{'─'*85}")
            print(f"  {r.name} ({r.symbol}) [{r.sector}]  当前价: {r.close:.2f}")
            print(f"{'─'*85}")

            if r.action == "BUY":
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📈 评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  🛑 止损: {r.stop_loss:.2f} | 🎯 止盈: {r.take_profit:.2f}")
                print(f"  ⚖️  盈亏比: 1:{r.risk_reward_ratio:.2f}")
            elif r.action == "SELL":
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📉 评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  🛑 止损: {r.stop_loss:.2f} | 🎯 止盈: {r.take_profit:.2f}")
                print(f"  ⚖️  盈亏比: 1:{r.risk_reward_ratio:.2f}")
            else:
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📊 评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  ⚠️  建议观望")

            if r.action_reasons:
                print(f"\n  📋 核心理由:")
                for i, reason in enumerate(r.action_reasons, 1):
                    print(f"     {i}. {reason}")

            print(f"\n  🟢 支撑线（前5）:")
            for i, l in enumerate(r.support_levels[:5], 1):
                print(f"     {i}. {l['name']:<18s} {l['price']:>10.2f} ({l['distance_pct']:+.2f}%)")

            print(f"\n  🔴 阻力线（前5）:")
            for i, l in enumerate(r.resistance_levels[:5], 1):
                print(f"     {i}. {l['name']:<18s} {l['price']:>10.2f} ({l['distance_pct']:+.2f}%)")

        print(f"\n{'='*85}")
        print("⚠️ 风险提示".center(85))
        print("="*85)
        print("  1. 以上分析基于技术指标，不构成投资建议")
        print("  2. 个股波动较大，建议单票仓位不超过总资金的10%")
        print("  3. 严格设置止损，个股止损建议放宽至-7%~-10%")
        print("  4. 关注板块轮动，COX板块受政策影响较大")
        print(f"{'='*85}")


def main():
    bot = COXBot()
    bot.run()


if __name__ == "__main__":
    main()
