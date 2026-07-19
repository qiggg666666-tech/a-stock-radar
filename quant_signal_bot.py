#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股指数技术分析信号系统 - 完整版
支持: 上证指数/深证成指/创业板指/上证50
输出: 交易建议(BUY/SELL/HOLD) + 支撑/阻力线(从高到低) + 可视化图表

GitHub Actions 部署:
1. 上传本文件到仓库根目录
2. 创建 .github/workflows/quant.yml (见文末)
3. 设置 Secrets: SERVERCHAN_KEY (可选)
4. 每天自动运行或手动触发

本地运行: python main.py
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List
from enum import Enum
import traceback

import numpy as np
import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
class Config:
    """运行配置 - 可修改此部分调整参数"""
    # 分析的指数: 代码 -> 名称
    SYMBOLS = {
        "000001": "上证指数",
        "399001": "深证成指",
        "399006": "创业板指",
        "000016": "上证50",
    }

    # baostock要求指数代码带交易所前缀，这里做映射（原akshare版本直接用不带前缀的代码，
    # 但akshare的index_zh_a_hist在GitHub Actions上频繁被东财接口拒绝连接，
    # 换成本仓库已验证稳定的baostock数据源）
    INDEX_CODE_MAP = {
        "000001": "sh.000001",
        "399001": "sz.399001",
        "399006": "sz.399006",
        "000016": "sh.000016",
    }

    # Server酱推送密钥 (从环境变量读取，可选)
    SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY', '')

    # 输出目录
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

    # 信号阈值
    RSI_BUY = 35          # RSI低于此值视为超卖
    RSI_SELL = 70         # RSI高于此值视为超买
    BIAS_BUY = -6.0       # BIAS低于此值视为超卖
    BIAS_SELL = 6.0       # BIAS高于此值视为超买
    VOL_CONFIRM = 1.1     # 成交量放大确认倍数
    ATR_STOP_MULT = 2.0   # ATR止损倍数

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
class SignalResult:
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

    def to_dict(self):
        return asdict(self)


# ==================== 技术指标 ====================
class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def ema(s, span):
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def sma(s, window):
        return s.rolling(window=window).mean()

    @staticmethod
    def rsi(close, n=14):
        d = close.diff()
        g = d.clip(lower=0)
        l = -d.clip(upper=0)
        ag = g.ewm(alpha=1/n, adjust=False).mean()
        al = l.ewm(alpha=1/n, adjust=False).mean()
        rs = ag / al
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df, n=14):
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ef = TechnicalIndicators.ema(close, fast)
        es = TechnicalIndicators.ema(close, slow)
        ml = ef - es
        sl = TechnicalIndicators.ema(ml, signal)
        return ml, sl, ml - sl

    @staticmethod
    def bollinger_bands(close, window=20, num_std=2.0):
        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        return ma + num_std*std, ma, ma - num_std*std

    @staticmethod
    def kdj(high, low, close, n=9, m1=3, m2=3):
        ll = low.rolling(window=n).min()
        hh = high.rolling(window=n).max()
        rsv = (close - ll) / (hh - ll) * 100
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        return k, d, 3*k - 2*d

    @staticmethod
    def adx(df, n=14):
        pdm = df['high'].diff()
        mdm = df['low'].diff().abs()
        pdm[pdm < 0] = 0
        mdm[mdm < 0] = 0
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        pdi = 100 * (pdm.ewm(alpha=1/n, adjust=False).mean() / atr)
        mdi = 100 * (mdm.ewm(alpha=1/n, adjust=False).mean() / atr)
        dx = (abs(pdi - mdi) / (pdi + mdi)) * 100
        return dx.ewm(alpha=1/n, adjust=False).mean()


# ==================== 成交量分布 ====================
class VolumeProfile:
    """成交量分布分析"""

    def __init__(self, df, price_col="close", volume_col="volume", bucket_size=10):
        self.df = df
        self.price_col = price_col
        self.volume_col = volume_col
        self.bucket_size = bucket_size
        self.profile = self._calculate()

    def _calculate(self):
        d = self.df[[self.price_col, self.volume_col]].dropna().copy()
        d["bucket"] = (d[self.price_col] / self.bucket_size).round() * self.bucket_size
        prof = d.groupby("bucket")[self.volume_col].sum().reset_index()
        prof = prof.sort_values(self.volume_col, ascending=False).reset_index(drop=True)
        prof["rank"] = prof.index + 1
        return prof.sort_values("bucket").reset_index(drop=True)

    def get_poc(self):
        return self.profile.sort_values(self.volume_col, ascending=False).iloc[0]['bucket']

    def get_value_area(self, volume_pct=0.7):
        total_vol = self.profile[self.volume_col].sum()
        target_vol = total_vol * volume_pct
        sorted_profile = self.profile.sort_values(self.volume_col, ascending=False)
        cumsum = 0
        va_prices = []
        for _, row in sorted_profile.iterrows():
            cumsum += row[self.volume_col]
            va_prices.append(row['bucket'])
            if cumsum >= target_vol:
                break
        return min(va_prices), max(va_prices)

    def get_top_levels(self, n=5):
        top = self.profile.sort_values(self.volume_col, ascending=False).head(n)
        return sorted(top['bucket'].tolist())


# ==================== 数据获取 ====================
class DataFetcher:
    """数据获取类"""

    def _query_with_timeout(self, fetch_fn, timeout=Config.QUERY_TIMEOUT_SEC):
        """给单次baostock查询包一层硬超时，防止网络卡顿导致整个job假死"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_fn)
            return future.result(timeout=timeout)

    def get_index_data(self, symbol, start_date=None, end_date=None, period="daily"):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        bs_code = Config.INDEX_CODE_MAP.get(symbol)
        if bs_code is None:
            raise ValueError(f"未找到 {symbol} 对应的baostock指数代码，请在Config.INDEX_CODE_MAP里补充映射")

        logger.info(f"获取 {symbol}({bs_code}) 数据: {start_date} - {end_date}")

        def _do_query():
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2"
            )
            return rs.get_data()

        df = self._query_with_timeout(_do_query)
        if df.empty:
            raise ValueError(f"{symbol}({bs_code}) baostock返回空数据")

        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


# ==================== 支撑线计算 ====================
class SupportCalculator:
    """支撑/阻力线计算器"""

    @staticmethod
    def calculate_all(df, symbol, name):
        indicators = TechnicalIndicators()
        out = df.copy()

        # 均线
        out["ma5"] = indicators.sma(out["close"], 5)
        out["ma10"] = indicators.sma(out["close"], 10)
        out["ma20"] = indicators.sma(out["close"], 20)
        out["ma60"] = indicators.sma(out["close"], 60)
        out["ma120"] = indicators.sma(out["close"], 120)
        out["ma250"] = indicators.sma(out["close"], 250)

        # 价格极值
        out["high5"] = out["high"].rolling(5).max()
        out["low5"] = out["low"].rolling(5).min()
        out["high20"] = out["high"].rolling(20).max()
        out["low20"] = out["low"].rolling(20).min()
        out["high60"] = out["high"].rolling(60).max()
        out["low60"] = out["low"].rolling(60).min()
        out["high120"] = out["high"].rolling(120).max()
        out["low120"] = out["low"].rolling(120).min()
        out["high250"] = out["high"].rolling(250).max()
        out["low250"] = out["low"].rolling(250).min()

        # 布林带
        ma20 = indicators.sma(out["close"], 20)
        std20 = out["close"].rolling(20).std()
        out["bb_upper"] = ma20 + 2 * std20
        out["bb_lower"] = ma20 - 2 * std20
        out["bb_upper_3"] = ma20 + 3 * std20
        out["bb_lower_3"] = ma20 - 3 * std20

        # ATR
        atr14 = indicators.atr(out, 14)
        out["atr14"] = atr14

        latest = out.iloc[-1]
        current = latest["close"]

        # 成交量分布
        vp = VolumeProfile(out.tail(250), bucket_size=max(10, current * 0.005))
        poc = vp.get_poc()
        va_low, va_high = vp.get_value_area(0.7)

        # 250日高低点 & 斐波那契
        high_250 = out["high"].rolling(250).max().iloc[-1]
        low_250 = out["low"].rolling(250).min().iloc[-1]
        fib_range = high_250 - low_250

        # 构建所有支撑/阻力线
        all_levels = []

        # 阻力区 (当前价之上)
        all_levels.append(("3倍ATR阻力", current + 3*atr14.iloc[-1], "resist"))
        all_levels.append(("2倍ATR阻力", current + 2*atr14.iloc[-1], "resist"))
        all_levels.append(("1倍ATR阻力", current + 1*atr14.iloc[-1], "resist"))
        all_levels.append(("布林上轨(3σ)", latest["bb_upper_3"], "resist"))
        all_levels.append(("布林上轨(2σ)", latest["bb_upper"], "resist"))
        all_levels.append(("近期高点(5日)", latest["high5"], "resist"))
        all_levels.append(("近期高点(20日)", latest["high20"], "resist"))
        all_levels.append(("斐波那契0%", high_250, "resist"))
        all_levels.append(("斐波那契23.6%", high_250 - 0.236*fib_range, "resist"))

        # 当前价
        all_levels.append(("当前收盘价", current, "current"))

        # 支撑区 (当前价之下)
        all_levels.append(("MA5", latest["ma5"], "support"))
        all_levels.append(("MA10", latest["ma10"], "support"))
        all_levels.append(("MA20", latest["ma20"], "support"))
        all_levels.append(("MA60", latest["ma60"], "support"))
        all_levels.append(("MA120", latest["ma120"], "support"))
        all_levels.append(("MA250", latest["ma250"], "support"))
        all_levels.append(("近期低点(5日)", latest["low5"], "support"))
        all_levels.append(("近期低点(20日)", latest["low20"], "support"))
        all_levels.append(("近期低点(60日)", latest["low60"], "support"))
        all_levels.append(("近期低点(120日)", latest["low120"], "support"))
        all_levels.append(("近期低点(250日)", latest["low250"], "support"))
        all_levels.append(("1倍ATR支撑", current - 1*atr14.iloc[-1], "support"))
        all_levels.append(("2倍ATR支撑", current - 2*atr14.iloc[-1], "support"))
        all_levels.append(("3倍ATR支撑", current - 3*atr14.iloc[-1], "support"))
        all_levels.append(("布林下轨(2σ)", latest["bb_lower"], "support"))
        all_levels.append(("布林下轨(3σ)", latest["bb_lower_3"], "support"))
        all_levels.append(("斐波那契38.2%", high_250 - 0.382*fib_range, "support"))
        all_levels.append(("斐波那契50%", high_250 - 0.500*fib_range, "support"))
        all_levels.append(("斐波那契61.8%", high_250 - 0.618*fib_range, "support"))
        all_levels.append(("斐波那契78.6%", high_250 - 0.786*fib_range, "support"))
        all_levels.append(("斐波那契100%", low_250, "support"))
        all_levels.append(("成交量POC", poc, "support"))
        all_levels.append(("价值区域上沿", va_high, "support"))
        all_levels.append(("价值区域下沿", va_low, "support"))

        # 去重并排序（从高到低）
        seen = set()
        unique_levels = []
        for label, price, level_type in sorted(all_levels, key=lambda x: x[1], reverse=True):
            price_rounded = round(price, 2)
            if price_rounded not in seen and price_rounded > 0:
                seen.add(price_rounded)
                unique_levels.append({
                    "name": label,
                    "price": price_rounded,
                    "type": level_type,
                    "distance_pct": round((price_rounded - current) / current * 100, 2)
                })

        resist = [l for l in unique_levels if l["type"] == "resist"]
        support = [l for l in unique_levels if l["type"] == "support"]

        return {
            "current": round(current, 2),
            "date": str(out["date"].iloc[-1].date()),
            "resistance_levels": resist,
            "support_levels": support,
            "summary": {
                "atr_14": round(atr14.iloc[-1], 2),
                "atr_pct": round(atr14.iloc[-1] / current * 100, 2),
                "high_250": round(high_250, 2),
                "low_250": round(low_250, 2),
                "poc": round(poc, 2),
                "va_low": round(va_low, 2),
                "va_high": round(va_high, 2),
            }
        }


# ==================== 信号引擎 ====================
class SignalEngine:
    """交易信号生成引擎"""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.cfg = Config()

    def calculate_all(self, df):
        out = df.copy()

        # 均线
        out["ma5"] = self.indicators.sma(out["close"], 5)
        out["ma10"] = self.indicators.sma(out["close"], 10)
        out["ma20"] = self.indicators.sma(out["close"], 20)
        out["ma60"] = self.indicators.sma(out["close"], 60)
        out["ma120"] = self.indicators.sma(out["close"], 120)
        out["ma250"] = self.indicators.sma(out["close"], 250)

        # 乖离率
        out["bias120"] = (out["close"] - out["ma120"]) / out["ma120"] * 100
        out["bias250"] = (out["close"] - out["ma250"]) / out["ma250"] * 100

        # RSI
        out["rsi6"] = self.indicators.rsi(out["close"], 6)
        out["rsi14"] = self.indicators.rsi(out["close"], 14)
        out["rsi24"] = self.indicators.rsi(out["close"], 24)

        # MACD
        out["macd"], out["macd_signal"], out["macd_hist"] = self.indicators.macd(out["close"])
        out["macd_hist_prev"] = out["macd_hist"].shift(1)

        # 布林带
        out["bb_upper"], out["bb_mid"], out["bb_lower"] = self.indicators.bollinger_bands(out["close"])
        out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

        # KDJ
        out["k"], out["d"], out["j"] = self.indicators.kdj(out["high"], out["low"], out["close"])

        # 成交量
        out["vol_ma5"] = out["volume"].rolling(5).mean()
        out["vol_ma20"] = out["volume"].rolling(20).mean()
        out["vol_ratio"] = out["volume"] / out["vol_ma20"]

        # ATR
        out["atr14"] = self.indicators.atr(out, 14)
        out["atr_pct"] = out["atr14"] / out["close"] * 100

        # 价格极值
        out["high20"] = out["high"].rolling(20).max()
        out["low20"] = out["low"].rolling(20).min()
        out["high60"] = out["high"].rolling(60).max()
        out["low60"] = out["low"].rolling(60).min()

        # ADX
        out["adx"] = self.indicators.adx(out, 14)

        return out

    def generate_signal(self, df, symbol, name, support_data):
        out = self.calculate_all(df)

        # 修复点：直接复用 support_data 里（SupportCalculator用动态bucket_size算好的）POC/VA，
        # 不再用固定bucket_size=10重新算一遍——原代码两处bucket_size不一致，
        # 导致"支撑位列表"和"关键指标"里显示的POC价格对不上号。
        poc = support_data["summary"]["poc"]
        va_low = support_data["summary"]["va_low"]
        va_high = support_data["summary"]["va_high"]

        latest = out.iloc[-1].copy()
        date = latest['date']
        current = latest['close']

        # 检查数据完整性
        required_cols = ["ma250", "rsi14", "macd_hist", "vol_ratio", "atr14"]
        if any(pd.isna(latest[col]) for col in required_cols):
            return SignalResult(
                date=str(date.date()), symbol=symbol, name=name,
                close=latest['close'], signal=SignalType.NO_DATA.value,
                action="HOLD", position_size=PositionSize.EMPTY.name,
                signal_score=0, buy_score=0, sell_score=0,
                confidence=0, stop_loss=0, take_profit=0,
                risk_reward_ratio=0, atr=0, atr_pct=0,
                indicators={}, reasons=["数据不足"], action_reasons=[],
                support_levels=support_data["support_levels"],
                resistance_levels=support_data["resistance_levels"]
            )

        buy_score = 0
        sell_score = 0
        reasons = []
        action_reasons = []

        # 1. 年线趋势
        if latest["close"] > latest["ma250"]:
            buy_score += 1
            reasons.append("✅ 价格在年线之上，长期趋势向上")
        else:
            sell_score += 1
            reasons.append("❌ 价格在年线之下，长期趋势向下")

        # 2. 均线排列
        if latest["ma5"] > latest["ma20"]:
            buy_score += 1
            reasons.append("✅ 短期均线金叉中期均线")
        elif latest["ma5"] < latest["ma20"]:
            sell_score += 1
            reasons.append("❌ 短期均线死叉中期均线")

        # 3. 乖离率
        if latest["bias120"] <= self.cfg.BIAS_BUY:
            buy_score += 2
            reasons.append(f"✅ BIAS120超卖({latest['bias120']:.2f}%)，均值回归概率大")
            action_reasons.append(f"乖离率超卖 {latest['bias120']:.2f}%，偏离均值过远，适合抄底")
        elif latest["bias120"] >= self.cfg.BIAS_SELL:
            sell_score += 2
            reasons.append(f"❌ BIAS120超买({latest['bias120']:.2f}%)，回调风险高")
            action_reasons.append(f"乖离率超买 {latest['bias120']:.2f}%，偏离均值过远，建议减仓")

        if latest["bias250"] <= -8.0:
            buy_score += 1
            reasons.append(f"✅ BIAS250深度超卖({latest['bias250']:.2f}%)")
        elif latest["bias250"] >= 8.0:
            sell_score += 1
            reasons.append(f"❌ BIAS250深度超买({latest['bias250']:.2f}%)")

        # 4. RSI
        if latest["rsi14"] < self.cfg.RSI_BUY:
            buy_score += 2
            reasons.append(f"✅ RSI14超卖({latest['rsi14']:.2f})，严重低估")
            action_reasons.append(f"RSI14 仅 {latest['rsi14']:.2f}，处于超卖区，建议买入")
        elif latest["rsi14"] < 45:
            buy_score += 1
            reasons.append(f"✅ RSI14偏低({latest['rsi14']:.2f})")
        elif latest["rsi14"] > self.cfg.RSI_SELL:
            sell_score += 2
            reasons.append(f"❌ RSI14超买({latest['rsi14']:.2f})，严重高估")
            action_reasons.append(f"RSI14 高达 {latest['rsi14']:.2f}，处于超买区，建议卖出")
        elif latest["rsi14"] > 55:
            sell_score += 1
            reasons.append(f"❌ RSI14偏高({latest['rsi14']:.2f})")

        # 5. MACD
        macd_bullish = latest["macd_hist"] > 0 and latest["macd_hist"] > latest["macd_hist_prev"]
        macd_bearish = latest["macd_hist"] < 0 and latest["macd_hist"] < latest["macd_hist_prev"]

        if macd_bullish:
            buy_score += 2
            reasons.append("✅ MACD柱状图转正且扩大，上涨动能增强")
            action_reasons.append("MACD 柱状图由负转正并扩大，趋势转多，建议买入")
        elif latest["macd_hist"] > 0:
            buy_score += 1
            reasons.append("✅ MACD柱状图为正")

        if macd_bearish:
            sell_score += 2
            reasons.append("❌ MACD柱状图转负且扩大，下跌动能增强")
            action_reasons.append("MACD 柱状图由正转负并扩大，趋势转空，建议卖出")
        elif latest["macd_hist"] < 0:
            sell_score += 1
            reasons.append("❌ MACD柱状图为负")

        # 6. 成交量确认
        if latest["vol_ratio"] >= self.cfg.VOL_CONFIRM:
            if buy_score > sell_score:
                buy_score += 1
                reasons.append(f"✅ 成交量放大({latest['vol_ratio']:.2f}倍)，买盘确认")
                action_reasons.append(f"成交量放大 {latest['vol_ratio']:.2f} 倍，买盘活跃，确认买入信号")
            elif sell_score > buy_score:
                sell_score += 1
                reasons.append(f"❌ 成交量放大({latest['vol_ratio']:.2f}倍)，卖盘确认")
                action_reasons.append(f"成交量放大 {latest['vol_ratio']:.2f} 倍，卖盘活跃，确认卖出信号")

        # 7. 价格位置
        near_low20 = pd.notna(latest["low20"]) and latest["close"] <= latest["low20"] * 1.03
        near_low60 = pd.notna(latest["low60"]) and latest["close"] <= latest["low60"] * 1.05
        near_high20 = pd.notna(latest["high20"]) and latest["close"] >= latest["high20"] * 0.97

        if near_low20 or near_low60:
            buy_score += 1
            reasons.append("✅ 价格接近近期低点，处于支撑区域")
            action_reasons.append("价格接近近期低点，支撑区域，适合逢低买入")
        if near_high20:
            sell_score += 1
            reasons.append("❌ 价格接近近期高点，处于阻力区域")
            action_reasons.append("价格接近近期高点，阻力区域，建议逢高减仓")

        # 8. KDJ
        if latest["j"] < 20 and latest["k"] < 30:
            buy_score += 1
            reasons.append(f"✅ KDJ超卖区(J={latest['j']:.2f})")
        elif latest["j"] > 80 and latest["k"] > 70:
            sell_score += 1
            reasons.append(f"❌ KDJ超买区(J={latest['j']:.2f})")

        # 9. 布林带
        if latest["bb_position"] < 0.1:
            buy_score += 1
            reasons.append("✅ 价格触及布林带下轨")
        elif latest["bb_position"] > 0.9:
            sell_score += 1
            reasons.append("❌ 价格触及布林带上轨")

        # 10. ADX趋势强度
        if latest["adx"] > 25:
            reasons.append(f"趋势较强(ADX={latest['adx']:.2f})")
        else:
            reasons.append(f"趋势较弱(ADX={latest['adx']:.2f})")

        # 信号判定
        signal_score = buy_score - sell_score

        if buy_score >= 5 and signal_score >= 3:
            signal = SignalType.STRONG_BUY
            position = PositionSize.HEAVY
            action = "BUY"
        elif buy_score >= 4 and signal_score >= 2:
            signal = SignalType.BUY
            position = PositionSize.MEDIUM
            action = "BUY"
        elif sell_score >= 5 and signal_score <= -3:
            signal = SignalType.STRONG_SELL
            position = PositionSize.EMPTY
            action = "SELL"
        elif sell_score >= 4 and signal_score <= -2:
            signal = SignalType.SELL
            position = PositionSize.LIGHT
            action = "SELL"
        else:
            signal = SignalType.HOLD
            if signal_score > 0:
                position = PositionSize.MEDIUM
            elif signal_score < 0:
                position = PositionSize.LIGHT
            else:
                position = PositionSize.MEDIUM
            action = "HOLD"

        # 止损止盈计算
        atr = latest["atr14"]
        if action == "BUY":
            stop_loss = current - self.cfg.ATR_STOP_MULT * atr
            take_profit = current + 2 * self.cfg.ATR_STOP_MULT * atr
        elif action == "SELL":
            stop_loss = current + self.cfg.ATR_STOP_MULT * atr
            take_profit = current - 2 * self.cfg.ATR_STOP_MULT * atr
        else:
            stop_loss = current - self.cfg.ATR_STOP_MULT * atr
            take_profit = current + self.cfg.ATR_STOP_MULT * atr

        risk = abs(current - stop_loss)
        reward = abs(take_profit - current)
        rr_ratio = reward / risk if risk > 0 else 0
        confidence = min(abs(signal_score) / 8 * 100, 95)

        indicators = {
            "ma20": round(latest["ma20"], 2),
            "ma60": round(latest["ma60"], 2),
            "ma120": round(latest["ma120"], 2),
            "ma250": round(latest["ma250"], 2),
            "bias120": round(latest["bias120"], 2),
            "bias250": round(latest["bias250"], 2),
            "rsi14": round(latest["rsi14"], 2),
            "macd_hist": round(latest["macd_hist"], 4),
            "vol_ratio": round(latest["vol_ratio"], 2),
            "atr_pct": round(latest["atr_pct"], 2),
            "bb_position": round(latest["bb_position"], 2),
            "k": round(latest["k"], 2),
            "d": round(latest["d"], 2),
            "j": round(latest["j"], 2),
            "adx": round(latest["adx"], 2),
            "poc": round(poc, 2),
            "va_low": round(va_low, 2),
            "va_high": round(va_high, 2)
        }

        return SignalResult(
            date=str(date.date()),
            symbol=symbol,
            name=name,
            close=round(current, 2),
            signal=signal.value,
            action=action,
            position_size=position.name,
            signal_score=signal_score,
            buy_score=buy_score,
            sell_score=sell_score,
            confidence=round(confidence, 1),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            atr=round(atr, 2),
            atr_pct=round(atr / current * 100, 2),
            indicators=indicators,
            reasons=reasons,
            action_reasons=action_reasons,
            support_levels=support_data["support_levels"],
            resistance_levels=support_data["resistance_levels"]
        )


# ==================== 通知推送 ====================
class Notifier:
    """消息推送类"""

    def __init__(self):
        self.key = Config.SERVERCHAN_KEY

    def send(self, title, content):
        if not self.key:
            logger.info("未配置 ServerChan Key，跳过推送")
            return False
        try:
            resp = requests.post(
                f"https://sctapi.ftqq.com/{self.key}.send",
                data={"title": title, "desp": content},
                timeout=10
            )
            if resp.status_code == 200:
                logger.info("推送成功")
                return True
            else:
                logger.error(f"推送失败: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"推送异常: {e}")
            return False


# ==================== 可视化 ====================
class Visualizer:
    """可视化类"""

    def __init__(self):
        # SimHei/Arial Unicode MS 是Windows/Mac字体，GitHub Actions的Ubuntu默认没有；
        # 需要在workflow yaml里加一步 apt-get install fonts-wqy-zenhei 安装文泉驿字体，
        # 否则图表里所有中文会渲染成方块。这里把文泉驿字体列在前面作为主选。
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_support_resistance(self, results, save_path):
        """绘制支撑/阻力全景图"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        colors_map = {'resist': '#e74c3c', 'support': '#27ae60', 'current': '#3498db'}

        for idx, r in enumerate(results):
            ax = axes[idx]
            current = r.close
            levels = r.resistance_levels + [{"name": "当前价", "price": current, "type": "current", "distance_pct": 0}] + r.support_levels

            # 绘制水平线
            for l in r.resistance_levels[:8]:
                ax.axhline(y=l["price"], color=colors_map["resist"], alpha=0.4, linewidth=1, linestyle="--")
            for l in r.support_levels[:10]:
                ax.axhline(y=l["price"], color=colors_map["support"], alpha=0.4, linewidth=1, linestyle="--")

            ax.axhline(y=current, color=colors_map["current"], alpha=0.9, linewidth=2.5)

            # 支撑区/阻力区背景
            if r.resistance_levels:
                ax.axhspan(current, min([l["price"] for l in r.resistance_levels]), alpha=0.05, color="red")
            if r.support_levels:
                ax.axhspan(max([l["price"] for l in r.support_levels]), current, alpha=0.05, color="green")

            ax.set_title(f"{r.name} ({r.symbol})\n当前价: {current:.2f} | 信号: {r.signal}", 
                        fontsize=13, fontweight="bold", pad=15)

            all_prices = [l["price"] for l in levels]
            ax.set_ylim(min(all_prices) * 0.98, max(all_prices) * 1.02)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_ylabel("Price", fontsize=11)
            ax.grid(True, alpha=0.2, axis="y")

            # 图例
            resist_patch = mpatches.Patch(color=colors_map["resist"], alpha=0.6, label="阻力区")
            support_patch = mpatches.Patch(color=colors_map["support"], alpha=0.6, label="支撑区")
            current_line = plt.Line2D([0], [0], color=colors_map["current"], linewidth=2.5, label="当前价")
            ax.legend(handles=[resist_patch, current_line, support_patch], loc="upper left", fontsize=9)

        plt.suptitle("A股主要指数 支撑/阻力线全景图", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"支撑/阻力图已保存: {save_path}")

    def plot_technical_analysis(self, df, result, save_path):
        """绘制四合一技术分析图"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.08)

        df_plot = df.tail(120).reset_index(drop=True)

        # K线+均线
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(f"{result.name} ({result.symbol}) 技术分析 ({result.date}) - 信号: {result.signal}", 
                     fontsize=14, fontweight="bold", pad=15)
        for i in range(len(df_plot)):
            row = df_plot.iloc[i]
            color = "#e74c3c" if row["close"] >= row["open"] else "#27ae60"
            ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            ax1.add_patch(Rectangle((i-0.35, min(row["open"], row["close"])), 0.7, 
                                     abs(row["close"]-row["open"]), facecolor=color, edgecolor=color, alpha=0.9))
        ax1.plot(df_plot.index, df_plot["close"].rolling(20).mean(), label="MA20", color="#e67e22", linewidth=1.2)
        ax1.plot(df_plot.index, df_plot["close"].rolling(60).mean(), label="MA60", color="#3498db", linewidth=1.2)
        ax1.axhline(y=result.stop_loss, color="red", linestyle="--", alpha=0.7, label=f"止损: {result.stop_loss}")
        ax1.axhline(y=result.take_profit, color="green", linestyle="--", alpha=0.7, label=f"止盈: {result.take_profit}")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.2, linestyle="--")

        # 成交量
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors = ["#e74c3c" if df_plot.iloc[i]["close"] >= df_plot.iloc[i]["open"] else "#27ae60" for i in range(len(df_plot))]
        ax2.bar(df_plot.index, df_plot["volume"], color=colors, alpha=0.6, width=0.8)
        ax2.plot(df_plot.index, df_plot["volume"].rolling(20).mean(), color="#3498db", linewidth=1.2, label="VOL_MA20")
        ax2.set_ylabel("Volume")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, linestyle="--")

        # MACD
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ema12 = df_plot["close"].ewm(span=12, adjust=False).mean()
        ema26 = df_plot["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        colors_macd = ["#e74c3c" if hist.iloc[i] >= 0 else "#27ae60" for i in range(len(hist))]
        ax3.bar(df_plot.index, hist, color=colors_macd, alpha=0.6, width=0.8)
        ax3.plot(df_plot.index, macd_line, color="#3498db", linewidth=1.2, label="MACD")
        ax3.plot(df_plot.index, signal_line, color="#e67e22", linewidth=1.2, label="Signal")
        ax3.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
        ax3.set_ylabel("MACD")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.2, linestyle="--")

        # RSI
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        delta = df_plot["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        ax4.plot(df_plot.index, rsi, color="#9b59b6", linewidth=1.5, label="RSI14")
        ax4.axhline(y=70, color="red", linestyle="--", alpha=0.6)
        ax4.axhline(y=30, color="green", linestyle="--", alpha=0.6)
        ax4.fill_between(df_plot.index, 30, 70, alpha=0.08, color="gray")
        ax4.set_ylabel("RSI")
        ax4.set_xlabel("Trading Days")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.2, linestyle="--")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"技术分析图已保存: {save_path}")


# ==================== 主程序 ====================
class QuantBot:
    """量化信号机器人"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.engine = SignalEngine()
        self.support_calc = SupportCalculator()
        self.notifier = Notifier()
        self.visualizer = Visualizer()
        self.results = []
        self.data_cache = {}  # symbol -> df，避免画图阶段重复拉取

    def run(self):
        logger.info("="*70)
        logger.info("A股指数技术分析信号系统启动")
        logger.info("="*70)

        bs.login()
        for symbol, name in Config.SYMBOLS.items():
            try:
                logger.info(f"\n分析: {name} ({symbol})")

                # 1. 获取数据
                df = self.fetcher.get_index_data(symbol)
                self.data_cache[symbol] = df  # 缓存，避免画图阶段重复请求

                # 2. 计算支撑/阻力线
                support_data = self.support_calc.calculate_all(df, symbol, name)

                # 3. 生成交易信号
                result = self.engine.generate_signal(df, symbol, name, support_data)
                self.results.append(result)

                logger.info(f"  信号: {result.signal} | 评分: {result.signal_score:+d} | 置信度: {result.confidence:.1f}%")
                logger.info(f"  支撑线: {len(result.support_levels)} 条 | 阻力线: {len(result.resistance_levels)} 条")

            except Exception as e:
                logger.error(f"分析 {symbol} 失败: {e}")
                logger.error(traceback.format_exc())
        bs.logout()

        self._save_results()
        self._generate_visualizations()
        self._send_notification()
        self._print_summary()

        return self.results

    def _save_results(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # JSON
        json_path = os.path.join(Config.OUTPUT_DIR, "signal_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存: {json_path}")

        # Markdown 报告
        md_path = os.path.join(Config.OUTPUT_DIR, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown())
        logger.info(f"报告已保存: {md_path}")

    def _generate_markdown(self):
        lines = [
            f"# A股指数技术分析报告",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 交易建议汇总",
            f"",
            f"| 指数 | 代码 | 收盘价 | 建议 | 仓位 | 止损 | 止盈 | 盈亏比 |",
            f"|------|------|--------|------|------|------|------|--------|",
        ]

        for r in self.results:
            action_emoji = "🟢" if r.action == "BUY" else "🔴" if r.action == "SELL" else "⚪"
            lines.append(
                f"| {r.name} | {r.symbol} | {r.close} | {action_emoji} {r.signal} | "
                f"{r.position_size} | {r.stop_loss} | {r.take_profit} | {r.risk_reward_ratio} |"
            )

        lines.append(f"")
        lines.append(f"## 详细分析")
        lines.append(f"")

        for r in self.results:
            lines.append(f"### {r.name} ({r.symbol})")
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
            lines.append(f"")
            lines.append(f"| 排名 | 名称 | 价格 | 距现价 |")
            lines.append(f"|------|------|------|--------|")
            for i, l in enumerate(r.resistance_levels, 1):
                lines.append(f"| {i} | {l['name']} | {l['price']:.2f} | {l['distance_pct']:+.2f}% |")
            lines.append(f"")

            lines.append(f"**支撑线（从高到低）**:")
            lines.append(f"")
            lines.append(f"| 排名 | 名称 | 价格 | 距现价 |")
            lines.append(f"|------|------|------|--------|")
            for i, l in enumerate(r.support_levels, 1):
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
        lines.append(f"2. 实际交易请结合基本面和市场情绪综合判断")
        lines.append(f"3. 严格设置止损，控制单笔亏损不超过本金的2%")
        lines.append(f"4. 分散投资，不要将所有资金投入单一指数")

        return "\n".join(lines)

    def _generate_visualizations(self):
        try:
            # 支撑/阻力全景图
            self.visualizer.plot_support_resistance(
                self.results,
                os.path.join(Config.OUTPUT_DIR, "support_resistance.png")
            )

            # 单个指数技术分析图
            for symbol, name in Config.SYMBOLS.items():
                try:
                    df = self.data_cache.get(symbol)  # 复用run()里已拉取的数据，不再重新请求
                    if df is None:
                        logger.warning(f"{symbol} 无缓存数据，跳过绘图")
                        continue
                    result = next(r for r in self.results if r.symbol == symbol)
                    self.visualizer.plot_technical_analysis(
                        df, result,
                        os.path.join(Config.OUTPUT_DIR, f"analysis_{symbol}.png")
                    )
                except Exception as e:
                    logger.error(f"绘制 {symbol} 图表失败: {e}")

        except Exception as e:
            logger.error(f"可视化生成失败: {e}")

    def _send_notification(self):
        if not self.results:
            return

        strong = [r for r in self.results if r.signal in ["强烈买入", "强烈卖出"]]

        # 修复点：原代码只在有"强烈买入/强烈卖出"信号时才推送，平时无从得知脚本是否正常运行、
        # 结果如何——现在改成每次都推送一份简要摘要，强烈信号单独突出显示
        title = f"【量化信号】{len(strong)}个强烈信号" if strong else "【量化信号】今日无强烈信号"
        content_lines = [f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

        if strong:
            content_lines.append("### ⚠️ 强烈信号")
            for r in strong:
                content_lines.append(f"## {r.name} ({r.symbol})")
                content_lines.append(f"信号: {r.signal}")
                content_lines.append(f"收盘价: {r.close} | 评分: {r.signal_score:+d}")
                content_lines.append(f"建议: {r.position_size} | 止损: {r.stop_loss} | 止盈: {r.take_profit}")
                content_lines.append("")

        content_lines.append("### 全部指数概览")
        for r in self.results:
            content_lines.append(f"- {r.name}({r.symbol})：{r.close} | {r.signal} | 评分{r.signal_score:+d}")

        self.notifier.send(title, "\n\n".join(content_lines))

    def _print_summary(self):
        print("\n" + "="*85)
        print("📊 A股主要指数 交易建议报告".center(85))
        print(f"📅 分析日期: {self.results[0].date if self.results else 'N/A'}".center(85))
        print("="*85)

        for r in self.results:
            print(f"\n{'─'*85}")
            print(f"  {r.name} ({r.symbol})  当前价: {r.close:.2f}")
            print(f"{'─'*85}")

            if r.action == "BUY":
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📈 信号评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  🛑 止损位: {r.stop_loss:.2f} (ATR: {r.atr:.2f}, {r.atr_pct:.2f}%)")
                print(f"  🎯 止盈位: {r.take_profit:.2f}")
                print(f"  ⚖️  盈亏比: 1:{r.risk_reward_ratio:.2f}")
            elif r.action == "SELL":
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📉 信号评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  🛑 止损位: {r.stop_loss:.2f} (ATR: {r.atr:.2f}, {r.atr_pct:.2f}%)")
                print(f"  🎯 止盈位: {r.take_profit:.2f}")
                print(f"  ⚖️  盈亏比: 1:{r.risk_reward_ratio:.2f}")
            else:
                print(f"\n  🎯 交易建议: {r.signal}")
                print(f"  💰 建议仓位: {r.position_size}")
                print(f"  📊 信号评分: 买入 {r.buy_score} 分 vs 卖出 {r.sell_score} 分 (净得分: {r.signal_score:+d})")
                print(f"  💡 置信度: {r.confidence:.1f}%")
                print(f"  ⚠️  建议观望，等待更明确信号")

            if r.action_reasons:
                print(f"\n  📋 核心交易理由:")
                for i, reason in enumerate(r.action_reasons, 1):
                    print(f"     {i}. {reason}")

            print(f"\n  📈 关键指标:")
            ind = r.indicators
            print(f"     MA20: {ind['ma20']:.2f} | MA60: {ind['ma60']:.2f} | MA120: {ind['ma120']:.2f} | MA250: {ind['ma250']:.2f}")
            print(f"     BIAS120: {ind['bias120']:.2f}% | BIAS250: {ind['bias250']:.2f}%")
            print(f"     RSI14: {ind['rsi14']:.2f} | MACD柱: {ind['macd_hist']:.4f} | 量比: {ind['vol_ratio']:.2f}")

            print(f"\n  🟢 支撑线（从高到低）:")
            for i, l in enumerate(r.support_levels[:10], 1):
                print(f"     {i:2d}. {l['name']:<20s} {l['price']:>10.2f} ({l['distance_pct']:+.2f}%)")

            print(f"\n  🔴 阻力线（从高到低）:")
            for i, l in enumerate(r.resistance_levels[:8], 1):
                print(f"     {i:2d}. {l['name']:<20s} {l['price']:>10.2f} ({l['distance_pct']:+.2f}%)")

        print(f"\n{'='*85}")
        print("⚠️ 风险提示".center(85))
        print("="*85)
        print("  1. 以上分析基于技术指标，不构成投资建议")
        print("  2. 实际交易请结合基本面和市场情绪综合判断")
        print("  3. 严格设置止损，控制单笔亏损不超过本金的2%")
        print("  4. 分散投资，不要将所有资金投入单一指数")
        print(f"{'='*85}")


def main():
    bot = QuantBot()
    bot.run()


if __name__ == "__main__":
    main()
