"""
MTF 共振选股系统 v4.0 (全市场版) — 趋势共振型
====================================================================
由 v3.0 固定 10 股升级而来, 保持文件名 mtf_resonance_screener.py 不变,
无缝兼容现有 workflow 的 mtf-resonance-screener job。

升级点:
  1. 固定10股 -> 全市场 akshare 实时快照初筛 + 并发精算
  2. 修复推送 bug: SERVERCHAN_KEY 从环境变量读, 兼容 SERVERCHAN_KEY/SENDKEY 两种命名
  3. 新增交易日判断 (本 job cron 含周末, 非交易日自动跳过)
  4. serverchan-sdk 优先 + requests 兜底
核心保留: MTFResonanceEngine 8 维共振评分模型 (一字未动)
依赖: 仓库现有 requirements.txt (akshare / pandas / requests / tqdm)
兼容: Python 3.10
====================================================================
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import random
import warnings
import logging
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==================== 配置 ====================

@dataclass
class Config:
    """系统配置"""
    UNIVERSE: str = os.environ.get('UNIVERSE', 'ACTIVE')        # ACTIVE/HS300/ZZ500/ALL
    MAX_CANDIDATES: int = int(os.environ.get('MAX_CANDIDATES', '500'))
    MAX_WORKERS: int = int(os.environ.get('MAX_WORKERS', '6'))
    TOP_N: int = int(os.environ.get('TOP_N', '30'))

    OUTPUT_DIR: str = "output"
    SCORE_THRESHOLD: int = 65
    STRONG_THRESHOLD: int = 78
    MIN_DATA_DAYS: int = 250
    REQUEST_DELAY: float = 0.0   # 并发下不再串行等待
    MAX_RETRIES: int = 3

    TURNOVER_MIN: float = 1.0
    AMOUNT_MIN: float = 1.0e8
    CAP_MIN: float = 2.0e9
    CAP_MAX: float = 3.0e11
    PRICE_MIN: float = 2.0
    LIMIT_PCT: float = 9.5

    # 推送 (修复: 从环境变量读, 兼容 SERVERCHAN_KEY 和 SENDKEY 两种命名)
    SERVERCHAN_KEY: str = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
    ENABLE_SERVERCHAN: bool = None

    def __post_init__(self):
        if self.ENABLE_SERVERCHAN is None:
            self.ENABLE_SERVERCHAN = bool(self.SERVERCHAN_KEY)


CONFIG = Config()

os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"{CONFIG.OUTPUT_DIR}/mtf_scanner_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)


# ==================== 工具函数 ====================

def send_serverchan(title: str, content: str, sendkey: str = "") -> bool:
    """Server酱推送: serverchan-sdk 优先, requests 兜底"""
    key = sendkey or CONFIG.SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断, 详见 output 报告)"
    try:  # 签名 sc_send(sendkey, title, desp); SDK 自动适配 key 版本
        from serverchan_sdk import sc_send
        sc_send(key, title, content)
        logger.info("serverchan-sdk 推送成功")
        return True
    except Exception as e:
        logger.warning(f"serverchan-sdk 失败, 回退 requests: {e}")
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        resp = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        return resp.json().get("code") == 0
    except Exception as e:
        logger.warning(f"requests 推送失败: {e}")
        return False


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """带指数退避的重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"{func.__name__} 第 {attempt + 1} 次失败，{delay:.1f}s 后重试: {e}")
                        time.sleep(delay)
                    else:
                        raise e
            return None
        return wrapper
    return decorator


def is_trading_day() -> bool:
    """非A股交易日(周末/节假日)直接跳过 (本 job cron 含周末, 此判断很重要)"""
    try:
        df = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        logger.warning(f"交易日历获取失败, 默认继续: {e}")
        return True


# ==================== 全市场初筛 (akshare) ====================

class Universe:
    """全市场实时快照初筛: 1 次接口拿全A, 向量化过滤后截断"""

    @classmethod
    def snapshot_filter(cls) -> List[Dict]:
        df = None
        for i in range(1, CONFIG.MAX_RETRIES + 1):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    break
            except Exception as e:
                logger.warning(f"  全A快照第{i}次失败: {e}")
            time.sleep(CONFIG.MAX_RETRIES + random.uniform(0, 2))
        if df is None or df.empty:
            logger.error("全A快照获取失败(东财可能限流), 无法选股")
            return []

        logger.info(f"  全A快照原始 {len(df)} 只")
        if len(df) < 1000:
            logger.warning(f"  快照行数异常偏少({len(df)}), 接口可能受限")

        df['代码'] = df['代码'].astype(str)
        for c in ['最新价', '涨跌幅', '换手率', '量比', '总市值', '流通市值', '成交额']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        m_code = df['代码'].str.match(r'^(60|00|30|68)')
        m_st = ~df['名称'].str.contains('ST|退', na=False, regex=True)
        m_act = (df['最新价'] >= CONFIG.PRICE_MIN) & (df['成交额'] >= CONFIG.AMOUNT_MIN)
        m_turn = df['换手率'] >= CONFIG.TURNOVER_MIN
        m_cap = (df['总市值'] >= CONFIG.CAP_MIN) & (df['总市值'] <= CONFIG.CAP_MAX)
        m_lim = df['涨跌幅'].abs() < CONFIG.LIMIT_PCT
        out = df[m_code & m_st & m_act & m_turn & m_cap & m_lim].copy()
        logger.info(f"  初筛后 {len(out)} 只")

        if CONFIG.UNIVERSE in ("HS300", "ZZ500"):
            idx = "000300" if CONFIG.UNIVERSE == "HS300" else "000905"
            try:
                cons = ak.index_stock_cons_csindex(symbol=idx)
                if cons is not None and not cons.empty:
                    code_col = '成分券代码' if '成分券代码' in cons.columns else cons.columns[0]
                    valid = set(cons[code_col].astype(str).str.zfill(6))
                    out = out[out['代码'].isin(valid)]
                    logger.info(f"  与{CONFIG.UNIVERSE}成分交集后 {len(out)} 只")
            except Exception as e:
                logger.warning(f"  获取{CONFIG.UNIVERSE}成分失败, 回退全A初筛: {e}")

        out = out.sort_values('成交额', ascending=False)
        if CONFIG.UNIVERSE != "ALL" and len(out) > CONFIG.MAX_CANDIDATES:
            logger.info(f"  截断至成交额 Top {CONFIG.MAX_CANDIDATES}")
            out = out.head(CONFIG.MAX_CANDIDATES)

        return out[['代码', '名称', '最新价', '涨跌幅', '换手率', '成交额']].to_dict('records')


# ==================== 数据获取 ====================

@retry_with_backoff(max_retries=CONFIG.MAX_RETRIES, base_delay=1.0)
def get_data(symbol: str, days: int = 600) -> pd.DataFrame:
    """获取股票历史数据（日线，后复权）"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days + 60)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="hfq")
    if df.empty:
        raise ValueError(f"{symbol} 返回空数据")

    col_mapping = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=col_mapping)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for col in ['date', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    available_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    cols = [c for c in available_cols if c in df.columns]
    return df[cols].copy()


@retry_with_backoff(max_retries=CONFIG.MAX_RETRIES, base_delay=1.0)
def get_industry_data(symbol: str) -> str:
    """获取股票所属行业 (仅对入选标的调用, 数量少)"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if not df.empty and 'item' in df.columns:
            row = df[df['item'].isin(['所属行业', '行业'])]
            if not row.empty:
                return row['value'].values[0]
    except Exception:
        pass
    return "未知"


# ==================== 技术指标计算 ====================

def calculate_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    macd = (dif - dea) * 2
    return pd.DataFrame({'dif': dif, 'dea': dea, 'macd': macd})

def calculate_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = calculate_ma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    return pd.DataFrame({'boll_mid': ma, 'boll_up': ma + num_std * std, 'boll_low': ma - num_std * std})

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['high'] if 'high' in df.columns else df['close']
    low = df['low'] if 'low' in df.columns else df['close']
    prev_close = df['close'].shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


# ==================== 共振评分引擎 (核心, 一字未动) ====================

class MTFResonanceEngine:
    """MTF 多时间框架共振评分引擎"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.df
        for window in [5, 10, 20, 60, 120, 250]:
            df[f'ma{window}'] = calculate_ma(df['close'], window)
        df['rsi'] = calculate_rsi(df['close'])
        macd_df = calculate_macd(df['close'])
        df['macd_dif'] = macd_df['dif']
        df['macd_dea'] = macd_df['dea']
        df['macd'] = macd_df['macd']
        boll_df = calculate_bollinger(df['close'])
        df['boll_mid'] = boll_df['boll_mid']
        df['boll_up'] = boll_df['boll_up']
        df['boll_low'] = boll_df['boll_low']
        if all(c in df.columns for c in ['high', 'low']):
            df['atr'] = calculate_atr(df)
        else:
            df['atr'] = df['close'].diff().abs().rolling(14, min_periods=1).mean()
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252)
        self.df = df

    def _safe_get(self, row_idx: int, col: str, default: float = 0) -> float:
        if row_idx < 0:
            row_idx = len(self.df) + row_idx
        if row_idx < 0 or row_idx >= len(self.df):
            return default
        val = self.df.iloc[row_idx].get(col, default)
        return val if pd.notna(val) else default

    def calculate_score(self) -> Tuple[float, str, Dict]:
        """计算 MTF 共振评分（满分100）"""
        if len(self.df) < CONFIG.MIN_DATA_DAYS:
            return 0, "数据不足", {}

        df = self.df
        latest_idx, prev_idx, prev3_idx = -1, -2, -3

        close = self._safe_get(latest_idx, 'close')
        ma20 = self._safe_get(latest_idx, 'ma20')
        ma20_prev = self._safe_get(prev_idx, 'ma20')
        ma20_prev3 = self._safe_get(prev3_idx, 'ma20')

        score = 0.0
        details = {}

        # 1. 位置接近 MA20 (15分)
        dist_pct = (close - ma20) / ma20 if ma20 != 0 else 999
        position_score = max(0, 1 - abs(dist_pct) / 0.085) * 15
        score += position_score
        details['位置接近MA20'] = round(position_score, 1)

        # 2. 上穿/回踩信号 (20分)
        prev_close = self._safe_get(prev_idx, 'close')
        today_cross = (self._safe_get(prev_idx, 'close') <= ma20_prev) and (close > ma20)
        recent_cross = False
        for i in range(-3, 0):
            if i >= -len(df):
                pc = self._safe_get(i-1, 'close')
                pm = self._safe_get(i-1, 'ma20')
                cc = self._safe_get(i, 'close')
                if pc <= pm and cc > pm:
                    recent_cross = True
                    break
        if today_cross:
            cross_score = 20
        elif recent_cross and dist_pct > -0.02:
            cross_score = 15
        elif dist_pct > -0.015:
            cross_score = 10
        elif dist_pct > -0.05:
            cross_score = 5
        else:
            cross_score = 0
        score += cross_score
        details['上穿/回踩信号'] = round(cross_score, 1)

        # 3. 短期动量 (15分)
        ma5 = self._safe_get(latest_idx, 'ma5')
        ma5_prev = self._safe_get(prev_idx, 'ma5')
        ma10 = self._safe_get(latest_idx, 'ma10')
        ma10_prev = self._safe_get(prev_idx, 'ma10')
        ma5_up = ma5 > ma5_prev
        ma10_up = ma10 > ma10_prev
        price_up = close > prev_close
        if ma5_up and ma10_up and price_up:
            mom_score = 15
        elif ma5_up and ma10_up:
            mom_score = 12
        elif ma5_up or ma10_up:
            mom_score = 8
        else:
            mom_score = 3
        score += mom_score
        details['短期动量'] = round(mom_score, 1)

        # 4. 中期趋势 (15分)
        ma20_slope = (ma20 - ma20_prev3) / ma20_prev3 if ma20_prev3 != 0 else 0
        if ma20 > ma20_prev3 and ma20_slope > 0.001:
            trend_score = 15
        elif ma20 > ma20_prev3:
            trend_score = 12
        elif ma20 > ma20_prev:
            trend_score = 8
        else:
            trend_score = 3
        score += trend_score
        details['MA20趋势'] = round(trend_score, 1)

        # 5. 成交量配合 (10分)
        vol_current = self._safe_get(latest_idx, 'volume')
        vol_prev = self._safe_get(prev_idx, 'volume', 1)
        vol_avg5 = df['volume'].tail(5).mean() if 'volume' in df.columns else vol_current
        vol_ratio = vol_current / vol_prev if vol_prev > 0 else 1
        vol_vs_avg = vol_current / vol_avg5 if vol_avg5 > 0 else 1
        if 1.2 <= vol_ratio <= 2.5 and vol_vs_avg > 1.0:
            vol_score = 10
        elif 1.0 <= vol_ratio <= 3.0:
            vol_score = 8
        elif vol_ratio > 0.8:
            vol_score = 5
        else:
            vol_score = 2
        score += vol_score
        details['成交量配合'] = round(vol_score, 1)

        # 6. 长线趋势支持 (15分)
        ma60 = self._safe_get(latest_idx, 'ma60')
        ma120 = self._safe_get(latest_idx, 'ma120')
        ma250 = self._safe_get(latest_idx, 'ma250')
        long_bull = (close > ma60 * 0.98 and ma60 > ma120 * 0.97 and
                     ma120 > ma250 * 0.97 and ma120 > ma250)
        above_long = close > ma120 * 0.96 and close > ma250 * 0.95
        if long_bull and above_long:
            long_score = 15
        elif above_long:
            long_score = 10
        elif close > ma250 * 0.95:
            long_score = 6
        else:
            long_score = 2
        score += long_score
        details['长线趋势'] = round(long_score, 1)

        # 7. RSI 健康度 (5分)
        rsi = self._safe_get(latest_idx, 'rsi', 50)
        if 40 <= rsi <= 65:
            rsi_score = 5
        elif 30 <= rsi < 40:
            rsi_score = 3
        elif 65 < rsi <= 75:
            rsi_score = 2
        else:
            rsi_score = 0
        score += rsi_score
        details['RSI健康度'] = round(rsi_score, 1)

        # 8. MACD 共振 (5分)
        macd_current = self._safe_get(latest_idx, 'macd')
        macd_prev = self._safe_get(prev_idx, 'macd')
        dif = self._safe_get(latest_idx, 'macd_dif')
        dea = self._safe_get(latest_idx, 'macd_dea')
        if dif > dea and macd_current > macd_prev and macd_current > 0:
            macd_score = 5
        elif dif > dea and macd_current > macd_prev:
            macd_score = 3
        elif dif > dea:
            macd_score = 1
        else:
            macd_score = 0
        score += macd_score
        details['MACD共振'] = round(macd_score, 1)

        final_score = min(100, round(score, 1))
        if final_score >= CONFIG.STRONG_THRESHOLD:
            level = '强信号'
        elif final_score >= CONFIG.SCORE_THRESHOLD:
            level = '中信号'
        elif final_score >= 50:
            level = '弱信号'
        else:
            level = '无信号'
        return final_score, level, details

    def get_summary(self) -> Dict:
        latest = self.df.iloc[-1]
        return {
            'close': round(latest['close'], 2),
            'ma20': round(latest.get('ma20', latest['close']), 2),
            'ma60': round(latest.get('ma60', latest['close']), 2),
            'rsi': round(latest.get('rsi', 50), 1),
            'macd': round(latest.get('macd', 0), 3),
            'volatility': round(latest.get('volatility', 0) * 100, 2),
        }


# ==================== 主程序 ====================

def analyze_stock(symbol: str, name: str) -> Optional[Dict]:
    """分析单只股票 (名称从快照带入; 行业留给 main 对入选标的补)"""
    try:
        df = get_data(symbol)
        if df is None or len(df) < CONFIG.MIN_DATA_DAYS:
            return None

        engine = MTFResonanceEngine(df)
        score, level, details = engine.calculate_score()
        if score < CONFIG.SCORE_THRESHOLD:
            return None

        summary = engine.get_summary()
        dist_pct = (summary['close'] - summary['ma20']) / summary['ma20'] * 100 if summary['ma20'] != 0 else 0

        return {
            '日期': datetime.now().strftime('%Y-%m-%d'),
            '代码': symbol,
            '名称': name,
            '行业': '',
            '总分': score,
            '级别': level,
            '收盘价': summary['close'],
            '距MA20%': round(dist_pct, 2),
            'RSI': summary['rsi'],
            'MACD': summary['macd'],
            '波动率%': summary['volatility'],
            '详细评分': json.dumps(details, ensure_ascii=False),
            '数据条数': len(df)
        }
    except Exception as e:
        logger.debug(f"分析 {symbol} 失败: {e}")
        return None


def main():
    logger.info("=" * 70)
    logger.info(f"MTF 共振选股系统 v4.0 (全市场) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"范围={CONFIG.UNIVERSE} | 并发={CONFIG.MAX_WORKERS} | 推送={'开' if CONFIG.ENABLE_SERVERCHAN else '关'}")
    logger.info("=" * 70)

    if not is_trading_day():
        logger.info("今日非A股交易日, 跳过选股")
        return

    candidates = Universe.snapshot_filter()
    if not candidates:
        logger.error("无候选股票, 退出")
        return

    logger.info(f"并发精算 {len(candidates)} 只 ...")
    results = []
    with ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS) as ex:
        futs = {ex.submit(analyze_stock, c['代码'], c['名称']): c for c in candidates}
        for f in tqdm(as_completed(futs), total=len(futs), desc="MTF共振扫描", unit="只"):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        logger.info("今日无符合条件的共振信号")
        return

    logger.info(f"为 {len(results)} 只入选标的补全行业 ...")
    for r in tqdm(results, desc="补行业", unit="只"):
        r['行业'] = get_industry_data(r['代码'])

    results.sort(key=lambda x: x['总分'], reverse=True)
    df_result = pd.DataFrame(results)

    csv_file = f"{CONFIG.OUTPUT_DIR}/MTF_共振选股_{datetime.now().strftime('%Y%m%d')}.csv"
    json_file = f"{CONFIG.OUTPUT_DIR}/MTF_共振选股_{datetime.now().strftime('%Y%m%d')}.json"
    df_result.to_csv(csv_file, index=False, encoding='utf-8-sig')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    display_cols = ['代码', '名称', '行业', '总分', '级别', '收盘价', '距MA20%', 'RSI']
    print("\n" + "=" * 70)
    print(f"📊 MTF 共振选股结果 (共 {len(results)} 个信号)")
    print("=" * 70)
    print(df_result[display_cols].head(CONFIG.TOP_N).to_string(index=False))
    print("=" * 70)
    print(f"📁 CSV: {csv_file}")
    print(f"📁 JSON: {json_file}")

    if CONFIG.ENABLE_SERVERCHAN:
        top = results[:CONFIG.TOP_N]
        push_content = "\n\n".join([
            f"**{r['代码']} {r['名称']}** [{r['行业']}] | 得分: {r['总分']} | {r['级别']} | "
            f"现价{r['收盘价']} 距MA20 {r['距MA20%']}%"
            for r in top
        ])
        push_content += f"\n\n*共 {len(results)} 个共振信号, 详见 output 报告。*"
        if send_serverchan(f"MTF共振选股 - {len(results)}个信号", push_content):
            logger.info("📲 Server酱推送已发送")


if __name__ == "__main__":
    main()
