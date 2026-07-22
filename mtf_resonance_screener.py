import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
import baostock as bs
from serverchan_sdk import sc_send
import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Tuple, Dict

"""
mtf_resonance_screener.py

MTF 多时间框架共振评分模型（改造自本地版本，接入仓库自动化流程）。
评分逻辑本身（位置/上穿回踩/短期动量/中期趋势/成交量/长线趋势/RSI/MACD共振，满分100）
未做改动，质量本来就不错。改造的是"适配仓库"这部分：

⚠️ 相比原始版本的修复/调整：
1. 默认只硬编码了10只股票，不是全市场扫描——改成从baostock批量拉全市场股票+行业列表。
2. get_stock_name/get_industry_data 原来各自单独调一次akshare接口（每只股票2次网络请求），
   改成主进程一次性从baostock批量拉取股票名称和行业分类，子进程不用重复请求。
3. 数据源从akshare换成baostock，全市场扫描用仓库已验证稳定的多进程+超时保护模式。
4. Server酱推送部分原代码本来就写对了（用的是正确的sctapi.ftqq.com），
   这里改用仓库统一的serverchan_sdk库，行为一致，默认启用。
"""

# ------------------ 评分参数 ------------------
SCORE_THRESHOLD = 65      # 达标阈值
STRONG_THRESHOLD = 78     # 强信号阈值
MIN_DATA_DAYS = 250       # 最少需要的历史数据天数（覆盖MA250）
MIN_PRICE = 5

LOOKBACK_DAYS = 900        # 日历天回溯窗口，覆盖MIN_DATA_DAYS+缓冲
NUM_PROCESSES = 3
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = 20


# ==================== 技术指标计算（与原版一致）====================
def calculate_ma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    macd = (dif - dea) * 2
    return pd.DataFrame({'dif': dif, 'dea': dea, 'macd': macd})


# ==================== 共振评分引擎（逻辑与原版一致，未改动核心算法）====================
class MTFResonanceEngine:
    def __init__(self, df):
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
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252)
        self.df = df

    def _safe_get(self, row_idx, col, default=0):
        if row_idx < 0:
            row_idx = len(self.df) + row_idx
        if row_idx < 0 or row_idx >= len(self.df):
            return default
        val = self.df.iloc[row_idx].get(col, default)
        return val if pd.notna(val) else default

    def calculate_score(self) -> Tuple[float, str, Dict]:
        if len(self.df) < MIN_DATA_DAYS:
            return 0, "数据不足", {}

        df = self.df
        latest_idx, prev_idx, prev3_idx = -1, -2, -3

        close = self._safe_get(latest_idx, 'close')
        ma20 = self._safe_get(latest_idx, 'ma20')
        ma20_prev = self._safe_get(prev_idx, 'ma20')
        ma20_prev3 = self._safe_get(prev3_idx, 'ma20')

        score = 0.0
        details = {}

        # 1. 位置接近MA20 (15分)
        dist_pct = (close - ma20) / ma20 if ma20 != 0 else 999
        position_score = max(0, 1 - abs(dist_pct) / 0.085) * 15
        score += position_score
        details['位置接近MA20'] = round(position_score, 1)

        # 2. 上穿/回踩信号 (20分)
        prev_close = self._safe_get(prev_idx, 'close')
        today_cross = (prev_close <= ma20_prev) and (close > ma20)

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

        # 7. RSI健康度 (5分)
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

        # 8. MACD共振 (5分)
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
        if final_score >= STRONG_THRESHOLD:
            level = '强信号'
        elif final_score >= SCORE_THRESHOLD:
            level = '中信号'
        elif final_score >= 50:
            level = '弱信号'
        else:
            level = '无信号'

        return final_score, level, details

    def get_summary(self):
        latest = self.df.iloc[-1]
        return {
            'close': round(latest['close'], 2),
            'ma20': round(latest.get('ma20', latest['close']), 2),
            'rsi': round(latest.get('rsi', 50), 1),
            'macd': round(latest.get('macd', 0), 3),
        }


# ==================== baostock 数据获取 ====================
def _init_worker():
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
    import random
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求可能持续报错")


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


def _process_one(args):
    """单只股票的抓取+评分逻辑，运行在子进程里"""
    code, name, industry = args
    try:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        df = _query_with_timeout(code, "date,open,high,low,close,volume", start_date)
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().sort_values('date').reset_index(drop=True)

        if df.empty or df['close'].iloc[-1] < MIN_PRICE:
            return None

        engine = MTFResonanceEngine(df)
        score, level, details = engine.calculate_score()
        if score < SCORE_THRESHOLD:
            return None

        summary = engine.get_summary()
        dist_pct = (summary['close'] - summary['ma20']) / summary['ma20'] * 100 if summary['ma20'] != 0 else 0

        return {
            "代码": code, "名称": name, "行业": industry,
            "总分": score, "级别": level,
            "收盘价": summary['close'], "距MA20%": round(dist_pct, 2),
            "RSI": summary['rsi'], "MACD": summary['macd'],
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_scan(limit=None):
    print("正在连接 Baostock（主进程，用于取股票列表+行业分类）...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1') &
        (~stock_df['code_name'].str.contains('ST|退', na=False))
    ]

    rs2 = bs.query_stock_industry()
    industry_df = rs2.get_data()
    code_to_industry = dict(zip(industry_df['code'], industry_df['industry']))
    bs.logout()

    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(code, code_to_name.get(code, ""), code_to_industry.get(code, "未知")) for code in target_stocks]

    results = []
    fail_count = 0
    print(f"开始MTF共振评分扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（{res['总分']}分/{res['级别']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("总分", ascending=False).reset_index(drop=True)
    return result_df


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- {row['名称']}（{row['代码']}）[{row['行业']}] {row['总分']}分/{row['级别']} "
            f"| 收盘{row['收盘价']} | 距MA20 {row['距MA20%']}% | RSI {row['RSI']}"
        )
    return "\n".join(lines)


def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}，message={response.get('message')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


if __name__ == "__main__":
    df = run_scan(limit=None)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            strong_count = (df["级别"] == "强信号").sum()
            title = f"MTF共振选股 命中{len(df)}只（强信号{strong_count}只）"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
