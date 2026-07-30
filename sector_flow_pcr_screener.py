# -*- coding: utf-8 -*-
"""
sector_flow_pcr_screener.py —— 板块资金流 + 个股现金流 + 大盘期权PCR情绪 多层选股 · 矩阵规格
三层共振:
  ① 市场层: 沪深300波动率做 regime 自适应权重(高波动重板块/低波动重个股) +
     【大盘期权PCR情绪】(上交所ETF期权 总认沽/总认购成交量比, 判断市场乐观/避险, 全局加减分)
  ② 板块层: 板块资金流(净流入) + 板块趋势/筹码 -> 板块现金流分 -> 选 Top N 板块
  ③ 个股层: Top板块成分股的 趋势(均线)+资金(量比/涨幅)+筹码(集中度/斜率) -> 个股现金流分
  综合 = regime权重×(板块归一分+个股归一分) + PCR情绪偏置 -> 最终评分排序。

【期权逻辑已修正-关键】原思路给【个股】算PCR是硬伤: A股个股无场内期权, option_daily_stats_sse
  只有ETF期权标的, 个股代码匹配不到→PCR恒为0→期权维度形同虚设。本版改为【大盘情绪因子】:
  取ETF期权市场总PCR(认沽/认购), 映射为乐观/中性/避险情绪, 给整体选股做加减分+仓位建议。
  这才是A股期权数据真正有意义的用法。拿不到期权数据(非交易日/限流)给中性, 不阻断选股。
【本版规格】双源baostock+东财+硬超时; 多进程(子进程独立登录baostock, 命门已修); append兼容补丁;
  风口共振🎯; 推送全发分页(严格检查返回); 存output/+收尾防护+sys.exit(0); 不拦交易日。
  指标全手写(去scipy/matplotlib依赖)。
⚠️ 概率性共振选股, 非预测; PCR情绪是市场参考非保证; 结合仓位/止损, 不构成投资建议。
"""
import os
import re
import sys
import json
import time
import random
import warnings
import traceback
import requests
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta

import pandas as pd

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错(防 get_data 在 pandas2.x 崩)
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==================== 参数配置 (env 可调) ====================
PARAMS = dict(
    TOP_SECTOR_N=5,            # 选前N个板块
    STOCK_PER_SECTOR=0,        # 每板块取成分股数(0=全取)
    SCAN_LIMIT=2000,           # 成分股总数上限(防板块太大扫不完)
    NUM_PROCESSES=3,
    # regime 波动率区间(沪深300 20日年化)
    VOL_LOW=0.15, VOL_HIGH=0.30,
    # PCR 情绪阈值(市场总认沽/认购成交量比)
    PCR_VERY_BULL=0.7, PCR_BULL=0.9, PCR_BEAR=1.1, PCR_VERY_BEAR=1.3,
    EMOTION_WEIGHT=8.0,        # PCR情绪偏置满分(±8分)
    # 数据
    LOOKBACK_DAYS=400, MIN_DATA_LEN=60,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', str(PARAMS['SCAN_LIMIT'])))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
FAIL_STATS = {"抓取失败": 0, "数据不足": 0}


# ------------------ 推送 (全发分页 + 严格检查返回) ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        if isinstance(ret, dict):
            ok = ret.get('code', ret.get('errno', -1)) == 0
        elif isinstance(ret, bool):
            ok = ret
        else:
            ok = ret not in (None, False)
        if ok:
            return True
        print(f"  sdk返回非成功({ret}), 回退requests")
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=15)
        j = r.json()
        if j.get('code') != 0:
            print(f"  requests返回非0: {j} (多为额度/限流/key问题)")
        return j.get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    lines = content.split("\n")
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur:
        chunks.append("\n".join(cur))
    if not chunks:
        chunks = [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        print(f"  推送第{i+1}/{len(chunks)}条 ({len(ch)}字符)")
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    if len(chunks) > 1:
        print(f"📲 共发送{len(chunks)}条(全发分页) {'✅全部成功' if ok else '⚠️存在失败(查额度/限流)'}")
    else:
        print("📲 推送成功 ✅" if ok else "⚠️ 推送返回失败(查Server酱额度/限流/微信端)")
    return ok


# ------------------ baostock 登录(命门已修) / 超时 ------------------
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    if _BS_LOGGED:
        return True
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True; return True
            print(f"  baostock 登录失败({getattr(lg,'error_msg','')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _bs_logout():
    global _BS_LOGGED
    try:
        if _BS_LOGGED:
            bs.logout()
    except Exception as e:
        print(f"  baostock 登出异常: {e}")
    finally:
        _BS_LOGGED = False


def _init_worker():
    global _BS_LOGGED
    time.sleep(random.uniform(0, 2))
    _BS_LOGGED = False
    _bs_login_ok()


def _bs_q(code, fields, sd, ed, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed,
                                            frequency="d", adjustflag="2").get_data()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c


def _clean(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")


# ------------------ 数学工具 (手写, 去 scipy) ------------------
def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def robust_z(s):
    s = pd.to_numeric(s, errors='coerce')
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    s2 = s.fillna(s.median())
    std = s2.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s2 - s2.mean()) / std).fillna(0.0)


def minmax_norm(series):
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s - lo) / (hi - lo) * 100).clip(0, 100)


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ------------------ 市场层: 波动率 regime + 大盘期权PCR情绪 ------------------
def get_market_volatility():
    """沪深300(sh.000300) 20日年化波动率; 拿不到返回None(用默认中性权重)"""
    sd = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    if _bs_login_ok():
        try:
            d = _bs_q("sh.000300", "date,close", sd, ed, timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d = d.dropna(subset=['close']).sort_values('date')
                vol = float(d['close'].pct_change().rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252))
                return vol if not pd.isna(vol) else None
        except Exception as e:
            print(f"  沪深300波动率获取失败: {e}")
    return None


def regime_weights(mkt_vol):
    """高波动重板块(板块趋势更稳), 低波动重个股(个股alpha)"""
    if mkt_vol is None or pd.isna(mkt_vol):
        mkt_vol = 0.20
    v = clip01((mkt_vol - PARAMS['VOL_LOW']) / (PARAMS['VOL_HIGH'] - PARAMS['VOL_LOW']))
    w_sector = 0.50 + 0.20 * v
    w_stock = 0.50 - 0.20 * v
    return {"sector": float(w_sector), "stock": float(w_stock),
            "regime_v": float(v), "mkt_vol": float(mkt_vol)}


def get_pcr_emotion():
    """【修正后的期权用法】取上交所ETF期权市场总PCR(总认沽/总认购成交量), 映射为大盘情绪分+标签。
    PCR是市场情绪指标, 用在大盘层面(非个股, A股个股无期权)。拿不到给中性, 不阻断选股。
    返回 (emotion分[-1,1], 标签, PCR值 or None)"""
    df = None
    for back in range(0, 7):
        d = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
        df = _call_with_timeout(ak.option_daily_stats_sse, date=d, timeout=AK_TIMEOUT)
        if df is not None and not df.empty:
            break
    if df is None or df.empty:
        print("  期权PCR: 数据缺失(非交易日/限流), 情绪按中性处理")
        return 0.0, "⚪数据缺失", None
    call_col = next((c for c in df.columns if '认购' in str(c) and ('成交' in str(c) or '量' in str(c))), None)
    put_col = next((c for c in df.columns if '认沽' in str(c) and ('成交' in str(c) or '量' in str(c))), None)
    if not call_col or not put_col:
        print(f"  期权PCR: 列名缺失(列={list(df.columns)[:8]}), 情绪按中性处理")
        return 0.0, "⚪列名缺失", None
    call_total = pd.to_numeric(df[call_col].astype(str).str.replace(',', '', regex=False), errors='coerce').sum()
    put_total = pd.to_numeric(df[put_col].astype(str).str.replace(',', '', regex=False), errors='coerce').sum()
    if not call_total or call_total <= 0:
        return 0.0, "⚪数据无效", None
    pcr = put_total / call_total
    if pcr < PARAMS['PCR_VERY_BULL']:
        emo, label = 1.0, "🟢极度乐观"
    elif pcr < PARAMS['PCR_BULL']:
        emo, label = 0.5, "🟢偏乐观"
    elif pcr <= PARAMS['PCR_BEAR']:
        emo, label = 0.0, "⚪中性"
    elif pcr <= PARAMS['PCR_VERY_BEAR']:
        emo, label = -0.5, "🔴偏避险"
    else:
        emo, label = -1.0, "🔴极度避险"
    print(f"  期权PCR: 市场总PCR={pcr:.3f} -> {label} (情绪分{emo:+.1f})")
    return round(emo, 2), label, round(pcr, 3)


# ------------------ 板块层: 资金流 + 趋势/筹码 ------------------
def get_sector_flow():
    attempts = [
        lambda: ak.stock_board_industry_fund_flow_rank(indicator="今日", sector_type="行业资金流"),
        lambda: ak.stock_board_industry_fund_flow_rank(indicator="今日"),
        lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业"),
    ]
    for fn in attempts:
        df = _call_with_timeout(fn, label="sector_flow")
        if df is not None and not df.empty:
            return df.copy()
    return pd.DataFrame()


def standardize_sector_flow(df):
    if df.empty:
        return df
    rename_map = {"板块名称": "sector", "名称": "sector", "板块": "sector", "行业": "sector",
                  "净流入": "net_inflow", "净流入/万": "net_inflow", "净额": "net_inflow",
                  "今日涨幅": "pct_change", "涨跌幅": "pct_change",
                  "领涨股-涨跌幅": "lead_stock_pct", "序号": "rank"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    for c in ["net_inflow", "pct_change", "lead_stock_pct", "rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df["sector"] = df["sector"].astype(str)
    fund = 0.0
    if "net_inflow" in df.columns:
        fund = fund + 0.6 * robust_z(df["net_inflow"])
    if "pct_change" in df.columns:
        fund = fund + 0.2 * robust_z(df["pct_change"])
    if "lead_stock_pct" in df.columns:
        fund = fund + 0.2 * robust_z(df["lead_stock_pct"])
    df["fund_score"] = fund
    return df


def _get_sector_hist_raw(sector):
    df = ak.stock_board_industry_hist_em(symbol=sector, adjust="")
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def get_sector_hist(sector):
    r = _call_with_timeout(_get_sector_hist_raw, sector, label=f"sector_hist:{sector}")
    return r if r is not None else pd.DataFrame()


def sector_features(hist):
    if hist.empty or len(hist) < 30 or 'volume' not in hist.columns:
        return {"trend_score": 0.0, "near_high_pct": np.nan, "vol_ratio": np.nan, "chip_score": 0.0}
    x = hist.copy()
    x["ma20"] = x["close"].rolling(20, min_periods=5).mean()
    x["ma60"] = x["close"].rolling(60, min_periods=10).mean()
    x["vol_ma20"] = x["volume"].rolling(20, min_periods=5).mean()
    x["vol_ratio"] = x["volume"] / x["vol_ma20"].replace(0, np.nan)
    x["high_250"] = x["close"].rolling(250, min_periods=20).max()
    x["near_high_pct"] = (x["high_250"] - x["close"]) / x["high_250"].replace(0, np.nan)
    cw = (x["close"].rolling(20, min_periods=5).max() - x["close"].rolling(20, min_periods=5).min()) / x["close"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    x["chip_concentration"] = (1 - cw).clip(lower=0, upper=1)
    x["chip_slope"] = x["close"].rolling(20, min_periods=5).mean().diff(10) / x["close"].rolling(20, min_periods=5).mean().shift(10)
    last = x.iloc[-1]
    trend = 0.0
    if pd.notna(last["close"]) and pd.notna(last["ma20"]) and last["close"] >= last["ma20"]:
        trend += 1
    if pd.notna(last["ma20"]) and pd.notna(last["ma60"]) and last["ma20"] >= last["ma60"]:
        trend += 1
    chip = 0.0
    if pd.notna(last["chip_concentration"]) and last["chip_concentration"] >= 0.65:
        chip += 2
    if pd.notna(cw.iloc[-1]) and cw.iloc[-1] <= 0.15:
        chip += 2
    if pd.notna(last["chip_slope"]):
        if last["chip_slope"] > 0.03:
            chip += 2
        elif last["chip_slope"] > 0.01:
            chip += 1
        elif last["chip_slope"] < -0.01:
            chip -= 1
    return {"trend_score": float(trend),
            "near_high_pct": float(last["near_high_pct"]) if pd.notna(last["near_high_pct"]) else np.nan,
            "vol_ratio": float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else np.nan,
            "chip_score": float(chip)}


def sector_cash_score(row):
    pos = 0.0
    if pd.notna(row.get("near_high_pct")):
        if row["near_high_pct"] >= 0.25:
            pos += 2
        elif row["near_high_pct"] >= 0.10:
            pos += 1
    if pd.notna(row.get("vol_ratio")) and row["vol_ratio"] >= 1.2:
        pos += 1
    return 0.45 * row.get("fund_score", 0.0) + 0.20 * row.get("trend_score", 0.0) + 0.20 * row.get("chip_score", 0.0) + 0.15 * pos


def sector_worker(chunk):
    rows = []
    for sector in chunk:
        feat = sector_features(get_sector_hist(sector))
        rows.append({"sector": sector, **feat})
    return pd.DataFrame(rows)


def _get_sector_members_raw(sector):
    for fn in [ak.stock_board_industry_cons_em, ak.stock_board_concept_cons_em]:
        try:
            df = fn(symbol=sector)
            if df is not None and not df.empty:
                return df.copy()
        except Exception as e:
            print(f"  板块成分 {sector} via {fn.__name__} 失败: {e}")
    return pd.DataFrame()


def get_sector_members(sector):
    r = _call_with_timeout(_get_sector_members_raw, sector, label=f"sector_members:{sector}")
    return r if r is not None else pd.DataFrame()


def normalize_members(df):
    if df.empty:
        return []
    code_col = next((c for c in ["代码", "code", "证券代码", "股票代码"] if c in df.columns), None)
    name_col = next((c for c in ["名称", "name", "证券简称", "股票名称"] if c in df.columns), None)
    if code_col is None:
        return []
    out = []
    for _, r in df.iterrows():
        code = str(r[code_col]).zfill(6)
        if not code.startswith(PARAMS['KEEP_PREFIX']):
            continue
        name = str(r[name_col]) if name_col else ""
        if any(t in name for t in PARAMS['EXCLUDE_NAME']):
            continue
        out.append((code, name))
    return out


# ------------------ 个股层: 现金流(趋势+资金+筹码), 双源日线 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['LOOKBACK_DAYS'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['MIN_DATA_LEN']:
                    return d
        except Exception:
            pass
    return _fetch_hist_em(code, sy, ey)


def stock_cash_score(df):
    if df is None or df.empty or len(df) < PARAMS['MIN_DATA_LEN']:
        return None
    df = df.copy()
    df['ma20'] = df['close'].rolling(20, min_periods=5).mean()
    df['ma60'] = df['close'].rolling(60, min_periods=10).mean()
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
    df['high_120'] = df['close'].rolling(120, min_periods=20).max()
    df['near_high_pct'] = (df['high_120'] - df['close']) / df['high_120'].replace(0, np.nan)
    last = df.iloc[-1]
    if float(last['close']) < PARAMS['MIN_PRICE']:
        return None
    trend = 0.0
    if pd.notna(last['close']) and pd.notna(last['ma20']) and last['close'] >= last['ma20']:
        trend += 1
    if pd.notna(last['ma20']) and pd.notna(last['ma60']) and last['ma20'] >= last['ma60']:
        trend += 1
    fund = 0.0
    if pd.notna(last['vol_ratio']) and last['vol_ratio'] >= 1.2:
        fund += 1
    if df['close'].tail(20).pct_change().sum() > 0:
        fund += 1
    width = (df['close'].rolling(20, min_periods=5).max() - df['close'].rolling(20, min_periods=5).min()) / df['close'].rolling(20, min_periods=5).mean().replace(0, np.nan)
    conc = (1 - width).clip(lower=0, upper=1)
    slope = df['close'].rolling(20, min_periods=5).mean().diff(10) / df['close'].rolling(20, min_periods=5).mean().shift(10)
    chip = 0.0
    if pd.notna(conc.iloc[-1]) and conc.iloc[-1] >= 0.65:
        chip += 2
    if pd.notna(slope.iloc[-1]) and slope.iloc[-1] > 0.01:
        chip += 2
    return {"trend": float(trend), "fund": float(fund), "chip": float(chip),
            "near_high_pct": round(float(last['near_high_pct']) * 100, 1) if pd.notna(last['near_high_pct']) else None,
            "vol_ratio": round(float(last['vol_ratio']), 2) if pd.notna(last['vol_ratio']) else None,
            "close": round(float(last['close']), 2)}


def _process_one(args):
    sector, code, name, sector_norm = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        cash = stock_cash_score(df)
        if cash is None:
            return {"__fail__": "数据不足"}
        time.sleep(PARAMS['SLEEP'])
        cash_score = 0.4 * cash['trend'] + 0.3 * cash['fund'] + 0.3 * cash['chip']
        return {"sector": sector, "代码": code, "名称": name, "sector_norm": sector_norm,
                "stock_cash_score": round(cash_score, 3),
                "趋势分": cash['trend'], "资金分": cash['fund'], "筹码分": cash['chip'],
                "近高%": cash['near_high_pct'], "量比": cash['vol_ratio'], "最新价": cash['close']}
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}


# ------------------ 主扫描 (市场层→板块层→个股层→综合) ------------------
def run_scan():
    global FAIL_STATS
    FAIL_STATS = {k: 0 for k in FAIL_STATS}

    # ---- 市场层: 主进程登录baostock拿指数波动率 ----
    print("连接 Baostock（主进程: 指数波动率）...")
    _bs_login_ok()
    mkt_vol = get_market_volatility()
    weights = regime_weights(mkt_vol)
    print(f"  沪深300波动率={weights['mkt_vol']:.1%} | regime权重 板块{weights['sector']:.0%}/个股{weights['stock']:.0%}")
    # 大盘期权PCR情绪
    emotion, emotion_label, pcr = get_pcr_emotion()
    _bs_logout()

    # ---- 板块层: 资金流 + 趋势/筹码 ----
    print("取板块资金流...")
    flow = get_sector_flow()
    if flow.empty:
        print("⚠️ 板块资金流获取失败(限流?), 本次跳过"); return None, None, None, None
    flow = standardize_sector_flow(flow)
    sectors = flow["sector"].dropna().astype(str).tolist()
    print(f"板块特征计算 {len(sectors)} 个板块（{PARAMS['NUM_PROCESSES']}进程）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES']) as pool:
        out = list(tqdm(pool.imap_unordered(sector_worker, list(chunk_list(sectors, 4))),
                        total=max(1, (len(sectors) + 3) // 4), desc="板块特征"))
    sec_feat = pd.concat([x for x in out if x is not None and not x.empty], ignore_index=True) if out else pd.DataFrame()
    sector_rank_df = flow.merge(sec_feat, on="sector", how="left")
    sector_rank_df["sector_cash_score"] = sector_rank_df.apply(lambda r: sector_cash_score(r.to_dict()), axis=1)
    sector_rank_df = sector_rank_df.sort_values("sector_cash_score", ascending=False).reset_index(drop=True)
    sector_rank_df["sector_norm"] = minmax_norm(sector_rank_df["sector_cash_score"])
    top_sectors = sector_rank_df.head(PARAMS['TOP_SECTOR_N'])["sector"].tolist()
    print(f"  Top{PARAMS['TOP_SECTOR_N']}板块: {top_sectors}")

    # ---- 个股层: Top板块成分股 ----
    print("取Top板块成分股...")
    sector_members_map = {}
    for s in top_sectors:
        members = normalize_members(get_sector_members(s))
        if PARAMS['STOCK_PER_SECTOR'] and len(members) > PARAMS['STOCK_PER_SECTOR']:
            members = members[:PARAMS['STOCK_PER_SECTOR']]
        sector_members_map[s] = members
        print(f"  {s}: {len(members)} 只成分股")

    tasks = []
    for s in top_sectors:
        sec_norm = float(sector_rank_df.loc[sector_rank_df["sector"] == s, "sector_norm"].iloc[0])
        for code, name in sector_members_map[s]:
            tasks.append((s, code, name, sec_norm))
    if SCAN_LIMIT and len(tasks) > SCAN_LIMIT:
        print(f"  成分股 {len(tasks)} 只 > 上限{SCAN_LIMIT}, 截断")
        tasks = tasks[:SCAN_LIMIT]

    results = []; fail_count = 0
    print(f"个股现金流扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="个股扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print(f"个股扫描完成 有效{len(results)} 失败{fail_count}")
    stock_df = pd.DataFrame(results)

    # ---- 综合评分: regime权重×(板块归一+个股归一) + PCR情绪偏置 ----
    if not stock_df.empty:
        stock_df["stock_norm"] = minmax_norm(stock_df["stock_cash_score"])
        stock_df["base_score"] = weights['sector'] * stock_df["sector_norm"] + weights['stock'] * stock_df["stock_norm"]
        emotion_bias = PARAMS['EMOTION_WEIGHT'] * emotion
        stock_df["情绪偏置"] = round(emotion_bias, 1)
        stock_df["final_score"] = (stock_df["base_score"] + emotion_bias).clip(0, 100).round(1)
        stock_df = stock_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    meta = {"weights": weights, "emotion": emotion, "emotion_label": emotion_label, "pcr": pcr,
            "emotion_bias": round(PARAMS['EMOTION_WEIGHT'] * emotion, 1)}
    return stock_df, sector_rank_df, top_sectors, meta


# ------------------ 风口共振 (Top板块是否在当日热点) ------------------
def mark_hot_sectors(sector_rank_df, top_sectors):
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty:
                break
        except Exception as e:
            print(f"  行业热度榜第{i+1}次失败: {e}")
        time.sleep(2 + i)
    hot = []
    if not heat.empty and '板块名称' in heat.columns and '涨跌幅' in heat.columns:
        h = heat.copy(); h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        hot = [(str(r['板块名称']), round(float(r['_chg']), 2)) for _, r in h.head(HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    hot_top = []
    for s in top_sectors:
        m = next((hh for hh in hot_names if hh and (hh == s or hh in s or s in hh)), "")
        if m:
            hot_top.append(s)
    print(f"🎯 Top板块命中风口: {hot_top or '无'}")
    return hot, hot_top


# ------------------ 推送 ------------------
def build_push(stock_df, sector_rank_df, top_sectors, meta, hot, hot_top):
    w = meta['weights']
    pos_advice = "正常偏进攻" if meta['emotion'] > 0 else ("均衡" if meta['emotion'] == 0 else "降仓防守")
    L = [f"**📊 板块资金流+个股现金流+大盘PCR情绪 多层选股** | 命中{len(stock_df)}只",
         f"*(市场情绪={meta['emotion_label']} PCR={meta['pcr']} | 沪深300波动率={w['mkt_vol']:.1%} | "
         f"regime权重 板块{w['sector']:.0%}/个股{w['stock']:.0%} | 情绪偏置{meta['emotion_bias']:+.1f}分)*",
         f"*(仓位建议: {pos_advice}; 概率性共振选股, 非预测; 结合止损)*", ""]
    if hot:
        L.append("🌪️ **当日风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    # Top板块
    L.append(f"### 🏆 Top{len(top_sectors)}板块 (现金流分)")
    top_df = sector_rank_df[sector_rank_df['sector'].isin(top_sectors)].head(len(top_sectors))
    for _, r in top_df.iterrows():
        flag = "🎯" if r['sector'] in hot_top else ""
        L.append(f"- {flag}**{r['sector']}** 现金流{r['sector_cash_score']:.2f} 趋势{r.get('trend_score',0):.0f} 筹码{r.get('chip_score',0):.0f}")
    L.append("")
    # Top个股
    L.append(f"### 📈 综合评分 Top{min(len(stock_df), PUSH_TOP)}")
    for _, r in stock_df.head(PUSH_TOP).iterrows():
        L.append(f"- **{r['名称']}({r['代码']})** [{r['sector']}] 综合{r['final_score']} | 现价{r['最新价']} "
                 f"趋{r['趋势分']}资{r['资金分']}筹{r['筹码分']} 量比{r['量比']} 近高{r['近高%']}%")
    if len(stock_df) > PUSH_TOP:
        L.append(f"\n*…另有{len(stock_df)-PUSH_TOP}只, 见output*")
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"📊 板块资金流+个股现金流+大盘PCR情绪 多层选股 | {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Top板块={PARAMS['TOP_SECTOR_N']}; 成分股上限={SCAN_LIMIT}; 进程={PARAMS['NUM_PROCESSES']}; 不拦交易日; 推送全列+分页")
    print("⚠️ 期权PCR用在大盘情绪(非个股, A股个股无期权); 概率性共振选股, 非预测")
    print("=" * 70)
    stock_df, sector_rank_df, top_sectors, meta = run_scan()
    if stock_df is None or stock_df.empty:
        print("\n本次未选出股票(板块资金流限流, 或Top板块成分股无有效数据)"); sys.exit(0)
    hot, hot_top = mark_hot_sectors(sector_rank_df, top_sectors)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        stock_df.to_csv(os.path.join(OUTPUT_DIR, f"sector_flow_pcr_{tag}.csv"), index=False, encoding="utf-8-sig")
        sector_rank_df.head(PARAMS['TOP_SECTOR_N'] * 3).to_csv(
            os.path.join(OUTPUT_DIR, f"sector_rank_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"sector_flow_pcr_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "meta": meta, "top_sectors": top_sectors, "hot": hot, "hot_top": hot_top,
                       "n": int(len(stock_df)), "fail_stats": FAIL_STATS,
                       "hits": stock_df.head(PUSH_TOP * 3).to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/sector_flow_pcr_{tag}.* 与 sector_rank_{tag}.csv")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = stock_df.copy()
        disp = disp.drop(columns=['sector_norm', 'stock_cash_score', 'stock_norm', 'base_score'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            send_serverchan(f"📊 板块+PCR情绪选股 命中{len(stock_df)}只 情绪{meta['emotion_label']}",
                            build_push(stock_df, sector_rank_df, top_sectors, meta, hot, hot_top))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_sector_flow_pcr<<<
