# -*- coding: utf-8 -*-
"""
vcp_screener_v2.py —— VCP(波动收缩形态) 全市场选股 · 向量化融合升级版

【融合精华】
1. 向量化全市场计算（fabtrader.in 风格）—— 比逐只循环快 10~50 倍
2. 加权多周期 RS Rating + 全市场百分位排名 —— 替代简单 120 日涨幅
3. Keltner Channel 宽度 + ATR + 价格范围 + 量缩 多维度 VCP 评分 —— 替代 argrelextrema 极值点
4. Pivot Breakout + Pocket Pivot + RVOL 三重突破确认 —— 新增
5. Isolation Forest 异常过滤 —— 排除突发消息/庄股干扰
6. 保留原版的：Stage2 预筛、本地 Parquet 缓存、双数据源、快照预筛、行业聚类、风口共振、ServerChan 推送

依赖: pandas numpy scipy akshare baostock tqdm pyarrow scikit-learn
"""
import os
import re
import sys
import json
import time
import random
import traceback
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import akshare as ak
import baostock as bs
from tqdm import tqdm

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# 兼容旧pandas
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# ====================== 参数区 ======================
PARAMS = dict(
    LOOKBACK_DAYS=500,
    MIN_REQUIRED=252,          # 向量化计算需要至少 252 日（1年）
    VCP_LOOKBACK=120,
    NUM_PROCESSES=4,           # 数据拉取进程数
    SLEEP=0.15,
    # Stage2 / Trend Template
    STAGE2_MIN_ABOVE_52W_LOW=0.30,
    STAGE2_MAX_DIST_52W_HIGH=0.30,
    # VCP 评分阈值
    VCP_SCORE_MIN=5,
    # 过滤阈值
    RS_RATING_MIN=70,
    TREND_SCORE_MIN=7,
    # 突破确认
    RVOL_THRESHOLD=1.5,
    BREAKOUT_TOLERANCE=0.995,  # pivot 突破允许 0.5% 误差
    # Keltner / ATR
    KC_TIGHTEN_PCT=0.85,       # KC 宽度收缩 15% 以上
    ATR_TIGHTEN_PCT=0.80,
    VOL_DRY_RATIO=0.65,
    PRICE_RANGE_TIGHTEN=0.60,
    # 异常过滤
    ANOMALY_CONTAMINATION=0.08,
    # 前置过滤
    KEEP_PREFIX=("0", "3", "6"),
    EXCLUDE_NAME=("ST", "退"),
    MIN_PRICE=3.0,
    PRE_AMOUNT_MIN=5.0e7,
    PRE_TURNOVER_MIN=0.3,
)

SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '15'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
CACHE_STALE_DAYS = int(os.environ.get('CACHE_STALE_DAYS', '3'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = Path(OUTPUT_DIR) / "price_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BS_LOGGED = False


# ====================== 推送相关 ======================
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content)
        return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            data={"title": title, "desp": content},
            timeout=15
        ).json().get("code") == 0
    except Exception as e:
        print(f"  requests推送失败: {e}")
        return False


def send_serverchan(title, content, sendkey=""):
    """全发: 超长自动按行切分多条发送"""
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    lines = content.split("\n")
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += lnlen
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
    print(f"📲 共发送{len(chunks)}条(全发分页)" if len(chunks) > 1 else "📲 推送成功")
    return ok


# ====================== Baostock 登录 ======================
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    if _BS_LOGGED:
        return True
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {i+1}/{retries}")
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


def _bs_q(code, fields, sd, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c


def _clean(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")


# ====================== 缓存相关 ======================
def get_cache_path(code: str) -> Path:
    safe = code.replace('.', '_').replace('/', '_')
    return CACHE_DIR / f"{safe}.parquet"


def load_cached_hist(code: str, min_days=PARAMS["MIN_REQUIRED"]) -> pd.DataFrame | None:
    path = get_cache_path(code)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) < min_days:
            return None
        last_date = pd.to_datetime(df['date']).max()
        if (datetime.now() - last_date).days > CACHE_STALE_DAYS:
            return None
        return df
    except Exception:
        return None


def save_hist_cache(code: str, df: pd.DataFrame):
    try:
        keep = ['date', 'open', 'high', 'low', 'close', 'volume']
        cols = [c for c in keep if c in df.columns]
        df[cols].to_parquet(get_cache_path(code), index=False)
    except Exception as e:
        print(f"  缓存写入失败 {code}: {e}")


# ====================== 快照预筛 ======================
def snapshot_prefilter(codes_with_prefix):
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫")
            return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (
            spot['代码'].str.startswith(PARAMS["KEEP_PREFIX"])
            & ~spot['名称'].astype(str).str.contains("|".join(PARAMS["EXCLUDE_NAME"]), na=False, regex=True)
            & (spot['最新价'] >= PARAMS["MIN_PRICE"])
        )
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS["PRE_AMOUNT_MIN"])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS["PRE_TURNOVER_MIN"])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (价≥{PARAMS['MIN_PRICE']}+活跃)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix


# ====================== 行业与风口 ======================
def fetch_industry(symbol):
    for attempt in range(2):
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            if info is not None and not info.empty and 'item' in info.columns:
                row = info[info['item'].isin(['行业', '所属行业'])]
                if not row.empty:
                    return row.iloc[0]['value']
        except Exception:
            time.sleep(1 + attempt)
    return "—"


def get_industry_heat():
    for i in range(3):
        try:
            d = ak.stock_board_industry_name_em()
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"  行业热度榜第{i+1}次失败: {e}")
        time.sleep(2 + i)
    return pd.DataFrame()


def get_hot_sectors(heat):
    if heat.empty or '板块名称' not in heat.columns or '涨跌幅' not in heat.columns:
        return []
    h = heat.copy()
    h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
    h = h[h['_chg'] >= HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
    return [(str(row['板块名称']), round(float(row['_chg']), 2))
            for _, row in h.head(HOT_SECTOR_TOP).iterrows()]


def match_sector(sector, hot_names):
    if not sector or sector in ('—', '未知', '') or not hot_names:
        return ""
    s = sector.strip()
    for h in hot_names:
        if h and h == s:
            return h
    for h in hot_names:
        if h and (h in s or s in h):
            return h
    return ""


# ====================== 单只数据获取（用于构建宽表）======================
def fetch_hist(code):
    """拉取单只股票历史数据，返回标准格式 DataFrame"""
    cached = load_cached_hist(code)
    if cached is not None:
        return cached

    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    sy = sd.replace('-', '')
    end_str = datetime.now().strftime("%Y%m%d")

    # 1. 优先 baostock
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    save_hist_cache(code, d)
                    return d
        except Exception:
            pass

    # 2. 兜底 akshare
    for attempt in range(2):
        try:
            d = _call_with_timeout(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=sy,
                end_date=end_str,
                adjust="qfq",
                timeout=AK_TIMEOUT
            )
            if d is not None and not d.empty:
                d = d.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high',
                    '最低': 'low', '收盘': 'close', '成交量': 'volume'
                })
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    save_hist_cache(code, d)
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))

    return None


def _fetch_one(args):
    """多进程 worker 用的包装"""
    code, name = args
    try:
        df = fetch_hist(code)
        if df is not None and len(df) >= PARAMS["MIN_REQUIRED"]:
            return (code, df)
    except FutureTimeoutError:
        pass
    except Exception as e:
        pass
    return None


# ====================== 向量化核心计算（融合精华）======================

def build_wide_frames(stock_data_dict: dict) -> tuple:
    """
    将 {code: df} 转换为宽表 DataFrame (index=date, columns=code)
    自动对齐日期，缺失值保持 NaN（后续指标计算会自动处理）
    """
    close, high, low, volume = {}, {}, {}, {}
    for code, df in stock_data_dict.items():
        if df is None or len(df) < PARAMS["MIN_REQUIRED"]:
            continue
        df = df.set_index('date').sort_index()
        close[code] = df['close']
        high[code]  = df['high']
        low[code]   = df['low']
        volume[code]= df['volume']

    if not close:
        return None, None, None, None

    C = pd.DataFrame(close)
    H = pd.DataFrame(high)
    L = pd.DataFrame(low)
    V = pd.DataFrame(volume)
    return C, H, L, V


def compute_rs_rating(close: pd.DataFrame) -> pd.Series:
    """
    加权多周期 RS + 全市场百分位排名 (0~100)
    权重: 40%×12月 + 20%×6月 + 20%×3月 + 20%×1月
    """
    r12 = close.pct_change(252)
    r6  = close.pct_change(126)
    r3  = close.pct_change(63)
    r1  = close.pct_change(21)
    rs = (0.4 * r12) + (0.2 * r6) + (0.2 * r3) + (0.2 * r1)
    latest_rs = rs.iloc[-1]
    rs_rating = latest_rs.rank(pct=True) * 100
    return rs_rating


def compute_trend_template(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """
    Minervini Trend Template 评分 (0~9 分)
    向量化计算全市场
    """
    sma50  = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high_52w = high.rolling(252).max()
    low_52w  = low.rolling(252).min()
    sma200_shift = sma200.shift(30)

    c = close.iloc[-1]
    s50 = sma50.iloc[-1]
    s150 = sma150.iloc[-1]
    s200 = sma200.iloc[-1]
    h52 = high_52w.iloc[-1]
    l52 = low_52w.iloc[-1]
    s200_prev = sma200_shift.iloc[-1]

    score = (
        (c > s50).astype(int) +
        (c > s150).astype(int) +
        (c > s200).astype(int) +
        (s50 > s150).astype(int) +
        (s50 > s200).astype(int) +
        (s150 > s200).astype(int) +
        (c >= l52 * (1 + PARAMS["STAGE2_MIN_ABOVE_52W_LOW"])).astype(int) +
        (c >= h52 * (1 - PARAMS["STAGE2_MAX_DIST_52W_HIGH"])).astype(int) +
        (s200 > s200_prev).astype(int)
    )
    return score


def compute_vcp_score(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                      volume: pd.DataFrame) -> pd.DataFrame:
    """
    多维度 VCP 评分 (0~10 分)
    - Keltner Channel 宽度收缩 (3分)
    - ATR 收缩 (2分)
    - 量缩 (2分)
    - 价格范围收缩 (2分)
    - 更高低点 (1分)
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).groupby(level=0, axis=1).max()

    atr14 = tr.rolling(14).mean()
    atr10 = tr.rolling(10).mean()

    # Keltner Channel 宽度 (归一化)
    ema20 = close.ewm(span=20).mean()
    kc_width = (4 * atr14) / ema20

    kc_w20 = kc_width.rolling(20).mean()
    kc_w40 = kc_width.rolling(40).mean()
    kc_contracting = kc_w20.iloc[-1] < kc_w40.iloc[-1] * PARAMS["KC_TIGHTEN_PCT"]

    # ATR 收缩
    atr_earlier = atr14.iloc[-70:-10].mean() if len(atr14) > 70 else atr14.mean()
    atr_contracting = atr10.iloc[-1] < atr_earlier * PARAMS["ATR_TIGHTEN_PCT"]

    # 量缩
    vol_avg5 = volume.rolling(5).mean()
    vol_avg50 = volume.rolling(50).mean()
    vol_dry = vol_avg5.iloc[-1] < vol_avg50.iloc[-1] * PARAMS["VOL_DRY_RATIO"]

    # 价格范围收缩
    range_20 = (high.rolling(20).max() - low.rolling(20).min()) / close
    range_60 = (high.rolling(60).max() - low.rolling(60).min()) / close
    price_tight = range_20.iloc[-1] < range_60.iloc[-1] * PARAMS["PRICE_RANGE_TIGHTEN"]

    # 更高低点
    low_20 = low.rolling(20).min()
    low_40 = low.rolling(40).min()
    higher_lows = low_20.iloc[-1] >= low_40.iloc[-21] * 0.98

    # 综合评分
    score = (
        kc_contracting.astype(int) * 3 +
        atr_contracting.astype(int) * 2 +
        vol_dry.astype(int) * 2 +
        price_tight.astype(int) * 2 +
        higher_lows.astype(int) * 1
    )

    return pd.DataFrame({
        'VCP_Score': score,
        'KC_Contracting': kc_contracting,
        'ATR_Contracting': atr_contracting,
        'Vol_Dry': vol_dry,
        'Price_Tight': price_tight,
        'Higher_Lows': higher_lows
    })


def compute_breakout_signals(close: pd.DataFrame, high: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    突破确认信号:
    - Pivot Breakout: 收盘价突破 60 日高点
    - Pocket Pivot: 当日成交量 > 过去10日最大下跌日成交量
    - RVOL: 相对成交量
    """
    base_high = high.rolling(60).max()
    pivot = base_high.iloc[-1]
    breakout = close.iloc[-1] > pivot * PARAMS["BREAKOUT_TOLERANCE"]

    down_days = close.diff() < 0
    down_vol = volume.where(down_days)
    max_down_vol = down_vol.rolling(10).max()
    pocket_pivot = volume.iloc[-1] > max_down_vol.iloc[-1]

    avg_vol50 = volume.rolling(50).mean()
    rvol = volume.iloc[-1] / avg_vol50.iloc[-1]

    valid_breakout = breakout & ((rvol > PARAMS["RVOL_THRESHOLD"]) | pocket_pivot)
    near_pivot = close.iloc[-1] >= pivot * 0.97

    return pd.DataFrame({
        'Pivot': pivot,
        'Breakout': breakout,
        'PocketPivot': pocket_pivot,
        'RVOL': rvol,
        'Valid_Breakout': valid_breakout,
        'Near_Pivot': near_pivot
    })


def filter_anomalies(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """Isolation Forest 异常过滤。True=正常, False=异常(应排除)"""
    if not SKLEARN_OK:
        returns = close.pct_change().iloc[-60:]
        vol_norm = volume / volume.rolling(50).mean()
        max_gap = returns.abs().max()
        vol_spike = vol_norm.iloc[-1]
        normal = (max_gap < 0.15) & (vol_spike < 5.0)
        return normal

    returns = close.pct_change().iloc[-60:]
    vol_norm = volume / volume.rolling(50).mean()

    features = pd.DataFrame({
        'volatility': returns.std(),
        'volume_spike': vol_norm.iloc[-1],
        'max_gap': returns.abs().max(),
        'skew': returns.skew()
    }).fillna(0)

    if len(features) < 10:
        return pd.Series(True, index=features.index)

    clf = IsolationForest(
        contamination=PARAMS["ANOMALY_CONTAMINATION"],
        random_state=42,
        n_estimators=100
    )
    pred = clf.fit_predict(features)
    return pd.Series(pred == 1, index=features.index)


def run_vectorized_scan(stock_data_dict: dict, name_map: dict) -> pd.DataFrame:
    """向量化主扫描"""
    print("构建宽表（向量化计算）...")
    C, H, L, V = build_wide_frames(stock_data_dict)
    if C is None or C.empty:
        print("宽表为空，无有效数据")
        return pd.DataFrame()

    print(f"  宽表: {C.shape[0]} 日 × {C.shape[1]} 只股票")

    print("异常过滤...")
    normal_mask = filter_anomalies(C, V)
    n_normal = normal_mask.sum()
    print(f"  正常股票: {n_normal}/{len(normal_mask)}")

    print("计算 RS Rating...")
    rs_rating = compute_rs_rating(C)

    print("计算 Trend Template...")
    trend_score = compute_trend_template(C, H, L)

    print("计算 VCP Score...")
    vcp_df = compute_vcp_score(C, H, L, V)

    print("计算 Breakout Signals...")
    bo_df = compute_breakout_signals(C, H, V)

    result = pd.DataFrame({
        '代码': C.columns,
        '名称': [name_map.get(c, '') for c in C.columns],
        '最新价': C.iloc[-1].values,
        'RS_Rating': rs_rating.reindex(C.columns).values,
        'Trend_Score': trend_score.reindex(C.columns).values,
        'VCP_Score': vcp_df['VCP_Score'].reindex(C.columns).values,
        'KC_Contracting': vcp_df['KC_Contracting'].reindex(C.columns).values,
        'ATR_Contracting': vcp_df['ATR_Contracting'].reindex(C.columns).values,
        'Vol_Dry': vcp_df['Vol_Dry'].reindex(C.columns).values,
        'Price_Tight': vcp_df['Price_Tight'].reindex(C.columns).values,
        'Higher_Lows': vcp_df['Higher_Lows'].reindex(C.columns).values,
        'Pivot': bo_df['Pivot'].reindex(C.columns).values,
        'Breakout': bo_df['Breakout'].reindex(C.columns).values,
        'PocketPivot': bo_df['PocketPivot'].reindex(C.columns).values,
        'RVOL': bo_df['RVOL'].reindex(C.columns).values,
        'Valid_Breakout': bo_df['Valid_Breakout'].reindex(C.columns).values,
        'Near_Pivot': bo_df['Near_Pivot'].reindex(C.columns).values,
        'Normal': normal_mask.reindex(C.columns).fillna(False).values
    })

    candidates = result[
        result['Normal'] &
        (result['RS_Rating'] >= PARAMS["RS_RATING_MIN"]) &
        (result['Trend_Score'] >= PARAMS["TREND_SCORE_MIN"]) &
        (result['VCP_Score'] >= PARAMS["VCP_SCORE_MIN"])
    ].copy()

    candidates = candidates.sort_values(
        ['Valid_Breakout', 'VCP_Score', 'RS_Rating'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"  命中候选: {len(candidates)} 只")
    return candidates


# ====================== 补充行业 + 风口 ======================
def enrich(df):
    if df.empty:
        return df, [], []

    targets = df.to_dict('records')
    print(f"为 {len(targets)} 只候选补行业 ...")

    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)

    with ThreadPoolExecutor(max_workers=PARAMS["NUM_PROCESSES"]) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [
        (n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()
    ] if labeled else []
    print(f"📐 VCP蓄势板块: {cluster or '无'}")

    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")

    cnt = 0
    for r in targets:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True
            r['resonance_sector'] = m
            cnt += 1
        else:
            r['resonance'] = False
            r['resonance_sector'] = ''

    print(f"🎯 VCP遇风口 {cnt} 只 (形态蓄势+板块催化)")

    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(
        ['resonance', 'Valid_Breakout', 'VCP_Score', 'RS_Rating'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    broke = df[df['Valid_Breakout'] == True] if 'Valid_Breakout' in df.columns else pd.DataFrame()

    L = [
        f"**📐 VCP波动收缩形态选股（向量化融合版）** | 命中{len(df)}只 🎯风口{len(reso)} 💥突破{len(broke)}",
        "*(加权RS + TrendTemplate + Keltner/ATR/Vol多维度VCP + PocketPivot/RVOL突破确认 + 异常过滤)*",
        ""
    ]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
        L.append("")
    if cluster:
        L.append("📐 **VCP蓄势板块**: " + "、".join(f"{n}({c})" for n, c in cluster))
        L.append("")
    if not broke.empty:
        L.append(f"### 💥 已突破(PocketPivot+放量) 共{len(broke)}只")
        for _, r in broke.head(PUSH_TOP).iterrows():
            L.append(
                f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] "
                f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 RS{r['RS_Rating']:.0f} "
                f"RVOL{r['RVOL']:.1f} pivot{r['Pivot']:.2f}"
            )
        L.append("")
    if not reso.empty:
        L.append(f"### 🎯 VCP遇风口 共{len(reso)}只")
        for _, r in reso.iterrows():
            L.append(
                f"- **{r['名称']}({r['代码']})** [🎯{r['resonance_sector']}] "
                f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 RS{r['RS_Rating']:.0f} "
                f"突破{'✓' if r['Valid_Breakout'] else '✗'}"
            )
        L.append("")
    L.append(f"### 📐 全部候选 共{len(df)}只")
    for _, r in df.iterrows():
        L.append(
            f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] "
            f"现价{r['最新价']:.2f} VCP{r['VCP_Score']}分 Trend{r['Trend_Score']} "
            f"RS{r['RS_Rating']:.0f} 突破{'✓' if r['Valid_Breakout'] else '✗'} "
            f"PP{'✓' if r['PocketPivot'] else '✗'} RVOL{r['RVOL']:.1f}"
        )
    return "\n".join(L)


# ====================== 主入口 ======================
def main():
    print("=" * 70)
    print(f"📐 VCP波动收缩形态选股（向量化融合版）| {datetime.now():%Y-%m-%d %H:%M}")
    print(f"回看{PARAMS['LOOKBACK_DAYS']}天 | 全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'} | 缓存加速 | 向量化计算")
    print("=" * 70)

    print("\n获取股票列表...")
    ind_map = {}
    stock_df = pd.DataFrame()

    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                ind_map = dict(zip(ind['code'], ind['industry'].fillna('')))
                print(f"  行业表 {len(ind_map)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception:
            stock_df = pd.DataFrame()
        _bs_logout()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        for attempt in range(3):
            try:
                d = ak.stock_info_a_code_name()
                if d is not None and not d.empty and 'code' in d.columns:
                    nc = 'name' if 'name' in d.columns else d.columns[1]
                    d = d[['code', nc]].copy()
                    d.columns = ['code', 'code_name']
                    d['code'] = d['code'].astype(str).str.zfill(6)
                    d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                    d['type'] = '1'
                    d['status'] = '1'
                    stock_df = d
                    break
            except Exception as e:
                print(f"  akshare列表第{attempt+1}次失败: {e}")
            time.sleep(2 + attempt)

    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表")
        sys.exit(1)

    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.'))
        & (stock_df['type'] == '1')
        & (stock_df['status'] == '1')
    ].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]

    codes = stock_df['code'].tolist()
    codes = snapshot_prefilter(codes)

    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]

    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    print(f"\n多进程拉取历史数据 ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    stock_data_dict = {}
    fail = 0
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=_init_worker) as pool:
        for res in tqdm(pool.imap_unordered(_fetch_one, tasks), total=len(tasks), desc="数据拉取", unit="只"):
            if res:
                stock_data_dict[res[0]] = res[1]
            else:
                fail += 1
    print(f"  成功 {len(stock_data_dict)} 只, 失败 {fail} 只")

    if not stock_data_dict:
        print("⚠️ 无有效数据")
        sys.exit(1)

    print("\n" + "=" * 50)
    df = run_vectorized_scan(stock_data_dict, name_map)
    print("=" * 50)

    if df is None or df.empty:
        print("\n本次无VCP命中 (条件严格, 全市场命中0只属正常, 非bug)")
        sys.exit(0)

    df, cluster, hot = enrich(df)

    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"vcp_v2_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"vcp_v2_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "date": tag,
                "params": PARAMS,
                "cluster": cluster,
                "n": int(len(df)),
                "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                "n_breakout": int(df['Valid_Breakout'].sum()) if 'Valid_Breakout' in df.columns else 0,
                "hits": df.to_dict('records')
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/vcp_v2_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        disp = df.copy()
        disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        drop_cols = ['行业', 'resonance', 'resonance_sector', 'Normal',
                     'KC_Contracting', 'ATR_Contracting', 'Price_Tight', 'Higher_Lows']
        disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")

    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_bo = int(df['Valid_Breakout'].sum()) if 'Valid_Breakout' in df.columns else 0
            send_serverchan(
                f"📐 VCP(v2向量化) 命中{len(df)}只 🎯风口{n_reso} 💥突破{n_bo}",
                build_push(df, cluster, hot)
            )
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()

    sys.exit(0)


if __name__ == "__main__":
    main()
