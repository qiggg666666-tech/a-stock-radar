# -*- coding: utf-8 -*-
"""
trend_judge_screener.py —— 综合走势判断 + 选股(多因子技术面强弱评分·精确版) · 矩阵规格
对每只票用真实日线, 综合5维度算【走势评分0-100】+【多空标签】, 全市场选出技术面走势最强的票。
  ① 趋势(30%): 均线多头排列(MA5>10>20>60)+站上MA20+MA20/MA60拐头向上
  ② 动量(20%): 近20日涨幅 + RSI健康区间(不超买不超卖)
  ③ 量价(15%): 量比放量 + OBV站上均量(量能向上)
  ④ MACD(20%): DIF>DEA + 零轴上方 + 近期金叉
  ⑤ 位置(15%): 距近20日高低点位置(越接近高点=越强/突破在即)
加权合成 -> 走势评分; 标签: ≥70强多/≥55偏多/≥40中性/≥25偏空/<25强空。

【本版精确化·解决"选太多"】原版仅"总分≥55"单一门槛, 牛市/反弹市大量票过线, 推送冗长且含"伪强势"
  (靠单维度凑分)。本版加6道质量门槛(全部env可调), 从"凑分即入"升级为"多维共振+趋势达标+有量+不追高+流动性":
  ① 趋势分≥MIN_TREND_SCORE(0.5, 灵魂维度必须达标); ② 至少MIN_STRONG_DIMS(3)个维度≥STRONG_DIM_THRESH(0.55)
  多维共振; ③ 量价分≥MIN_VOLUME_SCORE(0.4)有量支撑; ④ 近20日涨幅≤MAX_RET20(0.5)追高保护;
  ⑤ RSI≤MAX_RSI(80)过热保护; ⑥ 日均成交额≥MIN_AMOUNT(1亿)流动性; ⑦ 总分≥SCORE_MIN(55→65)。
  双源成交额统一为元(baostock amount / akshare 成交额), 修复 volume 单位(股/手差100倍)坑。
  推送加"共振X/5"与"★精选"(评分≥70且共振≥4维)分组。
  调松紧: 想更严->调高 SCORE_MIN/MIN_STRONG_DIMS/MIN_TREND_SCORE; 想放宽->调低, 或 MIN_STRONG_DIMS=2。

【诚实定位-必读】本脚本是【概率性技术面强弱评分, 非预测】。它给出"当前哪些票技术面最强/最弱",
  帮你缩小盯盘范围/提高胜率, 不等于"某只一定涨"; 多空标签是当下截面状态, 数据变标签也变。
  门槛提高后命中变少属正常(提质必然减量), 弱市可能0命中; 务必结合仓位/止损, 不构成投资建议。
【工程规格】双源baostock+东财+硬超时; 全市场多进程(每子进程独立登录baostock, 命门已修);
  append兼容补丁; 宽松快照预筛(失败退化全扫); baostock行业本地join+聚类+东财风口🎯;
  推送全发分页(严格检查返回); 存output/+收尾防护+sys.exit(0); 不拦交易日。指标全部手写, 无pandas_ta/matplotlib。
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


# ==================== 参数配置 ====================
PARAMS = dict(
    # 5维度权重(和=1.0)
    W_TREND=0.30, W_MOMENTUM=0.20, W_VOLUME=0.15, W_MACD=0.20, W_POSITION=0.15,
    # 多空标签阈值(走势评分0-100)
    LABEL_STRONG_BULL=70, LABEL_BULL=55, LABEL_NEUTRAL=40, LABEL_BEAR=25,
    # 选股门槛: 只推评分≥此值的票(本版55→65, 配合质量门槛提质减量)
    SCORE_MIN=65.0,
    # 数据
    lookback_days=400, min_data_len=120,   # MA60需60根+预热
    # 快照预筛(宽松, 失败退化全扫, 靠SCAN_LIMIT保底)
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
SCORE_MIN_ENV = float(os.environ.get('SCORE_MIN', str(PARAMS['SCORE_MIN'])))

# ---- 【本版精确化】质量门槛(全部env可调; 想更严调高, 想放宽调低) ----
MIN_TREND_SCORE = float(os.environ.get('MIN_TREND_SCORE', '0.5'))       # 趋势分下限(灵魂维度必须达标)
MIN_STRONG_DIMS = int(os.environ.get('MIN_STRONG_DIMS', '3'))           # 至少几个维度达标(多维共振)
STRONG_DIM_THRESH = float(os.environ.get('STRONG_DIM_THRESH', '0.55'))  # 单维度"达标"阈值
MIN_VOLUME_SCORE = float(os.environ.get('MIN_VOLUME_SCORE', '0.4'))     # 量价分下限(有量支撑)
MAX_RET20 = float(os.environ.get('MAX_RET20', '0.5'))                   # 近20日涨幅上限(追高保护)
MAX_RSI = float(os.environ.get('MAX_RSI', '80'))                        # RSI上限(过热保护)
MIN_AMOUNT = float(os.environ.get('MIN_AMOUNT', '1.0e8'))               # 日均成交额下限(流动性, 元)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "趋势不足": 0, "共振不足": 0, "量价不足": 0,
              "涨幅过大": 0, "RSI过热": 0, "流动性不足": 0, "评分不足": 0}


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


def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ==================== 指标 (全部手写, 无 pandas_ta) ====================
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _sma(s, n):
    return s.rolling(n).mean()


def _rsi(s, n):
    d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean(); al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-9))


def _obv(close, volume):
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


# ==================== 综合走势评分 (5维度 + 质量门槛) ====================
def _label(score):
    if score >= PARAMS['LABEL_STRONG_BULL']:
        return "🟢强多"
    if score >= PARAMS['LABEL_BULL']:
        return "🟢偏多"
    if score >= PARAMS['LABEL_NEUTRAL']:
        return "⚪中性"
    if score >= PARAMS['LABEL_BEAR']:
        return "🔴偏空"
    return "🔴强空"


def check_one_stock(df: pd.DataFrame):
    """返回 (命中dict 或 None, 失败原因 或 None)。5维度评分 + 6道质量门槛。"""
    if df is None or len(df) < PARAMS['min_data_len']:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"

    close = df['close'].astype(float); high = df['high'].astype(float)
    low = df['low'].astype(float); volume = df['volume'].astype(float)
    L = len(df) - 1

    ma5 = _sma(close, 5); ma10 = _sma(close, 10); ma20 = _sma(close, 20); ma60 = _sma(close, 60)
    if pd.isna(ma60.iloc[L]):
        return None, "数据不足"
    cL = float(close.iloc[L])

    # ---- 维度1 趋势 ----
    m5, m10, m20, m60 = float(ma5.iloc[L]), float(ma10.iloc[L]), float(ma20.iloc[L]), float(ma60.iloc[L])
    trend = 0.0
    if m5 > m10 > m20:
        trend += 0.3
    if m20 > m60:
        trend += 0.1
    if cL > m20:
        trend += 0.2
    if m20 > float(ma20.iloc[L - 6]):   # MA20拐头向上
        trend += 0.2
    if m60 > float(ma60.iloc[L - 6]):   # MA60拐头向上
        trend += 0.2

    # ---- 维度2 动量 ----
    ret20 = (cL / float(close.iloc[L - 20]) - 1) if close.iloc[L - 20] else 0.0
    ret_score = _clip((ret20 + 0.10) / 0.30)
    rsi = float(_rsi(close, 14).iloc[L])
    if 40 <= rsi <= 70:
        rsi_score = 1.0
    elif 30 <= rsi < 40 or 70 < rsi <= 80:
        rsi_score = 0.5
    else:
        rsi_score = 0.2
    momentum = 0.6 * ret_score + 0.4 * rsi_score

    # ---- 维度3 量价 ----
    vol_ma20 = float(_sma(volume, 20).iloc[L])
    vol_ratio = (float(volume.iloc[L]) / vol_ma20) if vol_ma20 > 0 else 0.0
    vol_score = _clip((vol_ratio - 0.5) / 1.5)
    obv = _obv(close, volume); obv_ma20 = _sma(obv, 20)
    obv_score = 1.0 if (pd.notna(obv_ma20.iloc[L]) and obv.iloc[L] > obv_ma20.iloc[L]) else 0.4
    volume_dim = 0.5 * vol_score + 0.5 * obv_score

    # ---- 维度4 MACD ----
    dif = _ema(close, 12) - _ema(close, 26); dea = _ema(dif, 9)
    dL, eL = float(dif.iloc[L]), float(dea.iloc[L])
    macd = 0.0
    if dL > eL:
        macd += 0.4
    if dL > 0:
        macd += 0.3
    cross = (dif.shift(1) <= dea.shift(1)) & (dif > dea)
    if bool(cross.iloc[-5:].any()):
        macd += 0.3

    # ---- 维度5 位置 ----
    h20 = float(high.iloc[-20:].max()); l20 = float(low.iloc[-20:].min())
    pos = _clip((cL - l20) / (h20 - l20)) if h20 > l20 else 0.5
    near_high_pct = round((h20 - cL) / h20 * 100, 1) if h20 else None

    # ---- 综合评分 ----
    total = (PARAMS['W_TREND'] * trend + PARAMS['W_MOMENTUM'] * momentum +
             PARAMS['W_VOLUME'] * volume_dim + PARAMS['W_MACD'] * macd +
             PARAMS['W_POSITION'] * pos)
    score = round(total * 100, 1)

    # ---- 【本版精确化】6道质量门槛(逐项, 任一不过即淘汰; 全部env可调) ----
    # ① 趋势是灵魂: 趋势分必须达标(挡掉靠动量/位置凑分的"伪强势")
    if trend < MIN_TREND_SCORE:
        return None, "趋势不足"
    # ② 多维共振: 至少MIN_STRONG_DIMS个维度≥STRONG_DIM_THRESH(挡掉单维偏科)
    dims = [trend, momentum, volume_dim, macd, pos]
    strong_dims = sum(1 for d in dims if d >= STRONG_DIM_THRESH)
    if strong_dims < MIN_STRONG_DIMS:
        return None, "共振不足"
    # ③ 量价确认: 有量能支撑(挡掉无量空涨)
    if volume_dim < MIN_VOLUME_SCORE:
        return None, "量价不足"
    # ④ 追高保护: 近20日涨幅过大(挡掉接盘)
    if ret20 > MAX_RET20:
        return None, "涨幅过大"
    # ⑤ 过热保护: RSI过高
    if rsi > MAX_RSI:
        return None, "RSI过热"
    # ⑥ 流动性: 日均成交额(双源统一为元; 缺amount列则跳过此项, 不误杀)
    avg_amount_yi = None
    if 'amount' in df.columns:
        amt = pd.to_numeric(df['amount'], errors='coerce').iloc[-20:].mean()
        if not pd.isna(amt):
            avg_amount_yi = round(float(amt) / 1e8, 2)
            if float(amt) < MIN_AMOUNT:
                return None, "流动性不足"
    # ⑦ 总分门槛
    if score < SCORE_MIN_ENV:
        return None, "评分不足"

    sig_date = pd.to_datetime(df['date'].iloc[L]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(cL, 2), "信号日期": sig_date,
            "走势评分": score, "多空标签": _label(score), "共振维度": strong_dims,
            "趋势分": round(trend, 2), "动量分": round(momentum, 2), "量价分": round(volume_dim, 2),
            "MACD分": round(macd, 2), "位置分": round(pos, 2),
            "RSI": round(rsi, 1), "量比": round(vol_ratio, 2), "日均成交额亿": avg_amount_yi,
            "近20日涨幅%": round(ret20 * 100, 1), "距20日高%": near_high_pct,
            "resonance": False, "resonance_sector": ""}, None


# ------------------ 历史双源 (含成交额amount, 统一为元; 修volume单位坑) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'})
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'] if c in d.columns]
                    return d[cols]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume,amount", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS['min_data_len']:
                    return d
        except Exception:
            pass
    return _fetch_hist_em(code, sy, ey)


def snapshot_prefilter(codes_with_prefix):
    if not PARAMS['SNAPSHOT_PRE']:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫")
            return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for col in ['最新价', '成交额', '换手率']:
            if col in spot.columns:
                spot[col] = pd.to_numeric(spot[col], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (宽松, 失败退化全扫)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < PARAMS['min_data_len']:
            return {"__fail__": "数据不足"}
        time.sleep(PARAMS['SLEEP'])
        info, reason = check_one_stock(df)
        if info is None:
            return {"__fail__": reason}
        info["代码"] = code; info["名称"] = name
        return info
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception:
        return {"__fail__": "抓取失败"}


# ------------------ 主扫描 ------------------
def run_scan():
    global _INDUSTRY_MAP, FAIL_STATS
    FAIL_STATS = {k: 0 for k in FAIL_STATS}
    print("连接 Baostock（行业表 + 列表 + 子进程登录）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
                for _, row in ind.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock国标, 本地join零接口)")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        _bs_logout()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切 akshare 兜底...")
        for attempt in range(3):
            try:
                d = ak.stock_info_a_code_name()
                if d is not None and not d.empty and 'code' in d.columns:
                    nc = 'name' if 'name' in d.columns else d.columns[1]
                    d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                    d['code'] = d['code'].astype(str).str.zfill(6)
                    d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                    d['type'] = '1'; d['status'] = '1'; stock_df = d; break
            except Exception as e:
                print(f"  akshare列表第{attempt+1}次失败: {e}")
            time.sleep(2 + attempt)
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = stock_df['code'].tolist()
    codes = snapshot_prefilter(codes)
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail_count = 0
    print(f"开始综合走势判断扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源）...")
    print(f"精确门槛: 总分≥{SCORE_MIN_ENV} 趋势≥{MIN_TREND_SCORE} 共振≥{MIN_STRONG_DIMS}维(每维≥{STRONG_DIM_THRESH}) "
          f"量价≥{MIN_VOLUME_SCORE} 涨幅≤{MAX_RET20*100:.0f}% RSI≤{MAX_RSI} 日均额≥{MIN_AMOUNT/1e8:.1f}亿")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="走势判断", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  {res['多空标签']} {res['代码']} {res['名称']} 评分{res['走势评分']} 共振{res['共振维度']}/5 趋{res['趋势分']}动{res['动量分']}量{res['量价分']}M{res['MACD分']}位{res['位置分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各淘汰原因统计：")
    for k, v in FAIL_STATS.items():
        if v:
            print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 淘汰{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['共振维度', '走势评分'], ascending=[False, False]).reset_index(drop=True)
    return df


# ------------------ 行业本地join + 聚类 + 风口🎯 ------------------
def enrich(df):
    targets = df.to_dict('records')
    mapped = 0
    for r in targets:
        ind = _INDUSTRY_MAP.get(r['代码'], '—')
        r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注(本地join): {mapped}/{len(targets)} 只有板块")
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"📊 走势强势板块: {cluster or '无'}")
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
        hot = [(str(row['板块名称']), round(float(row['_chg']), 2)) for _, row in h.head(HOT_SECTOR_TOP).iterrows()]
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in targets:
        sec = r.get('行业', '')
        m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh):
                    m = hh; break
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 走势强势遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '共振维度', '走势评分'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    jingxuan = df[(df['走势评分'] >= 70) & (df['共振维度'] >= 4)] if ('共振维度' in df.columns and '走势评分' in df.columns) else pd.DataFrame()
    strong = df[df['走势评分'] >= PARAMS['LABEL_STRONG_BULL']] if '走势评分' in df.columns else pd.DataFrame()
    bull = df[(df['走势评分'] >= PARAMS['LABEL_BULL']) & (df['走势评分'] < PARAMS['LABEL_STRONG_BULL'])] if '走势评分' in df.columns else pd.DataFrame()
    L = [f"**📊 综合走势判断(精确版)** | ★精选{len(jingxuan)} 强多{len(strong)} 偏多{len(bull)} 🎯风口{len(reso)} (评分≥{SCORE_MIN_ENV}+6道质量门槛)",
         "*(5维评分+趋势达标+多维共振+有量+不追高+流动性; 概率性强弱, 非预测; 结合仓位/止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📊 **走势强势板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        amt = f" 额{r['日均成交额亿']}亿" if r.get('日均成交额亿') is not None else ""
        return (f"- {r['多空标签']} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 评分{r['走势评分']} 共振{r['共振维度']}/5 | "
                f"现价{r['最新价']} 涨{r['近20日涨幅%']}% RSI{r['RSI']} 量比{r['量比']}{amt} | "
                f"趋{r['趋势分']}动{r['动量分']}量{r['量价分']}M{r['MACD分']}位{r['位置分']}")
    if not reso.empty:
        L.append(f"### 🎯 走势强势遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.head(PUSH_TOP).iterrows()]; L.append("")
    if not jingxuan.empty:
        L.append(f"### ★ 精选 共{len(jingxuan)}只 (评分≥70 且 共振≥4维)")
        L += [line(r) for _, r in jingxuan.head(PUSH_TOP).iterrows()]; L.append("")
    if not strong.empty:
        L.append(f"### 🟢 强多 共{len(strong)}只 (评分≥{PARAMS['LABEL_STRONG_BULL']})")
        L += [line(r) for _, r in strong.head(PUSH_TOP).iterrows()]; L.append("")
    if not bull.empty:
        L.append(f"### 🟢 偏多 共{len(bull)}只")
        L += [line(r) for _, r in bull.head(PUSH_TOP).iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"📊 综合走势判断+选股(精确版) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['lookback_days']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; 不拦交易日; 推送全列+分页")
    print("⚠️ 概率性技术面强弱评分, 非预测; 门槛提高后命中变少属正常(提质减量); 结合仓位/止损")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print(f"\n本次未发现通过质量门槛的票(市场偏弱或门槛严; 可调低 SCORE_MIN/MIN_STRONG_DIMS/MIN_TREND_SCORE 放宽)。")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"trend_judge_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"trend_judge_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"SCORE_MIN": SCORE_MIN_ENV, "MIN_TREND_SCORE": MIN_TREND_SCORE,
                       "MIN_STRONG_DIMS": MIN_STRONG_DIMS, "STRONG_DIM_THRESH": STRONG_DIM_THRESH,
                       "MIN_VOLUME_SCORE": MIN_VOLUME_SCORE, "MAX_RET20": MAX_RET20, "MAX_RSI": MAX_RSI, "MIN_AMOUNT": MIN_AMOUNT,
                       "weights": {"trend": PARAMS['W_TREND'], "momentum": PARAMS['W_MOMENTUM'],
                                   "volume": PARAMS['W_VOLUME'], "macd": PARAMS['W_MACD'], "position": PARAMS['W_POSITION']}},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/trend_judge_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', '距20日高%'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_strong = int((df['走势评分'] >= PARAMS['LABEL_STRONG_BULL']).sum()) if '走势评分' in df.columns else 0
            n_jx = int(((df['走势评分'] >= 70) & (df['共振维度'] >= 4)).sum()) if ('共振维度' in df.columns and '走势评分' in df.columns) else 0
            send_serverchan(f"📊 走势判断 ★精选{n_jx} 强多{n_strong} 共{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_trend_judge<<<
