# -*- coding: utf-8 -*-
"""
vegas_tunnel_screener.py —— Vegas通道趋势突破 全市场选股(日线扫描版) · 矩阵规格
源自 TradingView Pine strategy "Vegas Tunnel EMA12 Confirm + ATR Exit + Volume Confirm"。
Vegas通道: EMA144/169=中隧道, EMA576/676=大隧道(判大趋势方向), EMA12=快线确认。
做多=收盘+EMA12同时上穿EMA144(突破中隧道) + 中隧道多头(EMA144>EMA169) + 大隧道多头
  (EMA576>EMA676) + 收盘>EMA169 + EMA12>EMA169 + 放量。做空对称, A股无法做空->转"⚠️回避"。
对做多命中票, 用当前ATR给【初始建议位】: 止损=价-1.5*ATR, 止盈=价+3*ATR (盈亏比1:2)。

【移植适配-突破新鲜度】原策略"突破那根K线才入场"; 扫描若要求"今天恰好双穿"会几乎0命中,
  故改为【最近 BREAK_FRESH_BARS 根内发生过突破】+ 当前趋势/位置/量能配合, 抓"刚突破通道"的票。
【长历史要求】大通道 EMA576/676 需~700交易日预热才可信, 故拉2500天日线、min_data_len=700,
  次新股(上市不足约3年)会被过滤——Vegas日线策略固有要求, 非bug。
【指标手写】EMA(adjust=False)/ATR(Wilder)/均量 全部手写, 不依赖 pandas_ta; 不画图, 无 matplotlib。
【本版·快到中段趋势(防追高)】原策略要求"大隧道明确多头(EMA576>EMA676)", 选出的多是长期已涨、
  突破时已在高位的"追高票"。本版默认 MID_TREND_MODE=1 改为"快到中段趋势":
  ① 大隧道【走平/刚拐头/刚金叉EMA676】(长期趋势刚启动, 非已涨很久) 替代"明确多头";
  ② 中期突破中隧道 + 价格站上中轨 (保留);
  ③ 【位置不高】: 当前价距近HIGH_LOOKBACK(250)天高点 ≥ HIGH_PULLBACK_MIN(15%), 防追高。
  合起来抓"趋势启动早期、即将进入中段主升、但位置还不高"的票, 而非高位追高。
  MID_TREND_MODE=0 一键回退原突破追高逻辑; BIG_FLAT_TOL/BIG_GOLDEN_BARS/HIGH_PULLBACK_MIN 等 env 可调松紧。
【本版规格】双源baostock+东财+硬超时; 全市场多进程(每子进程独立登录baostock, 命门已修);
  宽松快照预筛; baostock行业本地join+聚类+东财风口🎯; 推送全发分页(严格检查返回);
  存output/+收尾防护+sys.exit(0); 不拦交易日(周末用上一交易日数据)。
⚠️ "快到中段"=趋势启动早期判定, 假启动(突破后回落)风险仍在, 非买入保证, 务必按建议位止损。
⚠️ 通道突破+多重趋势过滤, 信号稀少, 全市场0命中属正常, 非bug。
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

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import baostock as bs
except ImportError:
    raise ImportError("请先安装: pip install baostock")
import akshare as ak
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ==================== 参数配置 (对应Pine inputs) ====================
PARAMS = dict(
    # Vegas 通道 EMA
    emaFastLen=12, emaMid1Len=144, emaMid2Len=169, emaLong1Len=576, emaLong2Len=676,
    # ATR 风险
    atrLen=14, atrStopMult=1.5, atrTakeMult=3.0,
    # 量能确认
    volumeLookback=20, volumeMultiplier=1.0,
    # 移植适配
    BREAK_FRESH_BARS=5,            # 突破新鲜度窗口: 最近N根内发生过通道突破即视为有效
    INCLUDE_WARNINGS=False,        # True=空头警示票也输出; 默认只推做多
    # 【本版】快到中段趋势 (防追高), 全部 env 可调
    MID_TREND_MODE=os.environ.get('MID_TREND_MODE', '1').strip() in ('1', 'true', 'True'),
    BIG_FLAT_BARS=int(os.environ.get('BIG_FLAT_BARS', '20')),      # 大隧道走平观察窗口
    BIG_FLAT_TOL=float(os.environ.get('BIG_FLAT_TOL', '0.02')),    # 大隧道走平容差(2%内视为走平/微升)
    BIG_GOLDEN_BARS=int(os.environ.get('BIG_GOLDEN_BARS', '30')),  # 大隧道金叉新鲜度窗口
    HIGH_LOOKBACK=int(os.environ.get('HIGH_LOOKBACK', '250')),     # 位置参考窗口(近N天高点)
    HIGH_PULLBACK_MIN=float(os.environ.get('HIGH_PULLBACK_MIN', '0.15')),  # 距高点最小回撤(15%防追高; 越小越宽松)
    # 数据 (EMA676需长历史预热)
    lookback_days=2500, min_data_len=700,
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0}


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


# ==================== 指标 (全部手写, 无 pandas_ta) ====================
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _sma(s, n):
    return s.rolling(n).mean()


def _atr(df, n):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ==================== 入场信号检测 (日线, 只看最新一根 + 突破新鲜度窗口) ====================
def check_one_stock(df: pd.DataFrame):
    if df is None or len(df) < PARAMS['min_data_len']:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"

    close = df['close'].astype(float); volume = df['volume'].astype(float)
    ema12 = _ema(close, PARAMS['emaFastLen'])
    ema144 = _ema(close, PARAMS['emaMid1Len'])
    ema169 = _ema(close, PARAMS['emaMid2Len'])
    ema576 = _ema(close, PARAMS['emaLong1Len'])
    ema676 = _ema(close, PARAMS['emaLong2Len'])
    atr = _atr(df, PARAMS['atrLen'])
    avg_vol = _sma(volume, PARAMS['volumeLookback'])
    vol_confirm = volume > (avg_vol * PARAMS['volumeMultiplier'])   # 序列

    L = len(df) - 1
    if pd.isna(ema676.iloc[L]) or pd.isna(atr.iloc[L]) or pd.isna(avg_vol.iloc[L]):
        return None, "数据不足"
    aL = float(atr.iloc[L])
    if aL <= 0:
        return None, "数据不足"

    cL = float(close.iloc[L])
    e12, e144, e169 = float(ema12.iloc[L]), float(ema144.iloc[L]), float(ema169.iloc[L])
    e576, e676 = float(ema576.iloc[L]), float(ema676.iloc[L])

    # 趋势过滤 (最新一根)
    mid_long = e144 > e169; big_long = e576 > e676
    mid_short = e144 < e169; big_short = e576 < e676

    # 【本版】快到中段趋势判定: 大隧道企稳/拐头(趋势刚启动) + 位置不高(防追高)
    # (a) 大隧道走平或微升: 最近 BIG_FLAT_BARS 天 EMA576 未明显下行(跌幅<容差)
    big_flat_or_up = False
    if L >= PARAMS['BIG_FLAT_BARS']:
        e576_past = float(ema576.iloc[L - PARAMS['BIG_FLAT_BARS']])
        if e576_past > 0:
            big_flat_or_up = e576 >= e576_past * (1 - PARAMS['BIG_FLAT_TOL'])
    # (b) 大隧道刚金叉: 最近 BIG_GOLDEN_BARS 根内 EMA576 上穿 EMA676
    big_cross_up = (ema576.shift(1) <= ema676.shift(1)) & (ema576 > ema676)
    big_golden_recent = bool(big_cross_up.iloc[-PARAMS['BIG_GOLDEN_BARS']:].any())
    big_turning = bool(big_flat_or_up or big_golden_recent)
    big_state_tag = ("拐头/走平" if big_turning else ("多" if big_long else ("空" if big_short else "—")))
    # (c) 位置不高: 当前价距近 HIGH_LOOKBACK 天高点 >= HIGH_PULLBACK_MIN (防追高)
    high_ref = float(close.rolling(PARAMS['HIGH_LOOKBACK'], min_periods=60).max().iloc[L])
    pullback_pct = (high_ref - cL) / high_ref * 100 if high_ref > 0 else 0.0
    not_high = bool(pullback_pct >= PARAMS['HIGH_PULLBACK_MIN'] * 100)

    # 突破事件 (序列, 含放量同根)
    close_cross_up = (close.shift(1) <= ema144.shift(1)) & (close > ema144)
    ema12_cross_up = (ema12.shift(1) <= ema144.shift(1)) & (ema12 > ema144)
    long_break = close_cross_up & ema12_cross_up & vol_confirm
    close_cross_dn = (close.shift(1) >= ema169.shift(1)) & (close < ema169)
    ema12_cross_dn = (ema12.shift(1) >= ema169.shift(1)) & (ema12 < ema169)
    short_break = close_cross_dn & ema12_cross_dn & vol_confirm

    fb = PARAMS['BREAK_FRESH_BARS']
    long_break_recent = bool(long_break.iloc[-fb:].any())
    short_break_recent = bool(short_break.iloc[-fb:].any())

    # 状态条件 (最新一根) —— 【本版】做多按 MID_TREND_MODE 分支; 做空保持原样
    if PARAMS['MID_TREND_MODE']:
        # 快到中段: 大趋势刚启动(走平/拐头/刚金叉) + 中期多头 + 价格站上中轨 + 位置不高(防追高)
        long_state = big_turning and mid_long and (cL > e169) and (e12 > e169) and not_high
    else:
        # 原突破追高: 大隧道明确多头 + 中期多头 + 价格站上中轨
        long_state = mid_long and big_long and (cL > e169) and (e12 > e169)
    short_state = mid_short and big_short and (cL < e144) and (e12 < e144)

    long_signal = long_break_recent and long_state
    short_signal = short_break_recent and short_state

    is_long = bool(long_signal)
    is_warn = bool(short_signal)
    if not is_long and not (PARAMS['INCLUDE_WARNINGS'] and is_warn):
        return None, "无信号"

    entry = cL
    sl = round(entry - aL * PARAMS['atrStopMult'], 2)
    tp = round(entry + aL * PARAMS['atrTakeMult'], 2)
    vol_ratio = round(float(volume.iloc[L] / avg_vol.iloc[L]), 2) if avg_vol.iloc[L] else 0
    big_trend = "多" if big_long else ("空" if big_short else "—")
    sig_date = pd.to_datetime(df['date'].iloc[L]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    trig = ("上穿中隧道+放量" if is_long else "跌破中隧道+放量")
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(entry, 2), "信号日期": sig_date,
            "信号": "做多" if is_long else "⚠️空头警示", "触发": trig,
            "是否做多": is_long, "大趋势": big_trend, "大隧道": big_state_tag,
            "距高点%": round(pullback_pct, 1),
            "EMA12": round(e12, 2), "EMA144": round(e144, 2), "EMA169": round(e169, 2),
            "放量倍数": vol_ratio, "ATR": round(aL, 3),
            "建议止损": sl if is_long else round(entry + aL * PARAMS['atrStopMult'], 2),
            "建议止盈": tp if is_long else round(entry - aL * PARAMS['atrTakeMult'], 2),
            "score": vol_ratio, "resonance": False, "resonance_sector": ""}, None


# ------------------ 历史双源 (需 high/low/close/volume; 长历史供EMA576/676) ------------------
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
                if len(d) >= PARAMS['min_data_len']:
                    return d[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
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
    mode_tag = "快到中段(防追高)" if PARAMS['MID_TREND_MODE'] else "原突破追高"
    print(f"开始Vegas通道扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 模式={mode_tag}, 突破窗口={PARAMS['BREAK_FRESH_BARS']}根, 含警示={PARAMS['INCLUDE_WARNINGS']}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="vegas扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['信号']} 大隧道{res['大隧道']} 距高点{res['距高点%']}% 价={res['最新价']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['是否做多', 'score'], ascending=[False, False]).reset_index(drop=True)
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
    print(f"📈 通道突破板块: {cluster or '无'}")
    heat = pd.DataFrame()
    for i in range(3):
        try:
            heat = ak.stock_board_industry_name_em()
            if heat is not None and not heat.empty:
                return_heat_ok = True
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
    print(f"🎯 通道突破遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '是否做多', 'score'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    longs = df[df['是否做多'] == True] if '是否做多' in df.columns else pd.DataFrame()
    warns = df[df['是否做多'] == False] if '是否做多' in df.columns else pd.DataFrame()
    mode_tag = "快到中段·防追高" if PARAMS['MID_TREND_MODE'] else "突破追高"
    L = [f"**📈 Vegas通道({mode_tag})** | 做多{len(longs)}只 警示{len(warns)}只 🎯风口{len(reso)} (全发)",
         "*(快到中段=大隧道走平/拐头+中期突破+位置不高防追高; 建议位=入场那刻初始位, trailing需持仓后手动; 做空转回避; 信号稀少属正常)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📈 **通道突破板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        flag = "🟢" if r['是否做多'] else "⚠️"
        pos = f"止损{r['建议止损']}/止盈{r['建议止盈']}" if r['是否做多'] else "回避"
        return (f"- {flag} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] {r['触发']} | 现价{r['最新价']} "
                f"大隧道={r.get('大隧道','—')} 距高点{r.get('距高点%','—')}% 放量{r['放量倍数']} | {pos}")
    if not reso.empty:
        L.append(f"### 🎯 通道突破遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    if not longs.empty:
        L.append(f"### 🟢 做多机会 共{len(longs)}只 (按放量)")
        L += [line(r) for _, r in longs.iterrows()]; L.append("")
    if not warns.empty:
        L.append(f"### ⚠️ 空头警示 共{len(warns)}只 (回避)")
        L += [line(r) for _, r in warns.iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    mode_tag = "快到中段(防追高)" if PARAMS['MID_TREND_MODE'] else "原突破追高"
    print(f"📈 Vegas通道趋势突破 [{mode_tag}] (日线扫描+初始建议位) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['lookback_days']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; 突破窗口={PARAMS['BREAK_FRESH_BARS']}根; 距高点≥{PARAMS['HIGH_PULLBACK_MIN']*100:.0f}%防追高; 含警示={PARAMS['INCLUDE_WARNINGS']}; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次扫描未发现满足 Vegas通道(快到中段) 的信号。")
        print("通道突破+趋势启动+位置过滤本就极严, 0命中属正常; 可调: HIGH_PULLBACK_MIN调小(放宽位置)/BIG_FLAT_TOL调大(放宽大隧道)")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"vegas_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"vegas_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"MID_TREND_MODE": PARAMS['MID_TREND_MODE'],
                                               "BREAK_FRESH_BARS": PARAMS['BREAK_FRESH_BARS'],
                                               "HIGH_PULLBACK_MIN": PARAMS['HIGH_PULLBACK_MIN'],
                                               "INCLUDE_WARNINGS": PARAMS['INCLUDE_WARNINGS']},
                       "cluster": cluster, "n": int(len(df)),
                       "n_long": int(df['是否做多'].sum()) if '是否做多' in df.columns else 0,
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/vegas_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'EMA12', 'EMA144', 'EMA169', 'ATR'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_long = int(df['是否做多'].sum()) if '是否做多' in df.columns else 0
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📈 Vegas快到中段 做多{n_long}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_vegas<<<
