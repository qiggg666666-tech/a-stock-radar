# -*- coding: utf-8 -*-
"""
vagas_obv_screener.py —— VAGAS结构突破 + OBV-MACD量能确认 全市场选股(日线扫描版) · 矩阵规格
源自 TradingView Pine strategy "VAGAS + OBV MACD Template v2", 移植其【入场信号】为日线批量扫描。
信号=价格结构突破 × OBV量能确认 的双层共振:
  ① VAGAS结构突破: 收盘【刚】突破最近摆动高点(bullBreakNew, pivot左右各5根, 已确认不用未来数据)
  ② FVG缺口过滤(可选): low > high[2] (今日最低>前日最高, 向上跳空缺口, 代表强势)
  ③ OBV-MACD量能确认: OBV的MACD在信号线上方(obvBull) 且 近confirmWindow根内金叉(recentBullCross)
  三者同时 = longConfirm 做多。做空对称, A股无法做空 -> 转"⚠️风险警示/回避"。
对做多命中票, 用当前ATR给【初始建议位】: 止损=价-ATR*2, 止盈TP=价+ATR*4 (盈亏比1:2)。

【重要-扫描局限】Pine 是含持仓状态的 strategy; 本脚本是【截面扫描】, 只移植入场信号+初始建议位。
  原策略的 ATR止损止盈触发/反向信号平仓 是持仓后动态管理, 需持仓状态, 扫描【不执行】,
  拿到建议位后请手动管理, 或另做回测验证完整策略绩效。
【指标实现】OBV=成交量逐日累加(需较长历史, 故拉400天); OBV-MACD用EMA(12,26,9);
  摆动点为"已确认"局部极值近似 ta.pivothigh/pivotlow; ATR用Wilder平滑。
【本版规格】双源baostock+东财+硬超时; 全市场多进程(每子进程独立登录baostock, 命门已修);
  宽松快照预筛; baostock行业本地join+聚类+东财风口🎯; 推送全发分页(严格检查返回);
  存output/+收尾防护+sys.exit(0); 不拦交易日(周末用上一交易日数据)。
⚠️ 结构突破+量能确认双过滤, 信号稀少(尤其开FVG), 全市场0命中属正常, 非bug。
⚠️ 信号非买入保证; 突破易假突破, 务必按建议位止损。
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
    lookbackSwing=5, useFVG=True,
    fastLen=12, slowLen=26, signalLen=9,
    confirmWindow=3,
    atrLen=14, atrMultSL=2.0, atrMultTP=4.0,
    INCLUDE_WARNINGS=False,   # True=空头警示票也输出; 默认只推做多
    lookback_days=400, min_data_len=120,   # OBV需较长历史累加
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


# ==================== 指标 ====================
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high'].astype(float); low = df['low'].astype(float); close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()   # Wilder, 同 ta.atr


def calc_obv(df: pd.DataFrame) -> pd.Series:
    close = df['close'].astype(float); vol = df['volume'].astype(float)
    direction = np.sign(close.diff()).fillna(0)
    return (direction * vol).cumsum()


def last_swing_high(high_arr, left=5, right=5):
    """最近一个【已确认】摆动高点(右边right根确认, 不用未来数据); 近似 ta.pivothigh"""
    n = len(high_arr)
    for i in range(n - 1 - right, left - 1, -1):
        window = high_arr[i - left:i + right + 1]
        if high_arr[i] >= np.max(window):
            return float(high_arr[i])
    return None


def last_swing_low(low_arr, left=5, right=5):
    n = len(low_arr)
    for i in range(n - 1 - right, left - 1, -1):
        window = low_arr[i - left:i + right + 1]
        if low_arr[i] <= np.min(window):
            return float(low_arr[i])
    return None


# ==================== 入场信号检测 (日线) ====================
def check_one_stock(df: pd.DataFrame):
    """返回 (命中dict 或 None, 失败原因 或 None)。只看最新一根K线。"""
    if df is None or len(df) < PARAMS['min_data_len']:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    # OBV + OBV-MACD
    obv = calc_obv(df)
    obv_fast = obv.ewm(span=PARAMS['fastLen'], adjust=False).mean()
    obv_slow = obv.ewm(span=PARAMS['slowLen'], adjust=False).mean()
    obv_macd = obv_fast - obv_slow
    obv_signal = obv_macd.ewm(span=PARAMS['signalLen'], adjust=False).mean()
    obv_bull_cross = (obv_macd > obv_signal) & (obv_macd.shift(1) <= obv_signal.shift(1))
    obv_bear_cross = (obv_macd < obv_signal) & (obv_macd.shift(1) >= obv_signal.shift(1))

    # 摆动结构
    last_sh = last_swing_high(high.to_numpy(), PARAMS['lookbackSwing'], PARAMS['lookbackSwing'])
    last_sl = last_swing_low(low.to_numpy(), PARAMS['lookbackSwing'], PARAMS['lookbackSwing'])

    atr = calc_atr(df, PARAMS['atrLen'])

    n = len(df)
    c_now = float(close.iloc[-1]); c_prev = float(close.iloc[-2])
    bull_break = (last_sh is not None) and (c_now > last_sh)
    bear_break = (last_sl is not None) and (c_now < last_sl)
    bull_break_prev = (last_sh is not None) and (c_prev > last_sh)
    bear_break_prev = (last_sl is not None) and (c_prev < last_sl)
    bull_break_new = bull_break and not bull_break_prev
    bear_break_new = bear_break and not bear_break_prev

    # FVG 缺口
    bull_fvg = bool(low.iloc[-1] > high.iloc[-3])
    bear_fvg = bool(high.iloc[-1] < low.iloc[-3])

    vagas_bull = bull_break_new and (not PARAMS['useFVG'] or bull_fvg)
    vagas_bear = bear_break_new and (not PARAMS['useFVG'] or bear_fvg)

    obv_bull = bool(obv_macd.iloc[-1] > obv_signal.iloc[-1])
    obv_bear = bool(obv_macd.iloc[-1] < obv_signal.iloc[-1])
    long_signal = vagas_bull and obv_bull
    short_signal = vagas_bear and obv_bear

    cw = PARAMS['confirmWindow']
    recent_bull = bool(obv_bull_cross.iloc[-cw:].any())
    recent_bear = bool(obv_bear_cross.iloc[-cw:].any())
    long_confirm = long_signal and (bool(obv_bull_cross.iloc[-1]) or recent_bull)
    short_confirm = short_signal and (bool(obv_bear_cross.iloc[-1]) or recent_bear)

    is_long = bool(long_confirm)
    is_warn = bool(short_confirm)
    if not is_long and not (PARAMS['INCLUDE_WARNINGS'] and is_warn):
        return None, "无信号"

    at = float(atr.iloc[-1])
    if pd.isna(at) or at <= 0:
        return None, "数据不足"
    entry = c_now

    triggers = []
    if is_long:
        triggers.append("突破摆动高")
        if PARAMS['useFVG'] and bull_fvg: triggers.append("FVG缺口")
        if bool(obv_bull_cross.iloc[-1]): triggers.append("OBV金叉")
        elif recent_bull: triggers.append("OBV近期金叉")
    else:
        triggers.append("跌破摆动低")
        if PARAMS['useFVG'] and bear_fvg: triggers.append("FVG缺口")
        if bool(obv_bear_cross.iloc[-1]): triggers.append("OBV死叉")
        elif recent_bear: triggers.append("OBV近期死叉")

    sig_date = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(entry, 2), "信号日期": sig_date,
            "信号": "做多" if is_long else "⚠️空头警示",
            "触发": "+".join(triggers) if triggers else "—",
            "是否做多": is_long,
            "摆动高": round(last_sh, 2) if last_sh is not None else None,
            "摆动低": round(last_sl, 2) if last_sl is not None else None,
            "FVG": "↑缺口" if bull_fvg else ("↓缺口" if bear_fvg else "无"),
            "OBV状态": "多" if obv_bull else ("空" if obv_bear else "—"),
            "ATR": round(at, 3),
            "建议止损": round(entry - at * PARAMS['atrMultSL'], 2) if is_long else round(entry + at * PARAMS['atrMultSL'], 2),
            "建议止盈": round(entry + at * PARAMS['atrMultTP'], 2) if is_long else round(entry - at * PARAMS['atrMultTP'], 2),
            "score": round(at / entry * 100, 2), "resonance": False, "resonance_sector": ""}, None


# ------------------ 历史双源 (需 high/low/close/volume) ------------------
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
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
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
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
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
    print(f"开始VAGAS结构突破+OBV确认扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, FVG={'开' if PARAMS['useFVG'] else '关'}, 含警示={PARAMS['INCLUDE_WARNINGS']}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="vagas扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['信号']} 触发:{res['触发']} 价={res['最新价']}")
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
    print(f"🚀 结构突破板块: {cluster or '无'}")
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
    print(f"🎯 结构突破遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '是否做多', 'score'], ascending=[False, False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    longs = df[df['是否做多'] == True] if '是否做多' in df.columns else pd.DataFrame()
    warns = df[df['是否做多'] == False] if '是否做多' in df.columns else pd.DataFrame()
    L = [f"**🚀 VAGAS结构突破+OBV量能确认** | 做多{len(longs)}只 警示{len(warns)}只 🎯风口{len(reso)} (全发)",
         "*(结构突破×OBV量能确认; 建议位=入场那刻初始位, ATR止损止盈/反向平仓需持仓后手动管理或回测; 做空转回避警示; 信号稀少属正常)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🚀 **结构突破板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        flag = "🟢" if r['是否做多'] else "⚠️"
        pos = f"止损{r['建议止损']}/止盈{r['建议止盈']}" if r['是否做多'] else "回避"
        return (f"- {flag} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 触发:{r['触发']} | 现价{r['最新价']} "
                f"OBV={r['OBV状态']} FVG={r['FVG']} | {pos}")
    if not reso.empty:
        L.append(f"### 🎯 结构突破遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    if not longs.empty:
        L.append(f"### 🟢 做多机会 共{len(longs)}只")
        L += [line(r) for _, r in longs.iterrows()]; L.append("")
    if not warns.empty:
        L.append(f"### ⚠️ 空头警示 共{len(warns)}只 (回避)")
        L += [line(r) for _, r in warns.iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"🚀 VAGAS结构突破+OBV量能确认 (日线扫描+初始建议位) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['lookback_days']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; FVG={'开' if PARAMS['useFVG'] else '关'}; 含警示={PARAMS['INCLUDE_WARNINGS']}; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次扫描未发现满足 结构突破+OBV确认 的信号。")
        print("双过滤本就极严(尤其开FVG), 0命中属正常; 可调: useFVG改False放宽 / confirmWindow调大")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"vagas_obv_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"vagas_obv_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"useFVG": PARAMS['useFVG'], "confirmWindow": PARAMS['confirmWindow'],
                                               "INCLUDE_WARNINGS": PARAMS['INCLUDE_WARNINGS']},
                       "cluster": cluster, "n": int(len(df)),
                       "n_long": int(df['是否做多'].sum()) if '是否做多' in df.columns else 0,
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/vagas_obv_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', '摆动高', '摆动低', 'ATR'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_long = int(df['是否做多'].sum()) if '是否做多' in df.columns else 0
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"🚀 VAGAS结构突破 做多{n_long}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_vagas_obv<<<
