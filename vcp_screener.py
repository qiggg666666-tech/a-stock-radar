# -*- coding: utf-8 -*-
"""
vcp_screener.py —— VCP(波动收缩形态) 全市场选股 · 矩阵规格
Minervini式VCP: 多次收缩递减+量缩+贴近pivot+ATR收敛=突破前蓄势。
【本版】不拦交易日(周末可复盘); 推送全列+超长自动分页(所有命中都送达)。
【矩阵规格】双源+硬超时; baostock多进程命门已修; 快照预筛; 行业+聚类+风口🎯; 收尾防护。
⚠️ VCP条件严格, 命中常个位数甚至0, 属正常; 需等放量突破pivot确认, 严格止损。
依赖: requirements.txt 须含 scipy。
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

import numpy as np
import pandas as pd
import akshare as ak
import baostock as bs
from scipy.signal import argrelextrema
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

PARAMS = dict(
    LOOKBACK_DAYS=500, MIN_REQUIRED=120,
    VCP_LOOKBACK=120, MIN_CONTRACTIONS=2, MAX_CONTRACTIONS=5,
    MIN_CONTRACTION_PCT=0.03, MAX_FIRST_CONTRACTION=0.35, MAX_BASE_RANGE=0.35,
    VOLUME_DRY_RATIO=0.65, MAX_BASE_WEEKS=16, MIN_BASE_WEEKS=3, VCP_SCORE_MIN=5,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3, NUM_PROCESSES=3, SLEEP=0.3,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '15'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False


def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content); return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(f"https://sctapi.ftqq.com/{key}.send",
                             data={"title": title, "desp": content}, timeout=15).json().get("code") == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def send_serverchan(title, content, sendkey=""):
    """全发: 超长自动按行切分多条发送, 保证所有股票都送达"""
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
    print(f"📲 共发送{len(chunks)}条(全发分页)" if len(chunks) > 1 else "📲 推送成功")
    return ok


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
        m = (spot['代码'].str.startswith(PARAMS["KEEP_PREFIX"])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS["EXCLUDE_NAME"]), na=False, regex=True)
             & (spot['最新价'] >= PARAMS["MIN_PRICE"]))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS["PRE_AMOUNT_MIN"])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS["PRE_TURNOVER_MIN"])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (价≥{PARAMS['MIN_PRICE']}+活跃, 宽松保蓄势票)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix


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


def find_pivots(df, order=5):
    high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    return high_idx, low_idx


def detect_vcp_advanced(df, lookback=None, min_contractions=None, max_contractions=None,
                        min_contraction_pct=None, max_first_contraction=None, max_base_range=None,
                        volume_dry_ratio=None, max_base_weeks=None, min_base_weeks=None):
    lookback = lookback or PARAMS["VCP_LOOKBACK"]
    min_contractions = min_contractions if min_contractions is not None else PARAMS["MIN_CONTRACTIONS"]
    max_contractions = max_contractions if max_contractions is not None else PARAMS["MAX_CONTRACTIONS"]
    min_contraction_pct = min_contraction_pct if min_contraction_pct is not None else PARAMS["MIN_CONTRACTION_PCT"]
    max_first_contraction = max_first_contraction if max_first_contraction is not None else PARAMS["MAX_FIRST_CONTRACTION"]
    max_base_range = max_base_range if max_base_range is not None else PARAMS["MAX_BASE_RANGE"]
    volume_dry_ratio = volume_dry_ratio if volume_dry_ratio is not None else PARAMS["VOLUME_DRY_RATIO"]
    max_base_weeks = max_base_weeks if max_base_weeks is not None else PARAMS["MAX_BASE_WEEKS"]
    min_base_weeks = min_base_weeks if min_base_weeks is not None else PARAMS["MIN_BASE_WEEKS"]
    if df is None or len(df) < lookback:
        return False, {"reason": "数据不足"}
    recent = df.tail(lookback).copy().reset_index(drop=True)
    close = recent['close'].values; high = recent['high'].values
    low = recent['low'].values; volume = recent['volume'].values
    hi_max = recent['high'].max()
    base_range = (hi_max - recent['low'].min()) / hi_max if hi_max else 0
    if base_range > max_base_range:
        return False, {"reason": f"基地振幅过深: {base_range*100:.1f}%"}
    high_idx, low_idx = find_pivots(recent, order=4)
    if len(high_idx) < 2 or len(low_idx) < 2:
        return False, {"reason": "高低点不足"}
    swings = [('H', i, high[i]) for i in high_idx] + [('L', i, low[i]) for i in low_idx]
    swings.sort(key=lambda x: x[1])
    contractions = []
    for i in range(len(swings) - 1):
        if swings[i][0] == 'H' and swings[i + 1][0] == 'L':
            h_price, l_price = swings[i][2], swings[i + 1][2]
            if h_price > l_price:
                pct = (h_price - l_price) / h_price
                if pct >= min_contraction_pct:
                    contractions.append({'start_idx': swings[i][1], 'end_idx': swings[i + 1][1],
                                         'high': h_price, 'low': l_price, 'pct': pct})
    earliest_start = contractions[0]['start_idx'] if contractions else 0
    contractions = contractions[-max_contractions:]
    if len(contractions) < min_contractions:
        return False, {"reason": f"收缩次数不足: {len(contractions)}"}
    pcts = [c['pct'] for c in contractions]
    is_decreasing = all(pcts[i] > pcts[i + 1] * 0.85 for i in range(len(pcts) - 1))
    if not is_decreasing:
        return False, {"reason": "收缩未递减", "pcts": [round(p * 100, 1) for p in pcts]}
    if pcts[0] > max_first_contraction:
        return False, {"reason": f"首次收缩过深: {pcts[0]*100:.1f}%"}
    if pcts[-1] > 0.12:
        return False, {"reason": f"最终收缩不够紧: {pcts[-1]*100:.1f}%"}
    last_c = contractions[-1]
    vol_window = volume[last_c['start_idx']:last_c['end_idx'] + 1]
    avg_vol_50 = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
    vol_dry = bool(np.mean(vol_window) < avg_vol_50 * volume_dry_ratio) if len(vol_window) > 0 else False
    base_weeks = (lookback - earliest_start) / 5
    if base_weeks > max_base_weeks or base_weeks < min_base_weeks:
        return False, {"reason": f"整理时间不合适: {base_weeks:.1f}周"}
    pivot = last_c['high']; current_price = close[-1]
    near_pivot = bool(current_price >= pivot * 0.97)
    def calc_atr(h, l, c, period=14):
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else (np.mean(tr) if len(tr) else 0)
    atr_recent = calc_atr(high, low, close, 10)
    atr_earlier = calc_atr(high[:60], low[:60], close[:60], 10) if len(high) > 70 else atr_recent
    atr_contracting = bool(atr_recent < atr_earlier * 0.8)
    score = 0
    if is_decreasing: score += 2
    if vol_dry: score += 2
    if near_pivot: score += 2
    if atr_contracting: score += 1
    if pcts[-1] < 0.08: score += 1
    is_vcp = score >= PARAMS["VCP_SCORE_MIN"] and len(contractions) >= min_contractions
    info = {"contractions": len(contractions), "pcts": [round(p * 100, 1) for p in pcts],
            "pivot": round(pivot, 2), "current": round(current_price, 2),
            "near_pivot": near_pivot, "vol_dry": vol_dry, "atr_contracting": atr_contracting,
            "base_weeks": round(base_weeks, 1), "base_range": round(base_range * 100, 1),
            "score": score, "reason": "通过" if is_vcp else "分数不足"}
    return is_vcp, info


def fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    sy = sd.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,high,low,close,volume", sd)
            if d is not None and not d.empty:
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=code, period="daily",
                                   start_date=sy, end_date=datetime.now().strftime("%Y%m%d"),
                                   adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume'})
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return None


def _process_one(args):
    code, name = args
    try:
        h = fetch_hist(code)
        if h is None or len(h) < PARAMS["MIN_REQUIRED"]:
            return None
        is_vcp, info = detect_vcp_advanced(h)
        if not is_vcp:
            return None
        time.sleep(PARAMS["SLEEP"])
        return {"代码": code, "名称": name, "行业": "",
                "最新价": info["current"], "收缩次数": info["contractions"],
                "收缩幅度%": info["pcts"], "pivot": info["pivot"],
                "近pivot": info["near_pivot"], "量缩": info["vol_dry"],
                "ATR收敛": info["atr_contracting"], "基地周": info["base_weeks"],
                "基地振幅%": info["base_range"], "VCP分": info["score"],
                "resonance": False, "resonance_sector": ""}
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


def run_scan():
    print("连接 Baostock（行业表 + 列表 + 子进程登录）...")
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
    rows = []; fail = 0
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线算VCP ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="vcp扫描", unit="只")
        for r in pool.imap_unordered(_process_one, tasks):
            if r:
                if "__error__" in r:
                    fail += 1
                else:
                    rows.append(r)
                    pbar.write(f"  VCP {r['代码']} {r['名称']} 收缩{r['收缩次数']}次{r['收缩幅度%']} 分{r['VCP分']} pivot{r['pivot']}")
            pbar.update(1); pbar.set_postfix(命中=len(rows), 失败=fail)
    print(f"扫描完成 命中{len(rows)} 失败{fail}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def enrich(df):
    targets = df.to_dict('records')
    print(f"为 {len(targets)} 只VCP命中补行业 ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=PARAMS["NUM_PROCESSES"]) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"📐 VCP蓄势板块: {cluster or '无'}")
    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in targets:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 VCP遇风口 {cnt} 只 (形态蓄势+板块催化)")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', 'VCP分'], ascending=[False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**📐 VCP波动收缩形态选股** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         "*(多次收缩递减+量缩+贴近pivot+ATR收敛=突破前蓄势; 命中稀缺属正常; 需等放量突破pivot确认, 严格止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📐 **VCP蓄势板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    if not reso.empty:
        L.append(f"### 🎯 VCP遇风口 共{len(reso)}只 (形态蓄势+板块催化)")
        for _, r in reso.iterrows():
            L.append(f"- **{r['名称']}({r['代码']})** [🎯{r['resonance_sector']}] 现价{r['最新价']} 收缩{r['收缩次数']}次{r['收缩幅度%']} 分{r['VCP分']} pivot{r['pivot']}")
        L.append("")
    L.append(f"### 📐 全部VCP 共{len(df)}只")
    for _, r in df.iterrows():
        L.append(f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 现价{r['最新价']} 收缩{r['收缩次数']}次{r['收缩幅度%']} "
                 f"量缩{'✓' if r['量缩'] else '✗'} 近pivot{'✓' if r['近pivot'] else '✗'} ATR收敛{'✓' if r['ATR收敛'] else '✗'} 分{r['VCP分']}")
    return "\n".join(L)


def main():
    print("=" * 70)
    print(f"📐 VCP波动收缩形态选股 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("本次无VCP命中 (VCP条件严格, 全市场命中0只属正常, 非bug)")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"vcp_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"vcp_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/vcp_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📐 VCP形态 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_vcp<<<
