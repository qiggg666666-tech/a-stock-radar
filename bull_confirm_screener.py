# -*- coding: utf-8 -*-
"""
bull_confirm_screener.py —— 多指标共振翻多·右侧确认 全市场选股 · 矩阵规格
定位(与strong/mtf不重复): 抓"多头确认刚发生"=多指标同时翻多拐点。
【本版】不拦交易日; 推送全列+超长自动分页; 行业=东财优先+baostock国标本地兜底
  (命中再多enrich也秒级, 不卡东财, 解决周末限流补成全'—'或卡死超时)。
【矩阵规格】双源+硬超时; baostock多进程命门已修; 快照预筛; 风口共振🎯; 收尾防护。
⚠️ 右侧确认≠买入保证; 翻多后仍可能假突破, 需结合量能/止损。
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
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

PARAMS = dict(
    LOOKBACK_DAYS=500, MIN_REQUIRED=200,
    SCORE_MIN=6.0, EVENT_MIN=1, STRONG_MIN=8.0,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    PRE_AMOUNT_MIN=1.0e8, PRE_TURNOVER_MIN=1.0, NUM_PROCESSES=3, SLEEP=0.3,
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
_INDUSTRY_MAP = {}   # baostock国标行业映射, 东财补不上时兜底(本地join, 命中再多不卡)


def _send_one(title, content, key):
    """严格检查返回, 失败/超额不再误报成功"""
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
    if len(chunks) > 1:
        print(f"📲 共发送{len(chunks)}条(全发分页) {'✅全部成功' if ok else '⚠️存在失败(查额度/限流)'}")
    else:
        print("📲 推送成功 ✅" if ok else "⚠️ 推送返回失败(查Server酱额度/限流/微信端)")
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


def _bs_q(code, fields, sd, ed, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed, adjustflag="2").get_data()
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


def _clean_industry(s):
    """清洗 baostock 国标行业名: 去掉 'C39 ' 字母+数字前缀"""
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


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
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (价≥{PARAMS['MIN_PRICE']}+活跃)")
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


def fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,close,volume", sd, ed)
            if d is not None and not d.empty:
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=code, period="daily",
                                   start_date=sy, end_date=ey, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return None


def _f(v):
    return None if (v is None or pd.isna(v)) else float(v)


def compute_signal(df):
    if df is None or len(df) < PARAMS["MIN_REQUIRED"]:
        return None
    d = df.copy()
    d['close'] = d['close'].astype(float)
    d['MA50'] = d['close'].rolling(50).mean()
    d['MA200'] = d['close'].rolling(200).mean()
    e12 = d['close'].ewm(span=12, adjust=False).mean()
    e26 = d['close'].ewm(span=26, adjust=False).mean()
    d['DIF'] = e12 - e26
    d['DEA'] = d['DIF'].ewm(span=9, adjust=False).mean()
    d['MACD'] = (d['DIF'] - d['DEA']) * 2
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    ag = gain.rolling(14).mean(); al = loss.rolling(14).mean()
    d['RSI'] = 100 - 100 / (1 + ag / al.replace(0, 1e-9))
    d['VOL_MA5'] = d['volume'].rolling(5).mean()
    d['量比'] = d['volume'] / d['VOL_MA5']
    L = d.iloc[-1]; P = d.iloc[-2]
    close = _f(L['close']); ma50 = _f(L['MA50']); ma200 = _f(L['MA200'])
    dif = _f(L['DIF']); dea = _f(L['DEA']); macd = _f(L['MACD']); rsi = _f(L['RSI']); vr = _f(L['量比'])
    if close is None or ma200 is None:
        return None
    p_close = _f(P['close']); p_ma50 = _f(P['MA50']); p_ma200 = _f(P['MA200'])
    p_dif = _f(P['DIF']); p_dea = _f(P['DEA']); p_macd = _f(P['MACD']); p_rsi = _f(P['RSI'])
    above200 = close > ma200
    ma50_above = (ma50 is not None and ma50 > ma200)
    dif_pos = (dif is not None and dif > 0)
    rsi_above50 = (rsi is not None and rsi > 50)
    just_break200 = (p_close is not None and p_ma200 is not None and p_close <= p_ma200 and close > ma200)
    ma_golden = (p_ma50 is not None and p_ma200 is not None and ma50 is not None and p_ma50 <= p_ma200 and ma50 > ma200)
    macd_golden = (p_dif is not None and p_dea is not None and dif is not None and dea is not None and p_dif <= p_dea and dif > dea)
    hist_pos = (p_macd is not None and macd is not None and p_macd <= 0 and macd > 0)
    rsi_cross50 = (p_rsi is not None and rsi is not None and p_rsi <= 50 and rsi > 50)
    vol_up = (vr is not None and vr >= 1.5)
    rsi_os = (rsi is not None and rsi < 30)
    rsi_ob = (rsi is not None and rsi > 70)
    score = 0.0; n_events = 0
    if just_break200: score += 3; n_events += 1
    if ma_golden:     score += 2; n_events += 1
    if macd_golden:   score += 2; n_events += 1
    if hist_pos:      score += 2; n_events += 1
    if rsi_cross50:   score += 2; n_events += 1
    if above200:      score += 1
    if ma50_above:    score += 1
    if dif_pos:       score += 1
    if rsi_above50:   score += 1
    if vol_up and above200: score += 1
    if score < PARAMS["SCORE_MIN"] or n_events < PARAMS["EVENT_MIN"]:
        return None
    level = "强多确认" if score >= PARAMS["STRONG_MIN"] else ("偏多确认" if score >= PARAMS["SCORE_MIN"] else "中性")
    tags = []
    if just_break200: tags.append("🔥突破200")
    if ma_golden:     tags.append("均线金叉")
    if macd_golden:   tags.append("MACD金叉")
    if hist_pos:      tags.append("MACD转正")
    if rsi_cross50:   tags.append("RSI上穿50")
    if above200:      tags.append("站上200")
    if vol_up:        tags.append(f"放量{vr:.1f}")
    if rsi_os:        tags.append("RSI超卖")
    if rsi_ob:        tags.append("RSI超买")
    return {"代码": None, "名称": None, "行业": "", "最新价": round(close, 2),
            "MA50": round(ma50, 2) if ma50 is not None else None, "MA200": round(ma200, 2),
            "DIF": round(dif, 4) if dif is not None else None, "DEA": round(dea, 4) if dea is not None else None,
            "MACD柱": round(macd, 4) if macd is not None else None, "RSI": round(rsi, 2) if rsi is not None else None,
            "量比": round(vr, 2) if vr is not None else None,
            "多头得分": round(score, 1), "综合倾向": level, "拐点数": n_events, "tags": tags,
            "resonance": False, "resonance_sector": ""}


def _process_one(args):
    code, name = args
    try:
        h = fetch_hist(code)
        r = compute_signal(h)
        if r is None:
            return None
        time.sleep(PARAMS["SLEEP"])
        r["代码"] = code; r["名称"] = name
        return r
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业表 + 列表 + 子进程登录）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns and 'industry' in ind.columns:
                for _, row in ind.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock国标, 东财补不上时兜底)")
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
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线算翻多确认 ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="bull扫描", unit="只")
        for r in pool.imap_unordered(_process_one, tasks):
            if r:
                if "__error__" in r:
                    fail += 1
                else:
                    rows.append(r)
                    pbar.write(f"  🚀 {r['代码']} {r['名称']} {r['综合倾向']} 分{r['多头得分']} 拐点{r['拐点数']} {'/'.join(r['tags'][:3])}")
            pbar.update(1); pbar.set_postfix(命中=len(rows), 失败=fail)
    print(f"扫描完成 命中{len(rows)} 失败{fail}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def enrich(df):
    """东财行业优先(可读+匹配风口🎯) + baostock国标本地兜底(东财限流补不上的, 保证每只都有板块且不卡)"""
    targets = df.to_dict('records')
    print(f"为 {len(targets)} 只命中补行业 (东财优先, baostock兜底) ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=PARAMS["NUM_PROCESSES"]) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业(东财)", unit="只"))
    bs_filled = 0
    for r in targets:
        if not r.get('行业') or r['行业'] in ('—', '未知', ''):
            ind = _INDUSTRY_MAP.get(r['代码'], '')
            if ind and ind not in ('—', '未知', ''):
                r['行业'] = ind; bs_filled += 1
    has = sum(1 for r in targets if r.get('行业') not in ('—', '未知', '', None))
    print(f"🏷️ 行业标注: {has}/{len(targets)} 只有板块 (其中baostock兜底{bs_filled}只)")
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🚀 翻多确认板块: {cluster or '无'}")
    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    cnt = 0
    for r in targets:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 翻多遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '多头得分'], ascending=[False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    strong = df[df['综合倾向'] == '强多确认']
    L = [f"**🚀 多指标共振翻多·右侧确认** | 命中{len(df)}只 🎯风口{len(reso)} 强多{len(strong)} (全发)",
         "*(抓'多头确认刚发生': 多指标同时翻多拐点; 与strong已强/mtf将强不重叠; 右侧确认≠买入保证, 防假突破)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🚀 **翻多确认板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 现价{r['最新价']} 分{r['多头得分']} "
                f"RSI{r['RSI']} 量比{r['量比']} | {'·'.join(r['tags']) or '-'}")
    if not reso.empty:
        L.append(f"### 🎯 翻多遇风口 共{len(reso)}只 (翻多确认+板块催化)")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    L.append(f"### 🚀 强多确认 共{len(strong)}只 (拐点事件多+得分高)")
    L += [line(r) for _, r in strong.iterrows()] or ["今日无强多确认"]
    L.append("")
    rest = df[df['综合倾向'] != '强多确认']
    if not rest.empty:
        L.append(f"### 📋 偏多确认 共{len(rest)}只")
        L += [line(r) for _, r in rest.iterrows()]
    return "\n".join(L)


def main():
    print("=" * 70)
    print(f"🚀 多指标共振翻多·右侧确认 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 门槛 得分≥{PARAMS['SCORE_MIN']}+拐点≥{PARAMS['EVENT_MIN']}; 不拦交易日; 行业东财+baostock兜底; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("本次无翻多确认命中 (需多指标同时翻多拐点, 属正常)")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/bull_confirm_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp['信号'] = disp['tags'].apply(lambda t: '·'.join(t[:4]))
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'tags',
                                  'MA50', 'MA200', 'DIF', 'DEA', 'MACD柱'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            n_strong = int((df['综合倾向'] == '强多确认').sum())
            send_serverchan(f"🚀 翻多确认 命中{len(df)}只 🎯风口{n_reso} 强多{n_strong}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_bull2<<<
