# -*- coding: utf-8 -*-
"""
first_red_macd_combo.py —— 520首红 × 多周期MACD共振 结合选股
首红=左侧触发, 共振=右侧体检; 不做硬交集(首红当天周/月多死叉, 物理稀缺)。
对每只首红票做共振体检 -> 反转确认度分级 🟢强/弱/⚪纯左侧; 另附纯共振榜;
极少数"首红+五周期全共振"=🎯真双确认单独置顶(稀缺, 有则显示, 无则明说)。
一次拉~11年OHLCV日线(双源+超时), 本地同时算首红+共振, 零额外接口。
"""
import os
import re
import sys
import json
import time
import random
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

PARAMS = dict(
    LOW_WINDOW=520, VOL_WINDOW=20, NEW_LOW_TOLERANCE=1.02, SIGNAL_FRESH_DAYS=7,
    FAST=12, SLOW=26, SIGNAL=9, MIN_REQUIRED=35,
    MIN_SCORE=18.0, MIN_GOLDEN=3, NO_DEAD_IN_MAJOR=True,
    RSI_HEALTH=(30, 70), RSI_OVERHEAT=75, DIV_HALF=30,
    CONFIRM_STRONG=8, CONFIRM_WEAK=3,
    LOOKBACK_DAYS=4000,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    SNAPSHOT_PRE=False, AMOUNT_MIN=1.0e8, TURNOVER_MIN=1.0,
    TOP_N=20, NUM_PROCESSES=3, SLEEP=0.3,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
PERIOD_WEIGHT = {"daily": 8, "weekly": 12, "monthly": 15, "quarterly": 12, "yearly": 8}
PERIOD_NAME = {"daily": "日", "weekly": "周", "monthly": "月", "quarterly": "季", "yearly": "年"}
PERIOD_ORDER = ["daily", "weekly", "monthly", "quarterly", "yearly"]
RSI_PERIODS = {"daily", "weekly", "monthly"}
RESAMPLE_ALIAS = {"weekly": ["W-FRI"], "monthly": ["ME", "M"],
                  "quarterly": ["QE-DEC", "Q-DEC", "Q"], "yearly": ["YE-DEC", "Y-DEC", "Y", "A-DEC", "A"]}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '12'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))
if os.environ.get('SNAPSHOT_PRE', '').strip() in ('1', 'true', 'True'):
    PARAMS["SNAPSHOT_PRE"] = True
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}


def _pref(c6):
    c = str(c6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)

def _bs_login_ok(retries=5):
    global _BS_LOGGED
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

def _bs_q(code, fields, sd, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)

def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content); print("📲 推送成功"); return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(f"https://sctapi.ftqq.com/{key}.send",
                             data={"title": title, "desp": content}, timeout=10).json().get("code") == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False

def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

def is_trading_day():
    try:
        d = ak.tool_trade_date_hist_sina()
        return datetime.now().strftime('%Y-%m-%d') in set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  交易日历失败, 默认继续: {e}"); return True


def _fetch_list_akshare():
    for attempt in range(3):
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                nc = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', nc]].copy(); d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'; d['status'] = '1'; return d
        except Exception as e:
            print(f"  akshare列表第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])

def snapshot_codes():
    if not PARAMS["SNAPSHOT_PRE"]:
        return None
    for i in range(3):
        try:
            df = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=AK_TIMEOUT)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  快照第{i+1}次失败: {e}")
        time.sleep(2 + i)
    else:
        return None
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    for c in ['最新价', '成交额', '换手率']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    m = (df["代码"].str.startswith(PARAMS["KEEP_PREFIX"])
         & (df["最新价"] >= PARAMS["MIN_PRICE"])
         & (df["成交额"] >= PARAMS["AMOUNT_MIN"]) & (df["换手率"] >= PARAMS["TURNOVER_MIN"]))
    print(f"  快照初筛开启: {m.sum()} 只进入精算(注意: 可能漏低位首红)")
    return set(df.loc[m, "代码"])


def fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    sy = sd.replace("-", "")
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume", sd)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 60:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=code, period="daily",
                                   start_date=sy, end_date=datetime.now().strftime("%Y%m%d"),
                                   adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 60:
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


def detect_first_red(df):
    if len(df) < PARAMS["LOW_WINDOW"]:
        return pd.DataFrame()
    d = df.copy()
    for c in ['close', 'open', 'low', 'volume']:
        d[c] = d[c].astype(float)
    d['520_low'] = d['low'].rolling(PARAMS["LOW_WINDOW"]).min()
    d['avg_vol'] = d['volume'].rolling(PARAMS["VOL_WINDOW"]).mean()
    d['is_red'] = d['close'] > d['open']
    d['new_low'] = d['low'] <= d['520_low'].shift(1) * PARAMS["NEW_LOW_TOLERANCE"]
    sigs = []
    in_zone = False
    for i in range(PARAMS["LOW_WINDOW"], len(d)):
        if d['new_low'].iloc[i]:
            in_zone = True
        if in_zone and d['is_red'].iloc[i]:
            vr = d['volume'].iloc[i] / d['avg_vol'].iloc[i] if d['avg_vol'].iloc[i] > 0 else 0
            dp = (d['close'].iloc[i] - d['520_low'].iloc[i]) / d['520_low'].iloc[i] * 100
            sigs.append({'date': d['date'].iloc[i], 'close': d['close'].iloc[i],
                         'vol_ratio': round(vr, 2), 'dist520': round(dp, 2)})
            in_zone = False
    return pd.DataFrame(sigs)


def _resample_close(daily, period):
    if period == "daily":
        return daily.set_index("date")["close"]
    for alias in RESAMPLE_ALIAS[period]:
        try:
            r = daily.set_index("date")["close"].resample(alias).last().dropna()
            return r if len(r) >= 5 else None
        except Exception:
            continue
    return None

def _macd_full(cs):
    if cs is None or len(cs) < PARAMS["MIN_REQUIRED"]:
        return None, None
    c = cs.astype(float)
    dif = c.ewm(span=PARAMS["FAST"], adjust=False).mean() - c.ewm(span=PARAMS["SLOW"], adjust=False).mean()
    return dif, dif.ewm(span=PARAMS["SIGNAL"], adjust=False).mean()

def _rsi_last(cs, period=14):
    if cs is None or len(cs) < period + 1:
        return None
    c = cs.astype(float); delta = c.diff()
    g = delta.where(delta > 0, 0).rolling(period).mean()
    l = (-delta.where(delta < 0, 0)).rolling(period).mean()
    v = (100 - 100 / (1 + g / l.replace(0, 1e-9))).iloc[-1]
    return None if pd.isna(v) else round(float(v), 1)

def _divergence(cs, dif_s, half=None):
    half = half or PARAMS["DIV_HALF"]
    if cs is None or dif_s is None or len(cs) < 2 * half:
        return "无"
    c = cs.iloc[-2 * half:].astype(float).values; dd = dif_s.iloc[-2 * half:].astype(float).values
    if np.isnan(c).any() or np.isnan(dd).any():
        return "无"
    c1, c2, d1, d2 = c[:half], c[half:], dd[:half], dd[half:]
    p1, p2 = int(np.argmax(c1)), int(np.argmax(c2))
    if c2[p2] > c1[p1] * 0.999 and d2[p2] < d1[p1] * 0.98:
        return "顶背离⚠️"
    p1, p2 = int(np.argmin(c1)), int(np.argmin(c2))
    if c2[p2] < c1[p1] * 1.001 and d2[p2] > d1[p1] * 1.02:
        return "底背离🔥"
    return "无"

def score_resonance(daily):
    score = 0.0; detail = {"rsi": {}}; golden_n = 0; major_dead = False
    for period, w in PERIOD_WEIGHT.items():
        cs = _resample_close(daily, period)
        dif_s, dea_s = _macd_full(cs)
        if dif_s is None:
            detail[period] = {"status": "样本不足"}; continue
        dif, dea = float(dif_s.iloc[-1]), float(dea_s.iloc[-1])
        dif_p, dea_p = float(dif_s.iloc[-2]), float(dea_s.iloc[-2])
        golden = dif > dea; just_g = (dif_p <= dea_p) and golden; up0 = dif > 0; dif_up = dif > dif_p
        if golden:
            golden_n += 1
        if period in ("weekly", "monthly") and not golden:
            major_dead = True
        ps = (0.6 if golden else -0.6) + (0.4 if just_g else 0.0) + (0.3 if up0 else -0.3)
        rsi = _rsi_last(cs) if period in RSI_PERIODS else None
        if period in RSI_PERIODS:
            detail["rsi"][period] = rsi
        if rsi is not None:
            if period == "daily":
                if PARAMS["RSI_HEALTH"][0] <= rsi <= PARAMS["RSI_HEALTH"][1]:
                    ps += 3
                elif rsi > PARAMS["RSI_OVERHEAT"]:
                    ps -= 3
            elif period == "weekly" and 40 <= rsi <= 70:
                ps += 2
            elif period == "monthly" and rsi > 50:
                ps += 2
        score += ps * w
        detail[period] = {"dif": round(dif, 3), "dea": round(dea, 3), "golden": bool(golden),
                          "dif_up": bool(dif_up), "above_zero": bool(up0)}
        if period == "daily":
            detail["divergence"] = _divergence(cs, dif_s)
    return round(score, 1), detail, golden_n, major_dead


def confirm_level(detail):
    c = 0
    dd = detail.get("daily", {})
    if isinstance(dd, dict):
        if dd.get("golden"):
            c += 3
        elif dd.get("dif_up"):
            c += 2
    if isinstance(detail.get("weekly", {}), dict) and detail["weekly"].get("golden"):
        c += 4
    if isinstance(detail.get("monthly", {}), dict) and detail["monthly"].get("golden"):
        c += 5
    rsi_d = detail.get("rsi", {}).get("daily")
    if rsi_d is not None and 30 <= rsi_d <= 60:
        c += 2
    if "顶背离" in detail.get("divergence", ""):
        c -= 3
    if c >= PARAMS["CONFIRM_STRONG"]:
        return c, "🟢强确认"
    if c >= PARAMS["CONFIRM_WEAK"]:
        return c, "🟡弱确认"
    return c, "⚪纯左侧"


def _process_one(args):
    code, name = args
    try:
        h = fetch_hist(code)
        if h.empty:
            return None
        sigs = detect_first_red(h)
        is_fr = False; fr = {}
        if not sigs.empty:
            last = sigs.iloc[-1]
            if (pd.to_datetime(h['date'].iloc[-1]) - pd.to_datetime(last['date'])).days <= PARAMS["SIGNAL_FRESH_DAYS"]:
                is_fr = True
                fr = {"信号日期": str(last['date'])[:10], "量比": last['vol_ratio'], "距520低%": last['dist520']}
        score, detail, golden_n, major_dead = score_resonance(h)
        is_res = (score >= PARAMS["MIN_SCORE"] and golden_n >= PARAMS["MIN_GOLDEN"]
                  and (not PARAMS["NO_DEAD_IN_MAJOR"] or not major_dead))
        if not (is_fr or is_res):
            return None
        conf, conf_tag = confirm_level(detail)
        g = lambda p: detail.get(p, {}).get("golden") if isinstance(detail.get(p, {}), dict) else None
        return {"代码": code, "名称": name, "行业": "",
                "首红": bool(is_fr), "共振": bool(is_res), "双确认": bool(is_fr and is_res),
                "信号日期": fr.get("信号日期", ""), "量比": fr.get("量比"), "距520低%": fr.get("距520低%"),
                "共振分": score, "金叉数": golden_n,
                "确认度": conf, "确认分级": conf_tag,
                "日金叉": g("daily"), "周金叉": g("weekly"), "月金叉": g("monthly"),
                "日RSI": detail.get("rsi", {}).get("daily"),
                "背离": detail.get("divergence", "无"),
                "风险": bool(("顶背离" in detail.get("divergence", "")) or
                            (detail.get("rsi", {}).get("daily") is not None and detail["rsi"]["daily"] > PARAMS["RSI_OVERHEAT"])),
                "_detail": detail}
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业映射 + 子进程登录）...")
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                for _, r in ind.iterrows():
                    _INDUSTRY_MAP[r['code']] = _clean_industry(r.get('industry', ''))
                print(f"  行业映射 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        bs.logout()

    print("取股票列表...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception:
            stock_df = pd.DataFrame()
        bs.logout()
    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        stock_df = _fetch_list_akshare()
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表"); return pd.DataFrame()
    stock_df = stock_df[stock_df['code'].str.startswith(('sh.', 'sz.')) & (stock_df['type'] == '1') & (stock_df['status'] == '1')].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    codes = stock_df['code'].tolist()
    snap = snapshot_codes()
    if snap is not None:
        codes = [c for c in codes if c[3:] in snap or c in snap]
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail = 0
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线, 同时算首红+共振 ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=lambda: _bs_login_ok()) as pool:
        pbar = tqdm(total=len(tasks), desc="combo扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail += 1
                else:
                    results.append(res)
                    tag = "🎯双" if res["双确认"] else ("🔻首红" if res["首红"] else "📡共振")
                    pbar.write(f"  {tag} {res['代码']} {res['名称']} {res.get('确认分级','')} 共振分{res['共振分']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail)
    print(f"扫描完成 命中{len(results)} 失败{fail}")
    return pd.DataFrame(results) if results else pd.DataFrame()


def enrich(df):
    for _, r in df.iterrows():
        df.loc[df["代码"] == r["代码"], "行业"] = _INDUSTRY_MAP.get(r["代码"], "—")
    def cluster(sub):
        lab = sub[sub["行业"].isin([x for x in sub["行业"] if x not in ("—", "未知", "")])]
        if lab.empty:
            return []
        return [(n, int(c)) for n, c in lab["行业"].value_counts().head(CLUSTER_TOP).items()]
    fr_df = df[df["首红"] == True]
    res_df = df[df["共振"] == True]
    return df, cluster(fr_df), cluster(res_df)


def build_push(df, fr_cl, res_cl):
    P = PUSH_TOP
    dual = df[df["双确认"] == True].sort_values("共振分", ascending=False)
    fr = df[(df["首红"] == True) & (df["双确认"] == False)].sort_values("确认度", ascending=False)
    reso = df[(df["共振"] == True) & (df["首红"] == False)].sort_values("共振分", ascending=False)
    L = [f"**520首红 × 多周期共振 结合** | 首红{len(fr)+len(dual)} 共振{len(reso)+len(dual)} 🎯双确认{len(dual)}",
         "*(首红=左侧触发, 共振=右侧体检; 双确认稀缺属正常, 非bug)*", ""]
    if fr_cl:
        L.append("🔻 **首红见底板块**: " + "、".join(f"{n}({c})" for n, c in fr_cl))
    if res_cl:
        L.append("📡 **共振趋势板块**: " + "、".join(f"{n}({c})" for n, c in res_cl))
    L.append("")
    L.append(f"### 🎯 真·双确认 Top{min(len(dual), P)} (首红+五周期全共振, 置信天花板)")
    if dual.empty:
        L.append("今日无（首红当天周/月多死叉, 物理上极稀缺, 正常）")
    for _, r in dual.head(P).iterrows():
        L.append(f"- **{r['名称']}({r['代码']})** [{r['行业']}] {r['确认分级']} 共振分{r['共振分']} | 量比{r['量比']} 距520低{r['距520低%']}%")
    L.append("")
    L.append(f"### 🔻 首红·反转体检 Top{min(len(fr), P)} (左侧触发+共振分级)")
    for _, r in fr.head(P).iterrows():
        cyc = f"日{'✓' if r['日金叉'] else '✗'} 周{'✓' if r['周金叉'] else '✗'} 月{'✓' if r['月金叉'] else '✗'}"
        rk = " ⚠️" if r["风险"] else ""
        L.append(f"- {r['确认分级']} **{r['名称']}({r['代码']})** [{r['行业']}] {r['信号日期']} 量比{r['量比']} 距520低{r['距520低%']}% | {cyc} 日RSI{r['日RSI']} {r['背离']}{rk}")
    if len(fr) > P:
        L.append(f"  *…另有{len(fr)-P}只首红, 见output*")
    L.append("")
    L.append(f"### 📡 纯共振(右侧趋势, 无首红) Top{min(len(reso), 6)}")
    for _, r in reso.head(6).iterrows():
        L.append(f"- {r['名称']}({r['代码']}) [{r['行业']}] 共振分{r['共振分']} 金叉{r['金叉数']} 日RSI{r['日RSI']}")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"520首红 × 多周期共振 结合 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'} 快照初筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关(保首红)'}")
    print("=" * 70)
    if not is_trading_day():
        print("非交易日, 跳过"); sys.exit(0)
    df = run_scan()
    if df is not None and not df.empty:
        df, fr_cl, res_cl = enrich(df)
        tag = datetime.now().strftime("%Y%m%d")
        fr_all = df[df["首红"] == True].sort_values("确认度", ascending=False).drop(columns=["_detail"], errors="ignore")
        fr_all.to_csv(os.path.join(OUTPUT_DIR, f"combo_first_red_{tag}.csv"), index=False, encoding="utf-8-sig")
        reso_all = df[(df["共振"] == True) & (df["首红"] == False)].sort_values("共振分", ascending=False).drop(columns=["_detail"], errors="ignore")
        if not reso_all.empty:
            reso_all.to_csv(os.path.join(OUTPUT_DIR, f"combo_resonance_only_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"combo_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "fr_cluster": fr_cl, "res_cluster": res_cl,
                       "n_first_red": int(df["首红"].sum()), "n_resonance": int(df["共振"].sum()),
                       "n_dual": int(df["双确认"].sum()), "hits": df.drop(columns=["_detail"], errors="ignore").to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/combo_*_{tag}.*")
        disp = df.drop(columns=["_detail"], errors="ignore").copy()
        disp.insert(2, "板块", disp["行业"])
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
        if SERVERCHAN_KEY:
            send_serverchan(f"🔻首红×📡共振 结合 | 🎯双{int(df['双确认'].sum())} 首红{int(df['首红'].sum())} 共振{int(df['共振'].sum())}",
                            build_push(df, fr_cl, res_cl))
    else:
        print("本次无首红也无共振命中")
