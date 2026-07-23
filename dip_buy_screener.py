# -*- coding: utf-8 -*-
"""
dip_buy_screener.py —— 上升趋势·回调低点买入选股（基于 macd_resonance 的趋势框架）
====================================================================
"低点买入"的正确技术定义 = 回调买入(pullback): 在【已向上的趋势】里, 等价格【缩量回踩
支撑/超卖】时介入。不是接下跌趋势的飞刀(那是左侧抄底, 胜率极低)。

逻辑 = 趋势保护伞(必须, 否则排除) + 回调到位触发(缩量回踩+超卖+止跌):
  趋势向上: 收盘>MA60 且 (周线金叉或DIF>0 或 月线金叉或DIF>0)。不满足 -> 直接排除。
  回调到位: 近30日高回撤[MIN_DD,MAX_DD] + 缩量回踩 + 贴近MA20/MA60 + RSI从强转弱 + 止跌企稳。
  分级: 🟢强买点(缩量+回踩+止跌齐备) / 🟡观察(回调到位但无止跌, 等一等) /
        带⚠️风险(顶背离或放量下跌=可能出货非洗盘)的强买点降为🟡️。

⚠️ 诚实定位(必读): 本脚本提高"买在相对低位"的概率, 【不保证买在绝对最低】, 更不保证上涨。
  回调可能演变成反转下跌, 故趋势保护伞+止损是策略的一部分, 非可选。熊市/震荡市假信号多,
  主要靠"收盘>MA60+周月多头"过滤。所有判断用历史数据, 无前视。

数据源/工程: 复用矩阵加固(双源/超时/行业本地join/软导入推送/存output/多进程/交易日判断)。
  一次拉~6年OHLCV日线, 本地全算(均线/RSI/量能/下影线/回撤/周月MACD), 零额外接口。
  比首红(11年)快, timeout 100 够; 想更快设 SCAN_LIMIT 或 SNAPSHOT_PRE=1。
记号: 🛒=回调买点命中/聚类(矩阵未用过, 零撞色); 🟢🟡=买点分级; ⚠️=风险降级。
====================================================================
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

# ===================== 参数区 =====================
PARAMS = dict(
    FAST=12, SLOW=26, SIGNAL=9,
    LOOKBACK_DAYS=1500,            # ~6年: 够MA60+周月MACD+RSI+量能(比首红短, 更快)
    MA_TREND=60,                   # 趋势生命线
    MA_SHORT=20,
    RSI_P=14,
    # 回调触发阈值
    MIN_DD=5.0, MAX_DD=25.0,       # 近30日高回撤区间(%): 太小非回调, 太大可能破位
    NEAR_MA_PCT=3.0,               # 收盘距MA<此% 视为回踩支撑
    SHRINK_RATIO=0.8,              # 近5日均量<近20日*此 视为缩量(洗盘)
    HEAVY_RATIO=1.3,               # 近5日均量>近20日*此 且下跌 视为放量跌(出货嫌疑)
    RSI_LOW=45, RSI_WAS=55,        # RSI现在<LOW 且 近10日曾>WAS = 从强转弱的回调
    # 入选/分级门槛(累计分, 满分15)
    WEAK_SCORE=6, STRONG_SCORE=10,
    # 初筛
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    SNAPSHOT_PRE=False, AMOUNT_MIN=1.0e8, TURNOVER_MIN=1.0,
    TOP_N=20, NUM_PROCESSES=3, SLEEP=0.3,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))   # 0=全扫; 验逻辑先设1500

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


# ===================== 工具 =====================
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

def _clean(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

def is_trading_day():
    try:
        d = ak.tool_trade_date_hist_sina()
        return datetime.now().strftime('%Y-%m-%d') in set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  交易日历失败, 默认继续: {e}"); return True


# ===================== 取列表 / 快照初筛(可选) =====================
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
    m = (df["代码"].str.startswith(PARAMS["KEEP_PREFIX"]) & (df["最新价"] >= PARAMS["MIN_PRICE"])
         & (df["成交额"] >= PARAMS["AMOUNT_MIN"]) & (df["换手率"] >= PARAMS["TURNOVER_MIN"]))
    print(f"  快照初筛开启: {m.sum()} 只")
    return set(df.loc[m, "代码"])


# ===================== 个股历史(双源+超时, OHLCV) =====================
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
                if len(d) >= 80:
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
                if len(d) >= 80:
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


# ===================== 指标工具 =====================
def _resample_close(daily, rule_list):
    for alias in rule_list:
        try:
            r = daily.set_index("date")["close"].resample(alias).last().dropna()
            return r if len(r) >= 35 else None
        except Exception:
            continue
    return None

def _macd_golden_or_pos(cs):
    """周/月线: 金叉 或 DIF>0 = 多头; None=样本不足"""
    if cs is None or len(cs) < 35:
        return None
    c = cs.astype(float)
    dif = c.ewm(span=PARAMS["FAST"], adjust=False).mean() - c.ewm(span=PARAMS["SLOW"], adjust=False).mean()
    dea = dif.ewm(span=PARAMS["SIGNAL"], adjust=False).mean()
    return bool((dif.iloc[-1] > dea.iloc[-1]) or (dif.iloc[-1] > 0))

def _rsi_series(c, period=14):
    if len(c) < period + 1:
        return None
    delta = c.diff()
    g = delta.where(delta > 0, 0).rolling(period).mean()
    l = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-9))

def _detect_top_div(c, dif_s, half=30):
    if len(c) < 2 * half or dif_s is None or len(dif_s) < 2 * half:
        return False
    cv = c.iloc[-2 * half:].astype(float).values; dv = dif_s.iloc[-2 * half:].astype(float).values
    if np.isnan(cv).any() or np.isnan(dv).any():
        return False
    c1, c2, d1, d2 = cv[:half], cv[half:], dv[:half], dv[half:]
    p1, p2 = int(np.argmax(c1)), int(np.argmax(c2))
    return bool(c2[p2] > c1[p1] * 0.999 and d2[p2] < d1[p1] * 0.98)


# ===================== 回调买点判定(核心) =====================
def eval_dip(daily):
    """返回 dict(命中信息) 或 None(趋势不够/没回调/不入选)。全部历史数据, 无前视。"""
    if len(daily) < 80:
        return None
    c = daily['close'].astype(float); o = daily['open'].astype(float)
    hi = daily['high'].astype(float); lo = daily['low'].astype(float); v = daily['volume'].astype(float)
    ma20 = c.rolling(PARAMS["MA_SHORT"]).mean()
    ma60 = c.rolling(PARAMS["MA_TREND"]).mean()
    if pd.isna(ma60.iloc[-1]):
        return None
    # ---- 趋势保护伞(必须) ----
    trend_up = bool(c.iloc[-1] > ma60.iloc[-1])
    w_bull = _macd_golden_or_pos(_resample_close(daily, ["W-FRI"]))
    m_bull = _macd_golden_or_pos(_resample_close(daily, ["ME", "M"]))
    big_bull = (w_bull is True) or (m_bull is True)
    if not (trend_up and big_bull):
        return None   # 趋势不够 -> 排除(不接飞刀)
    # ---- 回撤 ----
    high30 = hi.iloc[-30:].max()
    dd = (high30 - c.iloc[-1]) / high30 * 100 if high30 else 999
    if not (PARAMS["MIN_DD"] <= dd <= PARAMS["MAX_DD"]):
        return None   # 没回调 或 回调过深
    # ---- 各回调条件(累加分 + 勾选) ----
    score = 0; checks = {}
    # 缩量回踩(洗盘核心)
    vol5 = v.iloc[-5:].mean(); vol20 = v.iloc[-20:].mean()
    shrink = bool(vol5 < vol20 * PARAMS["SHRINK_RATIO"]) if vol20 > 0 else False
    checks["缩量"] = shrink
    if shrink:
        score += 3
    # 回踩支撑(贴近MA20/MA60)
    d20 = abs(c.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] else 999
    d60 = abs(c.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1] * 100
    near20 = bool(d20 <= PARAMS["NEAR_MA_PCT"]); near60 = bool(d60 <= PARAMS["NEAR_MA_PCT"])
    checks["回踩MA20"] = near20; checks["回踩MA60"] = near60
    if near60:
        score += 3
    elif near20:
        score += 2
    # RSI 从强转弱(超卖回调)
    rsi_s = _rsi_series(c, PARAMS["RSI_P"])
    rsi_now = float(rsi_s.iloc[-1]) if rsi_s is not None and pd.notna(rsi_s.iloc[-1]) else None
    rsi_was = float(rsi_s.iloc[-10:].max()) if rsi_s is not None and len(rsi_s) >= 10 else None
    rsi_dip = bool(rsi_now is not None and rsi_was is not None and rsi_now < PARAMS["RSI_LOW"] and rsi_was > PARAMS["RSI_WAS"])
    checks["RSI回落"] = rsi_dip
    if rsi_dip:
        score += 2
    # 止跌企稳(收阳/长下影/收>前收)
    last_red = bool(c.iloc[-1] > o.iloc[-1])
    body = abs(c.iloc[-1] - o.iloc[-1]); lower_shadow = min(o.iloc[-1], c.iloc[-1]) - lo.iloc[-1]
    long_lower = bool(lower_shadow > body * 1.5 and lower_shadow > 0)
    up_prev = bool(c.iloc[-1] > c.iloc[-2])
    stop = last_red or long_lower or up_prev
    checks["止跌"] = stop
    if stop:
        score += 3
    # 回撤合适本身加分
    score += 2
    # ---- 风险标记 ----
    heavy = bool(vol5 > vol20 * PARAMS["HEAVY_RATIO"] and c.iloc[-1] < c.iloc[-5]) if vol20 > 0 else False
    checks["放量跌"] = heavy
    # 顶背离(日线)
    dif_s = None
    cs = daily.set_index("date")["close"].astype(float)
    dif_s = cs.ewm(span=PARAMS["FAST"], adjust=False).mean() - cs.ewm(span=PARAMS["SLOW"], adjust=False).mean()
    topdiv = _detect_top_div(cs, dif_s)
    checks["顶背离"] = topdiv
    risk = bool(heavy or topdiv)
    # ---- 分级 ----
    grade = None
    if score >= PARAMS["STRONG_SCORE"] and stop and shrink:
        grade = "🟢强买点"
    elif score >= PARAMS["WEAK_SCORE"]:
        grade = "🟡观察"
    if grade is None:
        return None
    if risk and grade == "🟢强买点":
        grade = "🟡️风险降级"   # 顶背离/放量跌 -> 强买点降级, 诚实
    vol_ratio = round(vol5 / vol20, 2) if vol20 > 0 else None
    return {"回撤%": round(dd, 1), "距MA20%": round(d20, 1), "距MA60%": round(d60, 1),
            "缩量比": vol_ratio, "RSI": round(rsi_now, 1) if rsi_now is not None else None,
            "周多头": w_bull, "月多头": m_bull, "收盘>MA60": True,
            "checks": checks, "score": score, "grade": grade, "risk": risk,
            "reason": " | ".join([f"{k}{'✓' if v else '✗'}" for k, v in checks.items()])}


# ===================== 单只处理(子进程) =====================
def _process_one(args):
    code, name = args
    try:
        h = fetch_hist(code)
        if h.empty:
            return None
        r = eval_dip(h)
        if r is None:
            return None
        time.sleep(PARAMS["SLEEP"])
        return {"代码": code, "名称": name, "行业": "", "最新价": round(float(h['close'].iloc[-1]), 2),
                "买点": r["grade"], "分": r["score"], "风险": r["risk"],
                "回撤%": r["回撤%"], "距MA20%": r["距MA20%"], "距MA60%": r["距MA60%"],
                "缩量比": r["缩量比"], "RSI": r["RSI"],
                "周": r["周多头"], "月": r["月多头"], "条件": r["reason"]}
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


# ===================== 主扫描 =====================
def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业映射 + 子进程登录）...")
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                for _, r in ind.iterrows():
                    _INDUSTRY_MAP[r['code']] = _clean(r.get('industry', ''))
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
        codes = [c for c in codes if c in snap]
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    rows = []; fail = 0; g_n = 0; o_n = 0
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线, 判上升趋势回调买点 ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=lambda: _bs_login_ok()) as pool:
        pbar = tqdm(total=len(tasks), desc="dip-buy", unit="只")
        for r in pool.imap_unordered(_process_one, tasks):
            if r:
                if "__error__" in r:
                    fail += 1
                else:
                    rows.append(r)
                    if "🟢" in r["买点"]:
                        g_n += 1
                    else:
                        o_n += 1
                    pbar.write(f"  🛒 {r['代码']} {r['名称']} {r['买点']} 回撤{r['回撤%']}% 缩量比{r['缩量比']} RSI{r['RSI']}")
            pbar.update(1); pbar.set_postfix(强=g_n, 观察=o_n, 失败=fail)
    print(f"扫描完成 强买点{g_n} 观察{o_n} 失败{fail}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===================== 行业join + 聚类 =====================
def enrich(df):
    for _, r in df.iterrows():
        df.loc[df["代码"] == r["代码"], "行业"] = _INDUSTRY_MAP.get(r["代码"], "—")
    lab = df[df["行业"].isin([x for x in df["行业"] if x not in ("—", "")])]
    cluster = [(n, int(c)) for n, c in lab["行业"].value_counts().head(CLUSTER_TOP).items()] if not lab.empty else []
    print(f"🛒 回调买点板块: {cluster or '无'}")
    return df, cluster


# ===================== 推送 =====================
def build_push(df, cluster):
    P = PUSH_TOP
    strong = df[df["买点"].str.contains("🟢")].sort_values("分", ascending=False)
    watch = df[~df["买点"].str.contains("🟢")].sort_values("分", ascending=False)
    L = [f"**🛒 上升趋势·回调低点买入** | 🟢强买点{len(strong)} 🟡观察{len(watch)}",
         "*(趋势向上+缩量回踩超卖; 非买绝对最低; 回调或转跌, 必设止损; 熊市假信号靠趋势伞过滤)*", ""]
    if cluster:
        L.append("🛒 **回调买点扎堆板块**: " + "、".join(f"{n}({c})" for n, c in cluster))
        L.append("")
    def line(r):
        rk = " ⚠️" if r["风险"] else ""
        return (f"- {r['买点']} **{r['名称']}({r['代码']})** [{r['行业']}] 现价{r['最新价']} 回撤{r['回撤%']}% "
                f"缩量比{r['缩量比']} RSI{r['RSI']} 周{'✓' if r['周'] else '✗'} 月{'✓' if r['月'] else '✗'}{rk}")
    L.append(f"### 🟢 强买点 Top{min(len(strong), P)} (缩量+回踩+止跌齐备)")
    L += [line(r) for _, r in strong.head(P).iterrows()] or ["今日无（齐备条件稀缺, 正常）"]
    L.append("")
    L.append(f"### 🟡 观察 Top{min(len(watch), P)} (回调到位但缺止跌/缩量, 等企稳信号再动)")
    L += [line(r) for _, r in watch.head(P).iterrows()] or ["无"]
    L.append("")
    L.append("*条件勾选见 csv 的'条件'列(缩量✓/回踩MA60✓/RSI回落✓/止跌✓/放量跌✗/顶背离✗)。*")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"🛒 上升趋势·回调低点买入 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 趋势伞=收盘>MA60+周/月多头; 回调=缩量回踩超卖+止跌")
    print("=" * 70)
    if not is_trading_day():
        print("非交易日, 跳过"); sys.exit(0)
    df = run_scan()
    if df is None or df.empty:
        print("本次无回调买点命中(趋势伞+回调门槛较严, 属正常)"); sys.exit(0)
    df, cluster = enrich(df)
    df = df.sort_values(["买点", "分"], ascending=[True, False]).reset_index(drop=True)
    tag = datetime.now().strftime("%Y%m%d")
    df.to_csv(os.path.join(OUTPUT_DIR, f"dip_buy_{tag}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, f"dip_buy_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump({"date": tag, "params": PARAMS, "cluster": cluster,
                   "n_strong": int(df["买点"].str.contains("🟢").sum()), "n_watch": int(~df["买点"].str.contains("🟢")).sum(),
                   "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 已存 output/dip_buy_{tag}.*")
    disp = df.copy(); disp.insert(2, "板块", disp["行业"])
    print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    if SERVERCHAN_KEY:
        send_serverchan(f"🛒 回调低点买入 | 🟢{int(df['买点'].str.contains('🟢').sum())} 🟡{int((~df['买点'].str.contains('🟢')).sum())}",
                        build_push(df, cluster))
