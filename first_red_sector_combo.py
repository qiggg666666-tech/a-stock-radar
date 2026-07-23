# -*- coding: utf-8 -*-
"""
first_red_sector_combo.py —— 520首红 × 板块(动量估值+轮动) 结合选股
====================================================================
把"520首红"(个股年线见底, 左侧) 与 板块状态(动量估值象限 + 轮动多空/接力, 外部确认) 结合:
  一次全市场扫描, 每只票本地同时算 ①首红 ②动量/估值(供板块聚合) ③站MA20/短动量(供轮动聚合);
  扫完按【baostock国标行业】groupby 内联出板块榜(象限+多空+强度+接力), 再把每只首红票
  挂到它所属行业的板块标签上 -> 交叉分级 🌟共振/🟡追高/⚠️逆风/⚪中性/❔缺失。

⚠️ 工程决策(为何内联重算板块榜, 而非读 sector_momentum/sector_rotation 的产物):
  GitHub Actions 每个 job 独立虚拟机, 文件不共享, 跨 job 读 json 需 artifact+依赖+时序, 脆;
  内联重算保证口径一致(同为国标行业)+自包含+一次扫描。代价: 与那两个板块脚本各自重复扫
  全市场(cron 错峰, 不影响正确性, 仅多耗免费额度)。

⚠️ 诚实定位: 首红=左侧触发, 板块=外部确认, 均非预测。⚠️逆风首红常见(年线新低票所在行业
  本易处看空区), 非bug, 正是结合要揭示的"板块未确认"风险。🌟共振首红=个股+板块双击, 稀缺。
  估值分位仅 baostock 路径有(东财兜底无历史PE/PB, 该票分位None, 板块均值skipna, 诚实)。

数据源/工程: 复用矩阵加固(双源/超时/行业本地join/软导入推送/存output/交易日判断/多进程)。
  字段一次拉 date,open,high,low,close,volume,peTTM,pbMRQ(东财兜底无pe/pb->估值降级)。
记号: 板块象限🟢🟡🔵⚪⚫ + 多空🐂🐻⚪ 与 sector_momentum/sector_rotation 推送【同语义同记号】,
  便于跨推送对照; 交叉分级🌟/🟡/⚠️/⚪/❔ 为本脚本独有。
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
    # 首红
    LOW_WINDOW=520, VOL_WINDOW=20, NEW_LOW_TOLERANCE=1.02, SIGNAL_FRESH_DAYS=7,
    # 板块聚合(内联自 sector_momentum + sector_rotation)
    MOMENTUM_DAYS=20, SHORT_MOMENTUM_DAYS=5, MIN_STOCKS_IN_SECTOR=5,
    BULLISH=65.0, BEARISH=35.0,                 # 多空强度分阈值
    PE_LOW=30.0, PE_HIGH=70.0,                  # 估值分位象限边界
    # 通用
    LOOKBACK_DAYS=4000,                          # ~11年: 够首红+估值分位历史
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    TOP_N=20, NUM_PROCESSES=3, SLEEP=0.3,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))   # 0=全扫; 验逻辑先设1500

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '12'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False

# 象限 emoji 集合(交叉判断用)
Q_DUAL, Q_MID_UP, Q_CHASE, Q_PIT, Q_AVOID = "🟢", "🚀", "🟡", "", "⚫"


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

def _pctrank(cur, hist):
    v = hist.dropna()
    if len(v) < 20 or pd.isna(cur):
        return None
    return float((v < cur).mean() * 100)

def _quadrant(mom, pe_pct):
    """动量方向 × PE分位 -> (emoji, 文字); 估值缺失标❔"""
    if pd.isna(mom):
        return ("❔", "动量缺失")
    up = bool(mom > 0)
    if pd.isna(pe_pct):
        return (Q_DUAL if up else Q_PIT, ("动量↑·估值?(兜底缺失)" if up else "动量↓·估值?(兜底缺失)"))
    band = 'low' if pe_pct < PARAMS["PE_LOW"] else ('high' if pe_pct > PARAMS["PE_HIGH"] else 'mid')
    return {
        (True, 'low'):  (Q_DUAL,  "动量↑·估值低(双优)"),
        (True, 'mid'):  (Q_MID_UP,"动量↑·估值中"),
        (True, 'high'): (Q_CHASE, "动量↑·估值高(追高警惕)"),
        (False,'low'):  (Q_PIT,   "动量↓·估值低(价值埋伏)"),
        (False,'mid'):  ("⚪",     "动量↓·估值中"),
        (False,'high'): (Q_AVOID, "动量↓·估值高(回避)"),
    }[(up, band)]


# ===================== 取列表 =====================
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


# ===================== 个股历史(双源+超时, 含 peTTM/pbMRQ) =====================
def fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    sy = sd.replace("-", "")
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume,peTTM,pbMRQ", sd)
            if d is not None and not d.empty:
                for c in ['open', 'high', 'low', 'close', 'volume', 'peTTM', 'pbMRQ']:
                    if c in d.columns:
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
                    return d   # 东财无历史pe/pb -> 估值分位将None(诚实降级)
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


# ===================== 首红内核(英文列) =====================
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
    sigs = []; in_zone = False
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


# ===================== 单只: 首红 + 动量/估值/MA20 (全票返回, 供板块聚合) =====================
def _process_one(args):
    code, name, ind_raw = args
    try:
        h = fetch_hist(code)
        if h.empty:
            return None
        c = h['close'].astype(float)
        # 首红
        sigs = detect_first_red(h)
        is_fr = False; fr = {}
        if not sigs.empty:
            last = sigs.iloc[-1]
            if (pd.to_datetime(h['date'].iloc[-1]) - pd.to_datetime(last['date'])).days <= PARAMS["SIGNAL_FRESH_DAYS"]:
                is_fr = True
                fr = {"信号日期": str(last['date'])[:10], "量比": last['vol_ratio'], "距520低%": last['dist520']}
        # 动量/短动量/站MA20 (板块聚合用)
        mom = (c.iloc[-1] / c.iloc[-1 - PARAMS["MOMENTUM_DAYS"]] - 1) * 100 if len(c) > PARAMS["MOMENTUM_DAYS"] else None
        smom = (c.iloc[-1] / c.iloc[-1 - PARAMS["SHORT_MOMENTUM_DAYS"]] - 1) * 100 if len(c) > PARAMS["SHORT_MOMENTUM_DAYS"] else None
        ma20 = c.rolling(20).mean().iloc[-1]
        above = bool(c.iloc[-1] > ma20) if pd.notna(ma20) else None
        # 估值历史分位 (仅 baostock 路径有 peTTM/pbMRQ)
        pe_pct = pb_pct = None
        if 'peTTM' in h.columns and 'pbMRQ' in h.columns:
            pe_pct = _pctrank(h['peTTM'].iloc[-1], h['peTTM'])
            pb_pct = _pctrank(h['pbMRQ'].iloc[-1], h['pbMRQ'])
        return {"代码": code, "名称": name, "行业_raw": ind_raw, "行业": _clean(ind_raw),
                "首红": bool(is_fr), "信号日期": fr.get("信号日期", ""),
                "量比": fr.get("量比"), "距520低%": fr.get("距520低%"),
                "动量": round(mom, 2) if pd.notna(mom) else None,
                "短动量": round(smom, 2) if pd.notna(smom) else None,
                "站MA20": above, "PE分位": round(pe_pct, 1) if pe_pct is not None else None,
                "PB分位": round(pb_pct, 1) if pb_pct is not None else None}
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


# ===================== 主扫描(全票) =====================
def run_scan():
    print("连接 Baostock（行业表 + 子进程登录）...")
    ind_map = {}
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                ind_map = dict(zip(ind['code'], ind['industry'].fillna('')))
                print(f"  行业表 {len(ind_map)} 条")
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
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, ""), ind_map.get(c, "")) for c in codes]

    rows = []; fail = 0; fr_n = 0
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线, 算首红+动量+估值 ({len(tasks)}只, {PARAMS['NUM_PROCESSES']}进程)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=lambda: _bs_login_ok()) as pool:
        pbar = tqdm(total=len(tasks), desc="sector-combo", unit="只")
        for r in pool.imap_unordered(_process_one, tasks):
            if r:
                if "__error__" in r:
                    fail += 1
                else:
                    rows.append(r)
                    if r["首红"]:
                        fr_n += 1
                        pbar.write(f"  🔻首红 {r['代码']} {r['名称']} [{r['行业']}] 量比{r['量比']} 距520低{r['距520低%']}%")
            pbar.update(1); pbar.set_postfix(有效=len(rows), 首红=fr_n, 失败=fail)
    print(f"扫描完成 有效{len(rows)} 首红{fr_n} 失败{fail}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===================== 内联板块榜(动量估值象限 + 轮动多空/接力) =====================
def build_sector_rank(df):
    g = df[df["行业_raw"].astype(str).str.strip() != ""].groupby("行业_raw").agg(
        股票数=("代码", "count"), 平均动量=("动量", "mean"), 平均短动量=("短动量", "mean"),
        站MA20占比=("站MA20", "mean"), 平均PE分位=("PE分位", "mean"), 平均PB分位=("PB分位", "mean")).reset_index()
    g = g[g["股票数"] >= PARAMS["MIN_STOCKS_IN_SECTOR"]]
    if g.empty:
        return g
    maxabs = g["平均动量"].abs().max() or 1
    g["强度分"] = (g["站MA20占比"] * 60 + (g["平均动量"] / maxabs) * 40 + 40).clip(0, 100).round(1)
    g["短期加速度"] = (g["平均短动量"] / PARAMS["SHORT_MOMENTUM_DAYS"] - g["平均动量"] / PARAMS["MOMENTUM_DAYS"]).round(3)
    q = g.apply(lambda r: _quadrant(r["平均动量"], r["平均PE分位"]), axis=1)
    g["象限"] = [x[0] for x in q]; g["象限说明"] = [x[1] for x in q]
    g["多空"] = g["强度分"].apply(lambda s: "🐂" if s >= PARAMS["BULLISH"] else ("🐻" if s <= PARAMS["BEARISH"] else "⚪"))
    g["接力"] = (g["强度分"] > PARAMS["BEARISH"]) & (g["强度分"] < PARAMS["BULLISH"]) & (g["短期加速度"] > 0)
    for col in ["平均动量", "平均短动量", "平均PE分位", "平均PB分位"]:
        g[col] = g[col].round(2)
    g["站MA20占比"] = (g["站MA20占比"] * 100).round(1)
    return g.sort_values("强度分", ascending=False).reset_index(drop=True)


# ===================== 交叉分级 =====================
def cross_tag(q_emoji, c_emoji):
    if q_emoji in (None, "❔") or c_emoji in (None, "❔"):
        return "❔板块缺失"
    if c_emoji == "🐻" or q_emoji == Q_AVOID:
        return "⚠️逆风"
    if q_emoji == Q_CHASE:
        return "🟡追高"
    if c_emoji == "🐂" and q_emoji in (Q_DUAL, Q_MID_UP, Q_PIT):
        return "🌟共振"
    return "⚪中性"


# ===================== 推送 =====================
def build_push(fr, rank):
    P = PUSH_TOP
    reso = fr[fr["交叉"] == "🌟共振"].sort_values("量比", ascending=False)
    chase = fr[fr["交叉"] == "🟡追高"].sort_values("量比", ascending=False)
    wind = fr[fr["交叉"] == "⚠️逆风"].sort_values("量比", ascending=False)
    neut = fr[fr["交叉"].isin(["⚪中性", "❔板块缺失"])].sort_values("量比", ascending=False)
    L = [f"**520首红 × 板块(动量估值+轮动) 结合** | 首红{len(fr)} 🌟共振{len(reso)} ⚠️逆风{len(wind)}",
         "*(首红=个股左侧见底, 板块=外部确认; ⚠️逆风常见非bug=板块未确认; 🌟=个股+板块双击, 稀缺)*", ""]
    # 共振首红聚类
    if not reso.empty:
        vc = reso[reso["行业"].isin([x for x in reso["行业"] if x not in ("—", "")])]["行业"].value_counts()
        cl = [(n, int(c)) for n, c in vc.head(CLUSTER_TOP).items()]
        if cl:
            L.append("🌟 **见底+景气共振板块**: " + "、".join(f"{n}({c})" for n, c in cl))
            L.append("")
    def line(r):
        return (f"- {r['交叉']} **{r['名称']}({r['代码']})** [{r['行业']}] {r['信号日期']} 量比{r['量比']} 距520低{r['距520低%']}% "
                f"| 板块{r.get('多空','❔')}{r.get('象限','❔')} 强度{r.get('强度分','—')}")
    L.append(f"### 🌟 共振首红 Top{min(len(reso), P)} (个股见底+板块看多/景气)")
    L += [line(r) for _, r in reso.head(P).iterrows()] or ["今日无（个股见底+板块双击稀缺, 正常）"]
    L.append("")
    L.append(f"### ⚠️ 逆风首红 Top{min(len(wind), 8)} (个股见底但板块看空/回避, 左侧更谨慎)")
    L += [line(r) for _, r in wind.head(8).iterrows()] or ["无"]
    L.append("")
    L.append(f"### 🟡 追高首红 {min(len(chase), 5)} (板块估值高, 警惕赶顶反弹)")
    L += [line(r) for _, r in chase.head(5).iterrows()] or ["无"]
    L.append("")
    # 板块全景附录
    if not rank.empty:
        L.append("### 📊 板块全景(内联重算, 与板块脚本同口径)")
        L.append("看多🐂: " + "、".join(f"{r['行业_raw'].split(' ',1)[-1] if ' ' in str(r['行业_raw']) else _clean(r['行业_raw'])}({r['强度分']})"
                                       for _, r in rank[rank["多空"] == "🐂"].head(6).iterrows()) or "无")
        L.append("双优🟢/动量🚀: " + "、".join(_clean(r["行业_raw"]) for _, r in rank[rank["象限"].isin([Q_DUAL, Q_MID_UP])].head(6).iterrows()) or "无")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"520首红 × 板块(动量估值+轮动) 结合 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 一次扫描内联重算板块榜(国标行业, 口径一致)")
    print("=" * 70)
    if not is_trading_day():
        print("非交易日, 跳过"); sys.exit(0)
    df = run_scan()
    if df is None or df.empty:
        print("无有效数据"); sys.exit(0)
    rank = build_sector_rank(df)
    quad_map = dict(zip(rank["行业_raw"], rank["象限"])) if not rank.empty else {}
    camp_map = dict(zip(rank["行业_raw"], zip(rank["多空"], rank["强度分"], rank["接力"]))) if not rank.empty else {}
    fr = df[df["首红"] == True].copy()
    if fr.empty:
        print("本次无首红命中"); sys.exit(0)
    fr["象限"] = fr["行业_raw"].map(quad_map).fillna("❔")
    fr["多空"] = fr["行业_raw"].map(lambda x: camp_map.get(x, ("❔", None, False))[0])
    fr["强度分"] = fr["行业_raw"].map(lambda x: camp_map.get(x, ("❔", None, False))[1])
    fr["接力"] = fr["行业_raw"].map(lambda x: camp_map.get(x, ("❔", None, False))[2])
    fr["交叉"] = [cross_tag(q, c) for q, c in zip(fr["象限"], fr["多空"])]
    fr = fr.sort_values("量比", ascending=False).reset_index(drop=True)

    tag = datetime.now().strftime("%Y%m%d")
    fr.drop(columns=["行业_raw", "动量", "短动量", "站MA20", "PE分位", "PB分位"], errors="ignore").to_csv(
        os.path.join(OUTPUT_DIR, f"combo_sector_first_red_{tag}.csv"), index=False, encoding="utf-8-sig")
    if not rank.empty:
        rank.assign(行业=rank["行业_raw"].apply(_clean)).drop(columns=["行业_raw"]).to_csv(
            os.path.join(OUTPUT_DIR, f"combo_sector_rank_{tag}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, f"combo_sector_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump({"date": tag, "n_first_red": int(len(fr)),
                   "cross_count": fr["交叉"].value_counts().to_dict(),
                   "sector_rank": (rank.assign(行业=rank["行业_raw"].apply(_clean)).drop(columns=["行业_raw"]).to_dict('records') if not rank.empty else []),
                   "first_red": fr.drop(columns=["行业_raw"], errors="ignore").to_dict('records')},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 已存 output/combo_sector_*_{tag}.*")
    print(f"\n交叉分布: {fr['交叉'].value_counts().to_dict()}")
    disp = fr.drop(columns=["行业_raw", "动量", "短动量", "站MA20", "PE分位", "PB分位", "接力"], errors="ignore")
    print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    if SERVERCHAN_KEY:
        send_serverchan(f"🔻首红×板块 结合 | 🌟{int((fr['交叉']=='🌟共振').sum())} ⚠️{int((fr['交叉']=='⚠️逆风').sum())} 首红{len(fr)}",
                        build_push(fr, rank))
