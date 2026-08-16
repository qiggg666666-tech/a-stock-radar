# -*- coding: utf-8 -*-
"""
bull_confirm_screener_v4_fast.py —— 多指标共振翻多·右侧确认 全市场选股 · 快速容错规格
定位(与strong/mtf不重复): 抓"多头确认刚发生"=多指标同时翻多拐点。
【本版加严·解决"命中太多+缩量水票"】原仅 SCORE_MIN=6+EVENT_MIN=1, 单拐点+状态凑分即进, 且量比≥1.5
  只是加分项非门槛 -> 缩量假突破(量比0.95/0.96)混入, 86只偏多偏水。本版加5旋钮(全env可调):
  ① MIN_VOL_RATIO(1.2)量比硬门槛砍缩量; ② EVENT_MIN(1→3)最少拐点砍单事件凑分; ③ SCORE_MIN(6→7);
  ④ STRONG_MIN(8→10)强多更硬; ⑤ MAX_RSI(75)过热保护; ⑥ PUSH_STRONG_ONLY(0;设1隐藏偏多段只推强多, csv仍全量)。
  【诚实撤回】不加"距高点回撤"位置过滤: 本脚本是右侧确认, 突破前高真强势与追高在位置上分不开, 加位置过滤
  会废右侧本色, 故只治"水+缩量+单事件+过热", 不碰位置。
【矩阵规格】不拦交易日; 推送全列+超长自动分页; 行业=东财优先+baostock国标本地兜底; 双源+硬超时;
  baostock多进程命门已修; 快照预筛; 风口共振🎯; 收尾防护。
⚠️ 右侧确认≠买入保证; 翻多后仍可能假突破, 需结合量能/止损。加严后弱市命中变少属正常(提质减量)。
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

def _bao_cursor_to_frame(response, context="BaoStock查询"):
    """逐行读取BaoStock结果，避免get_data()依赖pandas已删除的DataFrame.append。"""
    fields = list(getattr(response, "fields", []) or [])
    if not fields:
        raise RuntimeError(f"{context}字段为空")
    rows = []
    while response.next():
        row = list(response.get_row_data())
        if len(row) != len(fields):
            raise RuntimeError(f"{context}字段数量异常:{len(row)}/{len(fields)}")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields)

PARAMS = dict(
    LOOKBACK_DAYS=500, MIN_REQUIRED=200,
    # 【加严】三门槛改读env(原写死6/1/8): 提门槛提质减量
    SCORE_MIN=float(os.environ.get('SCORE_MIN', '7.0')),
    EVENT_MIN=int(os.environ.get('EVENT_MIN', '3')),
    STRONG_MIN=float(os.environ.get('STRONG_MIN', '10.0')),
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    PRE_AMOUNT_MIN=1.0e8, PRE_TURNOVER_MIN=1.0, NUM_PROCESSES=int(os.environ.get("NUM_WORKERS", "2")), SLEEP=0.05,
)
# 【加严】新增旋钮(全env可调)
MIN_VOL_RATIO = float(os.environ.get('MIN_VOL_RATIO', '1.2'))   # 量比硬门槛: 缩量翻多=假突破, 砍
MAX_RSI = float(os.environ.get('MAX_RSI', '75'))                # RSI过热保护: 极度过热=追高尾声, 砍
PUSH_STRONG_ONLY = os.environ.get('PUSH_STRONG_ONLY', '0').strip() in ('1', 'true', 'True')  # 1=只推强多段
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '15'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '12'))
BS_TIMEOUT = int(os.environ.get('BS_TIMEOUT', '8'))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '1'))
BS_FAILURE_LIMIT = int(os.environ.get('BS_FAILURE_LIMIT', '5'))
MAX_RUNTIME_SECONDS = int(os.environ.get('MAX_RUNTIME_SECONDS', '19200'))
SCAN_OFFSET = int(os.environ.get('SCAN_OFFSET', '0'))
ENABLE_EASTMONEY_INDUSTRY = os.environ.get('ENABLE_EASTMONEY_INDUSTRY', '0').strip() in ('1', 'true', 'True')
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_BS_FAILURES = 0
_BS_CIRCUIT_OPEN = False
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


def _hard_call(fn, *args, timeout: int, **kwargs):
    """真正的硬超时：退出时不等待卡住的后台线程。"""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        raise TimeoutError(f"{getattr(fn, '__name__', 'request')} 超时({timeout}s)")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _bs_q(code, fields, sd, ed, timeout=BS_TIMEOUT):
    def _do():
        response = bs.query_history_k_data_plus(code, fields, start_date=sd, end_date=ed, adjustflag="2")
        if getattr(response, "error_code", "1") != "0":
            raise RuntimeError(f"BaoStock日线失败:{getattr(response, 'error_msg', '')}")
        return _bao_cursor_to_frame(response, "BaoStock日线")
    return _hard_call(_do, timeout=timeout)


def _call_with_timeout(fn, *args, timeout=AK_TIMEOUT, **kwargs):
    return _hard_call(fn, *args, timeout=timeout, **kwargs)

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
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=AK_TIMEOUT)
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
    """默认禁用东财逐只行业请求；需要时才以一次短超时尝试。"""
    if not ENABLE_EASTMONEY_INDUSTRY:
        return "—"
    try:
        info = _call_with_timeout(ak.stock_individual_info_em, symbol=symbol, timeout=AK_TIMEOUT)
        if info is not None and not info.empty and "item" in info.columns:
            row = info[info["item"].isin(["行业", "所属行业"])]
            if not row.empty:
                return row.iloc[0]["value"]
    except Exception:
        pass
    return "—"

def get_industry_heat():
    try:
        return _call_with_timeout(ak.stock_board_industry_name_em, timeout=AK_TIMEOUT)
    except Exception as exc:
        print(f"  行业热度榜跳过: {exc}")
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


def _normalize_hist(df):
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume"})
    for col in ("close", "volume"):
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_hist(code):
    """AkShare 主取数；BaoStock 仅有限短超时回退，避免双源轮询拖垮全局任务。"""
    global _BS_FAILURES, _BS_CIRCUIT_OPEN
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime("%Y%m%d")
    ed = datetime.now().strftime("%Y%m%d")
    for _ in range(MAX_RETRIES):
        try:
            data = _call_with_timeout(ak.stock_zh_a_hist, symbol=code.split(".")[-1], period="daily", start_date=sd, end_date=ed, adjust="qfq", timeout=AK_TIMEOUT)
            norm = _normalize_hist(data)
            if norm is not None and len(norm) >= PARAMS["MIN_REQUIRED"]:
                return norm
        except Exception:
            pass
    if _BS_CIRCUIT_OPEN or not _BS_LOGGED:
        return None
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime("%Y-%m-%d")
        data = _bs_q(_pref(code), "date,close,volume", start, end, timeout=BS_TIMEOUT)
        norm = _normalize_hist(data)
        if norm is not None and len(norm) >= PARAMS["MIN_REQUIRED"]:
            _BS_FAILURES = 0
            return norm
    except Exception:
        _BS_FAILURES += 1
        if _BS_FAILURES >= BS_FAILURE_LIMIT:
            _BS_CIRCUIT_OPEN = True
            print("  BaoStock 连续失败达到阈值，本 worker 已关闭回退")
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
    # 【加严】量能硬门槛: 缩量翻多=假突破(如量比0.95/0.96), 砍; vr缺失视为无量亦砍
    if vr is None or vr < MIN_VOL_RATIO:
        return None
    # 【加严】RSI过热保护: 极度过热=追高尾声, 砍
    if rsi is not None and rsi > MAX_RSI:
        return None
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


def _save_checkpoint(rows, processed, total, reason):
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["代码", "名称", "多头得分", "综合倾向"])
    else:
        frame = frame.sort_values("多头得分", ascending=False)
    frame.to_csv(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}_checkpoint_{processed}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}_progress.json"), "w", encoding="utf-8") as handle:
        json.dump({"processed": processed, "total": total, "hits": len(rows), "reason": reason, "saved_at": datetime.now().isoformat()}, handle, ensure_ascii=False, indent=2)
    print(f"💾 检查点 {processed}/{total}，命中{len(rows)}，原因: {reason}")


def _load_pool_and_industry():
    global _INDUSTRY_MAP
    try:
        listed = _call_with_timeout(ak.stock_info_a_code_name, timeout=AK_TIMEOUT)
        names = "name" if "name" in listed.columns else listed.columns[1]
        work = listed[["code", names]].copy()
        work.columns = ["code", "code_name"]
        work["code"] = work["code"].astype(str).str.zfill(6)
        work = work[work["code"].str.match(r"^(00|30|60|68)", na=False)]
        work = work[~work["code_name"].astype(str).str.contains("ST|退", na=False, regex=True)]
        work["code"] = work["code"].apply(_pref)
        return work
    except Exception as exc:
        print(f"AkShare 股票池失败，回退 BaoStock: {exc}")
    if not _bs_login_ok(retries=1):
        return pd.DataFrame()
    try:
        industry_response = bs.query_stock_industry()
        if getattr(industry_response, "error_code", "1") != "0":
            raise RuntimeError(f"BaoStock行业失败:{getattr(industry_response, 'error_msg', '')}")
        industry = _bao_cursor_to_frame(industry_response, "BaoStock行业")
        if industry is not None and not industry.empty:
            for _, row in industry.iterrows():
                _INDUSTRY_MAP[row.get("code", "")] = _clean_industry(row.get("industry", ""))
        listed_response = bs.query_stock_basic()
        if getattr(listed_response, "error_code", "1") != "0":
            raise RuntimeError(f"BaoStock股票池失败:{getattr(listed_response, 'error_msg', '')}")
        listed = _bao_cursor_to_frame(listed_response, "BaoStock股票池")
        listed = listed[listed["code"].str.startswith(("sh.", "sz.")) & (listed["type"] == "1") & (listed["status"] == "1")].copy()
        listed = listed[~listed["code_name"].astype(str).str.contains("ST|退", na=False, regex=True)]
        return listed
    except Exception:
        return pd.DataFrame()
    finally:
        _bs_logout()


def run_scan():
    stock_df = _load_pool_and_industry()
    if stock_df is None or stock_df.empty:
        print("⚠️ 无股票列表")
        return pd.DataFrame()
    codes = snapshot_prefilter(stock_df["code"].tolist())
    codes = sorted(codes)
    if SCAN_OFFSET:
        codes = codes[SCAN_OFFSET:]
    if SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df["code"], stock_df["code_name"]))
    tasks = [(code, name_map.get(code, "")) for code in codes]
    total = len(tasks)
    print(f"逐只拉{PARAMS['LOOKBACK_DAYS']}天日线算翻多确认 ({total}只, {PARAMS['NUM_PROCESSES']}进程, offset={SCAN_OFFSET})")
    rows, fail, processed = [], 0, 0
    deadline = time.monotonic() + MAX_RUNTIME_SECONDS if MAX_RUNTIME_SECONDS > 0 else None
    timed_out = False
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=_init_worker)
    pbar = tqdm(total=total, desc="bull扫描", unit="只")
    try:
        for result in pool.imap_unordered(_process_one, tasks, chunksize=1):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                _save_checkpoint(rows, processed, total, "安全运行预算到达")
                print("达到安全运行预算，停止未开始任务并保留结果")
                break
            processed += 1
            pbar.update(1)
            if result:
                if "__error__" in result:
                    fail += 1
                else:
                    rows.append(result)
            if processed % 25 == 0:
                _save_checkpoint(rows, processed, total, "定期保存")
    finally:
        pbar.close()
        if timed_out:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    _save_checkpoint(rows, processed, total, "完成" if not timed_out else "预算退出")
    print(f"扫描结束 命中{len(rows)} 失败{fail} 已处理{processed}/{total}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def enrich(df):
    """默认只使用本地 BaoStock 行业映射，避免命中后逐只东财请求耗尽运行预算。"""
    targets = df.to_dict("records")
    for row in targets:
        row["行业"] = _INDUSTRY_MAP.get(row["代码"], "—")
    if ENABLE_EASTMONEY_INDUSTRY:
        print("行业补全已启用：仅对本地未知行业做单次短超时请求")
        for row in targets:
            if row["行业"] in ("—", "未知", ""):
                symbol = row["代码"].split(".")[-1]
                row["行业"] = fetch_industry(symbol)
    labeled = [row for row in targets if row.get("行业") not in ("—", "未知", "", None)]
    cluster = [(name, int(count)) for name, count in pd.Series([row["行业"] for row in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [name for name, _ in hot]
    for row in targets:
        match = match_sector(row.get("行业", ""), hot_names)
        row["resonance"] = bool(match)
        row["resonance_sector"] = match
    frame = pd.DataFrame(targets)
    frame = frame.sort_values(["resonance", "多头得分"], ascending=[False, False]).reset_index(drop=True)
    return frame, cluster, hot

def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    strong = df[df['综合倾向'] == '强多确认']
    mode = "仅推强多" if PUSH_STRONG_ONLY else "全发"
    L = [f"**🚀 多指标共振翻多·右侧确认(加严)** | 命中{len(df)}只 🎯风口{len(reso)} 强多{len(strong)} ({mode})",
         f"*(加严: 量比≥{MIN_VOL_RATIO}+拐点≥{PARAMS['EVENT_MIN']}+得分≥{PARAMS['SCORE_MIN']}+RSI≤{MAX_RSI}; 抓'多头确认刚发生'; 右侧确认≠买入保证, 防假突破)*", ""]
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
    # 【加严】PUSH_STRONG_ONLY=1 时隐藏偏多段(只推强多); csv仍存全量不丢数据
    if not PUSH_STRONG_ONLY:
        rest = df[df['综合倾向'] != '强多确认']
        if not rest.empty:
            L.append(f"### 📋 偏多确认 共{len(rest)}只")
            L += [line(r) for _, r in rest.iterrows()]
    else:
        n_rest = len(df) - len(strong)
        L.append(f"*(PUSH_STRONG_ONLY=1: 偏多确认{n_rest}只已隐藏, 仅推强多; 设0看全部, csv仍存全量)*")
    return "\n".join(L)


def main():
    print("=" * 70)
    print(f"🚀 多指标共振翻多·右侧确认(加严) | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 加严门槛 量比≥{MIN_VOL_RATIO}+拐点≥{PARAMS['EVENT_MIN']}+得分≥{PARAMS['SCORE_MIN']}+强多≥{PARAMS['STRONG_MIN']}+RSI≤{MAX_RSI}; 仅推强多={PUSH_STRONG_ONLY}; 不拦交易日; 行业东财+baostock兜底; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        tag = datetime.now().strftime("%Y%m%d")
        pd.DataFrame(columns=["代码", "名称", "多头得分", "综合倾向"]).to_csv(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}.csv"), index=False, encoding="utf-8-sig")
        print("本次无翻多确认命中，已保存空结果文件")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"bull_confirm_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"SCORE_MIN": PARAMS['SCORE_MIN'], "EVENT_MIN": PARAMS['EVENT_MIN'],
                       "STRONG_MIN": PARAMS['STRONG_MIN'], "MIN_VOL_RATIO": MIN_VOL_RATIO, "MAX_RSI": MAX_RSI,
                       "PUSH_STRONG_ONLY": PUSH_STRONG_ONLY},
                       "cluster": cluster, "n": int(len(df)),
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
            mode = "仅强多" if PUSH_STRONG_ONLY else "全发"
            send_serverchan(f"🚀 翻多确认(加严) 强多{n_strong} 共{len(df)} 🎯{n_reso} [{mode}]", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_bull2<<<
