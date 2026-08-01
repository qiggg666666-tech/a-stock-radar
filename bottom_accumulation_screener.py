# -*- coding: utf-8 -*-
"""
bottom_accumulation_screener.py —— 底部坑底/蓄势 全市场选股(日线扫描版) · 矩阵规格
【门槛】低位区(距低点≤20%+回撤≥30%) + 止跌宽松 + 坑底兜底(距低点≤5%无条件进, 保29.64)。
【本版·加MACD/布林/粘合收紧1742】原非坑底档太松(止跌六选一)致1742只。现分档加硬门槛:
  🟡横盘档: 粘合+挤压(原有) + 布林收窄 + MACD转强; 🟠止跌档: 止跌须含MACD转强;
  🔴/⚠️坑底档: 【不加任何硬门槛】(急跌底均线发散/布林宽, 硬卡粘合会漏29.64, 违背核心诉求)。
  三指标另作标签+加分排序。STRICT_NON_PIT=0 一键回宽松(1742)。
四档标签: 🔴坑底止跌 / ⚠️坑底探底(接飞刀) / 🟡低位横盘 / 🟠低位止跌。
【工程】双源+多进程+预筛+行业本地join+聚类+风口🎯+推送分页+对齐列+不拦交易日+收尾防护+append补丁+前复权。
⚠️ 坑底档=左侧极左接飞刀, 29.64事后是底事前未必; 非买入保证, 必止损。
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


# ==================== 参数配置 (全部 env 可调) ====================
BOTTOM_ZONE_PCT = float(os.environ.get('BOTTOM_ZONE_PCT', '0.20'))
DRAWDOWN_MIN = float(os.environ.get('DRAWDOWN_MIN', '0.30'))
LOW_LOOKBACK = int(os.environ.get('LOW_LOOKBACK', '250'))
PIT_PCT = float(os.environ.get('PIT_PCT', '0.05'))
RSI_LOW = float(os.environ.get('RSI_LOW', '35'))
MA_SPREAD_MAX = float(os.environ.get('MA_SPREAD_MAX', '0.02'))
SQUEEZE_LOOKBACK = int(os.environ.get('SQUEEZE_LOOKBACK', '10'))
SQUEEZE_MAX = float(os.environ.get('SQUEEZE_MAX', '0.06'))
# 【本版新增】布林收窄阈值 + 非坑底档收紧开关
BB_NARROW_MAX = float(os.environ.get('BB_NARROW_MAX', '0.12'))    # 布林带宽<此=收窄(横盘档硬门槛)
STRICT_NON_PIT = os.environ.get('STRICT_NON_PIT', '1').strip() in ('1', 'true', 'True')  # 非坑底档加MACD/布林硬门槛(默认开, 砍1742; 设0回宽松)
TURNOVER_MIN = float(os.environ.get('TURNOVER_MIN', '0.3'))
MIN_DATA_LEN = int(os.environ.get('MIN_DATA_LEN', '120'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '700'))
SNAPSHOT_PRE = os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True')
PRE_AMOUNT_MIN = float(os.environ.get('PRE_AMOUNT_MIN', '3.0e7'))
PRE_TURNOVER_MIN = float(os.environ.get('PRE_TURNOVER_MIN', '0.2'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
KEEP_PREFIX = ("0", "3", "6"); EXCLUDE_NAME = ("ST", "退"); MIN_PRICE = float(os.environ.get('MIN_PRICE', '3.0'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP = float(os.environ.get('SLEEP', '0.1'))
FETCH_TIMEOUT = int(os.environ.get('FETCH_TIMEOUT', '12'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无信号": 0}
_STAGE_ORDER = {"🔴坑底止跌": 0, "⚠️坑底探底": 1, "🟡低位横盘": 2, "🟠低位止跌": 3}


# ------------------ 推送 (全发分页 + 严格检查返回) ------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok:
            return True
        print(f"  sdk返回非成功({ret}), 回退requests")
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        j = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15).json()
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
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        print(f"  推送第{i+1}/{len(chunks)}条 ({len(ch)}字符)")
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print(f"📲 共发送{len(chunks)}条(全发分页) {'✅全部成功' if ok else '⚠️存在失败(查额度/限流)'}" if len(chunks) > 1
          else ("📲 推送成功 ✅" if ok else "⚠️ 推送返回失败(查Server酱额度/限流/微信端)"))
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


# ------------------ 历史双源 (前复权; 含换手率turn) ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume', '换手率': 'turn'})
                for col in ['open', 'high', 'low', 'close', 'volume', 'turn']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'turn'] if c in d.columns]
                    return d[cols]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,open,high,low,close,volume,turn", sd, ed, timeout=FETCH_TIMEOUT)
            if d is not None and not d.empty:
                for col in ['open', 'high', 'low', 'close', 'volume', 'turn']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume'])
                d = d[d['volume'] > 0].sort_values('date').reset_index(drop=True)
                if len(d) >= MIN_DATA_LEN:
                    return d
        except Exception:
            pass
    return _fetch_hist_em(code, sy, ey)


def snapshot_prefilter(codes_with_prefix):
    if not SNAPSHOT_PRE:
        return codes_with_prefix
    try:
        spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if spot is None or spot.empty or '代码' not in spot.columns:
            print("  快照预筛: 快照空, 退化全扫"); return codes_with_prefix
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        for col in ['最新价', '成交额', '换手率']:
            if col in spot.columns:
                spot[col] = pd.to_numeric(spot[col], errors='coerce')
        m = (spot['代码'].str.startswith(KEEP_PREFIX)
             & ~spot['名称'].astype(str).str.contains("|".join(EXCLUDE_NAME), na=False, regex=True)
             & (spot['最新价'] >= MIN_PRICE))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PRE_AMOUNT_MIN)
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PRE_TURNOVER_MIN)
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (宽松, 失败退化全扫)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}"); return codes_with_prefix


# ==================== 策略内核 (低位+止跌+坑底兜底 + 【本版】非坑底档MACD/布林硬门槛) ====================
def check_one_stock(df):
    if df is None or len(df) < MIN_DATA_LEN:
        return None, "数据不足"
    for c in ["high", "low", "close", "volume"]:
        if c not in df.columns:
            return None, "数据不足"

    close = df['close'].astype(float); high = df['high'].astype(float)
    low = df['low'].astype(float); opn = df['open'].astype(float); volume = df['volume'].astype(float)

    ma5 = close.rolling(5).mean(); ma10 = close.rolling(10).mean(); ma20 = close.rolling(20).mean()
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    # 【本版新增】布林带
    bb_mid = close.rolling(20).mean(); bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std; bb_lower = bb_mid - 2 * bb_std
    bb_bw = (bb_upper - bb_lower) / bb_mid
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
    # RSI14
    dlt = close.diff(); gain = dlt.where(dlt > 0, 0).rolling(14).mean(); loss = (-dlt.where(dlt < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    L = len(df) - 1
    if pd.isna(ma20.iloc[L]) or pd.isna(dif.iloc[L]) or pd.isna(rsi.iloc[L]):
        return None, "数据不足"
    cL = float(close.iloc[L]); oL = float(opn.iloc[L]); hL = float(high.iloc[L]); lL = float(low.iloc[L])
    if cL <= 0:
        return None, "数据不足"

    # 位置
    low_ref = float(low.rolling(LOW_LOOKBACK, min_periods=120).min().iloc[L])
    high_ref = float(high.rolling(LOW_LOOKBACK, min_periods=120).max().iloc[L])
    above_low = (cL - low_ref) / low_ref if low_ref > 0 else 999
    drawdown = (high_ref - cL) / high_ref if high_ref > 0 else 0.0
    in_zone = bool(above_low <= BOTTOM_ZONE_PCT and drawdown >= DRAWDOWN_MIN)
    if not in_zone:
        return None, "无信号"

    # 止跌信号 (宽松, 任一)
    yang = bool(cL > oL)
    body = abs(cL - oL); lower_shadow = min(oL, cL) - lL; amp = hL - lL
    long_lower = bool(amp > 0 and lower_shadow > max(body * 1.2, amp * 0.5))
    rsi_low = bool(rsi.iloc[L] < RSI_LOW)
    no_new_low = bool(low.iloc[-5:].min() >= low.iloc[-15:].min() * 0.99) if L >= 15 else False
    macd_turn = bool(pd.notna(macd_hist.iloc[L]) and pd.notna(macd_hist.iloc[L - 1]) and macd_hist.iloc[L] > macd_hist.iloc[L - 1])
    macd_golden = bool(pd.notna(dif.iloc[L]) and pd.notna(dea.iloc[L]) and dif.iloc[L] > dea.iloc[L] and dif.iloc[L - 1] <= dea.iloc[L - 1])
    macd_strong = bool(macd_turn or macd_golden)   # 【本版】MACD转强(止跌档硬门槛)
    above_ma5 = bool(cL > ma5.iloc[L])
    stop = bool(yang or long_lower or rsi_low or no_new_low or macd_turn or above_ma5)

    # 横盘标签
    ma_vals = [float(ma5.iloc[L]), float(ma10.iloc[L]), float(ma20.iloc[L])]
    ma_spread = (max(ma_vals) - min(ma_vals)) / cL
    squeeze = (float(high.iloc[-SQUEEZE_LOOKBACK:].max()) - float(low.iloc[-SQUEEZE_LOOKBACK:].min())) / cL
    is_sideways = bool(ma_spread < MA_SPREAD_MAX and squeeze < SQUEEZE_MAX)
    # 【本版新增】布林状态
    bb_narrow = bool(pd.notna(bb_bw.iloc[L]) and bb_bw.iloc[L] < BB_NARROW_MAX)
    bb_low = bool(pd.notna(bb_pct.iloc[L]) and bb_pct.iloc[L] < 0.3)

    # 坑底兜底: 距低点≤PIT_PCT 无条件进(保证29.64急跌底选出)
    is_pit = bool(above_low <= PIT_PCT)

    # 【本版改】分档硬门槛: 坑底无条件进; 非坑底档按 STRICT_NON_PIT 收紧(粘合与急跌矛盾, 故坑底不卡粘合)
    if is_pit:
        stage = "🔴坑底止跌" if stop else "⚠️坑底探底"
    elif is_sideways:
        if STRICT_NON_PIT and not (bb_narrow and macd_strong):   # 横盘档: 粘合+挤压+布林收窄+MACD转强
            return None, "无信号"
        stage = "🟡低位横盘"
    elif stop:
        if STRICT_NON_PIT and not macd_strong:                   # 止跌档: 止跌须含MACD转强
            return None, "无信号"
        stage = "🟠低位止跌"
    else:
        return None, "无信号"

    # 换手防僵尸 (可选)
    turn_now = None
    if 'turn' in df.columns:
        turn_now = float(pd.to_numeric(df['turn'], errors='coerce').iloc[L])
        if TURNOVER_MIN > 0 and (pd.isna(turn_now) or turn_now < TURNOVER_MIN):
            return None, "无信号"

    sig_date = pd.to_datetime(df['date'].iloc[L]).strftime('%Y-%m-%d') if 'date' in df.columns else ""
    score = round((1 - min(above_low, BOTTOM_ZONE_PCT) / BOTTOM_ZONE_PCT) * 40
                  + (25 if is_pit else 0) + (15 if stop else 0) + (10 if is_sideways else 0)
                  + (10 if macd_strong else 0) + (10 if bb_narrow else 0) + (10 if ma_spread < MA_SPREAD_MAX else 0), 1)
    return {"代码": None, "名称": None, "行业": "",
            "最新价": round(cL, 2), "信号日期": sig_date, "阶段": stage,
            "距低点%": round(above_low * 100, 1), "距高点回撤%": round(drawdown * 100, 1),
            "均线极差%": round(ma_spread * 100, 2), "近10日振幅%": round(squeeze * 100, 2),
            "RSI": round(float(rsi.iloc[L]), 1),
            "换手%": round(turn_now, 2) if turn_now is not None else None,
            "MACD状态": ("金叉" if macd_golden else ("转强" if macd_strong else "未转")),   # 【本版新增】
            "布林": ("收窄" if bb_narrow else ("下轨" if bb_low else "—")),                  # 【本版新增】
            "score": score, "resonance": False, "resonance_sector": ""}, None


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        if len(df) < MIN_DATA_LEN:
            return {"__fail__": "数据不足"}
        time.sleep(SLEEP)
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
            print(f"  baostock 取列表异常: {e}"); stock_df = pd.DataFrame()
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
    print(f"开始底部坑底/蓄势扫描 {len(tasks)} 只（{NUM_PROCESSES}进程, 双源; 非坑底收紧={'开' if STRICT_NON_PIT else '关'}; 坑底≤{PIT_PCT*100:.0f}%无条件进保29.64）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="底部坑底", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1; FAIL_STATS[res["__fail__"]] = FAIL_STATS.get(res["__fail__"], 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} {res['阶段']} MACD{res['MACD状态']} 布林{res['布林']} 价={res['最新价']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df['_so'] = df['阶段'].map(_STAGE_ORDER).fillna(9)
        df = df.sort_values(['_so', 'score'], ascending=[True, False]).drop(columns=['_so']).reset_index(drop=True)
    return df


# ------------------ 行业本地join + 聚类 + 风口🎯 ------------------
def enrich(df):
    targets = df.to_dict('records')
    mapped = 0
    for r in targets:
        ind = _INDUSTRY_MAP.get(r['代码'], '—'); r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注(本地join): {mapped}/{len(targets)} 只有板块")
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = [(n, int(c)) for n, c in pd.Series([r['行业'] for r in labeled]).value_counts().head(CLUSTER_TOP).items()] if labeled else []
    print(f"🪨 底部板块: {cluster or '无'}")
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
        sec = r.get('行业', ''); m = ""
        if sec and sec not in ('—', '未知', '') and hot_names:
            s = sec.strip()
            for hh in hot_names:
                if hh and (hh == s or hh in s or s in hh):
                    m = hh; break
        if m:
            r['resonance'] = True; r['resonance_sector'] = m; cnt += 1
    print(f"🎯 底部遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2['_so'] = df2['阶段'].map(_STAGE_ORDER).fillna(9)
    df2 = df2.sort_values(['resonance', '_so', 'score'], ascending=[False, True, False]).drop(columns=['_so']).reset_index(drop=True)
    return df2, cluster, hot


# ------------------ 实时对齐 (信号价 vs 现价) ------------------
def _fetch_spot_now():
    try:
        d = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
        if d is None or d.empty or '代码' not in d.columns:
            print("  实时对齐: 快照空(限流), 对齐列降级只显示信号时点"); return {}
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        if '最新价' in d.columns:
            d['最新价'] = pd.to_numeric(d['最新价'], errors='coerce')
        out = {r['代码']: float(r['最新价']) for _, r in d.iterrows() if pd.notna(r.get('最新价'))}
        print(f"  实时对齐: 取到 {len(out)} 只现价"); return out
    except Exception as e:
        print(f"  实时对齐: 快照失败({e}), 对齐列降级"); return {}


def _align_suffix(r, spot_now):
    sig_price = r.get('最新价'); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price):
        return ""
    head = f"🕒信号{sig_price}"
    if sig_date is not None and not pd.isna(sig_date):
        sd = str(sig_date)[:10]; head += f"@{sd[-5:]}"
        try:
            days = (datetime.now().date() - pd.to_datetime(sd).date()).days
            if days >= 0:
                head += f"(距今{days}天)"
        except Exception:
            pass
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return f" | {head} → 现价{now}@run({chg:+.1f}%)"
        except Exception:
            return f" | {head}"
    return f" | {head}"


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot, spot_now=None):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**🪨 底部坑底/蓄势** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         "*(🔴/⚠️坑底=左侧接飞刀; 🟡横盘=粘合+布林收窄+MACD转强; 🟠止跌=止跌+MACD转强; 非买入保证, 必止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("🪨 **底部板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- {r['阶段']} **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 距低点{r['距低点%']}% 回撤{r['距高点回撤%']}% "
                f"RSI{r['RSI']} MACD{r['MACD状态']} 布林{r['布林']} 换手{r['换手%']}%{_align_suffix(r, spot_now)}")
    for stage, head in [("🔴坑底止跌", "### 🔴 坑底止跌 (贴近低点+初现企稳, 左侧)"),
                        ("⚠️坑底探底", "### ⚠️ 坑底探底 (贴近低点但无止跌=接飞刀, 风险最高)"),
                        ("🟡低位横盘", "### 🟡 低位横盘蓄力 (粘合+布林收窄+MACD转强)"),
                        ("🟠低位止跌", "### 🟠 低位止跌 (止跌+MACD转强)")]:
        sub = df[df['阶段'] == stage] if '阶段' in df.columns else pd.DataFrame()
        if not sub.empty:
            L.append(f"{head} 共{len(sub)}只")
            L += [line(r) for _, r in sub.head(PUSH_TOP).iterrows()]; L.append("")
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"🪨 底部坑底/蓄势 (日线扫描) | {datetime.now():%Y-%m-%d %H:%M} | 回看{LOOKBACK_DAYS}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源; 多进程{NUM_PROCESSES}; 预筛={'开' if SNAPSHOT_PRE else '关'}; "
          f"非坑底收紧={'开' if STRICT_NON_PIT else '关'}(横盘+布林收窄+MACD转强 / 止跌+MACD转强); 坑底≤{PIT_PCT*100:.0f}%无条件进; 不拦交易日")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次扫描未发现满足 底部坑底/蓄势 的信号。")
        print("可调: STRICT_NON_PIT=0(回宽松) / BOTTOM_ZONE_PCT调大 / PIT_PCT调大")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"bottom_acc_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"bottom_acc_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"BOTTOM_ZONE_PCT": BOTTOM_ZONE_PCT, "PIT_PCT": PIT_PCT,
                                               "DRAWDOWN_MIN": DRAWDOWN_MIN, "STRICT_NON_PIT": STRICT_NON_PIT,
                                               "BB_NARROW_MAX": BB_NARROW_MAX},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "stage_counts": {s: int((df['阶段'] == s).sum()) for s in _STAGE_ORDER} if '阶段' in df.columns else {},
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/bottom_acc_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}"); traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = _fetch_spot_now()
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"🪨 底部坑底 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot, spot_now))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_bottom_acc<<<
