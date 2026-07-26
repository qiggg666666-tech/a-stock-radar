# -*- coding: utf-8 -*-
"""
macd_zigzag_divergence.py —— MACD零轴上方 + ZigZag底背离 全市场选股 · 矩阵规格
条件: MACD零轴上方 + 即将/刚金叉 + ZigZag底背离(已确认低点) + 量柱缩小 + 均线支撑。
ZigZag 只用"已确认"谷点, 避免信号随最新K线漂移(策略内核一字未动)。

【本版完善】单进程串行->多进程(每子进程独立登录baostock); baostock优先+东财兜底;
  全局FAIL_STATS改子进程回传__fail__+主进程累加(多进程安全); 加baostock行业本地join+
  聚类+东财风口🎯; 加推送全发分页(严格检查返回); 存output/+收尾防护+sys.exit(0)。
  不加快照预筛(缩量底背离票不活跃, 预筛会误杀)。
【不拦交易日】本脚本原无交易日判断, 保持不拦(周末可用上一交易日数据复盘), 横幅已注明。
⚠️ 底背离=左侧反转提示, 非买入保证; 需等金叉/放量确认, 严格止损。
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
import numpy as np

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


# ------------------ 参数 (env 可调) ------------------
PARAMS = dict(
    LOOKBACK_DAYS=300,            # 日线回看(底背离lookback120+MACD预热+余量)
    MIN_REQUIRED=100,
    ZIGZAG_THRESHOLD=0.05,        # ZigZag 反转幅度 5%
    GOLDEN_THRESHOLD=0.03,
    VOL_SHORT=5, VOL_LONG=20, VOL_RATIO=0.75,
    NUM_PROCESSES=3, SLEEP=0.1,
    FETCH_TIMEOUT=10, FETCH_RETRIES=3,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))   # 0=全扫; 仍超时设1500
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}   # baostock国标行业, 本地join零接口
# 失败统计(主进程累加, 子进程通过 __fail__ 回传, 多进程安全)
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "零轴上方": 0, "金叉条件": 0,
              "底背离": 0, "缩量": 0, "均线支撑": 0}


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
    """全发: 超长自动按行切分多条发送"""
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
    """子进程: 无条件清零标志(破除fork继承脏标志)再登录, 拿自己的socket"""
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


# ===================== 策略内核 (一字未动) =====================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    macd = 2 * (dif - dea)
    return dif, dea, macd


def calc_ma(close: pd.Series, windows=(5, 10, 20)):
    return {f"ma{w}": close.rolling(w).mean() for w in windows}


def zigzag(prices, threshold: float = 0.05):
    """真正的 ZigZag（百分比阈值）; 返回 list[(index, price, type, confirmed)]"""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 3:
        return []
    pivots = []
    last_pivot_price = prices[0]
    trend = 0
    extreme_idx = 0
    extreme_price = prices[0]
    for i in range(1, n):
        price = prices[i]
        if trend == 0:
            change = (price - last_pivot_price) / (abs(last_pivot_price) + 1e-12)
            if change >= threshold:
                trend = 1; extreme_idx = i; extreme_price = price
            elif change <= -threshold:
                trend = -1; extreme_idx = i; extreme_price = price
            continue
        if trend == 1:
            if price > extreme_price:
                extreme_idx = i; extreme_price = price
            else:
                drawdown = (extreme_price - price) / (abs(extreme_price) + 1e-12)
                if drawdown >= threshold:
                    pivots.append((extreme_idx, extreme_price, 1, True))
                    last_pivot_price = extreme_price; trend = -1
                    extreme_idx = i; extreme_price = price
        elif trend == -1:
            if price < extreme_price:
                extreme_idx = i; extreme_price = price
            else:
                rally = (price - extreme_price) / (abs(extreme_price) + 1e-12)
                if rally >= threshold:
                    pivots.append((extreme_idx, extreme_price, -1, True))
                    last_pivot_price = extreme_price; trend = 1
                    extreme_idx = i; extreme_price = price
    if trend != 0:
        pivots.append((extreme_idx, extreme_price, trend, False))
    return pivots


def get_zigzag_lows(prices, threshold: float = 0.05, confirmed_only: bool = True):
    pivots = zigzag(prices, threshold)
    return [(idx, price) for idx, price, typ, confirmed in pivots
            if typ == -1 and (confirmed or not confirmed_only)]


def detect_bottom_divergence_zigzag(close: pd.Series, dif: pd.Series,
                                    threshold: float = 0.05, lookback: int = 120):
    if len(close) < lookback:
        return False, None
    c = close.iloc[-lookback:].reset_index(drop=True)
    d = dif.iloc[-lookback:].reset_index(drop=True)
    lows = get_zigzag_lows(c.values, threshold=threshold, confirmed_only=True)
    if len(lows) < 2:
        return False, None
    i1, p1 = lows[-2]
    i2, p2 = lows[-1]
    if p2 >= p1:
        return False, None
    dif1 = float(d.iloc[i1])
    dif2 = float(d.iloc[i2])
    if dif2 <= dif1:
        return False, None
    detail = {"前低位置": i1, "前低价格": round(p1, 3), "前低DIF": round(dif1, 4),
              "后低位置": i2, "后低价格": round(p2, 3), "后低DIF": round(dif2, 4),
              "价格降幅%": round((p2 - p1) / p1 * 100, 2), "DIF抬高": round(dif2 - dif1, 4)}
    return True, detail


def is_about_to_golden_cross(dif: pd.Series, dea: pd.Series, threshold=0.03):
    if len(dif) < 5:
        return False
    above_zero = dif.iloc[-1] > 0
    just_cross = (dif.iloc[-2] < dea.iloc[-2]) and (dif.iloc[-1] >= dea.iloc[-1])
    gap = dif.iloc[-1] - dea.iloc[-1]
    prev_gap = dif.iloc[-2] - dea.iloc[-2]
    approaching = (gap <= 0) and (gap > prev_gap) and (abs(gap) < threshold or abs(gap) < abs(prev_gap) * 0.7)
    return above_zero and (just_cross or approaching)


def volume_shrinking(volume: pd.Series, short=5, long=20, ratio=0.75):
    if len(volume) < long:
        return False
    return volume.iloc[-short:].mean() < volume.iloc[-long:].mean() * ratio


def ma_support(close: pd.Series, ma_dict: dict):
    last = close.iloc[-1]
    if "ma5" not in ma_dict or "ma10" not in ma_dict:
        return False
    price_ok = last >= ma_dict["ma5"].iloc[-1] and last >= ma_dict["ma10"].iloc[-1]
    short_bull = ma_dict["ma5"].iloc[-1] >= ma_dict["ma10"].iloc[-1]
    return price_ok and short_bull


def check_one_stock(df: pd.DataFrame, zigzag_threshold: float = 0.05):
    """返回 (命中dict 或 None, 失败原因 或 None); 不在内部累加统计(多进程安全, 由主进程累加)"""
    if df is None or len(df) < PARAMS["MIN_REQUIRED"]:
        return None, "数据不足"
    if "close" not in df.columns or "volume" not in df.columns:
        return None, "数据不足"
    dif, dea, macd = calc_macd(df["close"])
    ma_dict = calc_ma(df["close"])
    zero_above = dif.iloc[-1] > 0
    golden = is_about_to_golden_cross(dif, dea, threshold=PARAMS["GOLDEN_THRESHOLD"])
    divergence, div_detail = detect_bottom_divergence_zigzag(
        df["close"], dif, threshold=zigzag_threshold, lookback=120)
    vol_ok = volume_shrinking(df["volume"], PARAMS["VOL_SHORT"], PARAMS["VOL_LONG"], PARAMS["VOL_RATIO"])
    ma_ok = ma_support(df["close"], ma_dict)
    # 第一个未通过条件(按检查顺序)
    if not zero_above:
        reason = "零轴上方"
    elif not golden:
        reason = "金叉条件"
    elif not divergence:
        reason = "底背离"
    elif not vol_ok:
        reason = "缩量"
    elif not ma_ok:
        reason = "均线支撑"
    else:
        reason = None
    if reason is not None:
        return None, reason
    result = {"DIF": round(dif.iloc[-1], 4), "DEA": round(dea.iloc[-1], 4),
              "收盘价": round(df["close"].iloc[-1], 3),
              "MA5": round(ma_dict["ma5"].iloc[-1], 3), "MA10": round(ma_dict["ma10"].iloc[-1], 3),
              "近5日均量": int(df["volume"].iloc[-5:].mean()),
              "近20日均量": int(df["volume"].iloc[-20:].mean()),
              "量缩比": round(df["volume"].iloc[-5:].mean() / df["volume"].iloc[-20:].mean(), 2)
              if df["volume"].iloc[-20:].mean() > 0 else None}
    if div_detail:
        result.update({"背离前低价": div_detail["前低价格"], "背离后低价": div_detail["后低价格"],
                       "背离前低DIF": div_detail["前低DIF"], "背离后低DIF": div_detail["后低DIF"],
                       "价格降幅%": div_detail["价格降幅%"]})
    return result, None


# ------------------ 历史双源 ------------------
def _fetch_hist_em(sym, start_y, end_y):
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d[['date', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    # 路径1: baostock (子进程已登录)
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,close,volume", sd, ed, timeout=PARAMS["FETCH_TIMEOUT"])
            if d is not None and not d.empty:
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
                if len(d) >= PARAMS["MIN_REQUIRED"]:
                    return d.tail(PARAMS["LOOKBACK_DAYS"]).copy()
        except Exception:
            pass
    # 路径2: 东财兜底
    return _fetch_hist_em(code, sy, ey)


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None:
            return {"__fail__": "抓取失败"}
        time.sleep(PARAMS["SLEEP"])
        info, reason = check_one_stock(df, zigzag_threshold=PARAMS["ZIGZAG_THRESHOLD"])
        if info is None:
            return {"__fail__": reason}
        return {"代码": code, "名称": name, "行业": "", **info,
                "resonance": False, "resonance_sector": ""}
    except cf.TimeoutError:
        return {"__fail__": "抓取失败"}
    except Exception as e:
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
    if SCAN_LIMIT and len(codes) > SCAN_LIMIT:
        codes = codes[:SCAN_LIMIT]
    name_map = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, name_map.get(c, "")) for c in codes]

    results = []; fail_count = 0
    print(f"开始MACD零轴上+ZigZag底背离扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 阈值{PARAMS['ZIGZAG_THRESHOLD']*100:.0f}%）...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="zigzag扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 收盘={res['收盘价']} DIF={res['DIF']} 背离降幅={res.get('价格降幅%','-')}%")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各条件失败次数统计（先失败先计数）：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("DIF", ascending=False).reset_index(drop=True)
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
    print(f"📉 底背离板块: {cluster or '无'}")
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
    print(f"🎯 底背离遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', 'DIF'], ascending=[False, False]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**📉 MACD零轴上+ZigZag底背离** | 命中{len(df)}只 🎯风口{len(reso)} (全发)",
         "*(零轴上方+即将金叉+已确认底背离+缩量+均线支撑=左侧反转提示; 非买入保证, 需等确认, 严格止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📉 **底背离板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        div = f"背离降幅{r.get('价格降幅%')}%" if r.get('价格降幅%') is not None else ""
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 现价{r['收盘价']} DIF{r['DIF']} "
                f"量缩比{r.get('量缩比')} {div}")
    if not reso.empty:
        L.append(f"### 🎯 底背离遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    L.append(f"### 📉 全部底背离 共{len(df)}只")
    L += [line(r) for _, r in df.iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"📉 MACD零轴上+ZigZag底背离 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天 阈值{PARAMS['ZIGZAG_THRESHOLD']*100:.0f}%")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源baostock+东财; 不拦交易日(周末可复盘); 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n扫描完成，未发现同时满足全部条件的股票。")
        print("可参考上面失败统计定位瓶颈: 降 ZIGZAG_THRESHOLD(如0.03)/放宽 VOL_RATIO/金叉阈值")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"macd_zigzag_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"macd_zigzag_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/macd_zigzag_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'MA5', 'MA10',
                                  'DEA', '近5日均量', '近20日均量', '背离前低价', '背离后低价',
                                  '背离前低DIF', '背离后低DIF'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📉 MACD零轴上+ZigZag底背离 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_macd_zigzag<<<
