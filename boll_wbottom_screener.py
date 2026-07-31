# -*- coding: utf-8 -*-
"""
boll_wbottom_screener.py —— 布林带 + W底 全市场选股 · 矩阵规格
形态: 突破上轨+放量(右肩确认) + 中间反弹触中轨(颈线) + 两低点触下轨且第二低点抬高(W底)
      + 两低点间隔[min_gap,max_gap] + 第二低点缩量 = W底买入信号; 只看最近recent_days天内信号。
策略内核四条件逻辑一字未改; 仅将循环内 pandas iloc 改 numpy 数组下标提速(数学等价),
并删除原"bandwidth<阈值: pass"死代码(无影响)。

【本版完善】删两个无用import(copy/matplotlib, 后者在Actions未装会ImportError崩);
  单源akshare->双源baostock+东财+硬超时; 写死10只->全市场多进程(每子进程独立登录baostock,
  命门已修); 加宽松快照预筛(突破放量票活跃, 不误杀, 失败退化全扫); 加baostock行业本地join+
  聚类+东财风口🎯; 加推送全发分页(严格检查返回); 存output/+收尾防护+sys.exit(0)。
【不拦交易日】原脚本无交易日判断, 保持不拦(周末可用上一交易日数据复盘), 横幅已注明。
【本版·放宽触轨容差】原 alpha=0.002(收盘价距布林轨道<0.2%才算"触轨")太严, 要求W底两低点+颈线
  三点都几乎精确落在轨道上(巧合级事件), 致全市场常只命中0-1只。本版 alpha 默认放宽到 0.008
  (0.8%容差, 介于严苛与宽松间的稳妥值: 既还原"接近轨道"的形态本意, 又保留较高标准度, 更放心);
  并把 alpha/min_gap/max_gap/vol_expand/vol_shrink/recent_days/bb_std/bb_period/lookback 改为
  env 可调(ALPHA/MIN_GAP/MAX_GAP/VOL_EXPAND/VOL_SHRINK/RECENT_DAYS/BB_STD/BB_PERIOD/LOOKBACK),
  无需改代码即可微调命中松紧。
  ⚠️ 权衡: 容差越宽命中越多但W底形态越不标准; W底本就稀有, 放宽后也不会爆炸增长。
⚠️ W底=左侧/突破形态提示, 非买入保证; 突破上轨放量是右侧确认但假突破常见, 需结合颈线/止损。
⚠️ 性能: 取数(网络)是主瓶颈; 计算端numpy提速后, 突破密集的牛市段内层循环会变多, 单只偏慢属正常。
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


# ==================== 参数配置 (形态参数 env 可调) ====================
PARAMS = dict(
    # 布林带
    bb_period=int(os.environ.get('BB_PERIOD', '20')),
    bb_std=float(os.environ.get('BB_STD', '2.0')),
    # W底形态
    lookback=int(os.environ.get('LOOKBACK', '75')),
    min_gap=int(os.environ.get('MIN_GAP', '8')),
    max_gap=int(os.environ.get('MAX_GAP', '45')),
    alpha=float(os.environ.get('ALPHA', '0.008')),   # 【本版】0.002→0.008: 原0.2%触轨容差太严, 放宽到0.8%(稳妥, 标准度较高)
    # 成交量
    volume_ma_period=int(os.environ.get('VOL_MA_PERIOD', '10')),
    volume_shrink_ratio=float(os.environ.get('VOL_SHRINK', '0.85')),
    volume_expand_ratio=float(os.environ.get('VOL_EXPAND', '1.5')),
    # 扫描
    recent_days=int(os.environ.get('RECENT_DAYS', '15')),
    min_data_len=100, lookback_days=400,
    # 快照预筛(宽松, 突破放量票活跃不误杀; 失败退化全扫)
    SNAPSHOT_PRE=True, PRE_AMOUNT_MIN=5.0e7, PRE_TURNOVER_MIN=0.3,
    KEEP_PREFIX=("0", "3", "6"), EXCLUDE_NAME=("ST", "退"), MIN_PRICE=3.0,
    NUM_PROCESSES=3, SLEEP=0.1, FETCH_TIMEOUT=10,
)
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))   # 0=全扫(预筛后); 仍超时设1500
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
FAIL_STATS = {"抓取失败": 0, "数据不足": 0, "无W底信号": 0, "信号过旧": 0}


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


# ==================== 布林带计算 (一字未动) ====================
def bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['std'] = data['close'].rolling(PARAMS['bb_period'], min_periods=PARAMS['bb_period']).std()
    data['mid'] = data['close'].rolling(PARAMS['bb_period'], min_periods=PARAMS['bb_period']).mean()
    data['upper'] = data['mid'] + PARAMS['bb_std'] * data['std']
    data['lower'] = data['mid'] - PARAMS['bb_std'] * data['std']
    data['bandwidth'] = (data['upper'] - data['lower']) / data['mid']
    data['vol_ma'] = data['volume'].rolling(PARAMS['volume_ma_period'], min_periods=1).mean()
    return data


# ==================== W底检测 (逻辑等价, iloc->numpy提速, 删pass死代码) ====================
def detect_w_bottom(df: pd.DataFrame) -> pd.DataFrame:
    """返回带 signal/coordinates 的 DataFrame。四条件判定逻辑与原脚本完全一致,
    仅将循环内 pandas iloc 访问改为 numpy 数组下标(提速), 并删除原 bandwidth pass 死代码。"""
    if len(df) < PARAMS['min_data_len']:
        return pd.DataFrame()

    data = bollinger_bands(df)
    n = len(data)
    lookback = PARAMS['lookback']
    alpha = PARAMS['alpha']
    min_gap = PARAMS['min_gap']; max_gap = PARAMS['max_gap']
    shrink = PARAMS['volume_shrink_ratio']; expand = PARAMS['volume_expand_ratio']

    # 提 numpy 数组(循环内用下标访问, 远快于 iloc)
    close = data['close'].to_numpy(dtype=float)
    volume = data['volume'].to_numpy(dtype=float)
    upper = data['upper'].to_numpy(dtype=float)
    mid = data['mid'].to_numpy(dtype=float)
    lower = data['lower'].to_numpy(dtype=float)
    vol_ma = data['vol_ma'].to_numpy(dtype=float)

    signal = np.zeros(n, dtype=int)
    coords = [''] * n

    for i in range(lookback, n):
        # 1. 突破上轨 + 放量
        if close[i] > upper[i] and volume[i] >= vol_ma[i] * expand:
            found = False
            lo_i = max(i - lookback, 0)
            for j in range(i - 1, lo_i, -1):
                # 条件2：中间反弹接近中轨
                if abs(close[j] - mid[j]) < alpha * close[j]:
                    for k in range(j - 1, lo_i, -1):
                        # 条件1：第一低点接近下轨
                        if abs(close[k] - lower[k]) < alpha * close[k]:
                            threshold = close[k]
                            for m in range(i - 1, j, -1):
                                # 条件3：第二低点接近下轨且更高
                                if (abs(close[m] - lower[m]) < alpha * close[m] and
                                        close[m] > lower[m] and close[m] > threshold * 0.995):
                                    gap = abs(m - k)
                                    if not (min_gap <= gap <= max_gap):
                                        continue
                                    vol_k = volume[k]; vol_m = volume[m]
                                    if vol_k <= 0 or vol_m >= vol_k * shrink:
                                        continue
                                    signal[i] = 1
                                    coords[i] = f"{k},{j},{m},{i}"
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
            # 原 bandwidth<阈值: pass 死代码已删除(无影响)

    data['signal'] = signal
    data['coordinates'] = coords
    return data


# ------------------ 历史双源 (只要 date/close/volume, W底不需high/low) ------------------
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
                if len(d) >= PARAMS['min_data_len']:
                    return d[['date', 'close', 'volume']]
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_hist(code):
    sd = (datetime.now() - timedelta(days=PARAMS['lookback_days'])).strftime('%Y-%m-%d')
    ed = datetime.now().strftime('%Y-%m-%d')
    sy = sd.replace('-', ''); ey = ed.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(_pref(code), "date,close,volume", sd, ed, timeout=PARAMS['FETCH_TIMEOUT'])
            if d is not None and not d.empty:
                d['close'] = pd.to_numeric(d['close'], errors='coerce')
                d['volume'] = pd.to_numeric(d['volume'], errors='coerce')
                d['date'] = pd.to_datetime(d['date'], errors='coerce')
                d = d.dropna(subset=['close', 'volume']).sort_values('date').reset_index(drop=True)
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
        for c in ['最新价', '成交额', '换手率']:
            if c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        m = (spot['代码'].str.startswith(PARAMS['KEEP_PREFIX'])
             & ~spot['名称'].astype(str).str.contains("|".join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)
             & (spot['最新价'] >= PARAMS['MIN_PRICE']))
        if '成交额' in spot.columns:
            m &= (spot['成交额'] >= PARAMS['PRE_AMOUNT_MIN'])
        if '换手率' in spot.columns:
            m &= (spot['换手率'] >= PARAMS['PRE_TURNOVER_MIN'])
        keep = set(spot.loc[m, '代码'])
        out = [c for c in codes_with_prefix if c[3:] in keep]
        print(f"  快照预筛: {len(codes_with_prefix)} → {len(out)} 只 (宽松, 突破放量票活跃不误杀)")
        return out if out else codes_with_prefix
    except Exception as e:
        print(f"  快照预筛失败, 退化全扫: {e}")
        return codes_with_prefix


def _process_one(args):
    code, name = args
    try:
        df = _fetch_hist(code)
        if df is None or len(df) < PARAMS['min_data_len']:
            return {"__fail__": "数据不足" if df is not None else "抓取失败"}
        time.sleep(PARAMS['SLEEP'])
        signal_df = detect_w_bottom(df)
        if signal_df.empty:
            return {"__fail__": "无W底信号"}
        buy = signal_df[signal_df['signal'] == 1]
        if buy.empty:
            return {"__fail__": "无W底信号"}
        latest = buy.iloc[-1]
        signal_date = latest['date']
        days_ago = (datetime.now().date() - pd.to_datetime(signal_date).date()).days
        if days_ago > PARAMS['recent_days']:
            return {"__fail__": "信号过旧"}
        return {"代码": code, "名称": name, "行业": "",
                "最新价": round(float(latest['close']), 2),
                "信号日期": pd.to_datetime(signal_date).strftime('%Y-%m-%d'),
                "距今天数": int(days_ago),
                "上轨": round(float(latest['upper']), 2),
                "中轨": round(float(latest['mid']), 2),
                "下轨": round(float(latest['lower']), 2),
                "coordinates": latest['coordinates'],
                "resonance": False, "resonance_sector": ""}
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
    print(f"开始布林带+W底扫描 {len(tasks)} 只（{PARAMS['NUM_PROCESSES']}进程, 双源, 近{PARAMS['recent_days']}天信号, alpha={PARAMS['alpha']}）...")
    with mp.Pool(processes=PARAMS['NUM_PROCESSES'], initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="w底扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__fail__" in res:
                    fail_count += 1
                    r = res["__fail__"]
                    FAIL_STATS[r] = FAIL_STATS.get(r, 0) + 1
                else:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} 信号{res['信号日期']} 距今{res['距今天数']}天 价={res['最新价']}")
            pbar.update(1); pbar.set_postfix(命中=len(results), 失败=fail_count)
    print("\n各失败原因统计：")
    for k, v in FAIL_STATS.items():
        print(f"  {k}: {v}")
    print(f"扫描完成 命中{len(results)} 失败{fail_count}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('距今天数').reset_index(drop=True)
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
    print(f"📈 W底板块: {cluster or '无'}")
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
    print(f"🎯 W底遇风口 {cnt} 只")
    df2 = pd.DataFrame(targets)
    df2 = df2.sort_values(['resonance', '距今天数'], ascending=[False, True]).reset_index(drop=True)
    return df2, cluster, hot


def _sec_tag(r):
    return ('🎯' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')


def build_push(df, cluster, hot):
    reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
    L = [f"**📈 布林带+W底突破** | 命中{len(df)}只 🎯风口{len(reso)} (全发, 近{PARAMS['recent_days']}天信号, alpha={PARAMS['alpha']})",
         "*(突破上轨放量+中轨颈线+两低点抬高+缩量=W底; 形态提示非买入保证, 假突破常见, 需颈线/止损)*", ""]
    if hot:
        L.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); L.append("")
    if cluster:
        L.append("📈 **W底板块**: " + "、".join(f"{n}({c})" for n, c in cluster)); L.append("")
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{_sec_tag(r.to_dict())}] 信号{r['信号日期']}(距今{r['距今天数']}天) "
                f"现价{r['最新价']} | 上{r['上轨']}/中{r['中轨']}/下{r['下轨']}")
    if not reso.empty:
        L.append(f"### 🎯 W底遇风口 共{len(reso)}只")
        L += [line(r) for _, r in reso.iterrows()]; L.append("")
    L.append(f"### 📈 全部W底 共{len(df)}只 (按新鲜度)")
    L += [line(r) for _, r in df.iterrows()]
    return "\n".join(L)


# ------------------ 主程序 (不拦交易日 + 收尾防护) ------------------
def main():
    print("=" * 70)
    print(f"📈 布林带+W底突破 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['lookback_days']}天 近{PARAMS['recent_days']}天信号 alpha={PARAMS['alpha']}")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 双源baostock+东财; 预筛={'开' if PARAMS['SNAPSHOT_PRE'] else '关'}; 不拦交易日; 推送全列+分页")
    print("=" * 70)
    df = run_scan()
    if df is None or df.empty:
        print("\n本次扫描未发现符合条件的近期W底信号。")
        print("可调: ALPHA 调大(如0.01~0.015) / RECENT_DAYS 调大 / 看失败统计定位瓶颈")
        sys.exit(0)
    df, cluster, hot = enrich(df)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"boll_wbottom_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"boll_wbottom_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"recent_days": PARAMS['recent_days'], "alpha": PARAMS['alpha']},
                       "cluster": cluster, "n": int(len(df)),
                       "n_resonance": int(df['resonance'].sum()) if 'resonance' in df.columns else 0,
                       "fail_stats": FAIL_STATS, "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/boll_wbottom_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy(); disp.insert(2, '板块', [_sec_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', 'resonance', 'resonance_sector', 'coordinates'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
            send_serverchan(f"📈 布林带+W底 命中{len(df)}只 🎯风口{n_reso}", build_push(df, cluster, hot))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
# >>>FILE_END_boll_wbottom<<<
