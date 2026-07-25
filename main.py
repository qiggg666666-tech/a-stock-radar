# -*- coding: utf-8 -*-
"""
main.py —— 主策略: 日/周/月/季 四周期"即将站上20日线"共振 + 周线宽松辅助
====================================================================
【本版核心修改】(应用户要求: 日线/周线/月线/季线即将站上20日线)
  原"三线共振(周/月/年 MA5/MA20 即将金叉)" -> "四周期(日/周/月/季)即将站上20日线共振"。
  CROSS_MODE 切换语义:
    price(默认) = 各周期收盘价逼近20均线(略低于MA20, 即将突破站上) —— 贴近"站上20日线"字面;
    ma          = 各周期 MA5 即将上穿 MA20(均线金叉临界) —— 原三线共振逻辑扩展到四周期。
  MIN_PERIODS(默认4) = 四周期中至少几个同时满足(4=严格全共振; 命中太少可设3=宽松)。
  季线 MA20 = 5年均线, 需≥5年季度数据, 故数据起始日 2020 -> 2019。
【保留】双信号结构(四周期主信号 + 周线宽松辅助)、双源baostock+东财、RSI打分、
  行业join、🌀共振板块聚类、🎯共振遇风口、推送、收尾防护。
⚠️ 四周期严格全共振条件极苛刻, 命中极少属正常; 实用建议 MIN_PERIODS=3 或放宽阈值。
====================================================================
"""
import os
import sys
import time
import random
import json
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ------------------ 阈值参数 (全部 env 可调) ------------------
# 四周期"即将站上20日线"语义与门槛
CROSS_MODE = os.environ.get('CROSS_MODE', 'price')               # price=收盘价逼近MA20 / ma=MA5金叉MA20
MIN_PERIODS = int(os.environ.get('MIN_PERIODS', '4'))            # 四周期至少几个满足(4=严格全共振, 3=宽松)
DAY_THRESHOLD = float(os.environ.get('DAY_THRESHOLD', '0.005'))      # 日线 gap 阈值 0.5%(最敏感)
WEEK_THRESHOLD = float(os.environ.get('WEEK_THRESHOLD', '0.008'))    # 周线 0.8%
MONTH_THRESHOLD = float(os.environ.get('MONTH_THRESHOLD', '0.012'))  # 月线 1.2%
QUARTER_THRESHOLD = float(os.environ.get('QUARTER_THRESHOLD', '0.02'))  # 季线 2.0%(最钝)
# 周线宽松辅助信号门槛
WEEKLY_ONLY_THRESHOLD = float(os.environ.get('WEEKLY_ONLY_THRESHOLD', '0.015'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '15'))
DATA_START_DASH = "2019-01-01"      # 季线MA20需≥5年季度数据, 起始日提到2019
DATA_START_Y = "20190101"
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '500'))        # 主策略默认500; 0=全扫
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
LABEL_TOP = int(os.environ.get('LABEL_TOP', '200'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))

# 多周期共振遇风口(记号🎯)
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 交易日 / 容错 ------------------
def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content)
        print("📲 serverchan-sdk 推送成功")
        return True
    except Exception as e:
        print(f"  serverchan-sdk 失败, 回退 requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=10)
        return r.json().get("code") == 0
    except Exception as e:
        print(f"  requests 推送失败: {e}")
        return False


def is_trading_day():
    try:
        d = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        print(f"  交易日历获取失败, 默认继续: {e}")
        return True


# ------------------ baostock 登录重试 ------------------
def _bs_login_ok(retries=5):
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _init_worker():
    time.sleep(random.uniform(0, 2))
    _bs_login_ok(retries=5)


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """单次baostock查询包硬超时, 防网络卡顿拖死进程池"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 数据兜底 ------------------
def _fetch_hist_em(sym, start_y):
    """东财 K 线兜底(前复权); 主策略只需 date,close"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close'})
            if 'close' not in d.columns:
                return None
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            return d[['date', 'close']] if len(d) >= 150 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_list_akshare():
    for attempt in range(3):
        try:
            d = ak.stock_info_a_code_name()
            if d is not None and not d.empty and 'code' in d.columns:
                name_col = 'name' if 'name' in d.columns else d.columns[1]
                d = d[['code', name_col]].copy()
                d.columns = ['code', 'code_name']
                d['code'] = d['code'].astype(str).str.zfill(6)
                d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                d['type'] = '1'
                d['status'] = '1'
                return d
        except Exception as e:
            print(f"  akshare 股票列表第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    return pd.DataFrame(columns=['code', 'code_name', 'type', 'status'])


# ------------------ 行业 / 风口 / 匹配 ------------------
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


def sec_tag(r):
    return ('🎯' + r.get('hot_sector', '')) if r.get('hot_meet') else (r.get('行业') or '—')


# ------------------ 周期重采样容错 ------------------
def _resample_close(df, rules):
    """按多个候选频率重采样取收盘价(兼容 pandas 新旧频率别名); 全失败返回空 Series"""
    for rule in rules:
        try:
            r = df.resample(rule, on='date')['close'].last().dropna()
            if len(r) > 0:
                return r
        except Exception:
            continue
    return pd.Series(dtype=float)


# ------------------ 策略内核: 四周期即将站上20日线 ------------------
def strategy_quad_cross(df):
    """日/周/月/季 四周期 即将站上20日线 共振。
    CROSS_MODE='price': 各周期收盘价逼近20均线(略低于MA20, 即将突破站上)。
    CROSS_MODE='ma':    各周期 MA5 即将上穿 MA20(均线金叉临界)。
    满足周期数 >= MIN_PERIODS 即命中(默认4=严格全共振)。返回各周期gap+满足情况供打分。"""
    try:
        if len(df) < 260:
            return None
        df = df.copy()
        df['close'] = df['close'].astype(float)
        df['date'] = pd.to_datetime(df['date'])

        # 四周期收盘价序列
        d_close = df['close']
        w_close = _resample_close(df, ['W-FRI'])
        m_close = _resample_close(df, ['ME', 'M'])
        q_close = _resample_close(df, ['QE-DEC', 'Q-DEC', 'QE', 'Q'])

        d_ma20 = d_close.rolling(20).mean(); d_ma5 = d_close.rolling(5).mean()
        w_ma20 = w_close.rolling(20).mean(); w_ma5 = w_close.rolling(5).mean()
        m_ma20 = m_close.rolling(20).mean(); m_ma5 = m_close.rolling(5).mean()
        q_ma20 = q_close.rolling(20).mean(); q_ma5 = q_close.rolling(5).mean()

        if any(len(s.dropna()) == 0 for s in [d_ma20, w_ma20, m_ma20, q_ma20]):
            return None

        if CROSS_MODE == 'ma':   # 均线金叉临界: gap=(MA20-MA5)/MA20
            d_gap = (d_ma20.iloc[-1] - d_ma5.iloc[-1]) / d_ma20.iloc[-1]
            w_gap = (w_ma20.iloc[-1] - w_ma5.iloc[-1]) / w_ma20.iloc[-1]
            m_gap = (m_ma20.iloc[-1] - m_ma5.iloc[-1]) / m_ma20.iloc[-1]
            q_gap = (q_ma20.iloc[-1] - q_ma5.iloc[-1]) / q_ma20.iloc[-1]
        else:                    # price: 收盘价逼近MA20: gap=(MA20-close)/MA20
            d_gap = (d_ma20.iloc[-1] - d_close.iloc[-1]) / d_ma20.iloc[-1]
            w_gap = (w_ma20.iloc[-1] - w_close.iloc[-1]) / w_ma20.iloc[-1]
            m_gap = (m_ma20.iloc[-1] - m_close.iloc[-1]) / m_ma20.iloc[-1]
            q_gap = (q_ma20.iloc[-1] - q_close.iloc[-1]) / q_ma20.iloc[-1]

        日即将 = bool((d_gap > 0) and (d_gap < DAY_THRESHOLD))
        周即将 = bool((w_gap > 0) and (w_gap < WEEK_THRESHOLD))
        月即将 = bool((m_gap > 0) and (m_gap < MONTH_THRESHOLD))
        季即将 = bool((q_gap > 0) and (q_gap < QUARTER_THRESHOLD))

        满足数 = sum([日即将, 周即将, 月即将, 季即将])
        if not (满足数 >= MIN_PERIODS and (df['close'].iloc[-1] > MIN_PRICE)):
            return None

        周期 = "+".join([n for n, ok in [("日", 日即将), ("周", 周即将), ("月", 月即将), ("季", 季即将)] if ok])
        return {"d_gap": d_gap, "w_gap": w_gap, "m_gap": m_gap, "q_gap": q_gap,
                "满足数": 满足数, "周期": 周期, "close": df['close'].iloc[-1]}
    except Exception:
        return None


def strategy_weekly_only(df):
    """辅助信号: 宽松版单周线即将金叉(MA5/MA20 + MA5抬头)。门槛低, 仅辅助参考, 非严格确认。"""
    try:
        if len(df) < 150:
            return None
        d = df.copy()
        d['close'] = d['close'].astype(float)
        d['date'] = pd.to_datetime(d['date'])
        df_week = d.resample('W-FRI', on='date')['close'].last().dropna()
        w_ma5 = df_week.rolling(5).mean().dropna()
        w_ma20 = df_week.rolling(20).mean().dropna()
        if len(w_ma5) < 2 or len(w_ma20) < 2:
            return None
        latest_w5, prev_w5 = w_ma5.iloc[-1], w_ma5.iloc[-2]
        latest_w20 = w_ma20.iloc[-1]
        gap = (latest_w20 - latest_w5) / latest_w20
        if latest_w5 < latest_w20 and 0 <= gap < WEEKLY_ONLY_THRESHOLD and latest_w5 > prev_w5:
            return {"gap": gap, "close": d['close'].iloc[-1]}
        return None
    except Exception:
        return None


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_signal_score(gaps, df):
    """四周期综合打分(0-100): 各周期临界程度加权(越接近金叉/站上分越高) + 满足数bonus + RSI。"""
    score = 0.0
    score += max(0, (1 - gaps["d_gap"] / DAY_THRESHOLD)) * 20      # 日线 20
    score += max(0, (1 - gaps["w_gap"] / WEEK_THRESHOLD)) * 22     # 周线 22
    score += max(0, (1 - gaps["m_gap"] / MONTH_THRESHOLD)) * 22    # 月线 22
    score += max(0, (1 - gaps["q_gap"] / QUARTER_THRESHOLD)) * 16  # 季线 16
    score += (gaps.get("满足数", 0) - MIN_PERIODS) * 3             # 超出最低要求的周期数 bonus
    rsi = calculate_rsi(df['close'])
    if rsi is not None:
        if 30 <= rsi <= 55:
            score += 10
        elif rsi > 70:
            score -= 10
        elif rsi < 20:
            score += 5
    return round(max(0, min(100, score)), 1)


# ------------------ 单只处理 (K线双源, 双信号结构保留) ------------------
def _process_one(args):
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    df = None
    timed_out = False

    # 路径1: baostock
    try:
        df = _query_with_timeout(code, "date,close", DATA_START_DASH)
        if df is None or df.empty or len(df) < 150:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时 -> 子进程内重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close", DATA_START_DASH)
                if df2 is not None and not df2.empty and len(df2) >= 150:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym, DATA_START_Y)

    if df is None or len(df) < 150:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        signals = []
        hit = {"代码": code, "名称": name, "行业": "", "评分": None, "周线宽松评分": None,
               "满足周期": "", "hot_meet": False, "hot_sector": ""}

        gaps = strategy_quad_cross(df)
        if gaps:
            signals.append("四周期共振")
            hit["评分"] = calculate_signal_score(gaps, df)
            hit["满足周期"] = gaps["周期"]
            hit["最新价"] = round(float(gaps["close"]), 2)

        weekly_res = strategy_weekly_only(df)
        if weekly_res:
            signals.append("周线宽松")
            hit["周线宽松评分"] = round(max(0, (1 - weekly_res["gap"] / WEEKLY_ONLY_THRESHOLD)) * 100, 1)
            hit.setdefault("最新价", round(float(weekly_res["close"]), 2))

        if not signals:
            return None

        hit["信号"] = "+".join(signals)
        hit["_排序权重"] = hit["评分"] if hit["评分"] is not None else (hit["周线宽松评分"] or 0) * 0.5
        return hit
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_all_strategies(limit=SCAN_LIMIT):
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        bs.logout()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切换 akshare 兜底取列表 ...")
        stock_df = _fetch_list_akshare()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("⚠️ 双源均无法获取股票列表, 本次跳过")
        return []

    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    if stock_df.empty:
        print("⚠️ 过滤后无股票, 本次跳过")
        return []

    codes = stock_df['code'].tolist()
    if limit and len(codes) > limit:
        codes = codes[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    results = []
    fail_count = 0
    print(f"开始检测 {len(tasks)} 只（{NUM_PROCESSES} 进程, 双源, 四周期模式={CROSS_MODE}, MIN_PERIODS={MIN_PERIODS}）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（{res['信号']} {res.get('满足周期','')}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    return results


# ------------------ 行业标注 + 共振聚类 + 共振遇风口 ------------------
def _weight(r):
    sc = r.get('评分')
    if sc is not None and pd.notna(sc):
        return float(sc)
    wk = r.get('周线宽松评分')
    return (float(wk) if wk is not None and pd.notna(wk) else 0.0) * 0.5


def enrich(results):
    if not results:
        return pd.DataFrame(), [], []
    targets = results[:LABEL_TOP]
    print(f"为 {len(targets)} 只命中标的补行业 ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"🌀 共振板块(四周期即将站上20日线扎堆, 板块级中线启动): {cluster or '无'}")

    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    meet_cnt = 0
    for r in results:
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['hot_meet'] = True
            r['hot_sector'] = m
            meet_cnt += 1
    print(f"🎯 共振遇风口 {meet_cnt} 只 (多周期共振+板块催化)")

    results.sort(key=lambda r: (1 if r.get('hot_meet') else 0, _weight(r)), reverse=True)
    df = pd.DataFrame(results).drop(columns=["_排序权重"], errors='ignore')
    return df, cluster, hot


def build_push_content(df, cluster, hot):
    P = PUSH_TOP
    lines = []
    if hot:
        lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
        lines.append("")
    if cluster:
        lines.append("🌀 **共振板块**(四周期即将站上20日线扎堆): " + "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    meet = df[df['hot_meet'] == True] if 'hot_meet' in df.columns else pd.DataFrame()
    if not meet.empty:
        lines.append(f"### 🎯 共振遇风口 Top{min(len(meet), P)} (多周期共振+板块催化)")
        for _, row in meet.head(P).iterrows():
            parts = [f"- {row['名称']}（{row['代码']}）[🎯{row['hot_sector']}] 最新价 {row.get('最新价')} | 信号: {row['信号']}"]
            if row.get('满足周期'):
                parts.append(f"周期 {row['满足周期']}")
            if pd.notna(row.get('评分')):
                parts.append(f"评分 {row['评分']}")
            if pd.notna(row.get('周线宽松评分')):
                parts.append(f"周线宽松 {row['周线宽松评分']}")
            lines.append(" | ".join(parts))
        lines.append("")
    lines.append(f"### 📋 全部共振 Top{min(len(df), P)}")
    for _, row in df.head(P).iterrows():
        parts = [f"- {row['名称']}（{row['代码']}）[{sec_tag(row.to_dict())}] 最新价 {row.get('最新价')} | 信号: {row['信号']}"]
        if row.get('满足周期'):
            parts.append(f"周期 {row['满足周期']}")
        if pd.notna(row.get('评分')):
            parts.append(f"评分 {row['评分']}")
        if pd.notna(row.get('周线宽松评分')):
            parts.append(f"周线宽松 {row['周线宽松评分']}")
        lines.append(" | ".join(parts))
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output 报告*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"主策略 四周期即将站上20日线 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print(f"模式={CROSS_MODE} | MIN_PERIODS={MIN_PERIODS} | 阈值 日{DAY_THRESHOLD}/周{WEEK_THRESHOLD}/月{MONTH_THRESHOLD}/季{QUARTER_THRESHOLD}")
    print("=" * 70)

    # 交易日拦截仅对定时触发(schedule)生效; 手动/本地不受限, 周末可用周五数据
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule' and not is_trading_day():
        print("非交易日且为定时触发(schedule), 跳过; 手动/本地运行不受此限")
        sys.exit(0)

    results = run_all_strategies(limit=SCAN_LIMIT)

    if results:
        df, cluster, hot = enrich(results)
        tag = datetime.now().strftime('%Y%m%d')
        # ---- 收尾全部包防护 ----
        try:
            csv_path = f"{OUTPUT_DIR}/main_screener_{tag}.csv"
            json_path = f"{OUTPUT_DIR}/main_screener_{tag}.json"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            df.to_json(json_path, orient='records', force_ascii=False, indent=2)
            print(f"\n结果已保存: {csv_path} (共 {len(df)} 只)")
        except Exception as e:
            print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        try:
            disp = df.head(PUSH_TOP).copy()
            disp.insert(2, '板块', [sec_tag(r) for r in df.head(PUSH_TOP).to_dict('records')])
            disp = disp.drop(columns=['行业', 'hot_meet', 'hot_sector'], errors='ignore')
            print("\n" + disp.to_string(index=False))
        except Exception as e:
            print(f"⚠️ 展示异常: {e}")
        if SERVERCHAN_KEY:
            try:
                meet_n = int(df['hot_meet'].sum()) if 'hot_meet' in df.columns else 0
                title = f"主策略 命中{len(df)}只 🌀共振{len(cluster)} 🎯精准{meet_n}"
                content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster, hot)
                send_serverchan(title, content)
            except Exception as e:
                print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
    else:
        print("本次未找到符合条件的股票 (四周期全共振条件极严, 命中0只属正常; 可设 MIN_PERIODS=3 放宽)")
    sys.exit(0)
