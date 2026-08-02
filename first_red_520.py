import os
import re
import sys
import time
import random
import json
import traceback
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak
import baostock as bs
from tqdm import tqdm

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ------------------ 阈值参数 ------------------
LOW_WINDOW = int(os.environ.get('LOW_WINDOW', '520'))
VOL_WINDOW = int(os.environ.get('VOL_WINDOW', '20'))
NEW_LOW_TOLERANCE = float(os.environ.get('NEW_LOW_TOLERANCE', '1.02'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
QUERY_TIMEOUT_SEC = 15
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))        # 0=全扫
# 【本版新增】共振开关: 0=全推首红+标老鸭头/W底共振; 1=只推有共振(老鸭头或W底)的票
RESONANCE_ONLY = os.environ.get('RESONANCE_ONLY', '0').strip() in ('1', 'true', 'True')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '10'))     # 见底板块聚类展示数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模块级行业映射: run 阶段主进程登录时一次拿全(baostock国标), enrich 本地 join, 零逐只接口零限流
_INDUSTRY_MAP = {}


# ------------------ 推送 (全列+分页) ------------------
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


def is_trading_day():
    """保留备用(当前 main 不调用, 即不拦交易日); 想恢复拦截把 main 里那两行 if 加回即可"""
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
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求将走东财兜底")


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 数据兜底 ------------------
def _fetch_hist_em(sym, start_y):
    """东财 K 线兜底(前复权); 返回 date/open/high/low/close/volume 或 None; 子进程内调用(进程安全)"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                  '最低': 'low', '收盘': 'close', '成交量': 'volume'})
            if 'close' not in d.columns:
                return None
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in d.columns]
            return d[cols] if len(d) >= 60 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _fetch_list_akshare():
    """akshare 兜底取股票列表; 构造与 baostock 同结构(code 带 sh./sz. 前缀, 含 type/status)"""
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


def _clean_industry(s):
    """清洗 baostock 国标行业名: 去掉 'C39 ' 这类字母+数字前缀, 留可读行业名"""
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


# ------------------ 实时对齐快照 + 时点后缀 ------------------
def _fetch_spot_now():
    """取一次全A实时快照, 返回 {六位代码: 现价}; 限流/失败返回空dict, 绝不阻断主流程。"""
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            d = ex.submit(ak.stock_zh_a_spot_em).result(timeout=25)
        if d is None or d.empty or '代码' not in d.columns:
            print("  实时对齐: 快照空(限流), 对齐列降级只显示信号时点")
            return {}
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        if '最新价' in d.columns:
            d['最新价'] = pd.to_numeric(d['最新价'], errors='coerce')
        out = {r['代码']: float(r['最新价']) for _, r in d.iterrows() if pd.notna(r.get('最新价'))}
        print(f"  实时对齐: 取到 {len(out)} 只现价(用于推送对齐列)")
        return out
    except Exception as e:
        print(f"  实时对齐: 快照失败({e}), 对齐列降级只显示信号时点")
        return {}


def _align_suffix(r, spot_now):
    """拼 '🕒信号价@信号日(距今N天) → 现价X@run(±Y%)'。逐层降级。"""
    sig_price = r.get('最新价')
    sig_date = r.get('信号日期')
    if sig_price is None or (hasattr(pd, 'isna') and pd.isna(sig_price)):
        return ""
    head = f"🕒信号{sig_price}"
    if sig_date is not None and not (hasattr(pd, 'isna') and pd.isna(sig_date)):
        sd = str(sig_date)[:10]
        head += f"@{sd[-5:]}"
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


# ------------------ 策略内核1: 520天首红 (一字未动) ------------------
def detect_first_red_to_520_low(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['520_low'] = df['low'].rolling(LOW_WINDOW).min()
    df['5_low'] = df['low'].rolling(5).min()
    df['avg_vol'] = df['volume'].rolling(VOL_WINDOW).mean()
    df['is_red'] = df['close'] > df['open']
    df['made_new_low_recently'] = df['low'] <= df['520_low'].shift(1) * NEW_LOW_TOLERANCE

    signals = []
    in_low_zone = False
    for i in range(LOW_WINDOW, len(df)):
        if df['made_new_low_recently'].iloc[i]:
            in_low_zone = True
        if in_low_zone and df['is_red'].iloc[i]:
            vol_ratio = (
                df['volume'].iloc[i] / df['avg_vol'].iloc[i]
                if df['avg_vol'].iloc[i] > 0 else 0
            )
            distance_pct = (df['close'].iloc[i] - df['520_low'].iloc[i]) / df['520_low'].iloc[i] * 100

            five_day_low = df['5_low'].iloc[i]
            distance_pct_5 = (
                (df['close'].iloc[i] - five_day_low) / five_day_low * 100
                if pd.notna(five_day_low) and five_day_low > 0 else None
            )

            signals.append({
                'date': df['date'].iloc[i],
                'close': df['close'].iloc[i],
                'vol_ratio': vol_ratio,
                'distance_pct': round(distance_pct, 2),
                '5日最低价': round(float(five_day_low), 2) if pd.notna(five_day_low) else None,
                '距5日低点%': round(distance_pct_5, 2) if distance_pct_5 is not None else None,
            })
            in_low_zone = False

    return pd.DataFrame(signals)


# ------------------ 策略内核2: 周线老鸭头 + 日线W底 (共振标记用, 简化版无scipy) ------------------
def _calc_macd(close, fast=12, slow=26, sig=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def _detect_weekly_duck(df_daily):
    """周线老鸭头(简化版, 无背离): 60周线向上+5/10上穿60+回调合适+短期多头+周线MACD金叉。返回(成立?, 得分)。"""
    try:
        w = df_daily.set_index('date').resample('W').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        if len(w) < 80:
            return False, 0
        w['ma5'] = w['close'].rolling(5).mean()
        w['ma10'] = w['close'].rolling(10).mean()
        w['ma60'] = w['close'].rolling(60).mean()
        if pd.isna(w['ma60'].iloc[-1]) or pd.isna(w['ma60'].iloc[-20]):
            return False, 0
        score = 0
        if (w['ma60'].iloc[-1] - w['ma60'].iloc[-20]) / 20 > 0:   # 60周线向上
            score += 1
        cross5 = (w['ma5'].shift(1) < w['ma60'].shift(1)) & (w['ma5'] > w['ma60'])
        cross10 = (w['ma10'].shift(1) < w['ma60'].shift(1)) & (w['ma10'] > w['ma60'])
        if (cross5 | cross10).iloc[-30:].any():                    # 5/10上穿60
            score += 1
        win = w.iloc[-25:]                                          # 回调幅度合适
        peak, trough = win['high'].max(), win['low'].min()
        if peak > 0 and 0.08 <= (peak - trough) / peak <= 0.35:
            score += 1
        if w['ma5'].iloc[-1] > w['ma10'].iloc[-1] and w['ma10'].iloc[-1] > w['ma60'].iloc[-1] * 0.98:  # 短期多头
            score += 1
        dif, dea, hist = _calc_macd(w['close'])                     # 周线MACD金叉
        cross = (dif.shift(1) < dea.shift(1)) & (dif > dea)
        if cross.iloc[-6:].any():
            score += 2
        elif dif.iloc[-1] > dea.iloc[-1]:
            score += 1
        return score >= 4, score
    except Exception:
        return False, 0


def _detect_daily_wbottom(df_daily):
    """日线W底(简化版, numpy找两底, 无scipy): 两底接近+突破颈线+放量+MACD金叉。返回(成立?, 得分)。"""
    try:
        if len(df_daily) < 60:
            return False, 0
        recent = df_daily.iloc[-40:].reset_index(drop=True)
        lows = recent['low'].to_numpy(float)
        # numpy 找局部极小值(比左右都低)
        peaks = []
        for i in range(1, len(lows) - 1):
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                peaks.append(i)
        # distance 过滤(至少隔8根)
        filt = []
        for p in peaks:
            if not filt or p - filt[-1] >= 8:
                filt.append(p)
        if len(filt) < 2:
            return False, 0
        p1, p2 = filt[-2], filt[-1]
        low1, low2 = lows[p1], lows[p2]
        if min(low1, low2) <= 0:
            return False, 0
        bottoms_close = abs(low1 - low2) / min(low1, low2) <= 0.05   # 两底接近
        neck = recent['high'].iloc[p1:p2 + 1].max()
        breakthrough = df_daily['close'].iloc[-1] > neck * 0.995     # 突破颈线
        vol_ma = df_daily['volume'].rolling(20).mean().iloc[-1]
        vol_break = df_daily['volume'].iloc[-1] > vol_ma * 1.5 if vol_ma > 0 else False  # 放量
        dif, dea, hist = _calc_macd(df_daily['close'].astype(float))
        cross = (dif.shift(1) <= dea.shift(1)) & (dif > dea)
        macd_cross = bool(cross.iloc[-6:].any())                     # MACD金叉
        score = 0
        if bottoms_close: score += 1
        if breakthrough: score += 1
        if vol_break: score += 1
        if macd_cross: score += 1
        return score >= 3, score
    except Exception:
        return False, 0


# ------------------ 单只处理 (K线双源 + 首红 + 老鸭头/W底共振) ------------------
def _process_one(args):
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=int(LOW_WINDOW * 1.6))).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录)
    try:
        df = _query_with_timeout(code, "date,open,high,low,close,volume", start_dash)
        if df is None or df.empty or len(df) < LOW_WINDOW:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时的空/异常 -> 子进程内重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,open,high,low,close,volume", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= LOW_WINDOW:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None or len(df) < LOW_WINDOW:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        signals = detect_first_red_to_520_low(df)
        if signals.empty:
            return None

        latest = signals.iloc[-1]
        if (df['date'].iloc[-1] != latest['date']) and \
           (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(latest['date'])).days > 7:
            return None

        # 【本版新增】周线老鸭头 + 日线W底 共振标记(不改变首红命中, 仅叠加形态信息)
        duck_ok, duck_score = _detect_weekly_duck(df)
        w_ok, w_score = _detect_daily_wbottom(df)
        resonance = bool(duck_ok or w_ok)

        # RESONANCE_ONLY=1 时, 只推有共振的票
        if RESONANCE_ONLY and not resonance:
            return None

        return {
            "代码": code, "名称": name, "行业": "",
            "信号日期": latest['date'],
            "最新价": round(float(latest['close']), 2),
            "量比": round(float(latest['vol_ratio']), 2),
            "距520日低点%": latest['distance_pct'],
            "5日最低价": latest['5日最低价'],
            "距5日低点%": latest['距5日低点%'],
            "周线老鸭头": "✓" if duck_ok else "—",
            "日线W底": "✓" if w_ok else "—",
            "鸭头分": duck_score, "W底分": w_score,
            "共振": resonance,
        }
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_first_red_520_scan(limit=SCAN_LIMIT):
    global _INDUSTRY_MAP
    print("正在连接 Baostock（主进程，用于取股票列表+行业映射）...")
    stock_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            stock_df = bs.query_stock_basic().get_data()
        except Exception as e:
            print(f"  baostock 取列表异常: {e}")
            stock_df = pd.DataFrame()
        try:
            ind_df = bs.query_stock_industry().get_data()
            if ind_df is not None and not ind_df.empty and 'code' in ind_df.columns and 'industry' in ind_df.columns:
                for _, row in ind_df.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row['industry'])
                print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条 (baostock国标行业, 一次拿全, 命中再多也不限流)")
        except Exception as e:
            print(f"  baostock 取行业表异常: {e}")
        bs.logout()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("  baostock 列表无效, 切换 akshare 兜底取列表 ...")
        stock_df = _fetch_list_akshare()

    if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
        print("⚠️ 双源均无法获取股票列表, 本次跳过")
        return pd.DataFrame()

    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ].copy()
    stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
    if stock_df.empty:
        print("⚠️ 过滤后无股票, 本次跳过")
        return pd.DataFrame()

    codes = stock_df['code'].tolist()
    if limit and len(codes) > limit:
        codes = codes[:limit]
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in codes]

    results = []
    fail_count = 0
    print(f"开始520天首红扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 进程并行, K线双源, 共振标记={'仅共振' if RESONANCE_ONLY else '全推+标记'}）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    tag = "★共振 " if res.get('共振') else ""
                    pbar.write(f"✅ {tag}命中: {res['代码']} {res['名称']}（量比 {res['量比']} 鸭头{res['周线老鸭头']}/W底{res['日线W底']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(["共振", "量比"], ascending=[False, False]).reset_index(drop=True)
    return result_df


# ------------------ 行业标注(本地join) + 见底聚类 ------------------
def enrich(results):
    if not results:
        return pd.DataFrame(), []

    mapped = 0
    for r in results:
        ind = _INDUSTRY_MAP.get(r['代码'], '—')
        r['行业'] = ind
        if ind not in ('—', '未知', ''):
            mapped += 1
    print(f"🏷️ 行业标注完成: {mapped}/{len(results)} 只命中行业映射 (baostock国标, 本地join)")

    labeled = [r for r in results if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"🔻 见底板块(首红扎堆, 板块级反转信号): {cluster or '无'}")

    # 终排序: 共振优先, 再按量比
    results.sort(key=lambda r: (1 if r.get('共振') else 0, r['量比']), reverse=True)
    return pd.DataFrame(results), cluster


def sec_tag(r):
    return r.get('行业') or '—'


def _resonance_tag(r):
    """共振标记: 双共振/鸭头/W底"""
    duck = r.get('周线老鸭头') == '✓'
    w = r.get('日线W底') == '✓'
    if duck and w:
        return " ★双共振"
    if duck:
        return " ★鸭头"
    if w:
        return " ★W底"
    return ""


def build_push_content(df, cluster, spot_now=None):
    n_reso = int(df['共振'].sum()) if '共振' in df.columns else 0
    lines = []
    lines.append("*(🕒每行末'信号价@日期→现价@run'=首红信号日收盘 vs 本次run实时价; ★鸭头/★W底/★双共振=首红叠加周线老鸭头/日线W底形态共振, 更强底部反转信号; 简化版无背离)*")
    lines.append("")
    if cluster:
        lines.append("🔻 **见底板块**(首红扎堆, 板块级反转): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    lines.append(f"### 📋 520天首红 共{len(df)}只 (★共振{n_reso}只, 全发, 含板块)")
    for _, r in df.iterrows():
        five = f" | 5日最低{r['5日最低价']}（距5日低点{r['距5日低点%']}%）" if pd.notna(r.get('5日最低价')) else ""
        base = (f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}]{_resonance_tag(r)} {r['信号日期']} 最新价 {r['最新价']} "
                f"| 量比 {r['量比']} | 距520日低点 {r['距520日低点%']}%{five}")
        lines.append(base + _align_suffix(r, spot_now))
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"520天首红扫描(+周线老鸭头/日线W底共振标记) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print(f"K线双源(baostock+东财); 板块=baostock国标行业本地join; 共振模式={'仅推共振票' if RESONANCE_ONLY else '全推首红+标共振'}; 不拦交易日; 推送全列+分页")
    print("【本版】首红逻辑不变, 叠加周线老鸭头+日线W底共振标记(简化版无scipy); 要完整老鸭头+W底+背离用独立 first_red_wbottom_screener")
    print("=" * 70)

    df = run_first_red_520_scan(limit=SCAN_LIMIT)

    if df is None or df.empty:
        print("本次未找到符合条件的股票" + ("(RESONANCE_ONLY=1且无共振票; 设RESONANCE_ONLY=0看全部首红)" if RESONANCE_ONLY else ""))
        sys.exit(0)

    df, cluster = enrich(df.to_dict('records'))
    tag = datetime.now().strftime('%Y%m%d')
    try:
        csv_path = f"{OUTPUT_DIR}/first_red_520_{tag}.csv"
        json_path = f"{OUTPUT_DIR}/first_red_520_{tag}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"\n结果已保存: {csv_path}")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存, 不影响结果): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.copy()
        disp.insert(2, '板块', [sec_tag(r) for r in df.to_dict('records')])
        disp = disp.drop(columns=['行业'], errors='ignore')
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            spot_now = _fetch_spot_now()
            n_reso = int(df['共振'].sum()) if '共振' in df.columns else 0
            title = f"520天首红 命中 {len(df)} 只 ★共振{n_reso}（全发, 含板块）"
            content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster, spot_now)
            send_serverchan(title, content)
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_first_red_520<<<
