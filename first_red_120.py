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

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ------------------ 阈值参数 ------------------
LOW_WINDOW = int(os.environ.get('LOW_WINDOW', '120'))   # 120日首红: 半年新低后首阳
VOL_WINDOW = int(os.environ.get('VOL_WINDOW', '20'))
NEW_LOW_TOLERANCE = float(os.environ.get('NEW_LOW_TOLERANCE', '1.02'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
QUERY_TIMEOUT_SEC = 15
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '10'))

# 【本版新增】MACD/布林/粘合 三指标 (首红: 粘合与首红矛盾, 故粘合只做加分+可选硬门槛, 不默认硬卡)
BB_NARROW_MAX = float(os.environ.get('BB_NARROW_MAX', '0.12'))   # 布林带宽<此=收窄
MA_TIGHT_MAX = float(os.environ.get('MA_TIGHT_MAX', '0.02'))     # MA5/10/20极差<此=粘合
REQUIRE_MACD_TURN = os.environ.get('REQUIRE_MACD_TURN', '1').strip() in ('1', 'true', 'True')   # 首红需MACD转强(默认开, 收紧275; 首红当天可满足, 不矛盾)
REQUIRE_BB_LOW = os.environ.get('REQUIRE_BB_LOW', '0').strip() in ('1', 'true', 'True')          # 首红需布林下轨区(更严, 默认关)
REQUIRE_MA_TIGHT = os.environ.get('REQUIRE_MA_TIGHT', '0').strip() in ('1', 'true', 'True')      # 首红需粘合(与首红矛盾! 默认关; 开=只要"横盘后首红"稀有形态)

os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 数据兜底 ------------------
def _fetch_hist_em(sym, start_y):
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
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"


# ------------------ 实时对齐快照 + 时点后缀 ------------------
def _fetch_spot_now():
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


# ------------------ 策略内核 (首红逻辑+【本版】MACD/布林/粘合三指标) ------------------
def detect_first_red_to_120_low(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['120_low'] = df['low'].rolling(LOW_WINDOW).min()
    df['5_low'] = df['low'].rolling(5).min()
    df['avg_vol'] = df['volume'].rolling(VOL_WINDOW).mean()
    df['is_red'] = df['close'] > df['open']
    df['made_new_low_recently'] = df['low'] <= df['120_low'].shift(1) * NEW_LOW_TOLERANCE

    # 【本版新增】均线/ MACD / 布林带 (用于加分+标签+可选硬门槛; 不改变首红判定本身)
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (df['dif'] - df['dea']) * 2
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_bw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

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
            distance_pct = (df['close'].iloc[i] - df['120_low'].iloc[i]) / df['120_low'].iloc[i] * 100

            five_day_low = df['5_low'].iloc[i]
            distance_pct_5 = (
                (df['close'].iloc[i] - five_day_low) / five_day_low * 100
                if pd.notna(five_day_low) and five_day_low > 0 else None
            )

            # 【本版新增】三指标状态 (在首红信号点 i 取值)
            _m5 = float(df['ma5'].iloc[i]); _m10 = float(df['ma10'].iloc[i]); _m20 = float(df['ma20'].iloc[i])
            _ma_tight = bool(pd.notna(_m20) and (max(_m5, _m10, _m20) - min(_m5, _m10, _m20)) / df['close'].iloc[i] < MA_TIGHT_MAX)
            _bb_pct = float(df['bb_pct'].iloc[i]) if pd.notna(df['bb_pct'].iloc[i]) else None
            _bb_narrow = bool(pd.notna(df['bb_bw'].iloc[i]) and df['bb_bw'].iloc[i] < BB_NARROW_MAX)
            _macd_turn = bool(i > 0 and pd.notna(df['macd_hist'].iloc[i]) and pd.notna(df['macd_hist'].iloc[i - 1])
                              and df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i - 1])
            _macd_golden = bool(i > 0 and pd.notna(df['dif'].iloc[i]) and pd.notna(df['dea'].iloc[i])
                                and df['dif'].iloc[i] > df['dea'].iloc[i] and df['dif'].iloc[i - 1] <= df['dea'].iloc[i - 1])

            signals.append({
                'date': df['date'].iloc[i],
                'close': df['close'].iloc[i],
                'vol_ratio': vol_ratio,
                'distance_pct': round(distance_pct, 2),
                '5日最低价': round(float(five_day_low), 2) if pd.notna(five_day_low) else None,
                '距5日低点%': round(distance_pct_5, 2) if distance_pct_5 is not None else None,
                'macd_turn': _macd_turn, 'macd_golden': _macd_golden,
                'bb_pct': round(_bb_pct, 2) if _bb_pct is not None else None,
                'bb_narrow': _bb_narrow, 'ma_tight': _ma_tight,
            })
            in_low_zone = False

    return pd.DataFrame(signals)


# ------------------ 单只处理 (K线双源 + 【本版】三指标硬门槛/加分) ------------------
def _process_one(args):
    code, name = args
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=int(LOW_WINDOW * 1.6))).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    try:
        df = _query_with_timeout(code, "date,open,high,low,close,volume", start_dash)
        if df is None or df.empty or len(df) < LOW_WINDOW:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

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

    if df is None:
        df = _fetch_hist_em(sym, start_y)

    if df is None or len(df) < LOW_WINDOW:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        signals = detect_first_red_to_120_low(df)
        if signals.empty:
            return None

        latest = signals.iloc[-1]
        if (df['date'].iloc[-1] != latest['date']) and \
           (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(latest['date'])).days > 7:
            return None

        # 【本版新增】三指标硬门槛 (粘合默认关, 因与首红矛盾; MACD转强默认开, 首红当天可满足)
        macd_ok = bool(latest.get('macd_turn')) or bool(latest.get('macd_golden'))
        if REQUIRE_MACD_TURN and not macd_ok:
            return None
        if REQUIRE_BB_LOW and not (latest.get('bb_pct') is not None and latest['bb_pct'] < 0.3):
            return None
        if REQUIRE_MA_TIGHT and not latest.get('ma_tight'):
            return None

        # 【本版新增】综合排序分 = 量比 + 三指标加分 (粘合/布林/金叉优先)
        score = round(float(latest['vol_ratio'])
                      + (10 if macd_ok else 0) + (15 if latest.get('ma_tight') else 0)
                      + (10 if latest.get('bb_narrow') else 0) + (10 if latest.get('macd_golden') else 0), 1)

        return {
            "代码": code, "名称": name, "行业": "",
            "信号日期": latest['date'],
            "最新价": round(float(latest['close']), 2),
            "量比": round(float(latest['vol_ratio']), 2),
            "距120日低点%": latest['distance_pct'],
            "5日最低价": latest['5日最低价'],
            "距5日低点%": latest['距5日低点%'],
            "MACD转强": macd_ok, "MACD金叉": bool(latest.get('macd_golden')),
            "均线粘合": bool(latest.get('ma_tight')), "布林收窄": bool(latest.get('bb_narrow')),
            "布林%B": latest.get('bb_pct'),
            "score": score,
        }
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_first_red_120_scan(limit=SCAN_LIMIT):
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
    print(f"开始120天首红扫描 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行, K线=baostock+东财双源）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（量比 {res['量比']} 分{res['score']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)  # 【本版】按综合分(量比+三指标)排序
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

    results.sort(key=lambda r: r.get('score', 0), reverse=True)  # 【本版】按综合分排序
    return pd.DataFrame(results), cluster


def sec_tag(r):
    return r.get('行业') or '—'


def _ind_tags(r):
    """【本版新增】三指标状态标签 (粘合/布林/ MACD), 拼到推送行"""
    t = []
    if r.get('MACD金叉'):
        t.append('MACD金叉')
    elif r.get('MACD转强'):
        t.append('MACD转强')
    if r.get('均线粘合'):
        t.append('均线粘合')
    if r.get('布林收窄'):
        t.append('布林收窄')
    elif r.get('布林%B') is not None and r['布林%B'] < 0.2:
        t.append('布林下轨')
    return (' 〔' + '·'.join(t) + '〕') if t else ''


def build_push_content(df, cluster, spot_now=None):
    lines = []
    lines.append("*(🕒每行末'信号价@日期→现价@run'=首红信号日收盘 vs 本次run实时价; 〔〕内=MACD/布林/粘合状态; 信号价=前复权日线, 现价=不复权实时, 除权则涨幅仅供参考)*")
    lines.append("")
    if cluster:
        lines.append("🔻 **见底板块**(首红扎堆, 板块级反转): " +
                     "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    lines.append(f"### 📋 120天首红 共{len(df)}只 (全发, 含板块, 按综合分)")
    for _, r in df.iterrows():
        five = f" | 5日最低{r['5日最低价']}（距5日低点{r['距5日低点%']}%）" if pd.notna(r.get('5日最低价')) else ""
        base = (f"- {r['名称']}（{r['代码']}）[{sec_tag(r.to_dict())}] {r['信号日期']} 最新价 {r['最新价']} "
                f"| 量比 {r['量比']} | 距120日低点 {r['距120日低点%']}%{five}{_ind_tags(r)}")
        lines.append(base + _align_suffix(r, spot_now))
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"120天首红扫描 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 上限={'全扫' if not SCAN_LIMIT else SCAN_LIMIT}")
    print(f"K线双源; 板块=baostock国标本地join; 不拦交易日; 推送全列+分页+对齐列")
    print(f"【三指标】MACD转强硬门槛={'开' if REQUIRE_MACD_TURN else '关'} 布林下轨硬门槛={'开' if REQUIRE_BB_LOW else '关'} 粘合硬门槛={'开' if REQUIRE_MA_TIGHT else '关(粘合与首红矛盾,默认仅加分)'}; 三指标加分排序")
    print("=" * 70)

    df = run_first_red_120_scan(limit=SCAN_LIMIT)

    if df is None or df.empty:
        print("本次未找到符合条件的股票")
        sys.exit(0)

    df, cluster = enrich(df.to_dict('records'))
    tag = datetime.now().strftime('%Y%m%d')
    try:
        csv_path = f"{OUTPUT_DIR}/first_red_120_{tag}.csv"
        json_path = f"{OUTPUT_DIR}/first_red_120_{tag}.json"
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
            title = f"120天首红信号 命中 {len(df)} 只（全发, 含板块）"
            content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(df, cluster, spot_now)
            send_serverchan(title, content)
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_first_red_120<<<
