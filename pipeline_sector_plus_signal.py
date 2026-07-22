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
# 不依赖 pandas 内部的 _append（该私有方法在 pandas 3.0+ 也被移除了），
# 直接用 pd.concat 重新实现，兼容任意 pandas 版本
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# ------------------ 板块选择参数 (全部 env 可调) ------------------
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
SECTOR_STOCK_LIMIT = int(os.environ.get('SECTOR_STOCK_LIMIT', '800'))   # 参与板块打分的股票池大小
TOP_N_SECTORS = int(os.environ.get('TOP_N_SECTORS', '5'))
MIN_SECTOR_SCORE = float(os.environ.get('MIN_SECTOR_SCORE', '55'))
MIN_STOCKS_IN_SECTOR = int(os.environ.get('MIN_STOCKS_IN_SECTOR', '5'))
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '15'))

# ------------------ 个股信号参数（与main.py保持一致）------------------
WEEK_THRESHOLD = float(os.environ.get('WEEK_THRESHOLD', '0.008'))
MONTH_THRESHOLD = float(os.environ.get('MONTH_THRESHOLD', '0.012'))
YEAR_THRESHOLD = float(os.environ.get('YEAR_THRESHOLD', '0.018'))
WEEKLY_ONLY_THRESHOLD = float(os.environ.get('WEEKLY_ONLY_THRESHOLD', '0.015'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 交易日 / 容错 ------------------
def send_serverchan(title, content, sendkey=""):
    """Server酱推送: serverchan-sdk 软导入优先, requests 兜底"""
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
    """每个子进程独立登录 baostock (非线程/进程安全)"""
    time.sleep(random.uniform(0, 2))
    _bs_login_ok(retries=5)


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 东财 K 线兜底 (含涨跌幅->pctChg, 两阶段通用) ------------------
def _fetch_hist_em(sym6, start_y):
    """东财逐只兜底(前复权); rename 含 涨跌幅->pctChg, 供第一阶段算'站上MA20'"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym6, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close', '涨跌幅': 'pctChg'})
            if 'close' not in d.columns:
                return None
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            if 'pctChg' in d.columns:
                d['pctChg'] = pd.to_numeric(d['pctChg'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'close', 'pctChg'] if c in d.columns]
            return d[cols] if len(d) >= 20 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _norm(code_pref):
    """baostock 带前缀代码 -> 6位 (东财用)"""
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


# ================== 第一阶段：板块打分 + 自动选择 ==================

def _get_stock_status(args):
    """单只状态: 涨跌幅 + 是否站上MA20; K线双源(baostock优先+东财兜底)"""
    code, name = args
    sym6 = _norm(code)
    start_dash = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock
    try:
        df = _query_with_timeout(code, "date,close,pctChg", start_dash)
        if df is None or df.empty or len(df) < 20:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None   # 板块打分阶段样本量大，超时的单只股票直接跳过不计分即可
    except Exception:
        df = None

    # 路径1.5: 非超时空/异常 -> 子进程重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close,pctChg", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= 20:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym6, start_y)

    if df is None or len(df) < 20:
        return None

    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if 'pctChg' not in df.columns or df['pctChg'].isna().all():
            # 东财兜底若缺涨跌幅, 用 close 自算
            df['pctChg'] = df['close'].pct_change() * 100
        else:
            df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
        df = df.dropna(subset=['close'])
        if len(df) < 20:
            return None
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        latest_close = df['close'].iloc[-1]
        latest_chg = df['pctChg'].iloc[-1]
        if pd.isna(ma20) or pd.isna(latest_chg):
            return None
        return {"代码": code, "名称": name, "涨跌幅%": latest_chg, "站上MA20": latest_close > ma20}
    except Exception:
        return None


def score_and_select_sectors():
    print("=" * 60)
    print("第一阶段：板块打分 + 自动选择强势板块 (行业分类=baostock国标, 无akshare全量替代)")
    print("=" * 60)

    # 主进程登录检查 (修复裸登录 -> 空表 KeyError)
    industry_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            industry_df = bs.query_stock_industry().get_data()
        except Exception as e:
            print(f"  baostock 取行业分类异常: {e}")
            industry_df = pd.DataFrame()
        bs.logout()

    if industry_df is None or industry_df.empty or 'code' not in industry_df.columns:
        print("⚠️ 行业分类表无效(登录失败/空), 本轮跳过 (akshare 无干净全量替代, 不强行兜底)")
        return pd.DataFrame(), [], {}, {}

    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]
    if SECTOR_STOCK_LIMIT:
        industry_df = industry_df.iloc[:SECTOR_STOCK_LIMIT]

    tasks = [(row['code'], row['code_name']) for _, row in industry_df.iterrows()]
    code_to_industry = dict(zip(industry_df['code'], industry_df['industry']))

    print(f"开始获取 {len(tasks)} 只股票的状态数据 (K线双源) ...")
    stock_status = []
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="板块打分进度", unit="只")
        for res in pool.imap_unordered(_get_stock_status, tasks):
            if res:
                res["行业"] = code_to_industry.get(res["代码"], "未知")
                stock_status.append(res)
            pbar.update(1)

    if not stock_status:
        print("未获取到有效数据")
        return pd.DataFrame(), [], {}, {}

    status_df = pd.DataFrame(stock_status)

    grouped = status_df.groupby("行业").agg(
        股票数=("代码", "count"),
        站上MA20占比=("站上MA20", "mean"),
        平均涨跌幅=("涨跌幅%", "mean")
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]

    if grouped.empty:
        print("没有板块满足最小股票数门槛")
        return grouped, [], {}, {}

    max_chg = grouped["平均涨跌幅"].abs().max() or 1
    grouped["板块强度分"] = (
        grouped["站上MA20占比"] * 60 +
        (grouped["平均涨跌幅"] / max_chg) * 40 + 40
    ).clip(0, 100).round(1)
    grouped = grouped.sort_values("板块强度分", ascending=False).reset_index(drop=True)

    print("\n全部板块强度排行：")
    print(grouped[["行业", "股票数", "板块强度分"]].to_string(index=False))

    top = grouped.head(TOP_N_SECTORS)
    selected = top[top["板块强度分"] >= MIN_SECTOR_SCORE]

    if selected.empty:
        print(f"\n没有板块强度分达到{MIN_SECTOR_SCORE}分门槛，本轮不选任何板块，跳过个股筛选")
        return grouped, [], {}, {}

    selected_names = selected["行业"].tolist()
    industry_score_map = dict(zip(selected["行业"], selected["板块强度分"]))
    print(f"\n自动选中板块：{selected_names}")

    candidate_df = status_df[status_df["行业"].isin(selected_names)]
    # candidates 带行业, 供第二阶段给个股标"所属强势板块"
    candidates = list(zip(candidate_df["代码"], candidate_df["名称"], candidate_df["行业"]))
    print(f"候选股票池：{len(candidates)} 只（来自 {len(selected_names)} 个强势板块）")

    return grouped, candidates, code_to_industry, industry_score_map


# ================== 第二阶段：个股信号筛选（复用main.py的策略）==================

def strategy_triple_cross(df):
    try:
        if len(df) < 260:
            return None
        df['close'] = df['close'].astype(float)
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma250'] = df['close'].rolling(250).mean()
        df['date'] = pd.to_datetime(df['date'])
        df_week = df.resample('W-FRI', on='date')['close'].last().dropna()
        df_month = df.resample('ME', on='date')['close'].last().dropna()
        w_ma5, w_ma20 = df_week.rolling(5).mean(), df_week.rolling(20).mean()
        m_ma5, m_ma20 = df_month.rolling(5).mean(), df_month.rolling(20).mean()
        if len(w_ma20.dropna()) == 0 or len(m_ma20.dropna()) == 0:
            return None
        w_gap = (w_ma20.iloc[-1] - w_ma5.iloc[-1]) / w_ma20.iloc[-1]
        m_gap = (m_ma20.iloc[-1] - m_ma5.iloc[-1]) / m_ma20.iloc[-1]
        y_gap = (df['ma250'].iloc[-1] - df['ma20'].iloc[-1]) / df['ma250'].iloc[-1]
        周即将 = (w_gap > 0) and (w_gap < WEEK_THRESHOLD)
        月即将 = (m_gap > 0) and (m_gap < MONTH_THRESHOLD)
        年即将 = (y_gap > 0) and (y_gap < YEAR_THRESHOLD)
        if not (周即将 and 月即将 and 年即将 and (df['close'].iloc[-1] > MIN_PRICE)):
            return None
        return {"w_gap": w_gap, "m_gap": m_gap, "y_gap": y_gap, "close": df['close'].iloc[-1]}
    except Exception:
        return None


def strategy_weekly_only(df):
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
    score = 0.0
    score += max(0, (1 - gaps["w_gap"] / WEEK_THRESHOLD)) * 30
    score += max(0, (1 - gaps["m_gap"] / MONTH_THRESHOLD)) * 30
    score += max(0, (1 - gaps["y_gap"] / YEAR_THRESHOLD)) * 20
    rsi = calculate_rsi(df['close'])
    if rsi is not None:
        if 30 <= rsi <= 55:
            score += 20
        elif rsi > 70:
            score -= 10
        elif rsi < 20:
            score += 5
    return round(max(0, min(100, score)), 1)


def _process_signal(args):
    """单只信号筛选; 解包(code,name,行业); K线双源"""
    code, name, industry = args
    sym6 = _norm(code)
    df = None
    timed_out = False

    # 路径1: baostock
    try:
        df = _query_with_timeout(code, "date,close", "2020-01-01")
        if df is None or df.empty or len(df) < 150:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时空/异常 -> 子进程重登重试一次
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close", "2020-01-01")
                if df2 is not None and not df2.empty and len(df2) >= 150:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(sym6, "20200101")

    if df is None or len(df) < 150:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        time.sleep(SLEEP_PER_STOCK)
        signals = []
        # 固定补齐所有字段(含行业), 避免 DataFrame 缺列产生 NaN 显示坑
        hit = {"代码": code, "名称": name, "行业": industry, "评分": None, "周线宽松评分": None}

        gaps = strategy_triple_cross(df)
        if gaps:
            signals.append("三线共振")
            hit["评分"] = calculate_signal_score(gaps, df)
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
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def screen_candidates(candidates):
    print("\n" + "=" * 60)
    print(f"第二阶段：在 {len(candidates)} 只候选股票中筛选三线共振/周线宽松信号 (K线双源)")
    print("=" * 60)

    if not candidates:
        return pd.DataFrame()

    results = []
    fail_count = 0
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(candidates), desc="信号筛选进度", unit="只")
        for res in pool.imap_unordered(_process_signal, candidates):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（{res['信号']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"筛选完成，共失败 {fail_count} 只")
    if not results:
        return pd.DataFrame()
    result_df = pd.DataFrame(results)
    # 保留 _排序权重, 由 main 结合板块强度分统一排序后再 drop
    return result_df


# ================== 推送 ==================

def _sector_tag(row):
    """个股所属强势板块标记: [行业·强度X分]; 不纳入统一emoji体系(口径不同)"""
    ind = row.get('行业', '')
    sc = row.get('板块强度分')
    if ind and ind not in ('—', '未知', '') and pd.notna(sc):
        return f"{ind}·强度{sc}"
    return ind or '—'


def build_push_content(sector_df, signal_df):
    P = PUSH_TOP
    lines = []
    if sector_df is not None and not sector_df.empty:
        lines.append("### 📊 强势板块榜 (自上而下优选)")
        for _, row in sector_df.head(TOP_N_SECTORS).iterrows():
            lines.append(f"- {row['行业']}：强度分 **{row['板块强度分']}** | 样本{row['股票数']}只 | 站MA20占比{round(row['站上MA20占比']*100,1)}%")
        lines.append("")
    lines.append(f"### 🎯 板块内个股信号 Top{min(len(signal_df), P) if not signal_df.empty else 0}")
    if signal_df.empty:
        lines.append("本次候选池内未筛出符合条件的股票")
    else:
        for _, row in signal_df.head(P).iterrows():
            parts = [f"- {row['名称']}（{row['代码']}）[{_sector_tag(row)}] 最新价 {row.get('最新价')} | 信号: {row['信号']}"]
            if pd.notna(row.get('评分')):
                parts.append(f"三线评分{row['评分']}")
            if pd.notna(row.get('周线宽松评分')):
                parts.append(f"周线宽松{row['周线宽松评分']}")
            lines.append(" | ".join(parts))
        if len(signal_df) > P:
            lines.append(f"\n*…另有 {len(signal_df)-P} 只, 详见 output 报告*")
    lines.append("\n*注: 本脚本板块=baostock国标行业(自上而下评出), 与东财风口口径不同, 故不叠⭐/🎯共振。*")
    return "\n".join(lines)


# ================== 主程序 ==================
if __name__ == "__main__":
    print("=" * 70)
    print(f"板块优选+信号流水线 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES}")
    print(f"两阶段: ①baostock国标行业评强势板块 ②板块内三线共振/周线宽松; K线双源")
    print("=" * 70)

    if not is_trading_day():
        print("今日非A股交易日, 跳过")
        sys.exit(0)

    grouped, candidates, code_to_industry, industry_score_map = score_and_select_sectors()
    signal_df = screen_candidates(candidates)

    # 给个股附"所属强势板块强度分", 并按(板块强度分, 个股权重)排序 -> 最强板块的票排最前
    if not signal_df.empty:
        signal_df['板块强度分'] = signal_df['行业'].map(industry_score_map).fillna(0)
        signal_df = signal_df.sort_values(['板块强度分', '_排序权重'], ascending=False).reset_index(drop=True)
        signal_df = signal_df.drop(columns=['_排序权重'], errors='ignore')

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    if not signal_df.empty:
        disp = signal_df.copy()
        disp.insert(2, '强势板块', [_sector_tag(r) for r in disp.to_dict('records')])
        disp = disp.drop(columns=['行业', '板块强度分'], errors='ignore')
        print(disp.head(PUSH_TOP).to_string(index=False))
    else:
        print("本次候选池内未筛出符合条件的股票")

    # 保存 (个股 + 板块榜, 均入 output/)
    tag = datetime.now().strftime('%Y%m%d')
    if not signal_df.empty:
        signal_df.to_csv(f"{OUTPUT_DIR}/pipeline_sector_{tag}.csv", index=False, encoding="utf-8-sig")
    if grouped is not None and not grouped.empty:
        grouped.to_csv(f"{OUTPUT_DIR}/pipeline_sector_rank_{tag}.csv", index=False, encoding="utf-8-sig")
    with open(f"{OUTPUT_DIR}/pipeline_sector_{tag}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "sector_rank": (grouped.to_dict('records') if grouped is not None and not grouped.empty else []),
            "signals": (signal_df.to_dict('records') if not signal_df.empty else []),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 结果已保存: {OUTPUT_DIR}/pipeline_sector_*_{tag}.*")

    # 推送
    if SERVERCHAN_KEY:
        n_sig = 0 if signal_df.empty else len(signal_df)
        title = f"板块优选+信号 命中{n_sig}只" if n_sig else "板块优选+信号 本轮无命中"
        content = f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + build_push_content(grouped, signal_df)
        send_serverchan(title, content)
