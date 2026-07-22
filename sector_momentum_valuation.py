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

"""
sector_momentum_valuation.py

板块"动量 + 估值分位"量化初筛。

⚠️ 重要局限（务必读完再用）：
板块上涨的完整依据通常有5层——价格动量、估值位置、订单/业绩数据、
一级市场融资/BD交易、政策催化。本脚本只能覆盖前两层（技术面+估值面），
因为baostock只提供价格和PE/PB这类结构化行情数据。
订单、融资、政策这三层依据藏在财报文字、新闻、研报里，不是结构化数字，
没有免费API能自动抓取，需要人工查证或让AI实时联网搜索核实。

本脚本输出的"动量强""估值低"只是技术面初筛信号，
不代表该板块有真实的基本面支撑，不构成投资建议。

【v2 升级说明】
- 修复主进程登录不检查导致的静默无产出; K线层加东财兜底(但东财无历史PE/PB,
  故东财兜底路径下估值分位诚实降级为缺失/象限❔, 仅动量+站MA20可用)。
- 原 build_report 的"动量+低估值同时具备/注意追高"半截note, 升维为完整6象限标签
  (🟢双优/🚀动量↑估值中/🟡追高/🔵埋伏/⚪弱势/⚫回避/❔估值缺失) + 双优榜/洼地榜/回避榜。
- 板块口径=baostock国标行业(自上而下), 与东财风口不同, 故不叠⭐/🎯共振。
- 不加交易日判断: 动量/估值为历史/快照性质, 非交易日仍有意义。
"""

# ------------------ 参数 (全部 env 可调) ------------------
SECTOR_STOCK_LIMIT = int(os.environ.get('SECTOR_STOCK_LIMIT', '1500'))  # 参与打分的股票池大小（越大越全面，也越慢）
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '950'))             # 约3年历史，用于计算估值历史分位
MOMENTUM_DAYS = int(os.environ.get('MOMENTUM_DAYS', '20'))              # 动量：最近N个交易日涨跌幅
MIN_STOCKS_IN_SECTOR = int(os.environ.get('MIN_STOCKS_IN_SECTOR', '5')) # 板块内样本数太少则该板块不参与排名
TOP_N_SECTORS = int(os.environ.get('TOP_N_SECTORS', '10'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '20'))      # 单次查询硬超时，防止网络卡死拖垮整个job

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

# 6象限 (动量方向 × PE分位区间); 估值缺失(东财兜底)单独标❔
_QUAD = {
    (True,  'low'):  ("🟢", "动量↑·估值低(双优)"),
    (True,  'mid'):  ("🚀", "动量↑·估值中"),
    (True,  'high'): ("🟡", "动量↑·估值高(追高警惕)"),
    (False, 'low'):  ("🔵", "动量↓·估值低(价值埋伏)"),
    (False, 'mid'):  ("⚪", "动量↓·估值中"),
    (False, 'high'): ("⚫", "动量↓·估值高(回避)"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 容错 (不加交易日判断: 动量/估值为历史/快照性质) ------------------
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
    _bs_login_ok(retries=5)


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致整个进程池假死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 东财 K 线兜底 (仅 date/close; 东财无历史PE/PB, 故估值维度兜底不了) ------------------
def _fetch_hist_em(sym6, start_y):
    """东财逐只兜底(前复权); 只取 date/close (动量+站MA20用); 估值分位无法兜底"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym6, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close'})
            if 'close' not in d.columns:
                return None
            d['close'] = pd.to_numeric(d['close'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            return d[['date', 'close']] if len(d) >= MOMENTUM_DAYS + 1 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _norm(code_pref):
    """baostock 带前缀代码 -> 6位 (东财用)"""
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


# ------------------ 估值历史分位 (原函数, 一字未动) ------------------
def _percentile_rank(current, history_series):
    """
    计算current在history_series历史序列中的分位：0=历史最低，100=历史最高。
    对空数据和"历史全同值"（除零）做防护，返回None表示无法计算。
    """
    valid = history_series.dropna()
    if len(valid) < 20 or pd.isna(current):
        return None
    return float((valid < current).mean() * 100)


# ------------------ 动量×估值 6象限 ------------------
def _quadrant(mom, pe_pct):
    """(动量, PE分位) -> (emoji, 文字); 估值缺失标❔"""
    if pd.isna(mom):
        return ("❔", "动量缺失")
    up = bool(mom > 0)
    if pd.isna(pe_pct):
        return ("🟢" if up else "🔵", ("动量↑·估值?(兜底缺失)" if up else "动量↓·估值?(兜底缺失)"))
    band = 'low' if pe_pct < 30 else ('high' if pe_pct > 70 else 'mid')
    return _QUAD[(up, band)]


# ------------------ 单只: 动量+估值分位 (K线双源; 估值列存在性防御) ------------------
def _process_one(args):
    """单只股票：拉取~3年数据，同时算动量和估值历史分位，一次查询搞定，运行在子进程里"""
    code, name, industry = args
    sym6 = _norm(code)
    start_dash_3y = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y_120 = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")  # 东财兜底只需动量+MA20, 省流量
    df = None
    timed_out = False
    src = 'bs'

    # 路径1: baostock 3年 (含 peTTM/pbMRQ, 估值分位可用)
    try:
        df = _query_with_timeout(code, "date,close,peTTM,pbMRQ", start_dash_3y)
        if df is None or df.empty or len(df) < MOMENTUM_DAYS + 1:
            df = None
    except FutureTimeoutError:
        timed_out = True
        df = None
    except Exception:
        df = None

    # 路径1.5: 非超时空/异常 -> 子进程重登重试一次 (baostock 3年)
    if df is None and not timed_out:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            if bs.login().error_code == '0':
                df2 = _query_with_timeout(code, "date,close,peTTM,pbMRQ", start_dash_3y)
                if df2 is not None and not df2.empty and len(df2) >= MOMENTUM_DAYS + 1:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底 120天 (无 pe/pb -> 估值分位将缺失)
    if df is None:
        df = _fetch_hist_em(sym6, start_y_120)
        src = 'em'

    if df is None or len(df) < MOMENTUM_DAYS + 1:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        if len(df) < MOMENTUM_DAYS + 1:
            return None

        latest_close = df['close'].iloc[-1]
        price_n_ago = df['close'].iloc[-1 - MOMENTUM_DAYS]
        momentum_pct = (latest_close - price_n_ago) / price_n_ago * 100 if price_n_ago > 0 else None

        ma20 = df['close'].rolling(20).mean().iloc[-1]
        above_ma20 = bool(latest_close > ma20) if pd.notna(ma20) else None

        # 估值历史分位: 仅 baostock 路径有 peTTM/pbMRQ 列; 东财兜底列不存在 -> 诚实降级为 None
        pe_percentile = pb_percentile = None
        if 'peTTM' in df.columns and 'pbMRQ' in df.columns:
            df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
            df['pbMRQ'] = pd.to_numeric(df['pbMRQ'], errors='coerce')
            current_pe = df['peTTM'].iloc[-1]
            current_pb = df['pbMRQ'].iloc[-1]
            pe_percentile = _percentile_rank(current_pe, df['peTTM'])
            pb_percentile = _percentile_rank(current_pb, df['pbMRQ'])

        if momentum_pct is None:
            return None

        return {
            "代码": code, "名称": name, "行业": industry,
            "动量%": round(momentum_pct, 2),
            "站上MA20": above_ma20,
            "PE历史分位": pe_percentile,
            "PB历史分位": pb_percentile,
            "_src": src,
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_sector_scan(limit=SECTOR_STOCK_LIMIT):
    print("=" * 60)
    print("板块动量+估值分位 初筛 (行业分类=baostock国标; K线双源, 估值仅baostock路径有)")
    print("=" * 60)

    # 主进程登录检查 (修复裸登录 -> 静默无产出)
    industry_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            industry_df = bs.query_stock_industry().get_data()
        except Exception as e:
            print(f"  baostock 取行业分类异常: {e}")
            industry_df = pd.DataFrame()
        bs.logout()

    if industry_df is None or industry_df.empty or 'code' not in industry_df.columns:
        print("⚠️ 行业分类表无效(登录失败/空), 本轮跳过 (akshare 无干净全量国标行业替代, 不强行兜底)")
        return pd.DataFrame()

    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]
    if limit:
        industry_df = industry_df.iloc[:limit]

    tasks = [(row['code'], row['code_name'], row['industry']) for _, row in industry_df.iterrows()]

    results = []
    fail_count = 0
    em_count = 0
    print(f"开始计算 {len(tasks)} 只股票的动量+估值分位（{NUM_PROCESSES} 个进程并行, K线双源）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    if res.get('_src') == 'em':
                        em_count += 1
                    results.append(res)
            pbar.update(1)
            pbar.set_postfix(样本=len(results), 失败=fail_count)

    print(f"个股数据抓取完成，共失败 {fail_count} 只；其中 {em_count} 只走东财兜底(估值分位缺失)")

    stock_df = pd.DataFrame(results).drop(columns=['_src'], errors='ignore')
    if stock_df.empty:
        print("未获取到有效数据")
        return pd.DataFrame()

    # 按行业聚合
    grouped = stock_df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均动量=("动量%", "mean"),
        站上MA20占比=("站上MA20", "mean"),
        平均PE分位=("PE历史分位", "mean"),
        平均PB分位=("PB历史分位", "mean"),
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]
    if grouped.empty:
        print("没有板块满足最小样本数门槛")
        return grouped

    grouped["平均动量"] = grouped["平均动量"].round(2)
    grouped["站上MA20占比"] = (grouped["站上MA20占比"] * 100).round(1)
    grouped["平均PE分位"] = grouped["平均PE分位"].round(1)
    grouped["平均PB分位"] = grouped["平均PB分位"].round(1)
    grouped = grouped.sort_values("平均动量", ascending=False).reset_index(drop=True)

    # 6象限标签
    q = grouped.apply(lambda r: _quadrant(r["平均动量"], r["平均PE分位"]), axis=1)
    grouped["象限"] = [x[0] for x in q]
    grouped["象限说明"] = [x[1] for x in q]

    return grouped


# ------------------ 报告 (象限 + 多榜; 原"双优/追高"note 升维为完整象限) ------------------
def build_report(grouped):
    N = TOP_N_SECTORS
    lines = []
    lines.append("⚠️ 仅覆盖技术面(动量)和估值面(历史分位)，不含订单/融资/政策等基本面依据，仅供技术面参考")
    lines.append("")

    # 动量 Top N (带象限)
    lines.append(f"### 🚀 动量最强 Top{N}")
    for _, row in grouped.head(N).iterrows():
        pe = row["平均PE分位"]; pb = row["平均PB分位"]
        lines.append(
            f"- {row['象限']} **{row['行业']}**：{MOMENTUM_DAYS}日动量{row['平均动量']}% "
            f"| 站MA20占比{row['站上MA20占比']}% | PE分位{pe if pd.notna(pe) else '—'} "
            f"| PB分位{pb if pd.notna(pb) else '—'} 〔{row['象限说明']}〕"
        )
    lines.append("")

    # 🟢 双优象限榜 (动量↑ + 估值低)
    dual = grouped[(grouped["平均动量"] > 0) & (grouped["平均PE分位"] < 30)].sort_values("平均动量", ascending=False)
    if not dual.empty:
        lines.append(f"### 🟢 双优象限 (动量↑+估值低, 最甜) Top{min(len(dual), N)}")
        for _, row in dual.head(N).iterrows():
            lines.append(f"- {row['行业']}：动量{row['平均动量']}% | PE分位{row['平均PE分位']} | 站MA20{row['站上MA20占比']}%")
        lines.append("")

    # 🔵 估值洼地榜 (PE分位最低, 不论动量 = 左侧埋伏)
    low = grouped[grouped["平均PE分位"].notna()].sort_values("平均PE分位").head(N)
    if not low.empty:
        lines.append(f"### 🔵 估值洼地 (PE历史分位最低, 左侧埋伏) Top{N}")
        for _, row in low.iterrows():
            lines.append(f"- {row['象限']} {row['行业']}：PE分位{row['平均PE分位']} | 动量{row['平均动量']}% 〔{row['象限说明']}〕")
        lines.append("")

    # ⚫ 回避 (动量↓ + 估值高)
    avoid = grouped[(grouped["平均动量"] <= 0) & (grouped["平均PE分位"] > 70)]
    if not avoid.empty:
        lines.append(f"### ⚫ 回避 (动量↓+估值高) {min(len(avoid), 5)}个")
        for _, row in avoid.head(5).iterrows():
            lines.append(f"- {row['行业']}：动量{row['平均动量']}% | PE分位{row['平均PE分位']}")
        lines.append("")

    lines.append("*注: 板块口径=baostock国标行业(自上而下), 与东财风口不同, 故不叠⭐/🎯共振; "
                 "标❔/估值?者=该板块样本多走东财兜底, 估值维度缺失, 仅看动量。*")
    return "\n".join(lines)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"板块动量+估值分位初筛 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 样本上限={SECTOR_STOCK_LIMIT}")
    print(f"不加交易日判断(动量/估值为历史/快照性质); 6象限=动量×估值交叉")
    print("=" * 70)

    grouped = run_sector_scan()
    if not grouped.empty:
        # 控制台 (带象限全表)
        print("\n全部板块 (按动量降序, 含象限):")
        print(grouped[["象限", "行业", "股票数", "平均动量", "站上MA20占比", "平均PE分位", "平均PB分位"]].to_string(index=False))

        # 保存 (output/ + JSON, 供汇总脚本读)
        tag = datetime.now().strftime('%Y%m%d')
        grouped.to_csv(f"{OUTPUT_DIR}/sector_momentum_valuation_{tag}.csv", index=False, encoding="utf-8-sig")
        with open(f"{OUTPUT_DIR}/sector_momentum_valuation_{tag}.json", 'w', encoding='utf-8') as f:
            json.dump(grouped.to_dict('records'), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 结果已保存: {OUTPUT_DIR}/sector_momentum_valuation_*_{tag}.*")

        report = build_report(grouped)
        print("\n" + report)

        if SERVERCHAN_KEY:
            n_dual = int(((grouped["平均动量"] > 0) & (grouped["平均PE分位"] < 30)).sum())
            n_low = int(grouped["平均PE分位"].notna().sum())
            title = f"板块动量+估值 🟢双优{n_dual} 🔵洼地榜已出"
            send_serverchan(title, f"扫描时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" + report)
    else:
        print("本次未获取到有效板块数据")
