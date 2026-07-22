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
sector_rotation_alert.py

板块轮动预警：看多/看空打分 + 潜力接力板块 + 热度TOP3。
参考了 oficcejo/aiagents-stock 仓库"智策板块"模块的报告结构
（板块多空/潜力接力板块/热度TOP3），但这是纯量化版本：

⚠️ 重要区别：原仓库的"核心机会"文字点评是调用AI(DeepSeek等)生成的自然语言分析，
本脚本没有接LLM，输出的是结构化数字和打分，不是AI生成的叙述性判断。
"潜力接力板块"的识别逻辑也是自己定义的技术面规则（短期动量加速），
不是原仓库可能包含的资金链/游资动向等更复杂的判断维度。

【v2 升级说明】
- 修复主进程裸登录(登录失败空表无列 -> 原代码会在 industry_df['code'] 处 KeyError 崩溃);
  加 _bs_login_ok 重试 + 空表 early return。行业分类表不做 akshare 兜底(无干净全量国标替代)。
- 逐只 K 线加东财兜底(东财有"换手率"列, 故热度榜维度不降级, rename 换手率->turn)。
- 修复 sc_send 硬导入(软导入+requests兜底); 结果存 output/ (csv+json+md 留痕)。
- 多空行加阵营标记 🐂看多/🐻看空 (金融多空标准符号, 矩阵内未用过, 不与🟢双优/⭐共振撞;
  属报告区块的多空阵营标记, 非个股共振记号)。逻辑公式(强度分/短期加速度)一字未动。
- 板块口径=baostock国标行业(自上而下), 与东财风口不同, 故不叠⭐/🎯共振。
- 不加交易日判断: 20/5日动量+换手率为历史/快照性质, 非交易日仍有意义(无个股时机信号)。
"""

# ------------------ 参数 (全部 env 可调) ------------------
SECTOR_STOCK_LIMIT = int(os.environ.get('SECTOR_STOCK_LIMIT', '1500'))  # 参与打分的股票池大小
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '60'))              # 拉取约2个月数据
LONG_MOMENTUM_DAYS = int(os.environ.get('LONG_MOMENTUM_DAYS', '20'))    # 长动量窗口
SHORT_MOMENTUM_DAYS = int(os.environ.get('SHORT_MOMENTUM_DAYS', '5'))   # 短动量窗口（识别"接力"）
MIN_STOCKS_IN_SECTOR = int(os.environ.get('MIN_STOCKS_IN_SECTOR', '5'))
BULLISH_SCORE_THRESHOLD = float(os.environ.get('BULLISH_SCORE_THRESHOLD', '65'))  # 强度分>=此值算看多
BEARISH_SCORE_THRESHOLD = float(os.environ.get('BEARISH_SCORE_THRESHOLD', '35'))  # 强度分<=此值算看空
TOP_N_HOT = int(os.environ.get('TOP_N_HOT', '3'))
TOP_N_POTENTIAL = int(os.environ.get('TOP_N_POTENTIAL', '5'))
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = 0.15
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '15'))

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 (软导入) / 登录重试 / 工具 ------------------
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


def _norm(code_pref):
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


# ------------------ 东财 K 线兜底 (含 换手率->turn, 热度榜维度不降级) ------------------
def _fetch_hist_em(sym6, start_y):
    """东财逐只兜底(前复权); rename 含 换手率->turn, 故热度榜维度可用"""
    end_y = datetime.now().strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym6, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty:
                return None
            d = d.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume', '换手率': 'turn'})
            if 'close' not in d.columns:
                return None
            for c in ['close', 'volume', 'turn']:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            cols = [c for c in ['date', 'close', 'volume', 'turn'] if c in d.columns]
            return d[cols] if len(d) >= LONG_MOMENTUM_DAYS + 1 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


# ------------------ 单只: 长/短动量+站MA20+换手率 (K线双源) ------------------
def _process_one(args):
    """单只股票：算长/短动量、是否站上MA20、换手率，运行在子进程里"""
    code, name, industry = args
    sym6 = _norm(code)
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    df = None
    timed_out = False

    # 路径1: baostock (子进程已登录; fields 含 turn)
    try:
        df = _query_with_timeout(code, "date,close,volume,turn", start_dash)
        if df is None or df.empty or len(df) < LONG_MOMENTUM_DAYS + 1:
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
                df2 = _query_with_timeout(code, "date,close,volume,turn", start_dash)
                if df2 is not None and not df2.empty and len(df2) >= LONG_MOMENTUM_DAYS + 1:
                    df = df2
        except Exception:
            pass

    # 路径2: 东财兜底 (含换手率, 不降级)
    if df is None:
        df = _fetch_hist_em(sym6, start_y)

    if df is None or len(df) < LONG_MOMENTUM_DAYS + 1:
        return {"__error__": f"{code} 双源均无足够数据, 已跳过"} if df is None else None

    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if 'turn' in df.columns:
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        if len(df) < LONG_MOMENTUM_DAYS + 1:
            return None

        latest_close = df['close'].iloc[-1]

        price_long_ago = df['close'].iloc[-1 - LONG_MOMENTUM_DAYS]
        long_momentum = (latest_close - price_long_ago) / price_long_ago * 100 if price_long_ago > 0 else None

        price_short_ago = df['close'].iloc[-1 - SHORT_MOMENTUM_DAYS] if len(df) > SHORT_MOMENTUM_DAYS else None
        short_momentum = (
            (latest_close - price_short_ago) / price_short_ago * 100
            if price_short_ago is not None and price_short_ago > 0 else None
        )

        ma20 = df['close'].rolling(20).mean().iloc[-1]
        above_ma20 = bool(latest_close > ma20) if pd.notna(ma20) else None

        avg_turnover = df['turn'].iloc[-5:].mean() if ('turn' in df.columns and df['turn'].notna().any()) else None

        if long_momentum is None:
            return None

        return {
            "代码": code, "名称": name, "行业": industry,
            "长动量%": long_momentum,
            "短动量%": short_momentum,
            "站上MA20": above_ma20,
            "换手率%": avg_turnover,
        }
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


# ------------------ 主扫描 ------------------
def run_sector_scan(limit=SECTOR_STOCK_LIMIT):
    print("=" * 60)
    print("板块轮动预警 (行业分类=baostock国标; K线双源含换手率; 逻辑公式未改)")
    print("=" * 60)

    # 主进程登录检查 (修复裸登录 -> 空表无列 -> KeyError: 'code' 崩溃)
    industry_df = pd.DataFrame()
    if _bs_login_ok():
        try:
            industry_df = bs.query_stock_industry().get_data()
        except Exception as e:
            print(f"  baostock 取行业分类异常: {e}")
            industry_df = pd.DataFrame()
        bs.logout()

    # 空表 early return: 避免后续 industry_df['code']/['industry'] 取列 KeyError
    if industry_df is None or industry_df.empty or 'code' not in industry_df.columns or 'industry' not in industry_df.columns:
        print("⚠️ 行业分类表无效(登录失败/空/缺列), 本轮跳过 (akshare 无干净全量国标行业替代, 不强行兜底)")
        return pd.DataFrame()

    industry_df = industry_df[industry_df['code'].str.startswith(('sh.', 'sz.'))]
    # 过滤掉行业字段为空的股票（新上市/未分类等），避免聚合出一个"无名板块"
    industry_df = industry_df[industry_df['industry'].astype(str).str.strip() != ""]
    if industry_df.empty:
        print("⚠️ 过滤后无有效行业分类, 本轮跳过")
        return pd.DataFrame()

    if limit:
        industry_df = industry_df.iloc[:limit]

    tasks = [(row['code'], row['code_name'], row['industry']) for _, row in industry_df.iterrows()]

    results = []
    fail_count = 0
    em_count = 0
    print(f"开始计算 {len(tasks)} 只股票的板块轮动数据（{NUM_PROCESSES} 个进程并行, K线双源）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                else:
                    results.append(res)
            pbar.update(1)
            pbar.set_postfix(样本=len(results), 失败=fail_count)

    print(f"个股数据抓取完成，共失败 {fail_count} 只")

    stock_df = pd.DataFrame(results)
    if stock_df.empty:
        print("未获取到有效数据")
        return pd.DataFrame()

    # 双重保险：即使前面漏网，这里再过滤一次空行业名
    stock_df = stock_df[stock_df["行业"].astype(str).str.strip() != ""]

    grouped = stock_df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均长动量=("长动量%", "mean"),
        平均短动量=("短动量%", "mean"),
        站上MA20占比=("站上MA20", "mean"),
        平均换手率=("换手率%", "mean"),
    ).reset_index()
    grouped = grouped[grouped["股票数"] >= MIN_STOCKS_IN_SECTOR]
    if grouped.empty:
        print("没有板块满足最小样本数门槛")
        return grouped

    max_abs_long = grouped["平均长动量"].abs().max() or 1
    grouped["强度分"] = (
        grouped["站上MA20占比"] * 60 +
        (grouped["平均长动量"] / max_abs_long) * 40 + 40
    ).clip(0, 100).round(1)

    grouped["短期加速度"] = (
        (grouped["平均短动量"] / SHORT_MOMENTUM_DAYS) - (grouped["平均长动量"] / LONG_MOMENTUM_DAYS)
    ).round(3)

    grouped["平均长动量"] = grouped["平均长动量"].round(2)
    grouped["平均短动量"] = grouped["平均短动量"].round(2)
    grouped["站上MA20占比"] = (grouped["站上MA20占比"] * 100).round(1)
    grouped["平均换手率"] = grouped["平均换手率"].round(2)

    # 多空阵营标记 (🐂/🐻/⚪; 矩阵内未用过, 不与🟢双优/⭐共振撞)
    def _camp(score):
        if score >= BULLISH_SCORE_THRESHOLD:
            return "🐂"
        if score <= BEARISH_SCORE_THRESHOLD:
            return "🐻"
        return "⚪"
    grouped["多空"] = grouped["强度分"].apply(_camp)

    return grouped.sort_values("强度分", ascending=False).reset_index(drop=True)


# ------------------ 报告 (四块结构保留; 多空行加🐂🐻; 加 md 保存) ------------------
def build_report(grouped):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"板块轮动预警报告 - {now}")
    lines.append("⚠️ 纯量化版本，不含AI生成的文字点评，仅供技术面参考，不构成投资建议")
    lines.append("")

    bullish = grouped[grouped["强度分"] >= BULLISH_SCORE_THRESHOLD].sort_values("强度分", ascending=False)
    bearish = grouped[grouped["强度分"] <= BEARISH_SCORE_THRESHOLD].sort_values("强度分", ascending=True)

    lines.append("📊 板块多空")
    if not bullish.empty:
        bull_str = "、".join(f"🐂{r['行业']}({r['强度分']:.0f}分)" for _, r in bullish.head(8).iterrows())
        lines.append(f"【看多】{bull_str}")
    else:
        lines.append("【看多】本次无板块达到看多阈值")
    if not bearish.empty:
        bear_str = "、".join(f"🐻{r['行业']}({r['强度分']:.0f}分)" for _, r in bearish.head(8).iterrows())
        lines.append(f"【看空】{bear_str}")
    else:
        lines.append("【看空】本次无板块达到看空阈值")
    lines.append("")

    mid_range = grouped[(grouped["强度分"] > BEARISH_SCORE_THRESHOLD) & (grouped["强度分"] < BULLISH_SCORE_THRESHOLD)]
    potential = mid_range[mid_range["短期加速度"] > 0].sort_values("短期加速度", ascending=False).head(TOP_N_POTENTIAL)

    lines.append("🔄 潜力接力板块（短期动量正在加速，尚未进入强势区间）")
    if not potential.empty:
        for _, row in potential.iterrows():
            lines.append(f"• {row['行业']}：强度分{row['强度分']:.0f}，短期加速度+{row['短期加速度']:.3f}")
    else:
        lines.append("本次未识别到明显的潜力接力板块")
    lines.append("")

    hot = grouped.dropna(subset=["平均换手率"]).sort_values("平均换手率", ascending=False).head(TOP_N_HOT)
    lines.append(f"🌡️ 热度TOP{TOP_N_HOT}（按平均换手率）")
    for i, (_, row) in enumerate(hot.iterrows(), 1):
        lines.append(f"{i}. {row['多空']} {row['行业']} - 换手率{row['平均换手率']:.2f}% | 强度分{row['强度分']:.0f}")

    lines.append("")
    lines.append("*注: 板块口径=baostock国标行业(自上而下), 与东财风口不同, 故不叠⭐/🎯共振; "
                 "🐂/🐻为多空阵营标记(非个股共振记号)。*")
    return "\n".join(lines)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"板块轮动预警 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 进程={NUM_PROCESSES} 样本上限={SECTOR_STOCK_LIMIT}")
    print(f"不加交易日判断(动量/换手率为历史/快照性质); 多空🐂+潜力接力+热度🌡️")
    print("=" * 70)

    grouped = run_sector_scan()
    if not grouped.empty:
        report = build_report(grouped)
        print("\n" + report)

        # 控制台全表 (带多空标记)
        print("\n全部板块 (按强度分降序):")
        print(grouped[["多空", "行业", "股票数", "强度分", "平均长动量", "平均短动量",
                       "短期加速度", "站上MA20占比", "平均换手率"]].to_string(index=False))

        # 留痕: csv + json(含四名单) + md
        tag = datetime.now().strftime('%Y%m%d')
        grouped.to_csv(f"{OUTPUT_DIR}/sector_rotation_{tag}.csv", index=False, encoding="utf-8-sig")
        bullish = grouped[grouped["强度分"] >= BULLISH_SCORE_THRESHOLD]["行业"].tolist()
        bearish = grouped[grouped["强度分"] <= BEARISH_SCORE_THRESHOLD]["行业"].tolist()
        mid = grouped[(grouped["强度分"] > BEARISH_SCORE_THRESHOLD) & (grouped["强度分"] < BULLISH_SCORE_THRESHOLD)]
        potential = mid[mid["短期加速度"] > 0].sort_values("短期加速度", ascending=False)["行业"].head(TOP_N_POTENTIAL).tolist()
        hot = grouped.dropna(subset=["平均换手率"]).sort_values("平均换手率", ascending=False)["行业"].head(TOP_N_HOT).tolist()
        with open(f"{OUTPUT_DIR}/sector_rotation_{tag}.json", 'w', encoding='utf-8') as f:
            json.dump({
                "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "thresholds": {"bullish": BULLISH_SCORE_THRESHOLD, "bearish": BEARISH_SCORE_THRESHOLD},
                "bullish": bullish, "bearish": bearish, "potential_relay": potential, "hot": hot,
                "all_sectors": grouped.to_dict('records'),
            }, f, ensure_ascii=False, indent=2, default=str)
        with open(f"{OUTPUT_DIR}/sector_rotation_{tag}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📁 结果已保存: {OUTPUT_DIR}/sector_rotation_*_{tag}.*")

        if SERVERCHAN_KEY:
            send_serverchan(f"板块轮动预警 {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                            f"(🐂{len(bullish)}/🐻{len(bearish)}/🔄{len(potential)})", report)
    else:
        print("本次未获取到有效板块数据")
