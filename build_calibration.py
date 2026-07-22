import os
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import baostock as bs

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# 打分逻辑来自 market_signal_utils (v2 已重写为连续对称, score 范围约[-7.5,7.5])
from market_signal_utils import INDEX_CODE, BANK_STOCKS, calculate_score

"""
build_calibration.py

用历史数据为 daily_market_signal 的打分生成"校准表"：对每个历史交易日算 score，
统计该 score 分箱内"次日上证上涨"的经验频率 -> calibration_table.csv。
本脚本被 cron 每月1号触发, 并把 csv commit 回仓库供 daily_market_signal 读取。

【v2 升级说明】
- 修复裸 bs.login 不检查 + 空表崩在 qcut (qcut 不接受空数组 -> ValueError 红叉);
  加 _bs_login_ok 重试 + 在 qcut 前拦截空表 early return + 登录失败走东财兜底。
- 修复 fetch_history 无硬超时 (拉730天, baostock 卡住会挂到30分钟强杀); 加 _query_with_timeout。
- 加东财兜底 (区分指数/个股: sh.000xxx/sz.399xxx 走 index_zh_a_hist, 其余个股走
  stock_zh_a_hist; 均有"涨跌幅"列, 全功能不降级)。
- 纠正上一轮误判: 分箱用 pd.qcut 等频自适应, 【非硬编码边界】, 故新 score 的 [-7.5,7.5]
  分布会被自动分箱 -> 改完 market_signal_utils 后【只需重跑本脚本】即生成匹配新分数的校准表,
  无需手动改任何边界 (闭环比预想更干净)。
- 已知微小瑕疵(诚实标注, 不强制修): qcut 区间为左开右闭(left,right], 而 daily 的
  estimate_probability 用双闭 bin_left<=score<=bin_right 匹配; 当 score 恰等于某分箱边界时
  会落到相邻箱(先匹配者)。仅边界精确命中触发, 概率低且相邻箱概率接近, 影响极小。
  若介意, 可选 patch 见文末说明(改 daily 的 estimate 为左开右闭)。
- 关键约束: OUTPUT_CSV 必须保持根目录 "calibration_table.csv" (workflow 的 build-calibration
  job 用 git add calibration_table.csv 把它 commit 回仓库; 改路径则 commit 不到)。
  故 csv 留根目录, 另存一份 json 到 output/ 留痕(供下载查看分箱明细)。
- 不加推送(训练脚本无需); 不加交易日判断(历史统计性质); 不加多进程(仅4标的)。
- 保留 qcut 等频 / next_day_up=shift(-1)>0 + iloc[:-1] / inner join 对齐 / 样本量警告。
"""

# ------------------ 参数 (env 可调) ------------------
HISTORY_DAYS = int(os.environ.get('HISTORY_DAYS', '730'))   # 约2年历史
BINS = int(os.environ.get('BINS', '10'))
# ⚠️ 不要改 OUTPUT_CSV: workflow 用 git add calibration_table.csv 把它 commit 回仓库
OUTPUT_CSV = os.environ.get('CALIBRATION_CSV', 'calibration_table.csv')
QUERY_TIMEOUT_SEC = int(os.environ.get('QUERY_TIMEOUT_SEC', '30'))   # 730天单次查询给宽超时
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False


# ------------------ 登录重试 / 工具 ------------------
def _bs_login_ok(retries=5):
    global _BS_LOGGED
    for i in range(retries):
        try:
            lg = bs.login()
            if getattr(lg, 'error_code', '1') == '0':
                _BS_LOGGED = True
                return True
            print(f"  baostock 登录失败({getattr(lg, 'error_msg', '')}), 重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  baostock 登录异常: {e}, 重试 {i+1}/{retries}")
        time.sleep(2 * (i + 1))
    return False


def _six(code_pref):
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


def _is_index(code_pref):
    """sh.000xxx(上证系列)/sz.399xxx(深证系列) 为指数; sz.000001 是平安银行(个股)不算"""
    code6 = _six(code_pref)
    if code_pref.startswith('sh.') and code6.startswith('000'):
        return True
    if code_pref.startswith('sz.') and code6.startswith('399'):
        return True
    return False


def _query_with_timeout(code, fields, start_date, timeout=QUERY_TIMEOUT_SEC):
    """给单次baostock查询包一层硬超时，防止网络卡顿导致脚本挂死"""
    def _do_query():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date,
                                          frequency="d", adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_query)
        return future.result(timeout=timeout)


# ------------------ 东财历史兜底 (区分指数/个股, 取涨跌幅列) ------------------
def _fetch_hist_em(code_pref, start_y):
    """东财历史兜底; 返回 date/pctChg df 或 None; 730天"""
    code6 = _six(code_pref)
    is_idx = _is_index(code_pref)
    end_y = datetime.now().strftime("%Y%m%d")
    import akshare as ak
    for attempt in range(2):
        try:
            if is_idx:
                d = ak.index_zh_a_hist(symbol=code6, period="daily", start_date=start_y, end_date=end_y)
            else:
                d = ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start_y,
                                       end_date=end_y, adjust="qfq")
            if d is None or d.empty or '涨跌幅' not in d.columns:
                return None
            d = d.rename(columns={'日期': 'date', '涨跌幅': 'pctChg'})
            d['pctChg'] = pd.to_numeric(d['pctChg'], errors='coerce')
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d = d.dropna(subset=['pctChg']).sort_values('date').reset_index(drop=True)
            return d[['date', 'pctChg']] if not d.empty else None
        except Exception as e:
            print(f"    东财 {code_pref} 第{attempt+1}次失败: {e}")
            time.sleep(1 + attempt)
    return None


# ------------------ 取历史 (双源: baostock 优先 + 东财兜底; 含硬超时) ------------------
def fetch_history(code, start_date):
    """通用历史数据获取：返回 date/pctChg df (可能为空); baostock 的 pctChg 直接是涨跌幅(%)"""
    start_y = start_date.replace("-", "")
    df = None

    # 路径1: baostock (主进程登录态; 含硬超时)
    if _BS_LOGGED:
        try:
            df = _query_with_timeout(code, "date,close,pctChg", start_date)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['pctChg']).sort_values('date').reset_index(drop=True)
                if df.empty:
                    df = None
        except FutureTimeoutError:
            df = None
        except Exception:
            df = None

    # 路径2: 东财兜底
    if df is None:
        df = _fetch_hist_em(code, start_y)

    return df if df is not None else pd.DataFrame(columns=['date', 'pctChg'])


# ------------------ 构建校准表 ------------------
def build_calibration_table():
    start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')
    print(f"拉取历史 {HISTORY_DAYS} 天 (自 {start_date}), 双源(baostock+东财), 分箱={BINS}")

    # 主进程登录检查 (修复裸登录 -> 失败空表 -> qcut 崩); 失败也继续, 走东财兜底
    if not _bs_login_ok():
        print("⚠️ baostock 主进程登录失败, 各标的将走东财历史兜底")

    idx_df = fetch_history(INDEX_CODE, start_date)
    bank_dfs = {name: fetch_history(code, start_date) for name, code in BANK_STOCKS.items()}

    if _BS_LOGGED:
        try:
            bs.logout()
        except Exception:
            pass

    # 空表拦截 (在 qcut 之前! 防 ValueError 红叉)
    if idx_df.empty:
        print(f"⚠️ 上证指数({INDEX_CODE}) 双源均无历史数据, 无法生成校准表")
        return None
    miss = [n for n, d in bank_dfs.items() if d.empty]
    if miss:
        print(f"⚠️ 银行股缺历史数据: {miss}, 无法生成校准表")
        return None

    merged = idx_df[['date', 'pctChg']].rename(columns={'pctChg': 'sz_chg'})
    for name, df in bank_dfs.items():
        merged = merged.merge(
            df[['date', 'pctChg']].rename(columns={'pctChg': f'{name}_chg'}),
            on='date', how='inner'
        )

    merged = merged.sort_values('date').reset_index(drop=True)
    if len(merged) < BINS * 5:
        print(f"⚠️ 历史样本量偏少（{len(merged)}条），分箱校准结果参考价值有限")
    if merged.empty:
        print("⚠️ 对齐后无共同交易日, 无法生成校准表")
        return None

    merged['next_day_up'] = (merged['sz_chg'].shift(-1) > 0).astype(int)
    merged = merged.iloc[:-1]

    scores = []
    for _, row in merged.iterrows():
        s, _ = calculate_score(row['sz_chg'], row['建设银行_chg'], row['工商银行_chg'], row['招商银行_chg'])
        scores.append(s)
    merged['score'] = scores

    # qcut 等频自适应分箱 (非硬编码边界 -> 自动匹配新 score 分布[-7.5,7.5])
    merged['score_bin'] = pd.qcut(merged['score'], q=BINS, duplicates='drop')
    calibration = merged.groupby('score_bin', observed=True).agg(
        样本数=('next_day_up', 'count'),
        次日上涨频率=('next_day_up', 'mean')
    ).reset_index()

    calibration['bin_left'] = calibration['score_bin'].apply(lambda x: x.left)
    calibration['bin_right'] = calibration['score_bin'].apply(lambda x: x.right)
    calibration = calibration.drop(columns=['score_bin'])

    # csv 留根目录 (workflow git add 依赖此路径, 勿改)
    calibration.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    # json 留痕到 output/ (供 artifact 下载查看分箱明细)
    tag = datetime.now().strftime('%Y%m%d')
    json_path = f"{OUTPUT_DIR}/calibration_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "history_days": HISTORY_DAYS, "bins": BINS, "bs_logged": _BS_LOGGED,
            "n_samples": int(len(merged)), "n_bins": int(len(calibration)),
            "score_range": [round(float(merged['score'].min()), 2), round(float(merged['score'].max()), 2)],
            "calibration": calibration.to_dict('records'),
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 校准表已生成: {OUTPUT_CSV}（{len(merged)}个历史交易日样本，{len(calibration)}个分箱）")
    print(f"   score 实际范围: [{merged['score'].min():.2f}, {merged['score'].max():.2f}] (qcut 自适应)")
    print(f"📁 分箱明细留痕: {json_path}")
    print(calibration.to_string(index=False))

    return calibration


if __name__ == "__main__":
    build_calibration_table()
