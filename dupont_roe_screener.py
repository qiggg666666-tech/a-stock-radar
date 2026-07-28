# -*- coding: utf-8 -*-
"""
dupont_roe_screener.py —— 杜邦ROE拆解 + 多因子选股（基本面×技术面）
ROE = 净利率 × 资产周转率 × 权益乘数; 选 高ROE+良好净利率+高成长+趋势放量 的小盘优质股。
策略内核(打分权重+硬条件)保留; 工程层对齐矩阵(快照粗筛/双源/超时/推送/行业/留痕/收尾防护)。

【本版完善】
  1. append 兼容补丁(核心): 原版缺此补丁, 导致 baostock 的 get_data() 在 pandas2.x 全报错被静默吞,
     "历史双源"实际只剩东财一条腿, 限流即空跑; 补上后 baostock 双源真正复活, 抗限流翻倍,
     同时消掉"取行业表 'DataFrame' object has no attribute 'append'"红字。
  2. 快照限流兜底: 快照挂时退化到 baostock/akshare 列表截断逐只算, 不再空跑;
     换手率用历史日线(baostock turn字段)兜底, 让硬条件退化时也能过。
  3. 软超时: 逐只财务接口慢, 到点主动收工保已算结果, 防撞 timeout。
  4. 财务接口加1次重试。
  5. 去掉交易日拦截(不限交易日, 周末用上一交易日数据)。
⚠️ 财务数据来自 akshare 逐只财务接口(季报, 有滞后); 该接口限流时杜邦因子全NaN, 硬条件过不了
   仍可能0命中——财务数据无可靠免费替代源, 此乃本策略天生局限, 交易日不限流时正常。
"""
import os
import sys
import json
import time
import random
import traceback
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

# 补丁：解决 baostock/akshare 调用已废弃的 DataFrame.append 报错(原版缺此, 致 baostock 双源失效)
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

# ------------------ 参数 (env 可调, 默认=原策略阈值) ------------------
MARKET_CAP_LIMIT = float(os.environ.get('MARKET_CAP_LIMIT', '200'))   # 市值上限(亿)
TOP_N = int(os.environ.get('TOP_N', '25'))
ROE_MIN = float(os.environ.get('ROE_MIN', '10'))
NET_MARGIN_MIN = float(os.environ.get('NET_MARGIN_MIN', '8'))
GROWTH_MIN = float(os.environ.get('GROWTH_MIN', '20'))
MOMENTUM_MIN = float(os.environ.get('MOMENTUM_MIN', '1.4'))
TURNOVER_MIN = float(os.environ.get('TURNOVER_MIN', '1.7'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '500'))   # 够算MA250
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
SLEEP_PER_STOCK = float(os.environ.get('SLEEP_PER_STOCK', '0.15'))
SCAN_LIMIT = int(os.environ.get('SCAN_LIMIT', '0'))
FIN_TIMEOUT = int(os.environ.get('FIN_TIMEOUT', '15'))        # 财务接口超时
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
# 【本版新增】快照兜底 + 软超时
SOFT_CAP = int(os.environ.get('SOFT_CAP', '1500'))            # 快照失败时列表截断只数
SOFT_TIMEOUT = int(os.environ.get('SOFT_TIMEOUT', '3000'))    # 逐只精算软超时(秒, 默认50分钟)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False
_INDUSTRY_MAP = {}


# ------------------ 推送 / 交易日 / 登录 / 超时 ------------------
def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content); print("📲 推送成功"); return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        return requests.post(f"https://sctapi.ftqq.com/{key}.send",
                             data={"title": title, "desp": content}, timeout=10).json().get("code") == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def is_trading_day():
    """保留备用; 【本版】main 不再调用, 即不限交易日"""
    try:
        d = ak.tool_trade_date_hist_sina()
        return datetime.now().strftime('%Y-%m-%d') in set(pd.to_datetime(d['trade_date']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  交易日历失败, 默认继续: {e}"); return True


def _bs_login_ok(retries=5):
    global _BS_LOGGED
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


def _init_worker():
    time.sleep(random.uniform(0, 2))
    _bs_login_ok()


def _bs_q(code, fields, sd, timeout=AK_TIMEOUT):
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


def _clean(s):
    import re
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")


# ------------------ 杜邦因子 (模糊列名匹配, 兼容akshare列名变化, 加1次重试) ------------------
def _find_col(idx, keywords):
    for col in idx:
        if all(k in str(col) for k in keywords):
            return col
    return None


def _to_num(val):
    if val is None:
        return np.nan
    if isinstance(val, str):
        val = val.replace('%', '').replace(',', '').strip()
    try:
        return float(val)
    except Exception:
        return np.nan


def get_dupont_roe_factors(symbol):
    """杜邦拆解ROE: 净利率 × 资产周转率 × 权益乘数; 模糊列名+超时+兼容新旧签名+1次重试"""
    for _retry in range(2):
        try:
            try:
                df = _call_with_timeout(ak.stock_financial_analysis_indicator,
                                        symbol=symbol, start_year=str(datetime.now().year - 1), timeout=FIN_TIMEOUT)
            except TypeError:
                df = _call_with_timeout(ak.stock_financial_analysis_indicator, symbol=symbol, timeout=FIN_TIMEOUT)
            if df is None or df.empty:
                time.sleep(1); continue
            latest = df.iloc[0]
            idx = latest.index
            roe = _to_num(latest[_find_col(idx, ['净资产收益率'])]) if _find_col(idx, ['净资产收益率']) else np.nan
            c = _find_col(idx, ['销售净利率']) or _find_col(idx, ['净利率'])
            net_margin = _to_num(latest[c]) if c else np.nan
            c = _find_col(idx, ['总资产周转率'])
            asset_turnover = _to_num(latest[c]) if c else np.nan
            c = _find_col(idx, ['权益乘数'])
            equity_multiplier = _to_num(latest[c]) if c else np.nan
            c = _find_col(idx, ['净利润增长率']) or _find_col(idx, ['净利润', '增长'])
            profit_growth = _to_num(latest[c]) if c else np.nan
            return roe, net_margin, asset_turnover, equity_multiplier, profit_growth
        except Exception:
            time.sleep(1)
    return (np.nan,) * 5


# ------------------ 历史 (双源, 取 close/volume/turn; turn 供快照失败时换手率兜底) ------------------
def _fetch_hist(code):
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    start_dash = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    start_y = start_dash.replace('-', '')
    if _BS_LOGGED:
        try:
            d = _bs_q(code, "date,close,volume,turn", start_dash)
            if d is not None and not d.empty:
                for c in ['close', 'volume', 'turn']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 200:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=sym, period="daily",
                                   start_date=start_y, adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'收盘': 'close', '成交量': 'volume', '日期': 'date', '换手率': 'turn'})
                for c in ['close', 'volume', 'turn']:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(d) >= 200:
                    cols = [c for c in ['date', 'close', 'volume', 'turn'] if c in d.columns]
                    return d[cols]
        except Exception:
            time.sleep(1 + attempt)
    return None


# ------------------ 单只: 技术面 + 杜邦 + 打分 (内核权重保留) ------------------
def _process_one(args):
    code, name, pe, turnover_spot = args
    hist = _fetch_hist(code)
    if hist is None or len(hist) < 200:
        return None
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code

    ma250 = hist['close'].rolling(250, min_periods=200).mean().iloc[-1]
    latest_close = float(hist['close'].iloc[-1])
    vol_ma5 = hist['volume'].rolling(5).mean().iloc[-1]
    momentum_score = float(hist['volume'].iloc[-1] / vol_ma5) if vol_ma5 and vol_ma5 > 0 else 0
    # 换手率: 快照优先; 快照失败(退化)时用历史近5日换手率兜底(baostock turn字段)
    if pd.notna(turnover_spot):
        turnover = float(min(turnover_spot, 30))
    elif 'turn' in hist.columns and hist['turn'].notna().any():
        turnover = float(min(hist['turn'].iloc[-5:].mean(), 30))
    else:
        turnover = 0

    roe, net_margin, asset_turnover, equity_multiplier, profit_growth = get_dupont_roe_factors(sym)
    time.sleep(SLEEP_PER_STOCK)

    # ---- 打分 (内核权重一字未动) ----
    trend_score = 1 if latest_close > ma250 else 0
    value_score = 1 / (pe + 1) if (pd.notna(pe) and pe > 0) else 0
    dupont_quality = 0
    if pd.notna(net_margin) and pd.notna(asset_turnover):
        dupont_quality = (net_margin / 10) + (asset_turnover * 2)
    growth_score = profit_growth / 15 if pd.notna(profit_growth) else 0
    composite_score = (
        trend_score * 22 +
        min(momentum_score, 5) * 18 +
        min(turnover, 20) * 18 +
        value_score * 12 +
        min(dupont_quality, 8) * 18 +
        min(growth_score, 6) * 12
    )

    # ---- 硬条件 (内核一字未动) ----
    if (trend_score == 1 and momentum_score > MOMENTUM_MIN and turnover >= TURNOVER_MIN
            and pd.notna(roe) and roe > ROE_MIN
            and pd.notna(net_margin) and net_margin > NET_MARGIN_MIN
            and pd.notna(profit_growth) and profit_growth > GROWTH_MIN):
        grade = "🟢优质" if (composite_score >= 110 and pd.notna(asset_turnover)) else "🟡入选"
        return {
            "代码": code, "名称": name, "行业": "",
            "最新价": round(latest_close, 2),
            "总市值(亿)": None,
            "PE": round(float(pe), 2) if pd.notna(pe) else None,
            "ROE(%)": round(float(roe), 2),
            "净利率(%)": round(float(net_margin), 2),
            "资产周转率": round(float(asset_turnover), 2) if pd.notna(asset_turnover) else None,
            "权益乘数": round(float(equity_multiplier), 2) if pd.notna(equity_multiplier) else None,
            "净利润增长率(%)": round(float(profit_growth), 2),
            "MA250": round(float(ma250), 2),
            "换手率(%)": round(turnover, 2),
            "放量倍数": round(momentum_score, 2),
            "杜邦质量分": round(dupont_quality, 2),
            "综合因子得分": round(composite_score, 2),
            "分级": grade,
            "_mv": None,
        }
    return None


# ------------------ 【本版新增】快照失败兜底列表 ------------------
def _fallback_list(cap):
    """快照失败(限流)兜底: baostock/akshare 列表截断 cap 只; pe/换手率快照拿不到,
    换手率由历史日线(baostock turn)兜底。返回 (tasks, mv_map)。"""
    print(f"  ⚠️ 快照失败, 启用兜底: 列表截断{cap}只(换手率用历史兜底, 无市值/PE, 质量降级)")
    if _bs_login_ok():
        try:
            d = bs.query_stock_basic().get_data()
            if d is not None and not d.empty and 'code' in d.columns:
                d = d[d['code'].str.startswith(('sh.', 'sz.')) & (d['type'] == '1') & (d['status'] == '1')]
                d = d[~d['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
                d = d.head(cap)
                tasks = [(r['code'], r['code_name'], np.nan, np.nan) for _, r in d.iterrows()]
                print(f"  兜底列表(baostock): {len(tasks)} 只")
                return tasks, {}
        except Exception as e:
            print(f"  baostock列表兜底失败: {e}")
    try:
        d = ak.stock_info_a_code_name()
        if d is not None and not d.empty and 'code' in d.columns:
            name_col = 'name' if 'name' in d.columns else d.columns[1]
            d = d[['code', name_col]].copy(); d.columns = ['code', 'code_name']
            d['code'] = d['code'].astype(str).str.zfill(6)
            d['code'] = d['code'].apply(lambda c: ('sh.' if c[0] in '69' else 'sz.') + c)
            d = d[~d['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
            d = d.head(cap)
            tasks = [(r['code'], r['code_name'], np.nan, np.nan) for _, r in d.iterrows()]
            print(f"  兜底列表(akshare): {len(tasks)} 只")
            return tasks, {}
    except Exception as e:
        print(f"  akshare列表兜底失败: {e}")
    return [], {}


# ------------------ 主扫描 (快照粗筛 -> 逐只精算; 快照失败兜底) ------------------
def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（行业表 + 子进程登录）...")
    if _bs_login_ok():
        try:
            ind = bs.query_stock_industry().get_data()
            if ind is not None and not ind.empty and 'code' in ind.columns:
                for _, r in ind.iterrows():
                    _INDUSTRY_MAP[r['code']] = _clean(r.get('industry', ''))
                print(f"  行业表 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        bs.logout()

    print("取全市场快照做粗筛...")
    spot = None
    for i in range(3):
        try:
            spot = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
            if spot is not None and not spot.empty:
                break
        except Exception as e:
            print(f"  快照第{i+1}次失败: {e}")
        time.sleep(2 + i)

    degraded = False
    if spot is None or spot.empty:
        # 【本版】快照失败兜底, 不再空跑
        tasks, mv_map = _fallback_list(SOFT_CAP)
        degraded = True
        if not tasks:
            print("⚠️ 快照与兜底均无候选, 本次空跑(限流严重)"); return pd.DataFrame()
    else:
        mv_col = next((c for c in spot.columns if '总市值' in c), None)
        pe_col = next((c for c in spot.columns if '市盈率' in c), None)
        turn_col = next((c for c in spot.columns if '换手率' in c), None)
        chg60_col = next((c for c in spot.columns if '60日' in c), None)
        price_col = next((c for c in spot.columns if c in ('最新价',)), None)
        for c in [mv_col, pe_col, turn_col, chg60_col, price_col]:
            if c and c in spot.columns:
                spot[c] = pd.to_numeric(spot[c], errors='coerce')
        spot['_code6'] = spot['代码'].astype(str).str.zfill(6)
        spot['代码'] = spot['_code6'].apply(lambda c: ('sh.' if c[0] in '69' else 'sz.') + c)

        m = ~spot['名称'].astype(str).str.contains('ST|退', na=False, regex=True)
        if mv_col:
            m &= (spot[mv_col] < MARKET_CAP_LIMIT * 1e8)
        if price_col:
            m &= (spot[price_col] >= MIN_PRICE)
        if turn_col:
            m &= (spot[turn_col] >= 1.0)
        if chg60_col:
            m &= (spot[chg60_col] > 0)
        cand = spot[m].copy()
        print(f"  粗筛: 全A {len(spot)} → 市值<{MARKET_CAP_LIMIT:.0f}亿+趋势+活跃 {len(cand)} 只")

        if SCAN_LIMIT and len(cand) > SCAN_LIMIT:
            cand = cand.head(SCAN_LIMIT)
        mv_map = dict(zip(cand['代码'], cand[mv_col] / 1e8)) if mv_col else {}
        tasks = [(r['代码'], r['名称'],
                  r[pe_col] if pe_col else np.nan,
                  r[turn_col] if turn_col else np.nan) for _, r in cand.iterrows()]

    results = []; fail = 0
    print(f"逐只精算(历史+杜邦财务) {len(tasks)} 只, {NUM_PROCESSES} 进程, 软超时{SOFT_TIMEOUT//60}分钟...")
    _t0 = time.time()
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="杜邦扫描", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                res['总市值(亿)'] = round(float(mv_map.get(res['代码'], 0)), 2) if mv_map.get(res['代码']) else None
                results.append(res)
                pbar.write(f"  🏅 {res['代码']} {res['名称']} {res['分级']} ROE{res['ROE(%)']} 净利率{res['净利率(%)']} 成长{res['净利润增长率(%)']}")
            pbar.update(1); pbar.set_postfix(命中=len(results))
            # 【本版】软超时主动收工, 保已算结果, 防财务接口慢撞 timeout
            if time.time() - _t0 > SOFT_TIMEOUT:
                print(f"\n⏱️ 软超时{SOFT_TIMEOUT//60}分钟到, 主动收工(已算{pbar.n}/{len(tasks)}), 保住已命中")
                break
    print(f"扫描完成 命中{len(results)}{' [退化兜底·无市值/PE]' if degraded else ''}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("综合因子得分", ascending=False).head(TOP_N * 3).reset_index(drop=True)
    return df


# ------------------ 行业join + 优质板块聚类 ------------------
def enrich(df):
    for _, r in df.iterrows():
        df.loc[df['代码'] == r['代码'], '行业'] = _INDUSTRY_MAP.get(r['代码'], '—')
    lab = df[df['行业'].isin([x for x in df['行业'] if x not in ('—', '')])]
    cluster = [(n, int(c)) for n, c in lab['行业'].value_counts().head(CLUSTER_TOP).items()] if not lab.empty else []
    print(f"🏅 优质基本面扎堆板块: {cluster or '无'}")
    return df, cluster


def build_push(df, cluster):
    P = PUSH_TOP
    L = [f"**🏅 杜邦ROE多因子选股** | 命中{len(df)}只 (ROE=净利率×周转率×权益乘数)",
         f"*(高ROE>{ROE_MIN}+净利率>{NET_MARGIN_MIN}+成长>{GROWTH_MIN}+趋势放量; 财务为季报有滞后; 非预测)*", ""]
    if cluster:
        L.append("🏅 **优质基本面板块**: " + "、".join(f"{n}({c})" for n, c in cluster))
        L.append("")
    L.append(f"### 🏅 杜邦优质 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        mv = f" {r['总市值(亿)']}亿" if pd.notna(r.get('总市值(亿)')) else ""
        L.append(f"- {r['分级']} **{r['名称']}({r['代码']})** [{r['行业']}{mv}] 现价{r['最新价']} "
                 f"ROE{r['ROE(%)']}% 净利率{r['净利率(%)']}% 成长{r['净利润增长率(%)']}% "
                 f"周转{r['资产周转率']} | 放量{r['放量倍数']} 综合{r['综合因子得分']}")
    if len(df) > P:
        L.append(f"\n*…另有{len(df)-P}只, 见output*")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"🏅 杜邦ROE多因子选股 | {datetime.now():%Y-%m-%d %H:%M} | 市值<{MARKET_CAP_LIMIT:.0f}亿")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 进程{NUM_PROCESSES}; 历史双源+财务逐只(超时{FIN_TIMEOUT}s); 不拦交易日; 软超时{SOFT_TIMEOUT//60}分钟")
    print("=" * 70)
    # 【本版】不再拦交易日(周末用上一交易日数据)
    df = run_scan()
    if df is None or df.empty:
        print("本次未找到满足杜邦+成长条件的股票(筛选严格, 或财务接口限流)"); sys.exit(0)
    df, cluster = enrich(df)
    df = df.sort_values("综合因子得分", ascending=False).head(TOP_N).reset_index(drop=True)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df.drop(columns=["_mv"], errors="ignore").to_csv(
            os.path.join(OUTPUT_DIR, f"dupont_roe_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"dupont_roe_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": {"市值上限亿": MARKET_CAP_LIMIT, "ROE_MIN": ROE_MIN,
                       "净利率_MIN": NET_MARGIN_MIN, "成长_MIN": GROWTH_MIN}, "cluster": cluster,
                       "n": int(len(df)), "hits": df.drop(columns=["_mv"], errors="ignore").to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/dupont_roe_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(命中已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()
    try:
        disp = df.drop(columns=["_mv", "权益乘数", "MA250"], errors="ignore").copy()
        disp.insert(2, "板块", disp["行业"])
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
        print("\n杜邦核心: ROE = 净利率 × 资产周转率 × 权益乘数 | 重点: 高ROE+良好净利率+高成长+趋势放量")
    except Exception as e:
        print(f"⚠️ 展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            send_serverchan(f"🏅 杜邦ROE多因子 命中{len(df)}只 🏅板块{len(cluster)}", build_push(df, cluster))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)
# >>>FILE_END_dupont<<<
