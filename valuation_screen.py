import os
import sys
import time
import random
import json
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime

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
PE_MIN = float(os.environ.get('PE_MIN', '0'))
PE_MAX = float(os.environ.get('PE_MAX', '40'))
PB_MIN = float(os.environ.get('PB_MIN', '0'))
PB_MAX = float(os.environ.get('PB_MAX', '8'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = int(os.environ.get('NUM_PROCESSES', '3'))
QUERY_TIMEOUT_SEC = 15
UNIVERSE = os.environ.get('UNIVERSE', 'ALL')          # 估值默认全A; 可 HS300/ZZ500 交集
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '20'))
TOP_N = int(os.environ.get('TOP_N', '50'))
LABEL_TOP = int(os.environ.get('LABEL_TOP', '100'))   # 补行业上限(聚类/💰基于"最便宜Top此数")
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '8'))

# 估值遇催化(低估值+板块风口=价值兑现提示; 记号💰, 非时机信号)
HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))
HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ 推送 / 容错 ------------------
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


def ak_retry(fn, *a, desc="", **kw):
    for i in range(1, 4):
        try:
            r = fn(*a, **kw)
            if r is not None and not (hasattr(r, 'empty') and r.empty):
                return r
            print(f"  [{desc}] 第{i}次返回空, 重试...")
        except Exception as e:
            print(f"  [{desc}] 第{i}次异常: {e}")
        time.sleep(3 + random.uniform(0, 2))
    return None


# ------------------ baostock 登录重试 (兜底路径用) ------------------
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


def _fetch_list_akshare():
    """akshare 兜底取股票列表 (baostock 列表失败时)"""
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


# ------------------ 行业 / 风口 / 匹配 (估值遇催化用) ------------------
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
    """展示用板块标记: 估值遇催化标💰, 否则标行业名"""
    return ('💰' + r.get('hot_sector', '')) if r.get('hot_meet') else (r.get('行业') or '—')


# ------------------ 主路径: 东财快照一次向量化 (秒级) ------------------
def _screen_via_em():
    """东财实时快照拿全市场 PE/PB, 向量化过滤; 1 次接口, 无需逐只/多进程"""
    df = ak_retry(ak.stock_zh_a_spot_em, desc="全A快照(估值)")
    if df is None or df.empty:
        return []
    pe_col = next((c for c in ['市盈率-动态', '市盈率'] if c in df.columns), None)
    pb_col = '市净率' if '市净率' in df.columns else None
    if not pe_col or not pb_col:
        print(f"  东财快照缺 PE/PB 列 (找到: {[c for c in df.columns if '市盈' in c or '市净' in c]}), 走兜底")
        return []

    df['代码'] = df['代码'].astype(str)
    close = pd.to_numeric(df.get('最新价'), errors='coerce')
    pe = pd.to_numeric(df[pe_col], errors='coerce')
    pb = pd.to_numeric(df[pb_col], errors='coerce')

    m_code = df['代码'].str.match(r'^(60|00|30|68)')
    m_st = ~df['名称'].str.contains('ST|退', na=False, regex=True)
    m_price = close >= MIN_PRICE
    m_pe = (pe > PE_MIN) & (pe < PE_MAX)
    m_pb = (pb > PB_MIN) & (pb < PB_MAX)
    out = df[m_code & m_st & m_price & m_pe & m_pb & pe.notna() & pb.notna()].copy()
    out = out.assign(_pe=pe, _pb=pb, _close=close)
    print(f"  东财快照估值初筛命中 {len(out)} 只")

    if UNIVERSE in ("HS300", "ZZ500"):
        idx = "000300" if UNIVERSE == "HS300" else "000905"
        cons = ak_retry(lambda: ak.index_stock_cons_csindex(symbol=idx), desc=f"成分{idx}")
        if cons is not None and not cons.empty:
            code_col = '成分券代码' if '成分券代码' in cons.columns else cons.columns[0]
            valid = set(cons[code_col].astype(str).str.zfill(6))
            out = out[out['代码'].isin(valid)]
            print(f"  与{UNIVERSE}成分交集后 {len(out)} 只")

    out = out.sort_values('_pe')
    codes = out['代码'].tolist(); names = out['名称'].tolist()
    closes = out['_close'].round(2).tolist(); pes = out['_pe'].round(2).tolist(); pbs = out['_pb'].round(2).tolist()
    return [{"代码": c, "名称": n, "最新价": cl, "PE": pe_, "PB": pb_,
             "行业": "", "hot_meet": False, "hot_sector": ""}
            for c, n, cl, pe_, pb_ in zip(codes, names, closes, pes, pbs)]


# ------------------ 兜底路径: baostock 逐只 (东财挂时才走) ------------------
def _process_one(args):
    code, name = args
    try:
        df = _query_with_timeout(
            code, "date,close,peTTM,pbMRQ",
            start_date=datetime.now().strftime('%Y-%m-01')
        )
        time.sleep(SLEEP_PER_STOCK)

        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        close = float(latest['close']) if latest['close'] else None
        pe = float(latest['peTTM']) if latest['peTTM'] else None
        pb = float(latest['pbMRQ']) if latest['pbMRQ'] else None

        if not close or close < MIN_PRICE or pe is None or pb is None:
            return None
        if not (PE_MIN < pe < PE_MAX):
            return None
        if not (PB_MIN < pb < PB_MAX):
            return None

        return {"代码": code, "名称": name, "最新价": round(close, 2), "PE": round(pe, 2), "PB": round(pb, 2),
                "行业": "", "hot_meet": False, "hot_sector": ""}
    except FutureTimeoutError:
        return {"__error__": f"{code} 查询超时（>{QUERY_TIMEOUT_SEC}s），已跳过"}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def _screen_via_bs():
    """baostock 逐只兜底 (修登录 bug + 列表双源); 仅东财主路径失败时调用"""
    print("  [兜底] 连接 Baostock 取股票列表 ...")
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
        return []

    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(c, code_to_name.get(c, "")) for c in stock_df['code'].tolist()]

    results = []
    fail_count = 0
    print(f"  [兜底] 估值筛选 {len(tasks)} 只（{NUM_PROCESSES} 进程并行）...")
    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="估值兜底", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                else:
                    results.append(res)
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)
    print(f"  兜底完成, 命中 {len(results)} / 失败 {fail_count}")
    results.sort(key=lambda r: r['PE'])
    return results


# ------------------ 行业标注 + 估值洼地聚类 + 估值遇催化 ------------------
def enrich(results):
    """补行业(并发, 仅最便宜Top LABEL_TOP) -> 洼地板块聚类(本地) -> 估值遇催化(热度榜1次)"""
    if not results:
        return [], [], []
    # results 已按 PE 升序; 补行业补最便宜的前 LABEL_TOP 只
    targets = results[:LABEL_TOP]
    print(f"为最便宜 Top {len(targets)} 只补行业 (聚类/💰基于此集) ...")
    def _q(r):
        sym = r['代码'][3:] if len(r['代码']) > 3 and r['代码'][2] == '.' else r['代码']
        r['行业'] = fetch_industry(sym)
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        list(tqdm(ex.map(_q, targets), total=len(targets), desc="补行业", unit="只"))

    # 估值洼地板块聚类: 最便宜 Top 的行业分布 (纯本地 groupby, 零接口)
    labeled = [r for r in targets if r.get('行业') and r['行业'] not in ('—', '未知', '')]
    cluster = []
    if labeled:
        vc = pd.Series([r['行业'] for r in labeled]).value_counts()
        cluster = [(name, int(cnt)) for name, cnt in vc.head(CLUSTER_TOP).items()]
    print(f"💎 估值洼地板块(最便宜Top{LABEL_TOP}的行业分布): {cluster or '无'}")

    # 估值遇催化: PE/PB合理 + 行业在风口 = 低估值遇板块催化 (热度榜1次)
    heat = get_industry_heat()
    hot = get_hot_sectors(heat)
    hot_names = [n for n, _ in hot]
    print(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in hot) or '(无)'}")
    meet_cnt = 0
    for r in results:   # 仅补了行业的(前LABEL_TOP)能匹配上
        m = match_sector(r.get('行业', ''), hot_names)
        if m:
            r['hot_meet'] = True
            r['hot_sector'] = m
            meet_cnt += 1
    print(f"💰 估值遇催化 {meet_cnt} 只 (低估值+板块催化, 价值兑现提示)")

    # 终排序: 遇催化优先, 再按 PE 升序
    results.sort(key=lambda r: (-(1 if r.get('hot_meet') else 0), r['PE']))
    return results, cluster, hot


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    print("=" * 70)
    print(f"估值筛选 (PE {PE_MIN}~{PE_MAX} / PB {PB_MIN}~{PB_MAX}) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"主路径=东财快照向量化(秒级); 兜底=baostock逐只; 范围={UNIVERSE}")
    print("=" * 70)

    # 主路径: 东财快照 (不依赖交易日, 周末亦可)
    results = _screen_via_em()
    via = 'em'
    if not results:
        print("  东财主路径无结果, 切换 baostock 兜底 ...")
        results = _screen_via_bs()
        via = 'bs'

    if results:
        results, cluster, hot = enrich(results)

        csv_path = f"{OUTPUT_DIR}/valuation_screen_{datetime.now().strftime('%Y%m%d')}.csv"
        json_path = f"{OUTPUT_DIR}/valuation_screen_{datetime.now().strftime('%Y%m%d')}.json"
        pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {csv_path} (共 {len(results)} 只, 来源={via})")

        # 控制台 (带板块标记)
        disp = pd.DataFrame(results).head(TOP_N).copy()
        disp.insert(2, '板块', [sec_tag(r) for r in results[:TOP_N]])
        disp = disp.drop(columns=['行业', 'hot_meet', 'hot_sector'], errors='ignore')
        print("\n" + disp.to_string(index=False))

        if SERVERCHAN_KEY:
            meet_n = sum(1 for r in results if r.get('hot_meet'))
            P = PUSH_TOP
            lines = [f"**PE {PE_MIN}~{PE_MAX} / PB {PB_MIN}~{PB_MAX} | 命中 {len(results)} 只 | 来源 {via}**", ""]
            if hot:
                lines.append("🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
                lines.append("")
            if cluster:
                lines.append("💎 **估值洼地板块**(最便宜Top的行业分布): " +
                             "、".join(f"{n}({c}只)" for n, c in cluster))
                lines.append("")
            meet = [r for r in results if r.get('hot_meet')]
            if meet:
                lines.append(f"### 💰 估值遇催化 Top{min(len(meet), P)} (低估值+板块催化)")
                for r in meet[:P]:
                    lines.append(f"- {r['名称']}（{r['代码']}）[💰{r['hot_sector']}] 现价{r['最新价']} | PE{r['PE']} | PB{r['PB']}")
                lines.append("")
            lines.append(f"### 📋 估值合理 Top{min(len(results), P)}")
            for r in results[:P]:
                lines.append(f"- {r['名称']}（{r['代码']}）[{sec_tag(r)}] 现价{r['最新价']} | PE{r['PE']} | PB{r['PB']}")
            if len(results) > P:
                lines.append(f"\n*…另有 {len(results)-P} 只, 详见 output 报告*")
            lines.append("\n*估值为安全垫维度, 非时机信号; 低PE/PB警惕价值陷阱, 需结合基本面。*")
            send_serverchan(f"估值筛选 命中{len(results)}只 💎洼地{len(cluster)} 💰催化{meet_n}",
                            "\n".join(lines))
    else:
        print("本次未找到符合条件的股票")
