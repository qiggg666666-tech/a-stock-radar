# -*- coding: utf-8 -*-
"""
macd_resonance_screener.py —— 多周期 MACD(DIF/DEA) + RSI + 背离 共振选股【完美融合版】
====================================================================
融合参考脚本的 3 个有价值维度(RSI / 背离 / 市值过滤), 修掉其全部 bug,
保留矩阵工程加固(双源/超时/行业本地join/季年诚实处理/DIF-DEA三处显示/多进程/正确推送)。

扫描个股, 在 日/周/月/季/年 五周期算 MACD 的 DIF/DEA + 日/周/月 RSI + 日线背离,
多周期累加打分(MACD主分+RSI辅助分), 高分+多周期金叉+大周期不死叉=共振看多入选;
顶背离/RSI过热 标⚠️风险(只提示不剔除, 因背离/RSI常失效, 剔除会漏票)。

⚠️ 诚实定位: 趋势状态筛选, 非预测; MACD滞后, 共振=趋势确认非领先; 背离/RSI为辅助提示。
⚠️ 周期硬限制: MACD(12,26,9)需≥35根; 季线需~9年/年线需~35年, A股1990开市,
  故年线仅极少数老票有效, 不足者中性不计分, 绝不凑数。

【vs 参考脚本 修复/不照搬清单】
 1 不拉5次历史: 拉1次日线+本地resample五周期(参考脚本3000次接口限流灾难)。
 2 不跨周期错配: 每周期用自己dif/dea判金叉(参考脚本月线DIF比日线DEA, 错)。
 3 背离真做: 近60根前后两半两峰/两谷比较(参考脚本传单元素列表, 背离恒"无")。
 4 推送正确: 软导入sc_send+读SENDKEY+不吞异常(参考脚本硬编码"你的SCKEY"+except pass, 从未成功)。
 5 全akshare调用硬超时 _call_with_timeout(参考脚本裸调, 卡死拖垮job)。
 6 双源: baostock优先+东财兜底(参考脚本纯东财)。
 7 存 output/(参考脚本存cwd, artifact收不到)。
 8 不吞warnings; resample用别名候选逐个try(参考脚本filterwarnings ignore掩盖QE/YE废弃)。
 9 年线缺失中性不计分(参考脚本年线权重最高却常缺失, 丢35分对新票不公)。
 10 加行业本地join + 市值列名容错(总市值/市值)。
 吸收: RSI进分(小权重辅助) + 背离/RSI过热标⚠️风险 + 市值初筛过滤。
 不照搬: 参考脚本的75阈值/130满分(分制不同, 用本脚本MIN_SCORE三门槛更稳健)。

记号: 📡=共振命中/共振板块; ⚠️=顶背离或RSI过热风险标记。
====================================================================
"""
import os
import re
import sys
import json
import time
import random
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from tqdm import tqdm

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

# ===================== 参数区 =====================
PARAMS = dict(
    FAST=12, SLOW=26, SIGNAL=9,
    LOOKBACK_DAYS=5475,                # 历史回看天数(默认15年)
    MIN_REQUIRED=35,                   # 某周期 resample 后至少此根数才算 MACD 可信

    # 入选门槛
    MIN_SCORE=18.0,                    # 共振总分≥此值(含RSI辅助分)
    MIN_GOLDEN=3,                      # 至少此数个周期处于金叉状态
    NO_DEAD_IN_MAJOR=True,             # 周+月不得死叉

    # 初筛(快照向量化砍量)
    KEEP_PREFIX=("0", "3", "6"),
    EXCLUDE_NAME=("ST", "退"),
    MIN_PRICE=3.0,
    AMOUNT_MIN=1.0e8,
    TURNOVER_MIN=1.0,
    NOT_LIMIT_PCT=9.5,
    CAP_MIN=30.0e8,                    # 市值下限(参考脚本30亿; 设0不过滤)
    CAP_MAX=1.0e13,                    # 市值上限(默认不过滤)
    TOP_N=30,

    # RSI 辅助(进分但权重小)
    RSI_HEALTH=(30, 70),               # 日线RSI健康区间加分
    RSI_OVERHEAT=75,                   # 日线RSI过热减分+风险标记

    # 背离窗口(日线, 前后各半)
    DIV_HALF=30,

    NUM_PROCESSES=3,
    SLEEP=0.3,
)

PERIOD_WEIGHT = {"daily": 8, "weekly": 12, "monthly": 15, "quarterly": 12, "yearly": 8}
PERIOD_NAME = {"daily": "日", "weekly": "周", "monthly": "月", "quarterly": "季", "yearly": "年"}
PERIOD_ORDER = ["daily", "weekly", "monthly", "quarterly", "yearly"]
RSI_PERIODS = {"daily", "weekly", "monthly"}   # 季/年样本少, RSI 不稳, 不算
RESAMPLE_ALIAS = {
    "weekly": ["W-FRI"],
    "monthly": ["ME", "M"],
    "quarterly": ["QE-DEC", "Q-DEC", "Q"],
    "yearly": ["YE-DEC", "Y-DEC", "Y", "A-DEC", "A"],
}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '15'))
CLUSTER_TOP = int(os.environ.get('CLUSTER_TOP', '10'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False
_INDUSTRY_MAP = {}


# ===================== 工具 =====================
def _pref(code6):
    c = str(code6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

def _call_with_timeout(fn, *args, timeout=AK_TIMEOUT, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result(timeout=timeout)

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

def _bs_query_with_timeout(code, fields, start_date, timeout=AK_TIMEOUT):
    def _do():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)

def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    if len(content) > 4000:
        content = content[:3900] + "\n\n...(已截断)"
    try:
        from serverchan_sdk import sc_send
        sc_send(key, title, content); print("📲 serverchan-sdk 推送成功"); return True
    except Exception as e:
        print(f"  serverchan-sdk 失败, 回退 requests: {e}")
    try:
        r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=10)
        return r.json().get("code") == 0
    except Exception as e:
        print(f"  requests 推送失败: {e}"); return False

def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    s = re.sub(r'^[A-Z]\d+\s*', '', s.strip())
    return s or "—"

def _dd_compact(detail, periods=PERIOD_ORDER):
    parts = []
    for p in periods:
        d = detail.get(p, {}) if isinstance(detail, dict) else {}
        if isinstance(d, dict) and d.get("dif") is not None:
            parts.append(f"{PERIOD_NAME[p]}{d['dif']}/{d['dea']}")
        else:
            parts.append(f"{PERIOD_NAME[p]}—")
    return " ".join(parts)


# ===================== 快照初筛(向量化砍量 + 市值容错) =====================
def snapshot_filter():
    df = None
    for i in range(3):
        try:
            df = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=AK_TIMEOUT)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  全A快照第{i+1}次失败: {e}")
        time.sleep(2 + i)
    if df is None or df.empty:
        print("⚠️ 全A快照获取失败(限流)")
        return []
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    for c in ['最新价', '涨跌幅', '换手率', '成交额']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 市值列名容错(ak版本间 '总市值'/'市值' 不一)
    cap_col = next((c for c in ['总市值', '市值'] if c in df.columns), None)
    if cap_col:
        df['_cap'] = pd.to_numeric(df[cap_col], errors='coerce')
        df['_cap_yi'] = (df['_cap'] / 1e8).round(2)
    else:
        df['_cap_yi'] = pd.NA
        print("  [注] 快照无市值列, 跳过市值过滤")
    m = (df["代码"].str.startswith(PARAMS["KEEP_PREFIX"])
         & ~df["名称"].astype(str).str.contains("|".join(PARAMS["EXCLUDE_NAME"]), na=False, regex=True)
         & (df["最新价"] >= PARAMS["MIN_PRICE"])
         & (df["成交额"] >= PARAMS["AMOUNT_MIN"])
         & (df["换手率"] >= PARAMS["TURNOVER_MIN"])
         & (df["涨跌幅"].abs() < PARAMS["NOT_LIMIT_PCT"]))
    if cap_col and PARAMS["CAP_MIN"] > 0:
        m &= (df['_cap'] >= PARAMS["CAP_MIN"]) & (df['_cap'] <= PARAMS["CAP_MAX"])
    out = df[m].sort_values("成交额", ascending=False)
    print(f"  快照初筛: 全A {len(df)} → 候选精算 {len(out)} 只 (市值≥{PARAMS['CAP_MIN']/1e8:.0f}亿)")
    return out[["代码", "名称", "_cap_yi"]].rename(columns={"_cap_yi": "市值亿"}).to_dict("records")


# ===================== 个股历史(双源+超时) =====================
def fetch_hist(code):
    start_dash = (datetime.now() - timedelta(days=PARAMS["LOOKBACK_DAYS"])).strftime('%Y-%m-%d')
    start_y = start_dash.replace("-", "")
    if _BS_LOGGED:
        try:
            d = _bs_query_with_timeout(_pref(code), "date,close", start_dash)
            if d is not None and not d.empty:
                d["close"] = pd.to_numeric(d["close"], errors="coerce")
                d["date"] = pd.to_datetime(d["date"])
                d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                if len(d) >= 60:
                    return d
        except Exception:
            pass
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=code, period="daily",
                                   start_date=start_y, end_date=datetime.now().strftime("%Y%m%d"),
                                   adjust="qfq", timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={"日期": "date", "收盘": "close"})
                d["close"] = pd.to_numeric(d["close"], errors="coerce")
                d["date"] = pd.to_datetime(d["date"])
                d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                if len(d) >= 60:
                    return d
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


# ===================== resample + MACD(全序列) + RSI + 背离 =====================
def _resample_close(daily, period):
    if period == "daily":
        return daily.set_index("date")["close"]
    for alias in RESAMPLE_ALIAS[period]:
        try:
            r = daily.set_index("date")["close"].resample(alias).last().dropna()
            return r if len(r) >= 5 else None
        except Exception:
            continue
    return None

def _macd_full(close_s, fast, slow, signal):
    """返回 (dif_series, dea_series) 或 (None,None); 长度不足返回 None"""
    if close_s is None or len(close_s) < PARAMS["MIN_REQUIRED"]:
        return None, None
    c = close_s.astype(float)
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea

def _rsi_last(close_s, period=14):
    if close_s is None or len(close_s) < period + 1:
        return None
    c = close_s.astype(float)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    v = rsi.iloc[-1]
    return None if pd.isna(v) else round(float(v), 1)

def _detect_divergence(close_s, dif_s, half=None):
    """真·两峰/两谷背离(日线): 近2*half根分前后两半, 比较价格峰/谷与对应dif峰/谷。
    顶背离=价格新高但dif没新高; 底背离=价格新低但dif没新低。O(n)无scipy依赖。"""
    half = half or PARAMS["DIV_HALF"]
    if close_s is None or dif_s is None or len(close_s) < 2 * half or len(dif_s) < 2 * half:
        return "无"
    c = close_s.iloc[-2 * half:].astype(float).values
    d = dif_s.iloc[-2 * half:].astype(float).values
    if np.isnan(c).any() or np.isnan(d).any():
        return "无"
    c1, c2 = c[:half], c[half:]
    d1, d2 = d[:half], d[half:]
    # 顶背离
    p1, p2 = int(np.argmax(c1)), int(np.argmax(c2))
    if c2[p2] > c1[p1] * 0.999 and d2[p2] < d1[p1] * 0.98:
        return "顶背离⚠️"
    # 底背离
    p1, p2 = int(np.argmin(c1)), int(np.argmin(c2))
    if c2[p2] < c1[p1] * 1.001 and d2[p2] > d1[p1] * 1.02:
        return "底背离🔥"
    return "无"


# ===================== 共振打分(MACD主分 + RSI辅助 + 背离标记) =====================
def score_resonance(daily):
    """返回 (score, reasons, detail, golden_n, major_dead)。
    detail[周期]={dif,dea,golden,just_golden,above_zero,score}; detail['rsi']={周期:rsi};
    detail['divergence']=日线背离; detail['risk']=顶背离或RSI过热。"""
    score = 0.0
    reasons = []
    detail = {"rsi": {}}
    golden_n = 0
    major_dead = False
    divergence = "无"
    for period, w in PERIOD_WEIGHT.items():
        cs = _resample_close(daily, period)
        dif_s, dea_s = _macd_full(cs, PARAMS["FAST"], PARAMS["SLOW"], PARAMS["SIGNAL"])
        pn = PERIOD_NAME[period]
        if dif_s is None:
            detail[period] = {"status": "样本不足"}
            reasons.append(f"·{pn}线样本不足")
            continue
        dif, dea = float(dif_s.iloc[-1]), float(dea_s.iloc[-1])
        dif_p, dea_p = float(dif_s.iloc[-2]), float(dea_s.iloc[-2])
        golden = dif > dea
        just_golden = (dif_p <= dea_p) and golden
        above_zero = dif > 0
        if golden:
            golden_n += 1
        if period in ("weekly", "monthly") and not golden:
            major_dead = True
        ps = (0.6 if golden else -0.6) + (0.4 if just_golden else 0.0) + (0.3 if above_zero else -0.3)
        ps *= w
        # RSI 辅助分(仅日/周/月; 权重小)
        rsi = _rsi_last(cs) if period in RSI_PERIODS else None
        if period in RSI_PERIODS:
            detail["rsi"][period] = rsi
        if rsi is not None:
            if period == "daily":
                if PARAMS["RSI_HEALTH"][0] <= rsi <= PARAMS["RSI_HEALTH"][1]:
                    ps += 3
                elif rsi > PARAMS["RSI_OVERHEAT"]:
                    ps -= 3; reasons.append(f"⚠️日RSI过热({rsi})")
            elif period == "weekly" and 40 <= rsi <= 70:
                ps += 2
            elif period == "monthly" and rsi > 50:
                ps += 2
        score += ps
        detail[period] = {"dif": round(dif, 3), "dea": round(dea, 3),
                          "golden": bool(golden), "just_golden": bool(just_golden),
                          "above_zero": bool(above_zero), "score": round(ps, 1)}
        if just_golden:
            reasons.append(f"▲{pn}线刚金叉(零轴{'上' if above_zero else '下'})")
        elif golden:
            reasons.append(f"▲{pn}线金叉")
        else:
            reasons.append(f"▼{pn}线死叉")
        # 日线额外算背离(用同一 dif_s / cs, 零额外网络)
        if period == "daily":
            divergence = _detect_divergence(cs, dif_s)
    detail["divergence"] = divergence
    rsi_d = detail["rsi"].get("daily")
    detail["risk"] = bool(("顶背离" in divergence) or (rsi_d is not None and rsi_d > PARAMS["RSI_OVERHEAT"]))
    if divergence != "无":
        reasons.append(f"{'⚠️' if '顶' in divergence else '🔥'}日线{divergence}")
    if not reasons:
        reasons.append("·无有效周期数据")
    return round(score, 1), reasons, detail, golden_n, major_dead


# ===================== 单只处理(子进程) =====================
def _process_one(args):
    code, name, cap = args
    try:
        h = fetch_hist(code)
        if h.empty:
            return None
        score, reasons, detail, golden_n, major_dead = score_resonance(h)
        ok = (score >= PARAMS["MIN_SCORE"] and golden_n >= PARAMS["MIN_GOLDEN"]
              and (not PARAMS["NO_DEAD_IN_MAJOR"] or not major_dead))
        if not ok:
            return None
        time.sleep(PARAMS["SLEEP"])
        return {"代码": code, "名称": name, "行业": "", "市值亿": cap,
                "共振分": score, "金叉周期数": golden_n,
                "日RSI": detail["rsi"].get("daily"),
                "背离": detail.get("divergence", "无"),
                "风险": bool(detail.get("risk")),
                "理由": " | ".join(reasons),
                "日": detail.get("daily", {}).get("golden"),
                "周": detail.get("weekly", {}).get("golden"),
                "月": detail.get("monthly", {}).get("golden"),
                "季": detail.get("quarterly", {}).get("status", detail.get("quarterly", {}).get("golden")),
                "年": detail.get("yearly", {}).get("status", detail.get("yearly", {}).get("golden")),
                "_detail": detail}
    except FutureTimeoutError:
        return {"__error__": f"{code} 超时"}
    except Exception as e:
        return {"__error__": f"{code} 失败: {e}"}


# ===================== 主扫描 =====================
def run_scan():
    global _INDUSTRY_MAP
    print("连接 Baostock（取行业映射 + 子进程登录）...")
    if _bs_login_ok():
        try:
            ind_df = bs.query_stock_industry().get_data()
            if ind_df is not None and not ind_df.empty and 'code' in ind_df.columns:
                for _, row in ind_df.iterrows():
                    _INDUSTRY_MAP[row['code']] = _clean_industry(row.get('industry', ''))
                print(f"  行业映射 {len(_INDUSTRY_MAP)} 条(一次拿全, 本地join不限流)")
        except Exception as e:
            print(f"  取行业表异常: {e}")
        bs.logout()

    cands = snapshot_filter()
    if not cands:
        return pd.DataFrame()

    results = []
    fail = 0
    print(f"逐只拉历史算五周期MACD+RSI+背离(约 {len(cands)*PARAMS['SLEEP']:.0f}s, 1次拉历史本地全算)...")
    with mp.Pool(processes=PARAMS["NUM_PROCESSES"], initializer=lambda: _bs_login_ok()) as pool:
        pbar = tqdm(total=len(cands), desc="共振扫描", unit="只")
        for res in pool.imap_unordered(_process_one, [(c["代码"], c["名称"], c.get("市值亿")) for c in cands]):
            if res:
                if "__error__" in res:
                    fail += 1
                else:
                    results.append(res)
                    rk = " ⚠️" if res["风险"] else ""
                    pbar.write(f"  📡 {res['代码']} {res['名称']} 分{res['共振分']} 金叉{res['金叉周期数']} 日RSI{res['日RSI']} {res['背离']}{rk}")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail)
    print(f"扫描完成, 命中 {len(results)} / 失败 {fail}")
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("共振分", ascending=False).reset_index(drop=True)


# ===================== 行业本地join + 聚类 =====================
def enrich(df):
    mapped = 0
    for _, r in df.iterrows():
        ind = _INDUSTRY_MAP.get(r["代码"], "—")
        df.loc[df["代码"] == r["代码"], "行业"] = ind
        if ind not in ("—", "未知", ""):
            mapped += 1
    print(f"🏷️ 行业标注 {mapped}/{len(df)} (baostock国标, 本地join)")
    labeled = df[df["行业"].isin([x for x in df["行业"] if x not in ("—", "未知", "")])]
    cluster = []
    if not labeled.empty:
        vc = labeled["行业"].value_counts()
        cluster = [(n, int(c)) for n, c in vc.head(CLUSTER_TOP).items()]
    print(f"📡 共振板块(多周期共振扎堆): {cluster or '无'}")
    return df, cluster


def _add_dif_dea_cols(df):
    for p in PERIOD_ORDER:
        cn = PERIOD_NAME[p]
        df[f"{cn}DIF"] = df["_detail"].apply(lambda d, p=p: d.get(p, {}).get("dif") if isinstance(d, dict) else None)
        df[f"{cn}DEA"] = df["_detail"].apply(lambda d, p=p: d.get(p, {}).get("dea") if isinstance(d, dict) else None)
    return df


def build_push(df, cluster):
    P = PUSH_TOP
    n_risk = int(df["风险"].sum()) if "风险" in df.columns else 0
    lines = [f"**多周期MACD+RSI+背离 共振选股** | 命中 {len(df)} 只 ⚠️风险 {n_risk}",
             f"门槛 分≥{PARAMS['MIN_SCORE']} 金叉≥{PARAMS['MIN_GOLDEN']}周期 | 市值≥{PARAMS['CAP_MIN']/1e8:.0f}亿",
             "*(趋势状态筛选, 非预测; 年/季样本不足中性; DIF/DEA=最新值; ⚠️=顶背离或RSI过热, 仅提示)*", ""]
    if cluster:
        lines.append("📡 **共振板块**: " + "、".join(f"{n}({c}只)" for n, c in cluster))
        lines.append("")
    lines.append(f"### 📡 共振命中 Top{min(len(df), P)}")
    for _, r in df.head(P).iterrows():
        cyc = f"日{'✓' if r['日'] else '✗'} 周{'✓' if r['周'] else '✗'} 月{'✓' if r['月'] else '✗'} 季{r['季']} 年{r['年']}"
        rk = " ⚠️风险" if r["风险"] else ""
        cap_s = f" {r['市值亿']}亿" if pd.notna(r.get('市值亿')) else ""
        lines.append(f"- **{r['名称']}({r['代码']})** [{r['行业']}{cap_s}] 分{r['共振分']} | {cyc}{rk}")
        lines.append(f"  └ DIF/DEA {_dd_compact(r.get('_detail', {}))}")
        lines.append(f"  └ 日RSI {r['日RSI']} | 背离 {r['背离']}")
        lines.append(f"  └ {r['理由']}")
    if len(df) > P:
        lines.append(f"\n*…另有 {len(df)-P} 只, 详见 output*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"多周期MACD+RSI+背离 共振选股【完美融合版】 | {datetime.now():%Y-%m-%d %H:%M}")
    print(f"回看{PARAMS['LOOKBACK_DAYS']}天 进程{PARAMS['NUM_PROCESSES']} 市值≥{PARAMS['CAP_MIN']/1e8:.0f}亿")
    print(f"周期 日/周/月/季/年; RSI=日/周/月; 背离=日线两峰两谷; 年线仅老票有效(诚实)")
    print("=" * 70)
    df = run_scan()
    if df is not None and not df.empty:
        df, cluster = enrich(df)
        df = _add_dif_dea_cols(df)
        tag = datetime.now().strftime("%Y%m%d")
        df.drop(columns=["_detail"], errors="ignore").to_csv(
            os.path.join(OUTPUT_DIR, f"macd_resonance_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"macd_resonance_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": PARAMS, "cluster": cluster,
                       "hits": df.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/macd_resonance_{tag}.*")
        disp = df.copy()
        disp["板块"] = disp["行业"]
        disp["DIF/DEA"] = disp["_detail"].apply(_dd_compact)
        disp = disp.drop(columns=["_detail", "行业", "理由", "大周期死叉", "季", "年"], errors="ignore")
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
        if SERVERCHAN_KEY:
            send_serverchan(f"📡 MACD+RSI+背离共振 命中{len(df)}只 ⚠️{int(df['风险'].sum())}", build_push(df, cluster))
    else:
        print("本次无共振命中(门槛较严, 属正常)")
