# -*- coding: utf-8 -*-
"""
zt_pre_screener.py —— 提前找涨停：盘前/盘后「涨停候选池」+ 可选盘中「逼近涨停监控」
（独立新脚本，不覆盖 quant_signal_bot.py 的收盘后超跌首板逻辑）
====================================================================
⚠️⚠️ 诚实定位（务必读完，这是本脚本的底线）：
  本工具【不预测涨停】。涨停由资金/消息/情绪博弈决定，没有任何免费结构化数据
  能可靠预测某只票今天会涨停；号称能预测的多为过拟合历史涨停样本，实盘必坑。
  本脚本只做两件"提前"的事，且都基于【已知历史数据/实时行情】，无前视：
   ① 候选池(MODE=candidate): 用昨日及之前数据, 圈"形态具备涨停条件"的票
      (逼近突破/多头/放量/股性活/超跌企稳), 缩小你盯盘的范围, 提高注意概率;
   ② 逼近监控(MODE=intraday): 盘中轮询, 当候选池里的票涨幅逼近涨停价时报警,
      在它封板【之前】发现它正在冲板。
  两者都是"缩小范围/监测逼近", 不是"保证涨停"。打分仅用于排序, 不设硬阈值保证。
  收盘后(candidate 模式)会自动回看"今日候选池 ∩ 今日涨停"的命中率, 作为诚实的自我验证。

【选股不依赖图形】候选池完全由【数据】算出: 东财快照初筛(数据表) + 逐只日线 + 数值形态打分。
  plot_5d 画的五日分时图只是【事后展示】, 不参与选股; 设 DRAW=False 关掉它, 选股照常工作。

【本版完善】
  1. 快照限流兜底: 东财全A快照挂掉时, 退化到 akshare 列表截断逐只拉日线算形态分。
  2. 软超时: 逐只精算到 SOFT_TIMEOUT 主动收工, 保住已算候选。
  3. append 兼容补丁。
  4. 【空跑也推送】空结果时推一条"空跑通知", 不再静默。
  5. 【实时价·午盘/收盘两次】新增腾讯实时行情(qt.gtimg.cn, 海外IP可访问, 不依赖东财快照)
     在推送前刷新候选池现价 -> 无论东财快照是否被墙, 午盘收盘后跑显示午盘实时价、
     下午收盘后跑显示收盘实时价; 并加场次标签(午盘盘后/收盘后)让每天两次推送可辨识。

运行环境适配：
  - MODE=candidate (默认): 跑一次即退, GitHub Actions cron 友好; 午盘收盘后跑=午盘实时价,
    下午收盘后跑=收盘实时价+回看命中。
  - MODE=intraday: 盘中 while 轮询, 【仅本地常驻】, 切勿放进 Actions cron。

数据源 akshare(东财) + 腾讯实时价。产物全部存 output/ 且前缀 zt_/zt_pre_/zt_candidates_。
====================================================================
"""
import os
import json
import time
import random
import requests
from datetime import datetime, timedelta, timezone
import akshare as ak
import pandas as pd

# 补丁：解决 baostock/akshare 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei',
                                   'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 参数区（想换风格只改这里）=====================
PARAMS = dict(
    MODE="candidate",
    KEEP_PREFIX=("0", "3", "6"),
    EXCLUDE_NAME=("ST", "退"),
    MIN_PRICE=3.0,
    AMOUNT_MIN=1.0e8,
    TURNOVER_MIN=1.0,
    NOT_LIMIT_PCT=9.5,
    LOOKBACK=70,
    NEAR_HIGH_PCT=3.0,
    REQUIRE_BULL=True,
    VOL_RATIO_MIN=1.3,
    ACTIVE_LOOKBACK=20,
    DD_MIN=10.0,
    W_BREAK=30, W_BULL=20, W_VOL=20, W_ACTIVE=15, W_REVERSE=15,
    TOP_N=30,
    FALLBACK_FULL=True,
    FALLBACK_CAP=1500,
    SOFT_TIMEOUT=1800,
    INTRADAY_INTERVAL=20,
    INTRADAY_APPROACH=7.0,
    INTRADAY_DURATION=240,
    DRAW=True,
    DRAW_TOP=8,
    SLEEP=0.4,
)

# ===================== 运行环境（env 可调）=====================
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '12'))
PARAMS["MODE"] = os.environ.get('MODE', PARAMS["MODE"])
PARAMS["FALLBACK_FULL"] = os.environ.get('FALLBACK_FULL', str(PARAMS["FALLBACK_FULL"])).strip() in ('1', 'true', 'True')
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BJ = timezone(timedelta(hours=8))


# ===================== 工具 =====================
def _col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _bj_now():
    return datetime.now(_BJ)


def _is_after_close_bj():
    return _bj_now().hour >= 15


def _session_tag():
    """区分午盘盘后 / 盘中 / 收盘后, 让每天两次推送可辨识。"""
    h = _bj_now().hour
    if h < 13:
        return "午盘盘后"
    elif h < 15:
        return "盘中"
    return "收盘后"


def _hist_limit_pct(code):
    return 19.5 if str(code).startswith(("30", "68")) else 9.8


def _fetch_realtime_tencent(codes):
    """腾讯实时行情(海外IP可访问, 不依赖东财), 分批查询, 返回 {6位代码: 现价}。失败返回 {}。"""
    out = {}
    try:
        syms = []
        for c in codes:
            c6 = str(c).zfill(6)
            if c6[:1] in ('6', '9'):
                pref = 'sh'
            elif c6[:1] in ('4', '8'):
                pref = 'bj'
            else:
                pref = 'sz'
            syms.append(f"{pref}{c6}")
        for i in range(0, len(syms), 50):
            batch = syms[i:i + 50]
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(batch), timeout=10)
                r.encoding = 'gbk'
                for line in r.text.strip().split(';'):
                    line = line.strip()
                    if '=' not in line:
                        continue
                    body = line.split('=', 1)[1].strip().strip('"')
                    f = body.split('~')
                    if len(f) > 4 and f[2]:
                        try:
                            price = float(f[3])
                            if price > 0:
                                out[f[2].zfill(6)] = price
                        except Exception:
                            pass
            except Exception as e:
                print(f"   [实时价] 批次{i // 50 + 1}失败: {e}")
            time.sleep(0.3)
    except Exception as e:
        print(f"  腾讯实时价异常: {e}")
    return out


def _refresh_realtime_price(pool):
    """用腾讯实时行情刷新候选池现价, 保证午盘/收盘两次推送的现价都是实时价。
    腾讯失败时沿用原快照价, 不影响主流程。"""
    if pool is None or pool.empty:
        return pool
    codes = pool["代码"].astype(str).str.zfill(6).tolist()
    rt = _fetch_realtime_tencent(codes)
    if not rt:
        print("  实时价刷新: 腾讯行情未取到, 沿用快照价")
        return pool
    pool = pool.copy()
    pool["实时价"] = pool["代码"].astype(str).str.zfill(6).map(lambda c: rt.get(c))
    pool["最新价"] = pool["实时价"].where(pool["实时价"].notna(), pool["最新价"])
    n_rt = int(pool["实时价"].notna().sum())
    print(f"  实时价刷新: 腾讯取到 {n_rt}/{len(pool)} 只实时价")
    return pool


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


# ===================== 空跑也推送 =====================
def _push_empty(reason):
    if not SERVERCHAN_KEY:
        return
    try:
        msg = (f"**⚠️ 涨停候选池·{_session_tag()} · 空跑** | {_bj_now():%m-%d %H:%M}\n\n"
               f"**原因**: {reason}\n\n"
               f"*(多为东财限流致快照/兜底无数据, 或当日形态门槛无命中, **非脚本故障**。"
               f"交易日错峰单跑通常恢复; 用 Actions 的 `Run workflow` 按钮手动重跑亦可验证。)*")
        send_serverchan(f"涨停候选池·{_session_tag()} {_bj_now():%m-%d} | ⚠️空跑", msg)
    except Exception as e:
        print(f"  空跑推送异常(不影响主流程): {e}")


# ===================== 快照初筛 =====================
def snapshot_filter():
    df = None
    for i in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  全A快照第{i + 1}次失败: {e}")
        time.sleep(2 + i)
    if df is None or df.empty:
        print("⚠️ 全A快照获取失败(东财可能限流)")
        return []
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    for c in ['最新价', '涨跌幅', '换手率', '量比', '成交额', '总市值']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    m_code = df["代码"].str.startswith(PARAMS["KEEP_PREFIX"])
    m_name = ~df["名称"].astype(str).str.contains("|".join(PARAMS["EXCLUDE_NAME"]), na=False, regex=True)
    m_price = df["最新价"] >= PARAMS["MIN_PRICE"]
    m_amt = df["成交额"] >= PARAMS["AMOUNT_MIN"]
    m_turn = df["换手率"] >= PARAMS["TURNOVER_MIN"]
    m_notlim = df["涨跌幅"].abs() < PARAMS["NOT_LIMIT_PCT"]
    out = df[m_code & m_name & m_price & m_amt & m_turn & m_notlim].copy()
    out = out.sort_values("成交额", ascending=False)
    print(f"  快照初筛: 全A {len(df)} → 候选精算 {len(out)} 只")
    return out[["代码", "名称", "最新价", "涨跌幅", "换手率", "量比", "成交额"]].to_dict("records")


# ===================== 快照失败兜底列表 =====================
def fallback_list(cap):
    print(f"  ⚠️ 快照失败, 启用兜底: akshare 列表截断{cap}只逐只算形态(无当日实时, 质量降级)")
    try:
        d = ak.stock_info_a_code_name()
        if d is None or d.empty or 'code' not in d.columns:
            print("  兜底列表为空"); return []
        name_col = 'name' if 'name' in d.columns else d.columns[1]
        d = d[['code', name_col]].copy(); d.columns = ['代码', '名称']
        d['代码'] = d['代码'].astype(str).str.zfill(6)
        d = d[d['代码'].str.startswith(PARAMS['KEEP_PREFIX'])]
        d = d[~d['名称'].astype(str).str.contains('|'.join(PARAMS['EXCLUDE_NAME']), na=False, regex=True)]
        recs = [{"代码": r['代码'], "名称": r['名称'], "最新价": None, "涨跌幅": None,
                 "换手率": None, "量比": None, "成交额": None} for _, r in d.head(cap).iterrows()]
        print(f"  兜底列表: {len(recs)} 只 (按代码序, 偏沪市)")
        return recs
    except Exception as e:
        print(f"  兜底列表失败: {e}")
        return []


# ===================== 历史日线(带重试) =====================
def fetch_hist(code, end_y, retries=3):
    start = (datetime.strptime(end_y, "%Y%m%d") - timedelta(days=PARAMS["LOOKBACK"] + 30)).strftime("%Y%m%d")
    for attempt in range(retries):
        try:
            d = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                                   end_date=end_y, adjust="qfq")
            if d is not None and not d.empty:
                d = d.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                      "最低": "low", "收盘": "close", "成交量": "volume"})
                for c in ["open", "high", "low", "close", "volume"]:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d["date"] = pd.to_datetime(d["date"])
                d = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                d["pct"] = d["close"].pct_change() * 100
                return d
        except Exception as e:
            print(f"   [hist] {code} 第{attempt + 1}次失败: {e}")
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


# ===================== 形态打分 =====================
def score_candidate(h, code):
    reasons = []
    detail = {}
    if len(h) < 25:
        return 0.0, ["数据不足"], detail
    last = h.iloc[-1]
    close = last["close"]
    score = 0.0

    h20 = h["high"].iloc[-20:].max()
    near = (h20 - close) / h20 * 100 if h20 else 999
    detail["距20日高%"] = round(near, 1) if pd.notna(near) else None
    if pd.notna(near) and near <= PARAMS["NEAR_HIGH_PCT"]:
        score += PARAMS["W_BREAK"]; reasons.append(f"▲逼近20日高点(差{near:.1f}%)")

    ma5 = h["close"].rolling(5).mean().iloc[-1]
    ma10 = h["close"].rolling(10).mean().iloc[-1]
    ma20 = h["close"].rolling(20).mean().iloc[-1]
    bull = pd.notna(ma20) and ma5 > ma10 > ma20
    detail["多头"] = bool(bull)
    if bull:
        score += PARAMS["W_BULL"]; reasons.append("▲均线多头(MA5>10>20)")
    elif PARAMS["REQUIRE_BULL"]:
        reasons.append("·非多头排列")

    v20 = h["volume"].iloc[-20:].mean()
    vr = last["volume"] / v20 if v20 else 0
    detail["量比"] = round(vr, 2)
    if vr >= PARAMS["VOL_RATIO_MIN"]:
        score += PARAMS["W_VOL"]; reasons.append(f"▲放量(量比{vr:.1f})")

    thr = _hist_limit_pct(code)
    recent_lim = int((h["pct"].iloc[-PARAMS["ACTIVE_LOOKBACK"]:] >= thr).sum()) if "pct" in h.columns else 0
    detail["近N日涨停数"] = recent_lim
    if recent_lim >= 1:
        score += PARAMS["W_ACTIVE"]; reasons.append(f"▲近{PARAMS['ACTIVE_LOOKBACK']}日有涨停(股性活)")

    h60 = h["high"].iloc[-60:].max() if len(h) >= 60 else h["high"].max()
    dd = (h60 - close) / h60 * 100 if h60 else 0
    p5 = (close / h["close"].iloc[-6] - 1) * 100 if len(h) >= 6 else 0
    detail["距60日高%"] = round(dd, 1)
    reverse = (dd >= PARAMS["DD_MIN"]) and (p5 > 0) and (vr > 1.0)
    if reverse:
        score += PARAMS["W_REVERSE"]; reasons.append(f"▲超跌企稳(回撤{dd:.0f}%+近5日涨+放量)")

    if not reasons:
        reasons.append("·无明显涨停形态")
    return round(min(100.0, score), 1), reasons, detail


# ===================== 五日分时图(仅展示) =====================
def plot_5d(code, name, save=True):
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period="1", adjust="")
    except Exception as e:
        print("   [图] 分钟数据失败:", e); return
    df = df.rename(columns={"时间": "t", "开盘": "o", "收盘": "c", "成交量": "v"})
    df["t"] = pd.to_datetime(df["t"]); df = df.sort_values("t").reset_index(drop=True)
    df["d"] = df["t"].dt.date
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0)
    df["amt"] = pd.to_numeric(df["成交额"], errors="coerce").fillna(0)
    g = df.groupby("d")
    df["avg"] = (g["amt"].cumsum() / (g["v"].cumsum() * 100).replace(0, pd.NA)).ffill()
    base = df["c"].iloc[0]; x = range(len(df))
    tp, tl = [], []
    for d, sub in df.groupby("d", sort=True):
        tp.append(sub.index[0]); tl.append(pd.Timestamp(d).strftime("%m-%d"))
    cols = ["#e84545" if c >= o else "#1aa260" for c, o in zip(df["c"], df["o"])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    a1.plot(x, df["c"], color="#1f6fd6", lw=1); a1.plot(x, df["avg"], color="#e8843c", lw=1)
    a1.axhline(base, color="#888", lw=.8, ls="--")
    for p in tp[1:]:
        a1.axvline(p, color="#ccc", lw=.6); a2.axvline(p, color="#ccc", lw=.6)
    a1.grid(alpha=.25); a2.bar(x, df["v"], color=cols, width=1.0); a2.grid(alpha=.25)
    a2.set_xticks(tp); a2.set_xticklabels(tl)
    fig.suptitle(f"{name} {code}  五日分时(涨停候选池)", color="#c0392b", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"zt_pre_5d_{code}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===================== 模式A: 候选池 =====================
def run_candidate():
    print("=" * 70)
    print(f"[MODE=candidate] 涨停候选池 | 北京 {_bj_now():%Y-%m-%d %H:%M} | 场次: {_session_tag()} | "
          f"{'收盘后(将回看命中)' if _is_after_close_bj() else '午盘盘后/盘中(备候选)'}")
    print("⚠️ 候选池=形态筛选缩小范围, 非预测涨停; 打分仅排序, 不保证; 选股不依赖图形")
    print("=" * 70)

    cands = snapshot_filter()
    degraded = False
    if not cands and PARAMS["FALLBACK_FULL"]:
        cands = fallback_list(PARAMS["FALLBACK_CAP"])
        degraded = bool(cands)
    if not cands:
        print("⚠️ 快照与兜底均无候选, 本次空跑(限流严重或 FALLBACK_FULL=0); 交易日重试")
        _push_empty("快照与兜底均无候选(东财限流或 FALLBACK_FULL=0); 交易日错峰或手动重跑通常恢复")
        return pd.DataFrame()

    rows = []
    _t0 = time.time()
    print(f"逐只拉日线算形态(约 {len(cands) * PARAMS['SLEEP']:.0f}s, 含重试, 软超时{PARAMS['SOFT_TIMEOUT'] // 60}分钟)...")
    for c in cands:
        if time.time() - _t0 > PARAMS["SOFT_TIMEOUT"]:
            print(f"\n⏱️ 软超时{PARAMS['SOFT_TIMEOUT'] // 60}分钟到, 主动收工(已算{len(rows)}/{len(cands)}只), 保住已算候选")
            break
        code, name = c["代码"], c["名称"]
        end_y = datetime.now().strftime("%Y%m%d")
        h = fetch_hist(code, end_y)
        if h.empty:
            time.sleep(PARAMS["SLEEP"]); continue
        score, reasons, detail = score_candidate(h, code)
        rows.append({
            "代码": code, "名称": name,
            "最新价": c.get("最新价"), "快照涨幅%": c.get("涨跌幅"),
            "潜力分": score, "理由": " | ".join(reasons),
            "距20日高%": detail.get("距20日高%"), "多头": detail.get("多头"),
            "量比": detail.get("量比"), "近N日涨停": detail.get("近N日涨停数"),
            "距60日高%": detail.get("距60日高%"),
        })
        time.sleep(PARAMS["SLEEP"])

    if not rows:
        print("⚠️ 无有效日线, 候选池为空")
        _push_empty("有候选但逐只拉日线全失败/无有效形态命中(限流或门槛严)")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("潜力分", ascending=False).reset_index(drop=True)
    pool = df.head(PARAMS["TOP_N"]).copy()

    # 【本版】实时价刷新: 腾讯行情(海外可访问), 保证午盘/收盘两次推送的现价都是实时价
    pool = _refresh_realtime_price(pool)

    pd.set_option("display.unicode.east_asian_width", True); pd.set_option("display.width", 240)
    mode_tag = " [退化兜底]" if degraded else ""
    print(f"\n[漏斗] 初筛 {len(cands)} → 有效日线 {len(df)} → 候选池 Top{len(pool)}{mode_tag}\n")
    show = ["代码", "名称", "潜力分", "最新价", "快照涨幅%", "量比", "距20日高%", "多头", "近N日涨停", "理由"]
    print("===== 涨停候选池(形态筛选, 仅排序, 非预测) =====")
    print(pool[show].to_string(index=False))

    tag = datetime.now().strftime("%Y%m%d")
    pool_csv = os.path.join(OUTPUT_DIR, f"zt_candidates_{tag}.csv")
    pool.to_csv(pool_csv, index=False, encoding="utf-8-sig")
    latest_json = os.path.join(OUTPUT_DIR, "zt_candidates_latest.json")
    with open(latest_json, 'w', encoding='utf-8') as f:
        json.dump({"build_time": _bj_now().strftime("%Y-%m-%d %H:%M"), "session": _session_tag(),
                   "degraded": degraded, "pool": pool.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(OUTPUT_DIR, f"zt_pre_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump({"date": tag, "mode": "candidate", "session": _session_tag(), "degraded": degraded,
                   "funnel": {"snap": len(cands), "valid": len(df), "pool": len(pool)},
                   "pool": pool.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 已存 {pool_csv} 与 {latest_json}")

    if _is_after_close_bj():
        hit_review(pool, tag)

    if PARAMS["DRAW"] and not pool.empty:
        print("\n画五日分时图(仅展示)：")
        for _, r in pool.head(PARAMS["DRAW_TOP"]).iterrows():
            plot_5d(r["代码"], r["名称"]); print("   saved output/zt_pre_5d_%s.png" % r["代码"]); time.sleep(1.0)

    if SERVERCHAN_KEY:
        sess = _session_tag()
        if not pool.empty:
            head = f"**候选池·{sess} {_bj_now():%m-%d %H:%M}** | 初筛{len(cands)}→有效{len(df)}→Top{len(pool)} | 现价=实时价"
            if degraded:
                head += " | ⚠️退化兜底(快照限流)"
            lines = [head, "*(形态筛选缩小范围, 非预测涨停; 现价=腾讯实时价; 选股不依赖图形)*", ""]
            for _, r in pool.head(PUSH_TOP).iterrows():
                price = r['最新价'] if pd.notna(r['最新价']) else '—'
                lines.append(f"- **{r['名称']}({r['代码']})** 分{r['潜力分']} 现价{price} | {r['理由']}")
            if len(pool) > PUSH_TOP:
                lines.append(f"\n*…另有 {len(pool) - PUSH_TOP} 只, 详见 output*")
            lines.append("\n*⚠️ 不保证涨停; 打板/埋伏高风险, 仅供参考, 不构成投资建议。*")
            send_serverchan(f"涨停候选池·{sess} {_bj_now():%m-%d} | Top{len(pool)}", "\n".join(lines))
        else:
            _push_empty(f"有效日线{len(df)}只, 但形态打分排序后候选池为空(门槛严)")

    return pool


def hit_review(pool, tag):
    try:
        zt = ak.stock_zt_pool_em(date=tag)
        if zt is None or zt.empty:
            print("\n[回看] 今日涨停池为空, 跳过命中回看"); return
        zt_codes = set(zt["代码"].astype(str).str.zfill(6))
        pool_codes = set(pool["代码"])
        hit = pool_codes & zt_codes
        rate = len(hit) / len(pool_codes) * 100 if pool_codes else 0
        print(f"\n[回看·诚实自验] 候选池 {len(pool_codes)} 只 ∩ 今日涨停 {len(zt_codes)} 只 = 命中 {len(hit)} 只 "
              f"(命中率 {rate:.1f}%)")
        if hit:
            names = dict(zip(pool["代码"], pool["名称"]))
            print("   命中: " + ", ".join(f"{c} {names.get(c, '')}" for c in hit))
        with open(os.path.join(OUTPUT_DIR, f"zt_hit_review_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "pool_n": len(pool_codes), "zt_n": len(zt_codes),
                       "hit_n": len(hit), "hit_rate": round(rate, 1),
                       "hit_codes": sorted(hit)}, f, ensure_ascii=False, indent=2)
        print("   (注: 命中率仅验证'形态筛选'的召回, 不代表可预测涨停; 多数日子命中率本就不高)")
    except Exception as e:
        print(f"[回看] 取今日涨停池失败: {e}")


# ===================== 模式B: 盘中逼近监控(仅本地常驻) =====================
def run_intraday():
    print("=" * 70)
    print(f"[MODE=intraday] 盘中逼近涨停监控 | 北京 {_bj_now():%H:%M:%S} | "
          f"间隔{PARAMS['INTRADAY_INTERVAL']}s 最长{PARAMS['INTRADAY_DURATION']}min")
    print("⚠️ 仅本地常驻! 切勿在 Actions 跑; 只盯候选池, 不扫全市场")
    print("=" * 70)

    latest_json = os.path.join(OUTPUT_DIR, "zt_candidates_latest.json")
    if not os.path.exists(latest_json):
        print("⚠️ 未找到 zt_candidates_latest.json, 先跑一次 MODE=candidate 生成候选池")
        return
    with open(latest_json, encoding='utf-8') as f:
        pool = json.load(f)["pool"]
    watch = {r["代码"]: r["名称"] for r in pool}
    print(f"监控候选池 {len(watch)} 只; 涨幅≥{PARAMS['INTRADAY_APPROACH']}% 触发报警")

    alerted = set()
    deadline = time.time() + PARAMS["INTRADAY_DURATION"] * 60
    round_n = 0
    while time.time() < deadline:
        round_n += 1
        try:
            snap = ak.stock_zh_a_spot_em()
            if snap is None or snap.empty:
                raise RuntimeError("空快照")
            snap["代码"] = snap["代码"].astype(str).str.zfill(6)
            snap["涨跌幅"] = pd.to_numeric(snap["涨跌幅"], errors="coerce")
            snap["最新价"] = pd.to_numeric(snap["最新价"], errors="coerce")
            sub = snap[snap["代码"].isin(watch)]
            now_s = _bj_now().strftime("%H:%M:%S")
            for _, r in sub.iterrows():
                chg = r["涨跌幅"]
                if pd.notna(chg) and chg >= PARAMS["INTRADAY_APPROACH"] and r["代码"] not in alerted:
                    alerted.add(r["代码"])
                    msg = (f"⚡[{now_s}] {watch[r['代码']]}({r['代码']}) 逼近涨停 涨幅{chg:.1f}% "
                           f"现价{r['最新价']}")
                    print(msg)
                    if SERVERCHAN_KEY:
                        send_serverchan(f"⚡逼近涨停 {watch[r['代码']]}",
                                        f"{msg}\n\n*(盘中逼近监测, 非封板保证; 仅供参考)*")
            print(f"  [轮{round_n} {now_s}] 监控{len(sub)}只 已报{len(alerted)}只")
        except Exception as e:
            print(f"  [轮{round_n}] 快照失败(限流?): {e}")
        time.sleep(PARAMS["INTRADAY_INTERVAL"])
    print(f"\n[监控结束] 共{round_n}轮, 触发{len(alerted)}只: {sorted(alerted)}")


# ===================== 主入口 =====================
def main():
    mode = PARAMS["MODE"].strip().lower()
    if mode == "intraday":
        run_intraday()
    else:
        run_candidate()


if __name__ == "__main__":
    main()
# >>>FILE_END_zt<<<
