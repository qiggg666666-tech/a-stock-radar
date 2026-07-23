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

运行环境适配：
  - MODE=candidate (默认): 跑一次即退, GitHub Actions cron 友好; 盘前跑=备今日, 盘后跑=备明日+回看。
  - MODE=intraday: 盘中 while 轮询, 【仅本地常驻】, 切勿放进 Actions cron(会卡到 timeout 红叉+限流);
    只盯候选池(先跑 candidate 存 zt_candidates_latest.json), 不扫全市场, 省接口。

数据源 akshare(东财)。产物全部存 output/ 且前缀 zt_/zt_pre_/zt_candidates_,
  与 quant_signal_bot.py 的 sel_*/quant_signal_* 产物互不撞名, 两脚本可并存。
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

import matplotlib
matplotlib.use("Agg")   # 强制非交互后端, CI 无显示环境画图更稳 (须在 import pyplot 之前)
import matplotlib.pyplot as plt

# 字体 fallback 链: 首位 WenQuanYi Zen Hei = workflow 所装 fonts-wqy-zenhei 的 family 名
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei',
                                   'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 参数区（想换风格只改这里）=====================
PARAMS = dict(
    MODE="candidate",     # candidate=盘前/盘后候选池(跑一次,Actions友好); intraday=盘中逼近监控(本地常驻,别在Actions设)

    # ---- 通用过滤 ----
    KEEP_PREFIX=("0", "3", "6"),   # 只留沪深，排除北交所
    EXCLUDE_NAME=("ST", "退"),     # 排除 ST/退市
    MIN_PRICE=3.0,
    AMOUNT_MIN=1.0e8,              # 初筛成交额下限(快照向量化砍量)
    TURNOVER_MIN=1.0,              # 初筛换手率下限
    NOT_LIMIT_PCT=9.5,             # 初筛排除"已涨停"(已涨停不算"提前")

    # ---- 候选池形态阈值(全部用基准日及之前数据, 无前视) ----
    LOOKBACK=70,                   # 历史日线回看天数
    NEAR_HIGH_PCT=3.0,             # 收盘距近20日高点 ≤ 此% 视为逼近突破
    REQUIRE_BULL=True,             # 是否要求 MA5>MA10>MA20 多头排列(加分项, 非硬卡)
    VOL_RATIO_MIN=1.3,             # 昨日量比(昨量/20日均量) ≥ 此 视为放量
    ACTIVE_LOOKBACK=20,            # 近N日内有过涨停=股性活(加分, 不硬卡)
    DD_MIN=10.0,                   # 超跌企稳维度: 距60日高点回撤≥此% 且近期企稳放量

    # ---- 打分权重(满分100; 仅用于排序, 不设硬阈值) ----
    W_BREAK=30, W_BULL=20, W_VOL=20, W_ACTIVE=15, W_REVERSE=15,
    TOP_N=30,                      # 候选池取前N

    # ---- 盘中逼近监控(仅 MODE=intraday) ----
    INTRADAY_INTERVAL=20,          # 轮询间隔秒(别太小, 防限流)
    INTRADAY_APPROACH=7.0,         # 涨幅≥此% 视为逼近涨停(主板涨停10%, 留缓冲; 近似, 见说明)
    INTRADAY_DURATION=240,         # 盘中监控最长分钟数(到点自停, 防本地忘关)

    # ---- 输出/绘图 ----
    DRAW=True,
    DRAW_TOP=8,
    SLEEP=0.4,                     # 逐只取日线间隔(秒)，限频用
)

# ===================== 运行环境（env 可调）=====================
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '12'))
# env 可覆盖 MODE: 本地盘中跑设 MODE=intraday; Actions 上务必保持 candidate
PARAMS["MODE"] = os.environ.get('MODE', PARAMS["MODE"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BJ = timezone(timedelta(hours=8))   # 北京时间(不依赖 runner 时区)


# ===================== 工具 =====================
def _col(df, *names):
    """容错取列名(ak 版本间列名偶有差异)"""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _bj_now():
    return datetime.now(_BJ)


def _is_after_close_bj():
    """北京时间≥15:00 视为收盘后(用于决定是否做命中回看)"""
    return _bj_now().hour >= 15


def _hist_limit_pct(code):
    """历史日线判定涨停的涨幅阈值(近似): 创业30/科创68≈19.5, 其余≈9.8"""
    return 19.5 if str(code).startswith(("30", "68")) else 9.8


def send_serverchan(title, content, sendkey=""):
    """可选推送: serverchan-sdk 软导入优先, requests 兜底; 无 key 静默不推"""
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


# ===================== 快照初筛(向量化, 5000->几百, 避免逐只拉历史) =====================
def snapshot_filter():
    """东财全A快照向量化初筛; 返回 list[dict]; 失败返回 []"""
    df = None
    for i in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  全A快照第{i+1}次失败: {e}")
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
    m_notlim = df["涨跌幅"].abs() < PARAMS["NOT_LIMIT_PCT"]   # 排除已涨停(不算提前)
    out = df[m_code & m_name & m_price & m_amt & m_turn & m_notlim].copy()
    out = out.sort_values("成交额", ascending=False)
    print(f"  快照初筛: 全A {len(df)} → 候选精算 {len(out)} 只")
    return out[["代码", "名称", "最新价", "涨跌幅", "换手率", "量比", "成交额"]].to_dict("records")


# ===================== 历史日线(带重试) =====================
def fetch_hist(code, end_y, retries=3):
    """逐只日线(带退避重试, 降低限流漏票); 失败返回空 df"""
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
            print(f"   [hist] {code} 第{attempt+1}次失败: {e}")
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


# ===================== 形态打分(可解释, 仅排序, 不保证; 接收 code 一次算对) =====================
def score_candidate(h, code):
    """用基准日(=h 最后一行, 即昨日收盘)及之前数据算涨停潜力分 + reasons; 无前视。
    返回 (score, reasons, detail)"""
    reasons = []
    detail = {}
    if len(h) < 25:
        return 0.0, ["数据不足"], detail
    last = h.iloc[-1]
    close = last["close"]
    score = 0.0

    # 维度1 逼近突破: 收盘距近20日高点
    h20 = h["high"].iloc[-20:].max()
    near = (h20 - close) / h20 * 100 if h20 else 999
    detail["距20日高%"] = round(near, 1) if pd.notna(near) else None
    if pd.notna(near) and near <= PARAMS["NEAR_HIGH_PCT"]:
        score += PARAMS["W_BREAK"]; reasons.append(f"▲逼近20日高点(差{near:.1f}%)")

    # 维度2 多头排列
    ma5 = h["close"].rolling(5).mean().iloc[-1]
    ma10 = h["close"].rolling(10).mean().iloc[-1]
    ma20 = h["close"].rolling(20).mean().iloc[-1]
    bull = pd.notna(ma20) and ma5 > ma10 > ma20
    detail["多头"] = bool(bull)
    if bull:
        score += PARAMS["W_BULL"]; reasons.append("▲均线多头(MA5>10>20)")
    elif PARAMS["REQUIRE_BULL"]:
        reasons.append("·非多头排列")

    # 维度3 放量
    v20 = h["volume"].iloc[-20:].mean()
    vr = last["volume"] / v20 if v20 else 0
    detail["量比"] = round(vr, 2)
    if vr >= PARAMS["VOL_RATIO_MIN"]:
        score += PARAMS["W_VOL"]; reasons.append(f"▲放量(量比{vr:.1f})")

    # 维度4 股性活: 近N日有过涨停(涨幅阈值按 code 近似)
    thr = _hist_limit_pct(code)
    recent_lim = int((h["pct"].iloc[-PARAMS["ACTIVE_LOOKBACK"]:] >= thr).sum()) if "pct" in h.columns else 0
    detail["近N日涨停数"] = recent_lim
    if recent_lim >= 1:
        score += PARAMS["W_ACTIVE"]; reasons.append(f"▲近{PARAMS['ACTIVE_LOOKBACK']}日有涨停(股性活)")

    # 维度5 超跌企稳: 距60日高回撤大 + 近5日收涨 + 昨日放量
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


# ===================== 五日分时图(存 output/, 前缀 zt_pre_5d_) =====================
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


# ===================== 模式A: 候选池(盘前/盘后, 跑一次) =====================
def run_candidate():
    print("=" * 70)
    print(f"[MODE=candidate] 涨停候选池 | 北京 {_bj_now():%Y-%m-%d %H:%M} | "
          f"{'收盘后(将回看命中)' if _is_after_close_bj() else '盘前/盘中(备候选)'}")
    print("⚠️ 候选池=形态筛选缩小范围, 非预测涨停; 打分仅排序, 不保证")
    print("=" * 70)

    cands = snapshot_filter()
    if not cands:
        return pd.DataFrame()

    rows = []
    print(f"逐只拉日线算形态(约 {len(cands)*PARAMS['SLEEP']:.0f}s, 含重试)...")
    for c in cands:
        code, name = c["代码"], c["名称"]
        end_y = datetime.now().strftime("%Y%m%d")
        h = fetch_hist(code, end_y)
        if h.empty:
            time.sleep(PARAMS["SLEEP"]); continue
        score, reasons, detail = score_candidate(h, code)   # 一次算对, 含 code 用于涨停阈值
        rows.append({
            "代码": code, "名称": name,
            "最新价": c["最新价"], "快照涨幅%": c["涨跌幅"],
            "潜力分": score, "理由": " | ".join(reasons),
            "距20日高%": detail.get("距20日高%"), "多头": detail.get("多头"),
            "量比": detail.get("量比"), "近N日涨停": detail.get("近N日涨停数"),
            "距60日高%": detail.get("距60日高%"),
        })
        time.sleep(PARAMS["SLEEP"])

    if not rows:
        print("⚠️ 无有效日线, 候选池为空")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("潜力分", ascending=False).reset_index(drop=True)
    pool = df.head(PARAMS["TOP_N"]).copy()

    # 打印漏斗 + 候选池
    pd.set_option("display.unicode.east_asian_width", True); pd.set_option("display.width", 240)
    print(f"\n[漏斗] 快照初筛 {len(cands)} → 有效日线 {len(df)} → 候选池 Top{len(pool)}\n")
    show = ["代码", "名称", "潜力分", "最新价", "快照涨幅%", "量比", "距20日高%", "多头", "近N日涨停", "理由"]
    print("===== 涨停候选池(形态筛选, 仅排序, 非预测) =====")
    print(pool[show].to_string(index=False))

    # 存 output (csv + latest json 供 intraday 读; 前缀与 quant 产物不撞)
    tag = datetime.now().strftime("%Y%m%d")
    pool_csv = os.path.join(OUTPUT_DIR, f"zt_candidates_{tag}.csv")
    pool.to_csv(pool_csv, index=False, encoding="utf-8-sig")
    latest_json = os.path.join(OUTPUT_DIR, "zt_candidates_latest.json")
    with open(latest_json, 'w', encoding='utf-8') as f:
        json.dump({"build_time": _bj_now().strftime("%Y-%m-%d %H:%M"),
                   "pool": pool.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(OUTPUT_DIR, f"zt_pre_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump({"date": tag, "mode": "candidate",
                   "funnel": {"snap": len(cands), "valid": len(df), "pool": len(pool)},
                   "pool": pool.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 已存 {pool_csv} 与 {latest_json}")

    # 收盘后命中回看(诚实自验: 昨日数据算的候选 vs 今日涨停结果)
    if _is_after_close_bj():
        hit_review(pool, tag)

    # 画五日分时(候选池)
    if PARAMS["DRAW"] and not pool.empty:
        print("\n画五日分时图：")
        for _, r in pool.head(PARAMS["DRAW_TOP"]).iterrows():
            plot_5d(r["代码"], r["名称"]); print("   saved output/zt_pre_5d_%s.png" % r["代码"]); time.sleep(1.0)

    # 推送
    if SERVERCHAN_KEY and not pool.empty:
        lines = [f"**候选池 {_bj_now():%m-%d %H:%M}** | 初筛{len(cands)}→有效{len(df)}→Top{len(pool)}",
                 "*(形态筛选缩小范围, 非预测涨停)*", ""]
        for _, r in pool.head(PUSH_TOP).iterrows():
            lines.append(f"- **{r['名称']}({r['代码']})** 分{r['潜力分']} 现价{r['最新价']} | {r['理由']}")
        if len(pool) > PUSH_TOP:
            lines.append(f"\n*…另有 {len(pool)-PUSH_TOP} 只, 详见 output*")
        lines.append("\n*⚠️ 不保证涨停; 打板/埋伏高风险, 仅供参考, 不构成投资建议。*")
        send_serverchan(f"涨停候选池 {_bj_now():%m-%d} | Top{len(pool)}", "\n".join(lines))

    return pool


def hit_review(pool, tag):
    """收盘后回看: 今日涨停池 ∩ 候选池代码 -> 命中率(验证候选池质量, 非筛选)"""
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
            print("   命中: " + ", ".join(f"{c} {names.get(c,'')}" for c in hit))
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

    # 读候选池(先跑 candidate 生成的 latest json)
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


# ===================== 主入口(按 MODE 分流) =====================
def main():
    mode = PARAMS["MODE"].strip().lower()
    if mode == "intraday":
        run_intraday()
    else:
        run_candidate()


if __name__ == "__main__":
    main()
