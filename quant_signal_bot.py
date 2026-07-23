# -*- coding: utf-8 -*-
"""
quant_signal_bot.py —— 收盘后选股器：抓「超跌/筑底后的首板涨停 + 板块共振」
（收盘后看结果版；与 zt_pre_screener.py 的"提前找涨停"并存，互不覆盖）
数据源：涨停池/连板数/封板资金/炸板次数 = akshare 东财独家(baostock 没有, 必须保留, 加超时保护);
        个股日线 hist = baostock 优先 + 东财兜底(双源更稳)。
个股只是参数/筛选结果，逻辑全在 PARAMS。

【本版整合修复(含对外部审查意见的吸收)】
 1. 删除死代码 sort_values(key=lambda s: ... if False else s)(审查点①: 该 sort 本质空操作且被下一行覆盖)。
 2. 中文字体补 WenQuanYi Zen Hei(审查点②: workflow 装 fonts-wqy-zenhei 对应此名, 否则 Actions 图中文变方块)。
 3. 【新增·审查点③】所有 akshare 调用加硬超时(_call_with_timeout 线程池包超时)+重试:
    stock_zt_pool_em / stock_board_industry_name_em / stock_zh_a_hist_min_em / stock_zh_a_hist,
    防止东财卡住拖死整个 job。超时后后台线程靠 socket 层最终返回 + job timeout 兜底。
 4. 接 Server 酱推送(审查点④: 软导入+requests 兜底, 有 key 才推, 仅推精选摘要)。
 5. 个股日线 hist 改双源: baostock 优先(主进程登录态+硬超时)+ 东财兜底(超时+重试);
    baostock 返回英文列 rename 成中文以兼容 reverse_feat(其用 h["日期"]/h["最高"]/h["收盘"])。
 6. 图/CSV/JSON 存 output/(对接整合 workflow 的 upload-artifact); import pyplot 前 use("Agg")。
 7. reverse_feat 的 rev=le_ma and(A or B) 经核对正确, 拆成 cond_dd/cond_d5 仅提升可读性, 未改判定。
 不覆盖 zt_pre_screener.py; 本脚本核心筛选逻辑(涨停池->首板->超跌反转->排序)原样保留。
"""
import os
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd
import baostock as bs

import matplotlib
matplotlib.use("Agg")   # 强制非交互后端, CI 无显示环境画图更稳 (须在 import pyplot 之前)
import matplotlib.pyplot as plt

# 字体 fallback 链: 首位 WenQuanYi Zen Hei = workflow 所装 fonts-wqy-zenhei 的 family 名
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei',
                                   'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 参数区（想换风格只改这里）=====================
PARAMS = dict(
    DATE=None,            # None=自动取最近有数据的交易日；或写 "20260723"
    KEEP_PREFIX=("0", "3", "6"),   # 只留沪深，排除北交所
    EXCLUDE_NAME=("ST", "退"),     # 排除 ST/退市
    DD_MIN=12.0,          # 拉板前距20日高点回撤 ≥ 此值 视为超跌
    D5_MAX=0.0,           # 拉板前5日累计涨幅 < 此值 视为近期在跌
    MA=20,                # 用 MA 几 判断"还在底部"
    BOARD_FILTER=0.0,     # >0 时硬卡行业涨幅≥此值(共振)；0=只展示不卡
    DRAW=True,            # 是否对入选票画五日分时图
    DRAW_TOP=10,          # 最多画几只(防太慢)
    SLEEP=0.4,            # 逐只取日线间隔(秒)，限频用
)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '10'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '25'))   # 单次 akshare 调用硬超时(秒)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_BS_LOGGED = False   # 主进程 baostock 登录态(hist 双源用)


# ===================== 工具 =====================
def _col(df, *names):
    """容错取列名(ak 版本间列名偶有差异)"""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _pref(code6):
    """6位代码 -> 带 sh./sz. 前缀(baostock 格式)"""
    c = str(code6).split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c


def _call_with_timeout(fn, *args, timeout=AK_TIMEOUT, **kwargs):
    """给单次 akshare 调用包硬超时(防东财卡死拖死 job); 超时抛 FutureTimeoutError。
    注: 超时后后台线程靠 socket 层最终返回, 极端情况由 job timeout 兜底。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)


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


def _bs_query_with_timeout(code, fields, start_date, timeout=AK_TIMEOUT):
    """baostock 单次查询硬超时(线程池包); 供 hist 的 baostock 路径用"""
    def _do():
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, adjustflag="2")
        return rs.get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


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


# ===================== 数据获取(全部带超时+重试) =====================
def resolve_date():
    if PARAMS["DATE"]:
        return PARAMS["DATE"]
    d = datetime.now()
    for i in range(8):                      # 自动回退找最近有数据的交易日
        ds = (d - timedelta(days=i)).strftime("%Y%m%d")
        try:
            t = _call_with_timeout(ak.stock_zt_pool_em, date=ds, timeout=20)
            if t is not None and not t.empty:
                return ds
        except FutureTimeoutError:
            print(f"   [resolve] {ds} 涨停池超时, 试下一天")
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("近8天都取不到涨停池，检查网络/akshare版本")


def fetch_zt(date):
    """涨停池(东财独家, 加超时+重试)"""
    df = None
    for attempt in range(3):
        try:
            df = _call_with_timeout(ak.stock_zt_pool_em, date=date, timeout=AK_TIMEOUT)
            if df is not None and not df.empty:
                break
        except FutureTimeoutError:
            print(f"   [涨停池] 第{attempt+1}次超时")
        except Exception as e:
            print(f"   [涨停池] 第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    if df is None or df.empty:
        raise RuntimeError(f"{date} 涨停池获取失败(限流/超时)")
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["连板数"] = pd.to_numeric(_col(df, "连板数"), errors="coerce")
    # 连板数缺失时，用"涨停统计"(形如 1/3)首段兜底
    miss = df["连板数"].isna()
    if miss.any() and "涨停统计" in df.columns:
        df.loc[miss, "连板数"] = df.loc[miss, "涨停统计"].astype(str).str.split("/").str[0]
    df["连板数"] = pd.to_numeric(df["连板数"], errors="coerce").fillna(1).astype(int)
    df["封成比"] = pd.to_numeric(_col(df, "封板资金"), errors="coerce") / \
                 pd.to_numeric(_col(df, "成交额"), errors="coerce").replace(0, pd.NA) * 100
    df["炸板"] = pd.to_numeric(_col(df, "炸板次数"), errors="coerce").fillna(0).astype(int)
    df["首封"] = _col(df, "首次封板时间").astype(str).str.replace(":", "", regex=False)
    df["首封_int"] = pd.to_numeric(df["首封"], errors="coerce").fillna(999999).astype(int)
    df["行业"] = _col(df, "所属行业").astype(str)
    return df


def base_filter(df):
    m = df["代码"].str.startswith(PARAMS["KEEP_PREFIX"])
    nm = df["名称"].astype(str)
    for kw in PARAMS["EXCLUDE_NAME"]:
        m &= ~nm.str.contains(kw, na=False)
    return df[m].copy()


def industry_chg():
    """行业涨幅(东财, 加超时+重试; 失败返回空 dict 不致命)"""
    for attempt in range(3):
        try:
            b = _call_with_timeout(ak.stock_board_industry_name_em, timeout=20)
            if b is not None and not b.empty:
                return dict(zip(b["板块名称"].astype(str), pd.to_numeric(b["涨跌幅"], errors="coerce")))
        except FutureTimeoutError:
            print(f"   [行业涨幅] 第{attempt+1}次超时")
        except Exception as e:
            print(f"   [行业涨幅] 第{attempt+1}次失败: {e}")
        time.sleep(2 + attempt)
    print("[warn] 行业涨幅取数失败(超时/限流), 行业涨幅列将为空")
    return {}


def hist(code, end, retries=2):
    """个股日线 双源: baostock 优先(主进程登录态+硬超时)+ 东财兜底(超时+重试)。
    返回中文列名 df(兼容 reverse_feat); 失败返回空 df。"""
    start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=75)).strftime("%Y%m%d")
    # 路径1: baostock
    if _BS_LOGGED:
        try:
            d = _bs_query_with_timeout(_pref(code), "date,open,high,low,close,volume", start)
            if d is not None and not d.empty:
                d = d.rename(columns={"date": "日期", "open": "开盘", "high": "最高",
                                      "low": "最低", "close": "收盘", "volume": "成交量"})
                for c in ["开盘", "最高", "最低", "收盘", "成交量"]:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d["日期"] = pd.to_datetime(d["日期"])
                d = d.dropna(subset=["收盘"]).sort_values("日期").reset_index(drop=True)
                if not d.empty:
                    return d
        except FutureTimeoutError:
            pass
        except Exception:
            pass
    # 路径2: 东财兜底(超时+重试)
    for attempt in range(retries):
        try:
            d = _call_with_timeout(ak.stock_zh_a_hist, symbol=code, period="daily",
                                   start_date=start, end_date=end, adjust="qfq", timeout=20)
            if d is not None and not d.empty:
                d["日期"] = pd.to_datetime(d["日期"])
                d = d.sort_values("日期").reset_index(drop=True)
                return d
        except FutureTimeoutError:
            print(f"   [hist] {code} 东财第{attempt+1}次超时")
        except Exception as e:
            print(f"   [hist] {code} 东财第{attempt+1}次失败: {e}")
        time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
    return pd.DataFrame()


def reverse_feat(h, date, ma):
    """算'超跌/筑底反转'特征，基于拉板前一日，避免用涨停当天污染判断"""
    t = h[h["日期"] == pd.Timestamp(date)]
    if t.empty or len(h) < ma + 2:
        return dict(prev5d=pd.NA, dd=pd.NA, close_ma=pd.NA, rev=False)
    i = t.index[0]
    prev = h.loc[i - 1] if i >= 1 else None
    if prev is None:
        return dict(prev5d=pd.NA, dd=pd.NA, close_ma=pd.NA, rev=False)
    win = h.loc[max(0, i - 20):i - 1]                 # 拉板前20日窗口
    high20 = win["最高"].max()
    dd = (high20 - prev["收盘"]) / high20 * 100 if high20 else pd.NA
    ma_v = h["收盘"].rolling(ma).mean().loc[i - 1]
    close_ma = prev["收盘"] / ma_v if ma_v else pd.NA
    p5 = h["收盘"].loc[i - 1] / h["收盘"].loc[i - 6] - 1 if i >= 6 else pd.NA
    p5d = p5 * 100 if pd.notna(p5) else pd.NA
    le_ma = pd.notna(close_ma) and close_ma <= 1.0
    # 原逻辑正确(and 优先级高于 or): 仍在底部 且 (超跌 或 近期在跌)。拆开仅为可读性, 未改判定。
    cond_dd = pd.notna(dd) and dd >= PARAMS["DD_MIN"]
    cond_d5 = pd.notna(p5d) and p5d < PARAMS["D5_MAX"]
    rev = le_ma and (cond_dd or cond_d5)
    return dict(prev5d=round(p5d, 1) if pd.notna(p5d) else pd.NA,
                dd=round(dd, 1) if pd.notna(dd) else pd.NA,
                close_ma=round(close_ma, 2) if pd.notna(close_ma) else pd.NA,
                rev=bool(rev))


# ===================== 五日分时图(存 output/, 分钟数据加超时) =====================
def plot_5d(code, name, save=True):
    try:
        df = _call_with_timeout(ak.stock_zh_a_hist_min_em, symbol=code, period="1", adjust="", timeout=AK_TIMEOUT)
    except FutureTimeoutError:
        print("   [图] 分钟数据超时"); return
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
    fig.suptitle(f"{name} {code}  五日分时(收盘后选股入选)", color="#c0392b", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"sel_5d_{code}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===================== 主流程 =====================
def main():
    # 主进程登录 baostock(hist 双源用; 失败也继续, hist 走东财兜底)
    if not _bs_login_ok():
        print("⚠️ baostock 登录失败, 个股日线将走东财兜底")

    date = resolve_date()
    print(f"交易日 = {date}\n")
    try:
        zt = fetch_zt(date)
    except RuntimeError as e:
        print(f"⚠️ {e}; 本次跳过")
        if _BS_LOGGED:
            try: bs.logout()
            except Exception: pass
        return
    first = base_filter(zt[zt["连板数"] == 1])           # 硬条件：首板 + 基础过滤
    ind_map = industry_chg()

    rows = []
    print(f"首板候选 {len(first)} 只，逐只取日线算反转特征(约 {len(first)*PARAMS['SLEEP']:.0f}s, 双源+超时)...")
    for _, r in first.iterrows():
        code, name = r["代码"], r["名称"]
        try:
            h = hist(code, date)
            f = reverse_feat(h, date, PARAMS["MA"])
        except Exception as e:
            print(f"   [skip] {code} 日线失败: {e}"); f = dict(prev5d=pd.NA, dd=pd.NA, close_ma=pd.NA, rev=False)
        time.sleep(PARAMS["SLEEP"])
        ichg = ind_map.get(r["行业"], float("nan"))
        rows.append({
            "代码": code, "名称": name,
            "涨幅%": round(float(r["涨跌幅"]), 2),
            "封成比%": round(r["封成比"], 1) if pd.notna(r["封成比"]) else pd.NA,
            "首封": r["首封"], "炸板": int(r["炸板"]),
            "行业": r["行业"],
            "行业涨幅%": round(ichg, 2) if pd.notna(ichg) else pd.NA,
            "前5日%": f["prev5d"], "20日回撤%": f["dd"], "收/MA": f["close_ma"],
            "反转": f["rev"],
        })

    df_all = pd.DataFrame(rows)
    # 板块共振硬卡(可选)
    if PARAMS["BOARD_FILTER"] > 0:
        df_all = df_all[df_all["行业涨幅%"].fillna(-999) >= PARAMS["BOARD_FILTER"]]
    df_sel = df_all[df_all["反转"] == True].copy()
    # 排序：封得早(首封去冒号字典序=时间序) + 炸板少 + 封成比高 (原死代码 sort 已删)
    if not df_sel.empty:
        df_sel = df_sel.sort_values(["首封", "炸板", "封成比%"],
                                    ascending=[True, True, False]).reset_index(drop=True)

    # ---- 打印漏斗 ----
    pd.set_option("display.unicode.east_asian_width", True); pd.set_option("display.width", 220)
    print(f"\n[漏斗] 当日涨停 {len(zt)} → 首板(沪深非ST) {len(first)} → 超跌反转精选 {len(df_sel)}\n")
    cols_show = ["代码", "名称", "涨幅%", "封成比%", "首封", "炸板", "行业", "行业涨幅%", "前5日%", "20日回撤%", "收/MA"]
    print("===== 精选：超跌/筑底 首板 (这就是截图那类票) =====")
    print(df_sel[cols_show].to_string(index=False) if not df_sel.empty else "(今日无符合，见下方全首板)")
    print("\n===== 参考：当日全部首板(含未满足反转) =====")
    print(df_all[cols_show + ["反转"]].to_string(index=False))

    # ---- 存 CSV + JSON 到 output/ ----
    tag = date
    df_sel.to_csv(os.path.join(OUTPUT_DIR, f"sel_reverse_first_{tag}.csv"), index=False, encoding="utf-8-sig")
    df_all.to_csv(os.path.join(OUTPUT_DIR, f"sel_all_first_{tag}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, f"quant_signal_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump({"date": tag, "mode": "post_close_reverse_first",
                   "funnel": {"zt": int(len(zt)), "first": int(len(first)), "sel": int(len(df_sel))},
                   "board_filter": PARAMS["BOARD_FILTER"],
                   "sel": df_sel.to_dict('records')}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 已存 output/sel_reverse_first_{tag}.csv / sel_all_first_{tag}.csv / quant_signal_{tag}.json")

    # ---- 画五日分时 ----
    if PARAMS["DRAW"] and not df_sel.empty:
        print("\n画五日分时图：")
        for _, r in df_sel.head(PARAMS["DRAW_TOP"]).iterrows():
            plot_5d(r["代码"], r["名称"]); print("   saved output/sel_5d_%s.png" % r["代码"]); time.sleep(1.0)

    # ---- 推送(有 key 才推, 仅精选摘要) ----
    if SERVERCHAN_KEY:
        if df_sel.empty:
            title = f"超跌首板选股 {date} | 今日无符合"
            content = f"交易日 {date}：涨停 {len(zt)} → 首板 {len(first)} → 超跌反转精选 0。\n\n(今日无超跌/筑底首板)"
        else:
            title = f"超跌首板选股 {date} | 精选 {len(df_sel)} 只"
            lines = [f"**交易日 {date}** | 涨停 {len(zt)} → 首板 {len(first)} → 精选 {len(df_sel)}", ""]
            for _, r in df_sel.head(PUSH_TOP).iterrows():
                ichg = r['行业涨幅%']
                ichg_s = f"{ichg}%" if pd.notna(ichg) else "—"
                lines.append(f"- **{r['名称']}({r['代码']})** [{r['行业']}·行业{ichg_s}] "
                             f"封成比{r['封成比%']}% 首封{r['首封']} 炸板{r['炸板']} | "
                             f"20日回撤{r['20日回撤%']}% 收/MA{r['收/MA']}")
            if len(df_sel) > PUSH_TOP:
                lines.append(f"\n*…另有 {len(df_sel)-PUSH_TOP} 只, 详见 output*")
            lines.append("\n*超跌/筑底首板涨停+板块共振; 打板高风险, 仅供参考, 不构成投资建议。*")
            content = "\n".join(lines)
        send_serverchan(title, content)

    if _BS_LOGGED:
        try: bs.logout()
        except Exception: pass


if __name__ == "__main__":
    main()
