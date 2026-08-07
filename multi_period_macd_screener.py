# -*- coding: utf-8 -*-
"""
多周期 MACD 综合扫描脚本（baostock 数据源 + 优化版 W 底检测）· 矩阵规格
条件：1.周线DIF<-1  2.年线DIF W底(周线近似)  3.月线DEA零轴  4.月线金叉  5.季线DIF W底  6.月线DIF W底
【矩阵化】新增 Server酱分页推送+baostock行业本地join+腾讯实时价对齐+存output/; 扫描/W底逻辑一字未动。
STOCK_POOL/START_DATE/OUTPUT_DIR 均 env 可调; 依赖 baostock pandas numpy tqdm requests。
"""
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
import atexit
import re
import os
import requests
from datetime import datetime
import warnings
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, as_completed,
    TimeoutError as FuturesTimeoutError,
)

warnings.filterwarnings("ignore")

# ==================== 条件开关 ====================
ENABLE_WEEKLY_DIF_LT_MINUS1   = True
ENABLE_YEARLY_W_BOTTOM        = True
ENABLE_MONTHLY_DEA_NEAR_ZERO  = True
ENABLE_MONTHLY_GOLDEN_CROSS   = True
ENABLE_QUARTERLY_W_BOTTOM     = True
ENABLE_MONTHLY_W_BOTTOM       = True

# ==================== 基础参数 ====================
WEEKLY_DIF_THRESHOLD = -1.0
MONTHLY_DEA_THRESHOLD = 0.08
FAST, SLOW, SIGNAL = 12, 26, 9
LOOKBACK_MONTHS_FOR_CROSS = 2

W_TOL = 0.20
W_MIN_REBOUND = 0.15
W_REQUIRE_BELOW_ZERO = True

YEARLY_W_MIN_DIST, YEARLY_W_MAX_DIST, YEARLY_W_MAX_AGE = 10, 70, 8
QUARTERLY_W_MIN_DIST, QUARTERLY_W_MAX_DIST, QUARTERLY_W_MAX_AGE = 3, 16, 3
MONTHLY_W_MIN_DIST, MONTHLY_W_MAX_DIST, MONTHLY_W_MAX_AGE = 4, 24, 4

STOCK_POOL = os.getenv("STOCK_POOL", "hs300")   # hs300/zz500/custom/其他=全市场
CUSTOM_CODES = []
START_DATE = os.getenv("START_DATE", "2015-01-01")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
SERVERCHAN_KEY = os.getenv("SERVERCHAN_KEY") or os.getenv("SENDKEY", "")
PUSH_TOP = int(os.getenv("PUSH_TOP", "30"))

FETCH_TIMEOUT_SEC = 15
FETCH_MAX_RETRIES = 2
FETCH_RETRY_BACKOFF = 1.5
NUM_PROCESSES = int(os.getenv("NUM_PROCESSES", "3"))
LOGIN_STAGGER_MAX_SEC = 2.0
LOGIN_MAX_RETRIES = 3
LOGIN_RETRY_BACKOFF = 2.0

_INDUSTRY_MAP = {}

# ---------------------- 推送(分页) ----------------------
def _send_one(title, content, key):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(key, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok:
            return True
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        j = requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": title, "desp": content}, timeout=15).json()
        return j.get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False

def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += lnlen
    if cur:
        chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        ok = _send_one(t, ch, key) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print("📲 推送完成" + (" ✅" if ok else " ⚠️存在失败"))
    return ok

# ---------------------- 实时价(腾讯)+对齐 ----------------------
def _fetch_realtime_tencent(codes):
    out = {}
    try:
        syms = []
        for c in codes:
            c6 = str(c).split('.')[-1].zfill(6)
            pref = 'sh' if c6[:1] in ('6', '9') else ('bj' if c6[:1] in ('4', '8') else 'sz')
            syms.append(f"{pref}{c6}")
        for i in range(0, len(syms), 50):
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(syms[i:i+50]), timeout=10)
                r.encoding = 'gbk'
                for line in r.text.strip().split(';'):
                    if '=' not in line:
                        continue
                    f = line.split('=', 1)[1].strip().strip('"').split('~')
                    if len(f) > 4 and f[2]:
                        try:
                            px = float(f[3])
                            if px > 0:
                                out[f[2].zfill(6)] = px
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(0.3)
    except Exception as e:
        print(f"  腾讯实时价异常: {e}")
    return out

def _refresh_realtime_price(df):
    if df is None or df.empty:
        return df, {}
    df = df.copy()
    if '最新价' not in df.columns:
        df['最新价'] = df['收盘价']
    if '信号价' not in df.columns:
        df['信号价'] = df['最新价']
    codes6 = [str(c).split('.')[-1].zfill(6) for c in df['代码']]
    rt = _fetch_realtime_tencent(codes6)
    if rt:
        df['实时价'] = [rt.get(c) for c in codes6]
        df['最新价'] = df['实时价'].where(df['实时价'].notna(), df['最新价'])
    return df, rt

def _align_suffix(r, spot_now):
    sig_price = r.get('信号价', r.get('最新价')); sig_date = r.get('信号日期')
    if sig_price is None or pd.isna(sig_price):
        return ""
    head = f"🕒信号{sig_price}"
    if sig_date and not pd.isna(sig_date):
        head += f"@{str(sig_date)[:10][-5:]}"
    code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
    now = spot_now.get(code6) if spot_now else None
    if now is not None:
        try:
            chg = (now - float(sig_price)) / float(sig_price) * 100
            return f" | {head} → 现价{now}@run({chg:+.1f}%)"
        except Exception:
            return f" | {head}"
    return f" | {head"}

def _clean_industry(s):
    if not s or not isinstance(s, str):
        return "—"
    return (re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—")

# ---------------------- baostock 登录/登出 ----------------------
def _login_with_retry(max_retries=LOGIN_MAX_RETRIES):
    lg = None
    for attempt in range(max_retries):
        lg = bs.login()
        if lg.error_code == "0":
            return True
        time.sleep(LOGIN_RETRY_BACKOFF * (attempt + 1))
    print(f"[警告] baostock 登录失败: {lg.error_msg if lg else '未知错误'}")
    return False

def _safe_logout():
    try:
        bs.logout()
    except Exception:
        pass

def _worker_init():
    time.sleep(random.uniform(0, LOGIN_STAGGER_MAX_SEC))
    _login_with_retry()
    atexit.register(_safe_logout)

def _to_bs_code(code):
    code = str(code).strip()
    if code.startswith(("sh.", "sz.", "bj.")):
        return code
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    if code.startswith(("4", "8")):
        return f"bj.{code}"
    return code

def _rs_to_df(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)

def get_stock_list():
    print("获取股票列表...")
    if STOCK_POOL == "hs300":
        df = _rs_to_df(bs.query_hs300_stocks()); codes = df["code"].tolist(); names = dict(zip(df["code"], df["code_name"]))
    elif STOCK_POOL == "zz500":
        df = _rs_to_df(bs.query_zz500_stocks()); codes = df["code"].tolist(); names = dict(zip(df["code"], df["code_name"]))
    elif STOCK_POOL == "custom":
        codes = [_to_bs_code(c) for c in CUSTOM_CODES]; names = {c: c for c in codes}
    else:
        df = _rs_to_df(bs.query_stock_basic())
        df = df[(df["type"] == "1") & (df["status"] == "1")]
        df = df[~df["code_name"].str.contains("ST|退", na=False)]
        codes = df["code"].tolist(); names = dict(zip(df["code"], df["code_name"]))
    print(f"股票数量: {len(codes)}")
    return codes, names

def calc_macd(close):
    ema_fast = close.ewm(span=FAST, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=SIGNAL, adjust=False).mean()
    return dif, dea

def _fetch_hist_raw(code):
    rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume",
        start_date=START_DATE, end_date=datetime.now().strftime("%Y-%m-%d"),
        frequency="d", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"baostock查询失败: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"})
    for col in ["开盘", "最高", "最低", "收盘", "成交量"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index().dropna(subset=["收盘"])
    if len(df) < 120:
        return None
    return df

def get_data(code, timeout=FETCH_TIMEOUT_SEC, max_retries=FETCH_MAX_RETRIES):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch_hist_raw, code)
                try:
                    df = future.result(timeout=timeout)
                    if df is None:
                        last_err = "数据不足或为空"
                    else:
                        return df, None
                except FuturesTimeoutError:
                    last_err = f"请求超时(>{timeout}s)"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            time.sleep(FETCH_RETRY_BACKOFF * (attempt + 1))
    return None, last_err

def resample_ohlc(df, rule):
    return df.resample(rule).agg({"开盘": "first", "最高": "max", "最低": "min", "收盘": "last", "成交量": "sum"}).dropna()

def to_weekly(df): return resample_ohlc(df, "W-FRI")
def to_monthly(df): return resample_ohlc(df, "ME")
def to_quarterly(df): return resample_ohlc(df, "QE")

# ---------------------- W底检测（优化版, 原逻辑保留）----------------------
def find_w_bottom(dif_series, min_dist=6, max_dist=40, tol=0.20, min_rebound=0.15,
                  max_age=12, require_below_zero=True, min_rebound_vol_mult=1.5):
    if len(dif_series) < 30:
        return False, None, None, None, None
    lookback = min(len(dif_series), 90)
    s = dif_series.iloc[-lookback:].copy()
    values = s.values; dates = s.index; n = len(values)
    noise_floor = min_rebound_vol_mult * float(np.std(np.diff(values)))
    bottoms = []
    window = 2
    for i in range(window, n - window):
        is_bottom = True
        for j in range(1, window + 1):
            if values[i] > values[i - j] or values[i] > values[i + j]:
                is_bottom = False; break
        if is_bottom:
            bottoms.append({"idx": i, "val": values[i], "date": dates[i]})
    if len(bottoms) < 2:
        return False, None, None, None, None
    for k in range(len(bottoms) - 1, 0, -1):
        b2 = bottoms[k]; b1 = bottoms[k - 1]
        dist = b2["idx"] - b1["idx"]
        if not (min_dist <= dist <= max_dist):
            continue
        age = n - 1 - b2["idx"]
        if age > max_age:
            continue
        if b2["val"] < b1["val"] * (1 - tol):
            continue
        if require_below_zero and (b1["val"] > 0.3 or b2["val"] > 0.5):
            continue
        mid_start = b1["idx"] + 1; mid_end = b2["idx"]
        if mid_end <= mid_start:
            continue
        mid_high = np.max(values[mid_start:mid_end])
        mid_high_idx = mid_start + np.argmax(values[mid_start:mid_end])
        rebound1 = mid_high - b1["val"]; rebound2 = mid_high - b2["val"]
        required_rebound = max(noise_floor, abs(b1["val"]) * min_rebound)
        if rebound1 < required_rebound or rebound2 < required_rebound:
            continue
        if (mid_high_idx - b1["idx"] < 2) or (b2["idx"] - mid_high_idx < 2):
            continue
        return True, round(b1["val"], 3), round(b2["val"], 3), b1["date"], b2["date"]
    return False, None, None, None, None

# ---------------------- 单只扫描 ----------------------
def scan_one(code, name):
    try:
        daily, err = get_data(code)
        if daily is None:
            return None, err or "数据获取失败"
        weekly = to_weekly(daily); monthly = to_monthly(daily); quarterly = to_quarterly(daily)
        result = {"代码": code, "名称": name, "收盘价": round(daily["收盘"].iloc[-1], 2),
                  "信号日期": daily.index[-1].strftime("%Y-%m-%d")}

        if ENABLE_WEEKLY_DIF_LT_MINUS1:
            dif_w, _ = calc_macd(weekly["收盘"])
            if len(dif_w) < 30 or dif_w.iloc[-1] >= WEEKLY_DIF_THRESHOLD:
                return None, None
            result["周DIF"] = round(dif_w.iloc[-1], 3)
        else:
            result["周DIF"] = None

        if ENABLE_YEARLY_W_BOTTOM:
            dif_w, _ = calc_macd(weekly["收盘"])
            ok, b1, b2, d1, d2 = find_w_bottom(dif_w, YEARLY_W_MIN_DIST, YEARLY_W_MAX_DIST, W_TOL, W_MIN_REBOUND, YEARLY_W_MAX_AGE, W_REQUIRE_BELOW_ZERO)
            if not ok:
                return None, None
            result["年W前"], result["年W后"], result["年W后日期"] = b1, b2, d2.strftime("%Y-%m-%d")
        else:
            result["年W前"] = result["年W后"] = result["年W后日期"] = None

        if ENABLE_MONTHLY_DEA_NEAR_ZERO:
            _, dea_m = calc_macd(monthly["收盘"])
            if len(dea_m) < 24 or abs(dea_m.iloc[-1]) >= MONTHLY_DEA_THRESHOLD:
                return None, None
            result["月DEA"] = round(dea_m.iloc[-1], 3)
        else:
            result["月DEA"] = None

        if ENABLE_MONTHLY_GOLDEN_CROSS:
            dif_m, dea_m = calc_macd(monthly["收盘"])
            crossed = False
            for i in range(1, min(LOOKBACK_MONTHS_FOR_CROSS + 1, len(dif_m))):
                if dif_m.iloc[-i] > dea_m.iloc[-i] and dif_m.iloc[-i - 1] <= dea_m.iloc[-i - 1]:
                    crossed = True
                    result["月金叉DIF"] = round(dif_m.iloc[-i], 3); result["月金叉DEA"] = round(dea_m.iloc[-i], 3)
                    break
            if not crossed:
                return None, None
        else:
            result["月金叉DIF"] = result["月金叉DEA"] = None

        if ENABLE_QUARTERLY_W_BOTTOM:
            dif_q, _ = calc_macd(quarterly["收盘"])
            ok, b1, b2, d1, d2 = find_w_bottom(dif_q, QUARTERLY_W_MIN_DIST, QUARTERLY_W_MAX_DIST, W_TOL, W_MIN_REBOUND, QUARTERLY_W_MAX_AGE, W_REQUIRE_BELOW_ZERO)
            if not ok:
                return None, None
            result["季W前"], result["季W后"], result["季W后日期"] = b1, b2, d2.strftime("%Y-%m-%d")
        else:
            result["季W前"] = result["季W后"] = result["季W后日期"] = None

        if ENABLE_MONTHLY_W_BOTTOM:
            dif_m2, _ = calc_macd(monthly["收盘"])
            ok, b1, b2, d1, d2 = find_w_bottom(dif_m2, MONTHLY_W_MIN_DIST, MONTHLY_W_MAX_DIST, W_TOL, W_MIN_REBOUND, MONTHLY_W_MAX_AGE, W_REQUIRE_BELOW_ZERO)
            if not ok:
                return None, None
            result["月W前"], result["月W后"], result["月W后日期"] = b1, b2, d2.strftime("%Y-%m-%d")
        else:
            result["月W前"] = result["月W后"] = result["月W后日期"] = None

        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ---------------------- 推送内容 ----------------------
def build_push(df, spot_now=None):
    lines = [f"**📊 多周期MACD底部共振** | 命中{len(df)}只 (现价=实时价)",
             "*(周DIF<-1超卖+年/季/月多级W底+月线金叉+月DEA零轴; 长周期左侧底部共振; 非预测, 必止损)*", ""]
    def line(r):
        return (f"- **{r['名称']}({r['代码']})** [{r.get('行业', '—')}] 现价{r.get('最新价')} | 周DIF{r.get('周DIF')} 月DEA{r.get('月DEA')} | "
                f"年W{r.get('年W后日期', '—')} 季W{r.get('季W后日期', '—')} 月W{r.get('月W后日期', '—')}{_align_suffix(r, spot_now)}")
    lines += [line(r) for _, r in df.head(PUSH_TOP).iterrows()]
    if len(df) > PUSH_TOP:
        lines.append(f"\n*…另有 {len(df) - PUSH_TOP} 只, 详见 output*")
    return "\n".join(lines)

# ---------------------- 主流程 ----------------------
def main():
    global _INDUSTRY_MAP
    if not _login_with_retry():
        print("主进程登录失败，无法继续。")
        return
    try:
        codes, names = get_stock_list()
        try:
            idf = _rs_to_df(bs.query_stock_industry())
            for _, row in idf.iterrows():
                _INDUSTRY_MAP[row['code']] = _clean_industry(row.get('industry', ''))
            print(f"  行业映射表加载 {len(_INDUSTRY_MAP)} 条")
        except Exception as e:
            print(f"  行业表异常: {e}")
    finally:
        _safe_logout()

    results = []; failures = []
    print("\n开始多条件扫描（含季线+月线 W底，优化版检测逻辑）...")
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=_worker_init) as executor:
        future_to_code = {executor.submit(scan_one, code, names.get(code, code)): code for code in codes}
        for future in tqdm(as_completed(future_to_code), total=len(codes), desc="扫描中"):
            code = future_to_code[future]
            try:
                res, err = future.result()
            except Exception as e:
                res, err = None, f"未捕获异常: {type(e).__name__}: {e}"
            if res:
                results.append(res)
            elif err:
                failures.append((code, names.get(code, code), err))

    if failures:
        print(f"\n有 {len(failures)} 只抓取/处理失败：")
        for code, name, reason in failures[:20]:
            print(f"  - {code} {name}: {reason}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    out_csv = os.path.join(OUTPUT_DIR, f"multi_period_macd_{tag}.csv")

    if not results:
        print("\n没有股票同时满足所有启用条件。")
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        if SERVERCHAN_KEY:
            send_serverchan("📊 多周期MACD底部共振 | 0命中", "**多周期MACD底部共振** | 本次无同时满足全部条件的票(长周期左侧条件极严, 0命中属正常)。")
        return

    df = pd.DataFrame(results)
    df['行业'] = df['代码'].map(lambda c: _INDUSTRY_MAP.get(c, '—'))
    df['最新价'] = df['收盘价']
    df, rt = _refresh_realtime_price(df)
    sort_col = "周DIF" if "周DIF" in df.columns and df["周DIF"].notna().any() else "收盘价"
    df = df.sort_values(sort_col)

    print(f"\n共找到 {len(df)} 只股票：\n")
    print(df.head(PUSH_TOP).to_string(index=False))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_csv}")

    if SERVERCHAN_KEY:
        send_serverchan(f"📊 多周期MACD底部共振 命中{len(df)}只", build_push(df, rt))

if __name__ == "__main__":
    main()
# >>>FILE_END_multi_period_macd<<<
