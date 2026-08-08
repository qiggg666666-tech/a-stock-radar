#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  【年线W底 + MACD零轴附近首穿 + 周线趋势过滤】质量优先复合策略扫描器
  GitHub Actions 版（数据源: baostock，进程池 + 单独登录 + 超时保护）
================================================================================
  依赖:
    pip install baostock pandas numpy

  用法:
    python scanner_final_wbottom_bs.py
    python scanner_final_wbottom_bs.py --code sh.600000
    python scanner_final_wbottom_bs.py --require-weekly-up
    python scanner_final_wbottom_bs.py --score 75
    python scanner_final_wbottom_bs.py --resume
    python scanner_final_wbottom_bs.py --clear-ckpt

  可用环境变量覆盖(方便 workflow_dispatch 手动跑时调参):
    MIN_SCORE, WORKERS, REQUIRE_WEEKLY_UP=1, SENDKEY(Server酱推送,可选)
    OUTPUT_DIR(结果目录, 默认 results; workflow 里设 output 以便上传 artifact)
================================================================================
"""

import os, sys, json, time, random, logging, argparse, atexit
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout
)

import pandas as pd
import numpy as np

# ========================= 质量优先配置 =========================
CFG = {
    "min_score": 72,
    "workers": 3,                 # 与仓库既有 NUM_PROCESSES=3 模式保持一致
    "batch_size": 40,

    # 高质量W底参数
    "valley_window": 7,
    "min_dist": 25,
    "max_dist": 90,
    "bottom_tolerance": 0.028,
    "neck_min_rise": 0.05,
    "neck_proximity": 0.025,
    "volume_shrink_ratio": 0.70,

    # MACD
    "dif_zero_range": 0.28,

    # 位置
    "ma250_near_pct": 0.045,

    # 周线
    "require_weekly_up": False,

    # 数据
    "history_days": 420,
    "retry_times": 4,
    "query_timeout": 15,          # 单次 baostock 查询超时(秒)
    "login_stagger_max": 2.0,     # 每个 worker 登录前的随机等待,避免并发登录被限
    "min_price": 1.0,
}

# 环境变量覆盖(workflow_dispatch 手动触发调参用)
CFG["min_score"] = int(os.getenv("MIN_SCORE", CFG["min_score"]))
CFG["workers"] = int(os.getenv("WORKERS", CFG["workers"]))
if os.getenv("REQUIRE_WEEKLY_UP", "") == "1":
    CFG["require_weekly_up"] = True

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, os.environ.get("OUTPUT_DIR", "results"))
CKPT = os.path.join(BASE, ".checkpoint")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
log = logging.getLogger("scanner")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
lf = logging.FileHandler(os.path.join(RESULTS, f"scan_{datetime.now():%Y%m%d}.log"), encoding="utf-8")
lf.setFormatter(fmt)
log.addHandler(lf)


# ========================= 数据结构 =========================
@dataclass
class Sig:
    code: str = ""
    name: str = ""
    date: str = ""
    score: int = 0
    ma250_w: bool = False
    ma250_left: str = ""
    ma250_right: str = ""
    ma250_neck: float = 0.0
    ma250_cur: float = 0.0
    ma250_brk: float = 0.0
    ma250_wd: int = 0
    ma250_diff: float = 0.0
    ma250_tr: str = ""
    ma250_quality: int = 0
    price_w: bool = False
    price_left: str = ""
    price_right: str = ""
    price_neck: float = 0.0
    price_quality: int = 0
    macd_fr: bool = False
    dif: float = 0.0
    dea: float = 0.0
    macd_h: float = 0.0
    dif_nz: bool = False
    vol_shrink: bool = False
    vol_break: bool = False
    above: bool = False
    near: bool = False
    divergence: bool = False
    weekly_tr: str = ""
    filtered_out: str = ""   # 命中了哪个硬过滤条件(调试用,不影响 score>=min_score 判定)
    err: str = ""

    def to_dict(self):
        return asdict(self)


# ========================= 核心算法(与数据源无关,未改动核心逻辑) =========================
def valleys(s: pd.Series, w: int = 7):
    r = s.rolling(2 * w + 1, center=True).min()
    m = (s == r)
    m.iloc[:w] = False
    m.iloc[-w:] = False
    return m


def macd(c: pd.Series):
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d = e12 - e26
    a = d.ewm(span=9, adjust=False).mean()
    return d, a, (d - a) * 2


def to_weekly_optimized(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) < 5:
        return pd.DataFrame()
    df = df.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    weekly = df.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])
    if weekly.empty:
        return weekly
    real_last_days = df.groupby(pd.Grouper(freq='W-FRI')).apply(lambda x: x.index[-1])
    weekly.index = real_last_days.reindex(weekly.index).values
    weekly = weekly[~weekly.index.isna()]
    weekly = weekly[~weekly.index.duplicated(keep='last')]
    return weekly.sort_index()


def weekly_trend(df: pd.DataFrame, lookback: int = 8) -> str:
    if len(df) < 60:
        return "unknown"
    try:
        w = to_weekly_optimized(df)
        if len(w) < 20:
            return "unknown"
        close = w['close']
        ma20 = close.rolling(20, min_periods=12).mean()
        dif, _, hist = macd(close)
        recent_ma = ma20.iloc[-lookback:].dropna()
        recent_dif = dif.iloc[-lookback:].dropna()
        recent_hist = hist.iloc[-lookback:].dropna()
        if len(recent_ma) < 4:
            return "unknown"
        ma_slope = recent_ma.iloc[-1] / recent_ma.iloc[0] - 1
        ma_up = ma_slope > 0.008
        ma_down = ma_slope < -0.008
        dif_rising = recent_dif.iloc[-1] > recent_dif.iloc[0]
        hist_pos = (recent_hist.iloc[-1] > 0) and (recent_hist.iloc[-2] >= 0)
        above_ma = close.iloc[-1] > ma20.iloc[-1] * 0.995
        if ma_up and (dif_rising or hist_pos) and above_ma:
            return "up"
        if ma_down and not above_ma:
            return "down"
        if abs(ma_slope) <= 0.008:
            return "flat"
        return "up" if ma_slope > 0 else "down"
    except Exception as e:
        log.debug(f"weekly_trend 异常: {e}")
        return "unknown"


def detect_w_bottom(df: pd.DataFrame,
                     price_col: str = "low",
                     vol_col: str = "volume",
                     window: int = 7,
                     min_dist: int = 25,
                     max_dist: int = 90,
                     tolerance: float = 0.028,
                     min_rise: float = 0.05,
                     vol_shrink: float = 0.70,
                     near_neck: float = 0.025) -> dict | None:
    """高质量 W 底自动检测"""
    if len(df) < max_dist + 30:
        return None

    s = df[price_col]
    v = df[vol_col] if vol_col in df.columns else None

    rolling_min = s.rolling(2 * window + 1, center=True).min()
    is_valley = (s == rolling_min)
    is_valley.iloc[:window] = False
    is_valley.iloc[-window:] = False
    valleys_idx = s.index[is_valley].tolist()

    best = None
    best_score = -1

    for i in range(len(valleys_idx)):
        for j in range(i + 1, len(valleys_idx)):
            left = valleys_idx[i]
            right = valleys_idx[j]
            dist = df.index.get_loc(right) - df.index.get_loc(left)

            if dist < min_dist or dist > max_dist:
                continue

            lp = float(s.loc[left])
            rp = float(s.loc[right])
            base = min(lp, rp)
            if base <= 0:
                continue
            height_diff = abs(lp - rp) / base

            if height_diff > tolerance:
                continue

            mid = s.loc[left:right]
            neck = float(mid.max())
            if neck < base * (1 + min_rise):
                continue

            vol_ok = False
            if v is not None:
                try:
                    lv = float(v.loc[left])
                    if lv > 0:
                        vol_ok = float(v.loc[right]) < lv * vol_shrink
                except Exception:
                    pass

            cur = float(s.iloc[-1])
            near = abs(cur - neck) / neck <= near_neck if neck else False

            score = 0
            if height_diff < 0.015:
                score += 30
            elif height_diff < 0.025:
                score += 20
            else:
                score += 10

            if 30 <= dist <= 70:
                score += 25
            elif 25 <= dist <= 90:
                score += 15

            if vol_ok:
                score += 20
            if near:
                score += 15
            if rp >= lp * 0.985:
                score += 10

            if score > best_score:
                best_score = score
                best = {
                    "left_date": str(left)[:10],
                    "right_date": str(right)[:10],
                    "left_price": round(lp, 2),
                    "right_price": round(rp, 2),
                    "neckline": round(neck, 2),
                    "width": dist,
                    "height_diff": round(height_diff * 100, 2),
                    "volume_shrink": vol_ok,
                    "near_neck": near,
                    "quality_score": score
                }

    return best if best_score >= 50 else None


def simple_divergence(price: pd.Series, dif: pd.Series, lookback: int = 40):
    if len(price) < lookback + 10:
        return False
    p = price.iloc[-lookback:]
    d = dif.iloc[-lookback:]
    p_vals = p[valleys(p, 5)]
    d_vals = d[valleys(d, 5)]
    if len(p_vals) < 2 or len(d_vals) < 2:
        return False
    if p_vals.iloc[-1] <= p_vals.iloc[-2] * 1.01 and d_vals.iloc[-1] > d_vals.iloc[-2]:
        return True
    return False


def analyze(df: pd.DataFrame, code: str, name: str, hard_filter: bool = True):
    """
    hard_filter=False 时跳过周线硬过滤,但仍记录 filtered_out 字段,
    用于 --code 单股调试时能看到完整诊断信息而不是被直接吃掉。
    """
    s = Sig(code=code, name=name)
    try:
        if len(df) < 260:
            s.err = "数据不足"
            return None
        df = df.sort_index()
        C, L, V = df["close"], df["low"], df["volume"]
        cur = float(C.iloc[-1])
        if cur < CFG["min_price"]:
            s.err = "价格过低"
            return None

        ma250 = C.rolling(250).mean()
        dif, dea, hist = macd(C)
        ma250c = float(ma250.iloc[-1])
        if pd.isna(ma250c) or ma250c <= 0:
            s.err = "年线数据不足"
            return None
        s.ma250_cur = round(ma250c, 2)

        m15 = ma250.iloc[-15:].dropna()
        if len(m15) >= 8:
            if m15.iloc[-1] > m15.iloc[0] * 1.015:
                s.ma250_tr = "up"
            elif m15.iloc[-1] < m15.iloc[0] * 0.985:
                s.ma250_tr = "down"
            else:
                s.ma250_tr = "flat"

        w = detect_w_bottom(
            df.assign(ma250=ma250),
            price_col="ma250",
            vol_col="volume",
            window=CFG["valley_window"],
            min_dist=CFG["min_dist"],
            max_dist=CFG["max_dist"],
            tolerance=CFG["bottom_tolerance"],
            min_rise=CFG["neck_min_rise"],
            vol_shrink=CFG["volume_shrink_ratio"],
            near_neck=CFG["neck_proximity"]
        )
        if w:
            s.ma250_w = True
            s.ma250_left = w["left_date"]
            s.ma250_right = w["right_date"]
            s.ma250_neck = w["neckline"]
            s.ma250_wd = w["width"]
            s.ma250_diff = w["height_diff"]
            s.ma250_quality = w["quality_score"]
            s.ma250_brk = round((ma250c - w["neckline"]) / w["neckline"] * 100, 2)
            s.vol_shrink = w["volume_shrink"]

        pw = detect_w_bottom(
            df,
            price_col="low",
            vol_col="volume",
            window=CFG["valley_window"],
            min_dist=CFG["min_dist"] - 5,
            max_dist=CFG["max_dist"],
            tolerance=CFG["bottom_tolerance"] * 1.3,
            min_rise=CFG["neck_min_rise"] * 0.9,
            vol_shrink=CFG["volume_shrink_ratio"],
            near_neck=CFG["neck_proximity"] * 1.2
        )
        if pw:
            s.price_w = True
            s.price_left = pw["left_date"]
            s.price_right = pw["right_date"]
            s.price_neck = pw["neckline"]
            s.price_quality = pw["quality_score"]
            if not s.vol_shrink:
                s.vol_shrink = pw["volume_shrink"]

        s.macd_fr = (hist.iloc[-2] <= 0) and (hist.iloc[-1] > 0)
        s.dif = round(float(dif.iloc[-1]), 3)
        s.dea = round(float(dea.iloc[-1]), 3)
        s.macd_h = round(float(hist.iloc[-1]), 3)
        s.dif_nz = abs(s.dif) < CFG["dif_zero_range"]

        s.divergence = simple_divergence(C, dif)

        s.above = cur > ma250c
        s.near = (cur / ma250c < 1 + CFG["ma250_near_pct"]) and (cur / ma250c > 1 - CFG["ma250_near_pct"] * 0.6)

        vm5 = V.iloc[-5:].mean()
        rv = float(V.iloc[-1])
        s.vol_break = (vm5 > 0) and (rv > vm5 * 1.25) and (rv < vm5 * 4.5) and (rv > float(V.iloc[-2]) * 1.15)

        s.weekly_tr = weekly_trend(df)

        if CFG.get("require_weekly_up", False):
            if s.weekly_tr != "up":
                s.filtered_out = "require_weekly_up"
                if hard_filter:
                    return None
        else:
            if s.weekly_tr == "down":
                s.filtered_out = "weekly_down"
                if hard_filter:
                    return None

        sc = 0
        if s.ma250_w:
            sc += 35 + min(s.ma250_quality // 5, 15)
        if s.price_w:
            sc += 12 + min(s.price_quality // 8, 8)
        if s.macd_fr:
            sc += 20
            if s.dif_nz:
                sc += 12
            if s.dif > -0.1:
                sc += 4
            if s.macd_h > 0.06:
                sc += 3
        if s.divergence:
            sc += 8
        if s.above:
            sc += 4
        if s.near:
            sc += 14
        if s.vol_shrink:
            sc += 10
        if s.vol_break:
            sc += 4
        if s.ma250_tr == "up":
            sc += 8
        elif s.ma250_tr == "flat":
            sc += 3
        if s.weekly_tr == "up":
            sc += 15
        elif s.weekly_tr == "flat":
            sc += 5

        s.score = sc
        s.date = str(df.index[-1])[:10]
        return s
    except Exception as e:
        s.err = str(e)[:120]
        log.debug(f"{code} analyze 异常: {e}")
        return None


# ========================= baostock 数据获取(带超时保护) =========================
def _with_timeout(fn, *args, timeout=15, **kwargs):
    """用单线程池给可能卡死的 baostock 调用加超时,超时返回 None。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except FutTimeout:
            return None
        except Exception as e:
            log.debug(f"baostock 调用异常: {e}")
            return None


def _worker_login():
    """ProcessPoolExecutor 的 initializer:每个子进程各自登录一次 baostock,
    登录前随机等待,避免并发登录被限流(与仓库既有 per-worker 登录模式一致)。"""
    import baostock as bs
    time.sleep(random.uniform(0, CFG["login_stagger_max"]))
    for attempt in range(3):
        lg = bs.login()
        if lg.error_code == '0':
            atexit.register(lambda: bs.logout())
            return
        time.sleep(1.0 + attempt)
    log.warning("baostock 子进程登录失败,该 worker 之后的查询会持续失败")


def to_bs_code(code: str) -> str:
    """将裸代码(600000/000001/300001/688001)转换成 baostock 的 sh./sz. 前缀格式。
    若已经带前缀则原样返回。"""
    if "." in code:
        return code
    if code.startswith("6"):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    return f"bj.{code}"  # 北交所,后续会被 universe 过滤掉


def get_data_bs(bs_code: str, retry_times: int = None) -> pd.DataFrame | None:
    import baostock as bs
    retry_times = retry_times or CFG["retry_times"]
    st = (datetime.now() - timedelta(days=CFG["history_days"] + 80)).strftime("%Y-%m-%d")
    ed = datetime.now().strftime("%Y-%m-%d")

    for attempt in range(retry_times):
        try:
            rs = _with_timeout(
                bs.query_history_k_data_plus,
                bs_code, "date,open,high,low,close,volume",
                start_date=st, end_date=ed, frequency="d", adjustflag="2",
                timeout=CFG["query_timeout"]
            )
            if rs is None or rs.error_code != '0':
                time.sleep(0.8 + attempt * 0.6)
                continue

            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=rs.fields)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close", "volume"])
            if len(df) < 250:
                return None
            df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.debug(f"{bs_code} 获取异常(第{attempt + 1}次): {e}")
            time.sleep(0.8 + attempt * 0.6)
    return None


def get_stock_list_bs() -> pd.DataFrame:
    """通过 baostock 拉全市场股票列表,过滤指数/ST/退市/北交所。
    从今天往前找最近一个有数据的交易日(节假日/未开盘时 query_all_stock 会返回空)。"""
    import baostock as bs
    day = datetime.now()
    raw = pd.DataFrame()
    for back in range(10):
        d = (day - timedelta(days=back)).strftime("%Y-%m-%d")
        rs = _with_timeout(bs.query_all_stock, day=d, timeout=CFG["query_timeout"])
        if rs is None or rs.error_code != '0':
            continue
        rows = []
        while (rs.error_code == '0') and rs.next():
            rows.append(rs.get_row_data())
        if rows:
            raw = pd.DataFrame(rows, columns=rs.fields)
            break

    if raw.empty:
        log.error("获取股票列表失败(近10天都没有交易日数据)")
        return pd.DataFrame(columns=["code", "name"])

    raw = raw[raw["tradeStatus"] == "1"]

    def valid(code: str) -> bool:
        # 仅保留沪深主板/中小板/创业板/科创板,排除指数与北交所
        return (code.startswith("sh.60") or code.startswith("sh.688")
                or code.startswith("sz.00") or code.startswith("sz.30"))

    raw = raw[raw["code"].apply(valid)]
    raw = raw[~raw["code_name"].str.contains("ST|退", na=False)]
    raw = raw.rename(columns={"code_name": "name"})
    return raw[["code", "name"]].drop_duplicates().reset_index(drop=True)


# ========================= 断点续传 =========================
def ckpt_p():
    return os.path.join(CKPT, f"ckpt_{datetime.now():%Y%m%d}.json")


def load_ckpt():
    p = ckpt_p()
    if os.path.exists(p):
        try:
            with open(p) as f:
                return set(json.load(f).get("done", []))
        except Exception as e:
            log.debug(f"读取断点失败: {e}")
    return set()


def save_ckpt(done):
    try:
        with open(ckpt_p(), "w") as f:
            json.dump({"done": list(done), "t": datetime.now().isoformat()}, f)
    except Exception as e:
        log.debug(f"保存断点失败: {e}")


# ========================= 微信推送(Server酱,可选) =========================
def push_wechat(title: str, content: str):
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        return
    try:
        import urllib.request
        import urllib.parse
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
        data = urllib.parse.urlencode({"title": title, "desp": content}).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as e:
        log.debug(f"微信推送失败: {e}")


# ========================= 单个股票的 worker 任务 =========================
def _scan_one(bs_code: str, name: str):
    """在子进程中运行:必须是模块级函数才能被 ProcessPoolExecutor pickle。"""
    df = get_data_bs(bs_code)
    if df is None:
        return bs_code, name, None
    sig = analyze(df, bs_code, name)
    if sig is None:
        return bs_code, name, None
    return bs_code, name, sig.to_dict()


# ========================= 扫描 =========================
def scan_all():
    log.info("=" * 70)
    log.info("  年线W底 + MACD零轴附近首穿 + 周线趋势  (baostock / GitHub Actions 版)")
    log.info(f"  min_score={CFG['min_score']} | workers={CFG['workers']} | "
             f"require_weekly_up={CFG['require_weekly_up']}")
    log.info("=" * 70)

    import baostock as bs
    lg = bs.login()
    try:
        if lg.error_code != '0':
            log.error(f"baostock 登录失败: {lg.error_msg}")
            return pd.DataFrame()

        log.info("获取股票列表...")
        stocks = get_stock_list_bs()
        if stocks.empty:
            log.error("获取股票列表失败")
            return pd.DataFrame()
        total = len(stocks)
        log.info(f"共 {total} 只股票")
    finally:
        bs.logout()

    done = load_ckpt()
    if done:
        log.info(f"断点续传: 已扫描 {len(done)} 只")

    todo = stocks[~stocks["code"].isin(done)]
    results = []
    scanned = len(done)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=CFG["workers"], initializer=_worker_login) as ex:
        futures = {
            ex.submit(_scan_one, row["code"], row["name"]): row["code"]
            for _, row in todo.iterrows()
        }
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                c, n, sig_dict = fut.result()
            except Exception as e:
                log.debug(f"{code} worker 异常: {e}")
                c, sig_dict = code, None

            scanned += 1
            done.add(c)
            if scanned % CFG["batch_size"] == 0:
                save_ckpt(done)
                log.info(f"进度: {scanned}/{total} ({scanned / total * 100:.1f}%)")

            if sig_dict and sig_dict.get("score", 0) >= CFG["min_score"]:
                log.info(f"[信号] {sig_dict['code']} {sig_dict['name']} 评分:{sig_dict['score']} "
                         f"年线W底:{sig_dict['ma250_w']} 质量:{sig_dict['ma250_quality']} "
                         f"MACD:{sig_dict['macd_fr']} 周线:{sig_dict['weekly_tr']}")
                results.append(sig_dict)

    elapsed = time.time() - t0
    log.info(f"完成: {elapsed:.1f}秒, 平均 {elapsed / max(total, 1):.2f}秒/股, 信号 {len(results)} 只")
    save_ckpt(done)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(
        ["score", "ma250_quality", "weekly_tr"], ascending=[False, False, True]
    ).reset_index(drop=True)


# ========================= 保存 =========================
def save(df: pd.DataFrame):
    d = datetime.now().strftime("%Y%m%d")
    csv = os.path.join(RESULTS, f"results_{d}.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info(f"CSV: {csv}")

    js = os.path.join(RESULTS, f"results_{d}.json")
    df.to_json(js, orient="records", force_ascii=False, indent=2)

    md = os.path.join(RESULTS, f"report_{d}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# 高质量W底复合策略扫描报告 {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"**符合信号**: {len(df)} 只 | **最低评分**: {CFG['min_score']}\n\n")
        f.write("| 排名 | 代码 | 名称 | 评分 | 年线W底 | W质量 | MACD首穿 | 零轴 | 周线 | DIF | 年线趋势 | 位置 |\n")
        f.write("|------|------|------|------|---------|-------|----------|------|------|-----|----------|------|\n")
        for idx, r in df.head(40).iterrows():
            pos = "站上" if r.get("above") else ""
            if r.get("near"):
                pos += "贴近"
            f.write(f"| {idx + 1} | {r['code']} | {r['name']} | **{r['score']}** | "
                    f"{'✓' if r.get('ma250_w') else ''} | {r.get('ma250_quality', 0)} | "
                    f"{'✓' if r.get('macd_fr') else ''} | {'✓' if r.get('dif_nz') else ''} | "
                    f"{r.get('weekly_tr', '')} | {r.get('dif', 0)} | {r.get('ma250_tr', '')} | {pos} |\n")

    if not df.empty:
        print("\n" + "=" * 120)
        print(f"  🎯 高质量W底 TOP 结果 (共 {len(df)} 只符合)")
        print("=" * 120)
        cols = ["code", "name", "score", "ma250_w", "ma250_quality", "macd_fr", "dif_nz",
                "weekly_tr", "dif", "near", "ma250_tr"]
        top = df[[c for c in cols if c in df.columns]].head(20).copy()
        for c in ["ma250_w", "macd_fr", "dif_nz", "near"]:
            if c in top.columns:
                top[c] = top[c].map({True: "✓", False: ""})
        print(top.to_string(index=False))
        print("=" * 120)
        print("\n📁 结果已保存到结果目录")

        lines = [f"{i+1}. {r['code']} {r['name']} 评分{r['score']}" for i, r in df.head(15).iterrows()]
        push_wechat(
            f"W底扫描完成,信号 {len(df)} 只",
            "\n\n".join(lines)
        )
    else:
        push_wechat("W底扫描完成", "本次未发现符合条件的信号")


# ========================= 入口 =========================
def main():
    p = argparse.ArgumentParser(description="高质量W底 + MACD + 周线 扫描器 (baostock版)")
    p.add_argument("--code", help="单股代码,如 sh.600000 或 600000")
    p.add_argument("--name", default="", help="股票名称")
    p.add_argument("--score", type=int, help=f"最低评分 (默认{CFG['min_score']})")
    p.add_argument("--workers", type=int, help=f"进程数 (默认{CFG['workers']})")
    p.add_argument("--resume", action="store_true", help="断点续传(默认行为,保留兼容)")
    p.add_argument("--clear-ckpt", action="store_true", help="清除断点")
    p.add_argument("--require-weekly-up", action="store_true", help="强制周线向上")
    args = p.parse_args()

    if args.score is not None:
        CFG["min_score"] = args.score
    if args.workers is not None:
        CFG["workers"] = args.workers
    if args.require_weekly_up:
        CFG["require_weekly_up"] = True

    if args.clear_ckpt:
        cp = ckpt_p()
        if os.path.exists(cp):
            os.remove(cp)
        log.info("断点已清除")
        return

    if args.code:
        import baostock as bs
        bs_code = to_bs_code(args.code)
        log.info(f"===== 单股检测: {bs_code} =====")
        lg = bs.login()
        try:
            if lg.error_code != '0':
                print(f"baostock 登录失败: {lg.error_msg}")
                return
            df = get_data_bs(bs_code)
        finally:
            bs.logout()

        if df is not None:
            # 单股调试模式下跳过硬过滤,便于看到完整诊断信息
            sig = analyze(df, bs_code, args.name, hard_filter=False)
            if sig:
                print("\n" + "=" * 70)
                for k, v in sig.to_dict().items():
                    print(f"  {k:20s}: {v}")
                print("=" * 70)
                passed = sig.score >= CFG["min_score"] and not sig.filtered_out
                print(f"\n  评分: {sig.score}/100  {'✅ 符合' if passed else '❌ 未达标/被过滤'}")
                if sig.filtered_out:
                    print(f"  (注: 命中硬过滤条件 {sig.filtered_out},正式扫描中会被排除)")
            else:
                print(f"\n{bs_code} 未能计算信号(数据不足或异常)")
        else:
            print(f"\n{bs_code} 数据获取失败")
        return

    df = scan_all()
    save(df)
    log.info("===== 完成 =====")


if __name__ == "__main__":
    main()
# >>>FILE_END_scanner_final_wbottom<<<
