# -*- coding: utf-8 -*-
"""
market_anomaly_monitor.py —— 市场异常行为检测(微观结构风控监控) · 矩阵规格
【定位】这不是选股脚本, 是"市场异常/操纵行为监控": 检测成交额Top票的成交量/盘口微观结构异常
  (本福特=成交量数字造假 / 深度错配=盘口操纵 / 碎单=拆单 / 大单集中 / OFI订单流失衡 /
   价格远离POC成交密集区 / tick失衡)。输出"交易行为可疑"提示, 非买入信号; 异常≠操纵定论, 需人工核实。
【本版完善·只升级工程规格, 不动检测算法/阈值默认值】
  ① 推送: markdown表格(微信错位)→列表格式 + 超3800自动分页 + serverchan_sdk/requests双通道;
  ② 8个风险阈值(原硬编码)全env可调, 默认值不变; SCAN_COUNT/MAX_WORKERS/TICK_LIMIT env可调;
  ③ benfordslaw 顶层裸import→try/except降级(缺库本福特维度跳过、其它维度照跑、不崩);
  ④ 加 output/market_anomaly_*.csv/json 存盘; ⑤ results取值容错(.get防KeyError)。
【依赖】需 pip install benfordslaw (并加进 requirements.txt); 缺失则本福特维度降级跳过。
【数据单位待验证】DEPTH_UNIT_SCALE: tick成交量常为"股", 盘口买卖量常为"手"(=100股),
  若确认如此请设 DEPTH_UNIT_SCALE=100 让"深度错配比"同单位可比; 设前先打印两者原始值核实。
【性能】tick接口(腾讯)重且限流: SCAN_COUNT默认50(别设太大), MAX_WORKERS默认5(别>8), 否则大量拉取失败。
⚠️ 非交易日检测的是上一交易日数据; 异常提示仅供风控参考, 非操纵定论、非买卖依据。
"""
import os
import math
import time
import traceback
import requests
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# benfordslaw 降级保护: 缺库不崩, 本福特维度跳过
try:
    from benfordslaw import benfordslaw
    BENFORD_AVAILABLE = True
except ImportError:
    benfordslaw = None
    BENFORD_AVAILABLE = False
    print("⚠️ benfordslaw 未安装, 本福特检测降级跳过 (pip install benfordslaw 启用; 不影响其它维度)")

# ------------------ 配置 (env 可调) ------------------
SCAN_COUNT = int(os.environ.get("SCAN_COUNT", "50"))    # 检测成交额Top N只(原20; tick拉取重, 别设太大防限流)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "5"))   # 并发拉tick线程(腾讯接口限流, 别>8)
TICK_LIMIT = int(os.environ.get("TICK_LIMIT", "500"))   # 每只取最近N笔tick
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
SERVERCHAN_KEY = os.environ.get("SERVERCHAN_KEY") or os.environ.get("SENDKEY", "")
BENFORD_MIN_SAMPLES = int(os.environ.get("BENFORD_MIN_SAMPLES", "50"))
BENFORD_ALPHA = float(os.environ.get("BENFORD_ALPHA", "0.05"))
# 成交量(股) vs 盘口深度(手=100股) 单位换算; 确认单位后设100让深度错配比同单位可比
DEPTH_UNIT_SCALE = float(os.environ.get("DEPTH_UNIT_SCALE", "1"))
# 【完善】8个风险阈值全env可调(默认值=原硬编码, 只给调节能力)
BENFORD_MAD_MAX = float(os.environ.get("BENFORD_MAD_MAX", "0.05"))
DEPTH_MISMATCH_MAX = float(os.environ.get("DEPTH_MISMATCH_MAX", "5"))
ODD_RATIO_MAX = float(os.environ.get("ODD_RATIO_MAX", "0.3"))
TOP5_VOL_MAX = float(os.environ.get("TOP5_VOL_MAX", "0.5"))
OFI_Z_MAX = float(os.environ.get("OFI_Z_MAX", "2"))
DIST_POC_MAX = float(os.environ.get("DIST_POC_MAX", "0.03"))
IMBALANCE_MAX = float(os.environ.get("IMBALANCE_MAX", "0.6"))
PUSH_TOP = int(os.environ.get("PUSH_TOP", "30"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Benford ------------------
def check_benford_v2(data, code=None):
    if not BENFORD_AVAILABLE:
        return None
    if len(data) < BENFORD_MIN_SAMPLES:
        return None

    clean_data = [abs(int(x)) for x in data if abs(int(x)) > 0]
    if len(clean_data) < BENFORD_MIN_SAMPLES:
        return None

    try:
        bl = benfordslaw(alpha=BENFORD_ALPHA, verbose=False)
        bl.fit(clean_data)
        results = bl.results

        # MAD类偏离统计的key名随benfordslaw版本/配置变化, 多key兼容(原仅试rad会静默KeyError)
        mad_val = None
        for key in ("MAD", "mad", "rad", "RAD"):
            if key in results:
                mad_val = results[key]
                break
        if mad_val is None:
            print(f"[benford] no MAD-like key found in results for {code}; keys={list(results.keys())}")
            return None

        return {
            "mad": round(mad_val, 4),
            "pvalue": round(results.get("pvalue", 0.0), 4),     # 【完善】.get容错防KeyError
            "rmse": round(results.get("rmse", 0.0), 4),
            "is_normal": bool(results.get("pvalue", 0.0) >= BENFORD_ALPHA),
        }
    except Exception as e:
        print(f"[benford] failed for {code}: {e}")
        return None

# ------------------ 圆整度 ------------------
def check_roundness(data):
    if len(data) < 50:
        return None

    total = len(data)
    round_100 = sum(1 for x in data if x % 100 == 0)
    round_500 = sum(1 for x in data if x % 500 == 0)
    round_1000 = sum(1 for x in data if x % 1000 == 0)
    odd_ratio = (total - round_100) / total

    return {
        "round_100_ratio": round(round_100 / total, 4),
        "round_500_ratio": round(round_500 / total, 4),
        "round_1000_ratio": round(round_1000 / total, 4),
        "odd_ratio": round(odd_ratio, 4),
    }

# ------------------ 聚类 ------------------
def check_clustering(prices, volumes, price_bins=50):
    if len(prices) < 50:
        return None

    # 先分箱再算熵: 原始tick价格近乎唯一, 不分箱熵会塌缩到log(n)失去区分度;
    # 分箱还原"交易集中在少数价位"的信号。
    prices_arr = np.asarray(prices, dtype=float)
    pmin, pmax = prices_arr.min(), prices_arr.max()
    if pmin == pmax:
        price_entropy = 0.0
    else:
        edges = np.linspace(pmin, pmax, price_bins + 1)
        idx = np.clip(np.searchsorted(edges, prices_arr, side="right") - 1, 0, price_bins - 1)
        counts = pd.Series(idx).value_counts(normalize=True)
        price_entropy = -sum(p * math.log(p) for p in counts if p > 0)

    sorted_vols = sorted(volumes, reverse=True)
    top_5_pct_idx = max(1, int(len(volumes) * 0.05))
    top_5_vol_ratio = sum(sorted_vols[:top_5_pct_idx]) / sum(volumes) if sum(volumes) > 0 else 0

    return {
        "price_entropy": round(price_entropy, 4),
        "top_5_vol_ratio": round(top_5_vol_ratio, 4),
    }

# ------------------ OFI ------------------
def calc_ofi_from_spot(row, df_tick=None):
    bid_depth = sum(float(row.get(f"买{i}量", 0)) for i in range(1, 6))
    ask_depth = sum(float(row.get(f"卖{i}量", 0)) for i in range(1, 6))
    depth_sum = bid_depth + ask_depth
    static_ofi = (bid_depth - ask_depth) / depth_sum if depth_sum > 0 else 0.0

    tick_ofi = None
    tick_ofi_z = None
    signed_vol_ratio = None

    if df_tick is not None and not df_tick.empty and "成交价格" in df_tick.columns and "成交量" in df_tick.columns:
        prices = df_tick["成交价格"].astype(float).values
        vols = df_tick["成交量"].astype(float).values

        if len(prices) >= 2:
            dpx = np.diff(prices)
            sign = np.sign(dpx)
            sign = np.where(sign == 0, 0, sign)
            signed_vol = sign * vols[1:]

            tick_ofi = float(np.sum(signed_vol))
            denom = float(np.sum(vols[1:])) if np.sum(vols[1:]) > 0 else 1.0
            signed_vol_ratio = float(tick_ofi / denom)

            if len(signed_vol) >= 20:
                mu = float(np.mean(signed_vol))
                sd = float(np.std(signed_vol)) if float(np.std(signed_vol)) > 1e-12 else 1.0
                tick_ofi_z = float((signed_vol[-1] - mu) / sd)
            else:
                tick_ofi_z = 0.0

    return {
        "static_ofi": round(static_ofi, 4),
        "tick_ofi": round(tick_ofi, 2) if tick_ofi is not None else None,
        "tick_ofi_z": round(tick_ofi_z, 4) if tick_ofi_z is not None else None,
        "signed_vol_ratio": round(signed_vol_ratio, 4) if signed_vol_ratio is not None else None,
    }

# ------------------ Volume Profile ------------------
def calc_volume_profile(df_tick, bins=20, value_area_pct=0.7):
    if df_tick is None or df_tick.empty:
        return None
    if "成交价格" not in df_tick.columns or "成交量" not in df_tick.columns:
        return None

    prices = df_tick["成交价格"].astype(float).values
    vols = df_tick["成交量"].astype(float).values

    if len(prices) < 20 or np.sum(vols) <= 0:
        return None

    pmin = float(np.min(prices))
    pmax = float(np.max(prices))
    if pmin == pmax:
        return {
            "poc_price": round(pmin, 4),
            "vah": round(pmax, 4),
            "val": round(pmin, 4),
            "profile_entropy": 0.0,
            "distance_to_poc": 0.0,
        }

    edges = np.linspace(pmin, pmax, bins + 1)
    hist = np.zeros(bins, dtype=float)

    idx = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, bins - 1)
    for i, v in zip(idx, vols):
        hist[i] += v

    total_vol = float(hist.sum())
    if total_vol <= 0:
        return None

    poc_idx = int(np.argmax(hist))
    poc_price = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)

    left = right = poc_idx
    acc = float(hist[poc_idx])
    target = total_vol * value_area_pct

    while acc < target and (left > 0 or right < bins - 1):
        left_vol = hist[left - 1] if left > 0 else -1
        right_vol = hist[right + 1] if right < bins - 1 else -1
        if right_vol >= left_vol:
            right += 1
            acc += float(hist[right])
        else:
            left -= 1
            acc += float(hist[left])

    val = float((edges[left] + edges[left + 1]) / 2)
    vah = float((edges[right] + edges[right + 1]) / 2)

    p = hist / total_vol
    profile_entropy = float(-np.sum([x * np.log(x) for x in p if x > 0]))

    latest_price = float(prices[-1])
    distance_to_poc = float(abs(latest_price - poc_price) / poc_price) if poc_price != 0 else 0.0

    return {
        "poc_price": round(poc_price, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "profile_entropy": round(profile_entropy, 4),
        "distance_to_poc": round(distance_to_poc, 4),
    }

# ------------------ Tick Imbalance Bars ------------------
def calc_tick_imbalance_bars(df_tick, ewma_span=50, min_bar_ticks=20):
    if df_tick is None or df_tick.empty:
        return None
    if "成交价格" not in df_tick.columns or "成交量" not in df_tick.columns:
        return None

    prices = df_tick["成交价格"].astype(float).values
    vols = df_tick["成交量"].astype(float).values
    if len(prices) < min_bar_ticks:
        return None

    sign = np.sign(np.diff(prices))
    sign = np.where(sign == 0, 0, sign)
    signed_vol = sign * vols[1:]

    if len(signed_vol) == 0:
        return None

    ewma_abs = pd.Series(np.abs(signed_vol)).ewm(span=ewma_span, adjust=False).mean().values
    threshold = float(np.nanmean(ewma_abs[-min(20, len(ewma_abs)):])) * min_bar_ticks
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.mean(np.abs(signed_vol))) * min_bar_ticks

    bars = 0
    cum_imb = 0.0
    current_ticks = 0
    bar_lengths = []

    for x in signed_vol:
        cum_imb += float(x)
        current_ticks += 1
        if abs(cum_imb) >= threshold and current_ticks >= min_bar_ticks:
            bars += 1
            bar_lengths.append(current_ticks)
            cum_imb = 0.0
            current_ticks = 0

    imbalance_ratio = float(np.sum(np.abs(signed_vol)) / np.sum(vols[1:])) if np.sum(vols[1:]) > 0 else 0.0
    avg_bar_len = float(np.mean(bar_lengths)) if bar_lengths else float(len(signed_vol))

    return {
        "bars_count": int(bars),
        "imbalance_ratio": round(imbalance_ratio, 4),
        "avg_bar_len": round(avg_bar_len, 2),
        "threshold": round(threshold, 2),
    }

# ------------------ 单股分析 ------------------
def analyze_stock(row):
    code = row["代码"]
    name = row["名称"]
    symbol = ("sh" if code.startswith(("6", "9")) else "sz") + code

    try:
        bid_depth = sum(float(row.get(f"买{i}量", 0)) for i in range(1, 6))
        ask_depth = sum(float(row.get(f"卖{i}量", 0)) for i in range(1, 6))
        total_depth = bid_depth + ask_depth

        df_tick = ak.stock_zh_a_tick_tx_js(symbol=symbol)
        if df_tick is None or df_tick.empty:
            return None

        df_tick = df_tick.tail(TICK_LIMIT)
        vols = df_tick["成交量"].astype(float).tolist()
        prices = df_tick["成交价格"].astype(float).tolist()

        benford_result = check_benford_v2(vols, code=code)
        roundness = check_roundness(vols)
        clustering = check_clustering(prices, vols)
        ofi = calc_ofi_from_spot(row, df_tick)
        vp = calc_volume_profile(df_tick, bins=24, value_area_pct=0.7)
        tib = calc_tick_imbalance_bars(df_tick, ewma_span=50, min_bar_ticks=20)

        avg_tick_vol = sum(vols[-50:]) / 50 if len(vols) >= 50 else sum(vols) / len(vols)
        # 深度按盘口单位(常为手); 与tick成交量(股)比较前按 DEPTH_UNIT_SCALE 换算同单位。
        avg_depth = (total_depth / 10) * DEPTH_UNIT_SCALE if total_depth > 0 else 0
        depth_mismatch = avg_tick_vol / avg_depth if avg_depth > 0 else 0

        result = {
            "代码": code,
            "名称": name,
            "最新价": row["最新价"],
            "本福特-MAD": benford_result["mad"] if benford_result else None,
            "本福特-P值": benford_result["pvalue"] if benford_result else None,
            "本福特正常": benford_result["is_normal"] if benford_result else None,
            "非圆整比例": roundness["odd_ratio"] if roundness else 0,
            "圆整100比例": roundness["round_100_ratio"] if roundness else 0,
            "大单集中度": clustering["top_5_vol_ratio"] if clustering else 0,
            "价格熵": clustering["price_entropy"] if clustering else None,
            "深度错配比": round(depth_mismatch, 2),
            "静态OFI": ofi["static_ofi"],
            "逐笔OFI": ofi["tick_ofi"],
            "逐笔OFI-Z": ofi["tick_ofi_z"],
            "签名量比例": ofi["signed_vol_ratio"],
            "POC价": vp["poc_price"] if vp else None,
            "VAH": vp["vah"] if vp else None,
            "VAL": vp["val"] if vp else None,
            "剖面熵": vp["profile_entropy"] if vp else None,
            "距POC": vp["distance_to_poc"] if vp else None,
            "失衡Bar数": tib["bars_count"] if tib else None,
            "失衡比例": tib["imbalance_ratio"] if tib else None,
            "平均Bar长度": tib["avg_bar_len"] if tib else None,
            "风险提示": "",
        }

        return result

    except Exception:
        print(f"[analyze_stock] failed for {code} {name}:")
        traceback.print_exc()
        return None

# ------------------ 推送 (列表格式 + 分页 + 双通道) ------------------
def _send_one(title, content):
    try:
        from serverchan_sdk import sc_send
        ret = sc_send(SERVERCHAN_KEY, title, content)
        ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
        if ok:
            return True
        print(f"  sdk返回非成功({ret}), 回退requests")
    except Exception as e:
        print(f"  sdk失败回退requests: {e}")
    try:
        j = requests.post(f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send",
                          data={"title": title, "desp": content}, timeout=15).json()
        if j.get('code') != 0:
            print(f"  requests返回非0: {j} (多为额度/限流/key问题)")
        return j.get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def send_to_serverchan(df_warnings, total_n):
    if not SERVERCHAN_KEY:
        return False
    lines = [f"**🚨 市场异常行为检测** | 异常{len(df_warnings)}只 / 扫描{total_n}只",
             "*(本福特/深度错配/碎单/大单集中/OFI/POC/tick失衡 多维微观结构异常; 检测成交额Top票; 非交易日=上一交易日数据; 异常≠操纵定论, 需人工核实)*", ""]
    for _, r in df_warnings.head(PUSH_TOP).iterrows():
        lines.append(f"- 🚨 **{r['名称']}({r['代码']})** 价{r['最新价']} | {r['风险提示']}")
        lines.append(f"  MAD={r['本福特-MAD']} P={r['本福特-P值']} 深度错配={r['深度错配比']} 碎单={r['非圆整比例']} "
                     f"大单={r['大单集中度']} OFI-Z={r['逐笔OFI-Z']} 距POC={r['距POC']} 失衡={r['失衡比例']}")
    if len(df_warnings) > PUSH_TOP:
        lines.append(f"\n*…另有 {len(df_warnings)-PUSH_TOP} 只, 详见 output 报告*")
    content = "\n".join(lines)

    LIMIT = 3800
    chunks, cur, cur_len = [], [], 0
    for ln in content.split("\n"):
        lnlen = len(ln) + 1
        if cur_len + lnlen > LIMIT and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(ln); cur_len += ln_len if False else lnlen
    if cur:
        chunks.append("\n".join(cur))
    chunks = chunks or [""]
    ok = True
    title = f"🚨 市场异常监控 ({len(df_warnings)}只)"
    for i, ch in enumerate(chunks):
        t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
        print(f"  推送第{i+1}/{len(chunks)}条 ({len(ch)}字符)")
        ok = _send_one(t, ch) and ok
        if i < len(chunks) - 1:
            time.sleep(1)
    print(f"📲 推送 {'✅' if ok else '⚠️失败'} ({len(chunks)}条)")
    return ok

# ------------------ 主流程 ------------------
def run_detection():
    print("=" * 70)
    print(f"🚨 市场异常行为检测 | {datetime.now():%Y-%m-%d %H:%M} | 扫描成交额Top{SCAN_COUNT} | 并发{MAX_WORKERS} | 每只{TICK_LIMIT}笔tick")
    print(f"本福特={'启用' if BENFORD_AVAILABLE else '降级跳过(未装benfordslaw)'}; 阈值 MAD>{BENFORD_MAD_MAX}/深度错配>{DEPTH_MISMATCH_MAX}/碎单>{ODD_RATIO_MAX}/大单>{TOP5_VOL_MAX}/|OFI-Z|>{OFI_Z_MAX}/距POC>{DIST_POC_MAX}/失衡>{IMBALANCE_MAX}")
    print("=" * 70)
    try:
        df_spot = ak.stock_zh_a_spot_em()
        df_spot = df_spot.sort_values("成交额", ascending=False).head(SCAN_COUNT)
    except Exception as e:
        print(f"获取快照失败: {e}")
        return

    results = []
    print(f"正在分析 {len(df_spot)} 只高成交股票...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(analyze_stock, row) for _, row in df_spot.iterrows()]
        for f in futures:
            res = f.result()
            if res:
                warnings = []

                # 【完善】阈值全用env变量(默认值=原硬编码)
                if res["本福特-MAD"] is not None and res["本福特-MAD"] > BENFORD_MAD_MAX:
                    warnings.append("本福特偏离过高")
                if res["本福特正常"] is False:
                    warnings.append("本福特显著异常")
                if res["深度错配比"] > DEPTH_MISMATCH_MAX:
                    warnings.append("成交/深度严重错配")
                if res["非圆整比例"] > ODD_RATIO_MAX:
                    warnings.append("碎单比例异常")
                if res["大单集中度"] > TOP5_VOL_MAX:
                    warnings.append("大单集中")
                if res["逐笔OFI-Z"] is not None and abs(res["逐笔OFI-Z"]) > OFI_Z_MAX:
                    warnings.append("逐笔OFI异常")
                if res["距POC"] is not None and res["距POC"] > DIST_POC_MAX:
                    warnings.append("价格远离成交密集区")
                if res["失衡比例"] is not None and res["失衡比例"] > IMBALANCE_MAX:
                    warnings.append("tick失衡偏高")

                res["风险提示"] = " | ".join(warnings)
                results.append(res)

    if not results:
        print("未发现显著异常(或tick拉取全部失败, 查限流)。")
        return

    df_res = pd.DataFrame(results)
    df_res["风险计数"] = df_res["风险提示"].apply(lambda x: len(x.split(" | ")) if x else 0)
    df_res = df_res.sort_values("风险计数", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("异常检测结果汇总")
    print("=" * 80)

    df_warnings = df_res[df_res["风险提示"] != ""]
    if not df_warnings.empty:
        cols = [
            "代码", "名称", "最新价", "本福特-MAD", "本福特-P值",
            "非圆整比例", "大单集中度", "深度错配比",
            "静态OFI", "逐笔OFI-Z", "距POC", "失衡比例", "风险提示"
        ]
        print(df_warnings[cols].to_string(index=False))
    else:
        print("未发现显著异常股票")

    # None填0再比较, 防 object-dtype 列 None>x 抛 TypeError 崩在汇总块
    mad_col = df_res["本福特-MAD"].fillna(0)
    ofi_z_col = df_res["逐笔OFI-Z"].fillna(0)
    poc_col = df_res["距POC"].fillna(0)
    imbalance_col = df_res["失衡比例"].fillna(0)

    print(f"\n全部统计 ({len(df_res)}只股票):")
    print(f"  - 本福特异常 (MAD>{BENFORD_MAD_MAX}): {len(df_res[mad_col > BENFORD_MAD_MAX])} 只")
    print(f"  - 本福特异常 (P<{BENFORD_ALPHA}): {len(df_res[df_res['本福特正常'] == False])} 只")
    print(f"  - 深度错配 (>{DEPTH_MISMATCH_MAX}): {len(df_res[df_res['深度错配比'] > DEPTH_MISMATCH_MAX])} 只")
    print(f"  - 碎单过高 (>{ODD_RATIO_MAX*100:.0f}%): {len(df_res[df_res['非圆整比例'] > ODD_RATIO_MAX])} 只")
    print(f"  - 大单集中 (>{TOP5_VOL_MAX*100:.0f}%): {len(df_res[df_res['大单集中度'] > TOP5_VOL_MAX])} 只")
    print(f"  - 逐笔OFI异常 (|Z|>{OFI_Z_MAX}): {len(df_res[ofi_z_col.abs() > OFI_Z_MAX])} 只")
    print(f"  - 距POC过远 (>{DIST_POC_MAX*100:.0f}%): {len(df_res[poc_col > DIST_POC_MAX])} 只")
    print(f"  - tick失衡偏高 (>{IMBALANCE_MAX}): {len(df_res[imbalance_col > IMBALANCE_MAX])} 只")

    # 【完善】存盘 (可回看历史异常)
    tag = datetime.now().strftime("%Y%m%d")
    try:
        df_res.to_csv(os.path.join(OUTPUT_DIR, f"market_anomaly_{tag}.csv"), index=False, encoding="utf-8-sig")
        df_res.to_json(os.path.join(OUTPUT_DIR, f"market_anomaly_{tag}.json"), orient="records", force_ascii=False, indent=2)
        print(f"\n📁 已存 {OUTPUT_DIR}/market_anomaly_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常(结果已在内存): {type(e).__name__}: {e}")
        traceback.print_exc()

    if SERVERCHAN_KEY and not df_warnings.empty:
        send_to_serverchan(df_warnings, len(df_res))

if __name__ == "__main__":
    run_detection()
# >>>FILE_END_market_anomaly<<<
