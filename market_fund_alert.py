import os
import json
import time
import random
import requests
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak
import baostock as bs

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

"""
market_fund_alert.py

市场资金异动监控：北向资金 + 重点个股成交量放大。
⚠️ 单次运行架构（不是 while+schedule 常驻进程），靠 GitHub Actions 的 cron 定时重复触发。

【v2 升级说明】
- 修复 sc_send 硬导入(软导入+requests兜底); 重点个股 baostock 登录检查+东财兜底;
  结果每次存 output/ json 留痕(含"无异动"事实+北向原始快照+个股量比), 仅触发时推送。
- 北向资金: 保留原作者"宽松列名探测"(抗akshare版本差异, 精华), 扩充候选列名;
  加单位启发式(|原始值|>10000 视为"元"自动/1e8转亿, 否则视为"亿"); alert 同时打印
  原始值与换算值, 并把原始快照存 json —— 把作者故意留下的"单位/列名未确认"不确定性
  显式化、可核对, 而非假装解决。第一次跑后请看 output json 的 north_raw_* 字段核对。
- 本脚本是"事件监控"非选股/选板块, 故不并入统一emoji记号体系, 不加板块维度。
- 不加交易日判断: 北向/量比为快照/历史性质, 非交易日不会误报(接口空->无异动)。
"""

# ------------------ 参数 (全部 env 可调) ------------------
THRESHOLD_NORTH = float(os.environ.get('THRESHOLD_NORTH', '50'))    # 北向净流入异动阈值(单位: 亿元; 配合单位启发式)
THRESHOLD_VOLUME = float(os.environ.get('THRESHOLD_VOLUME', '1.5')) # 成交量放大倍数阈值
ALERT_STOCKS = {           # 重点监控个股，baostock格式代码（默认）
    "sh.600519": "贵州茅台",
    "sz.000001": "平安银行",
}
# 追加重点个股(env, 逗号分隔, 每段 code:name; code 可带或不带 sh./sz. 前缀)
# 例: ALERT_STOCKS_EXTRA="sh.601318:中国平安,sz.000858:五粮液"
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_alert_stocks():
    """默认 dict + env ALERT_STOCKS_EXTRA 追加"""
    stocks = dict(ALERT_STOCKS)
    extra = os.environ.get('ALERT_STOCKS_EXTRA', '').strip()
    if extra:
        for seg in extra.split(','):
            seg = seg.strip()
            if not seg:
                continue
            code, _, name = seg.partition(':')
            code = code.strip(); name = name.strip()
            if not code:
                continue
            if not code.startswith(('sh.', 'sz.')):
                code = _norm6(code)
            stocks[code] = name or code
    return stocks


# ------------------ 推送 (软导入) / 登录重试 / 工具 ------------------
def send_serverchan(title, content, sendkey=""):
    """Server酱推送: serverchan-sdk 软导入优先, requests 兜底"""
    key = sendkey or SERVERCHAN_KEY
    if not key:
        print("未配置 SERVERCHAN_KEY/SENDKEY，仅打印不推送")
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


def _norm6(code):
    """6位/带前缀 -> 带 sh./sz. 前缀 (baostock 格式)"""
    c = str(code).strip()
    if c.startswith(('sh.', 'sz.')):
        return c
    c = c.split('.')[-1].zfill(6)
    return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c


def _six(code_pref):
    """带前缀 -> 6位 (东财用)"""
    return code_pref[3:] if len(code_pref) > 3 and code_pref[2] == '.' else code_pref


def _to_yi(net_flow):
    """单位启发式: |原始值|>10000 视为'元'->/1e8; 否则视为'亿'。返回 (亿值, 说明)。
    分界10000逻辑自洽: 北向单日净流入不可能>10000亿, 也不可能'元'单位下<10000(=0.0001亿无意义)。"""
    if net_flow is None or pd.isna(net_flow):
        return None, "无数据"
    if abs(net_flow) > 10000:
        return net_flow / 1e8, "原始值>1万, 判为'元'已/1e8转亿(启发式)"
    return float(net_flow), "原始值≤1万, 判为'亿'(原值)"


# ------------------ 东财 K 线兜底 (量比只需 volume) ------------------
def _fetch_vol_em(sym6, days=15):
    """东财逐只兜底; 返回 volume Series(最近days天) 或 None"""
    end_y = datetime.now().strftime("%Y%m%d")
    start_y = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = ak.stock_zh_a_hist(symbol=sym6, period="daily",
                                   start_date=start_y, end_date=end_y, adjust="qfq")
            if d is None or d.empty or '成交量' not in d.columns:
                return None
            v = pd.to_numeric(d['成交量'], errors='coerce').dropna()
            return v if len(v) >= 6 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


# ------------------ 北向资金异动 (保留宽松列名探测 + 单位启发式 + 原始快照) ------------------
def check_north_flow():
    """
    北向资金净流入异动检查。
    正确函数是 stock_hsgt_fund_flow_summary_em()（不传symbol参数）。
    返回结构可能是"沪股通/深股通/港股通"分行的表格，具体列名和数值单位尚未100%确认，
    这里第一次运行会把完整数据打印出来，供人工核对后再精确处理列名匹配和单位换算逻辑。
    返回: (alerts, snapshot_dict)  # snapshot 供存 json 核对
    """
    alerts = []
    snapshot = {"columns": [], "rows": [], "id_col": None, "value_col": None,
                "net_raw": None, "net_yi": None, "unit_note": "", "matched_rows": 0}
    try:
        print(f"当前akshare版本: {ak.__version__}")
        if not hasattr(ak, "stock_hsgt_fund_flow_summary_em"):
            print("⚠️ 当前akshare版本没有 stock_hsgt_fund_flow_summary_em 函数，"
                  "需要在 requirements.txt 里把 akshare 升级到较新版本")
            return alerts, snapshot

        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            print("⚠️ 沪深港通资金流向接口返回空数据")
            return alerts, snapshot

        snapshot["columns"] = list(df.columns)
        try:
            snapshot["rows"] = df.head(20).to_dict('records')
        except Exception:
            snapshot["rows"] = []

        print(f"沪深港通资金流向接口返回列名: {list(df.columns)}")
        print("完整数据（用于第一次核对列名/结构/单位）:")
        print(df.to_string(index=False))

        # 尝试识别"类型/板块"这类标识列，过滤出沪股通+深股通（=北向，港资买A股）
        # 港股通是南向（内地资金买港股），不计入北向异动
        # (v2: 扩充候选列名, 抗更多 akshare 版本; 保留作者宽松探测逻辑)
        id_col = None
        for col in ['类型', '板块', '资金方向', '名称', '资金流向', '板块名称']:
            if col in df.columns:
                id_col = col
                break

        value_col = None
        for col in ['成交净买额', '净买额', '净流入', 'value', '当日资金流入',
                    '成交净买额-净买额', '当日净买入-净买额', '净买额-成交净买额', '北向资金']:
            if col in df.columns:
                value_col = col
                break

        snapshot["id_col"] = id_col
        snapshot["value_col"] = value_col

        if id_col is None or value_col is None:
            print("⚠️ 未能自动识别出'类型'列或'净流入金额'列，本次仅打印数据供人工核对，不做阈值判断")
            return alerts, snapshot

        north_rows = df[df[id_col].astype(str).str.contains('沪股通|深股通|北向', na=False)]
        snapshot["matched_rows"] = len(north_rows)
        if north_rows.empty:
            uniq = df[id_col].astype(str).unique().tolist()
            print(f"⚠️ 在 {id_col} 列中未匹配到沪股通/深股通/北向相关行，本次不做阈值判断")
            print(f"   该列现有取值(供核对): {uniq}")
            return alerts, snapshot

        net_flow = pd.to_numeric(north_rows[value_col], errors='coerce').sum()
        # 注意：单位可能是"元"而非"亿元"，第一次运行后需要人工核对是否要除以1e8
        # (v2: 单位启发式 + 显式标注, 把不确定性显式化)
        net_yi, unit_note = _to_yi(net_flow)
        snapshot["net_raw"] = None if pd.isna(net_flow) else float(net_flow)
        snapshot["net_yi"] = net_yi
        snapshot["unit_note"] = unit_note
        print(f"北向资金(沪股通+深股通)净流入合计: 原始={net_flow} | 换算={net_yi}亿 | {unit_note}")

        if net_yi is not None and abs(net_yi) > THRESHOLD_NORTH:
            direction = "净流入" if net_yi > 0 else "净流出"
            alerts.append(
                f"🌊 北向资金大{direction}: {net_yi:+.2f} 亿元 "
                f"（原始接口值 {net_flow:.2f}，{unit_note}，阈值±{THRESHOLD_NORTH}亿）"
            )

    except Exception as e:
        print(f"⚠️ 北向资金检查失败: {e}")

    return alerts, snapshot


# ------------------ 重点个股成交量放大 (登录检查 + K线双源) ------------------
def check_stock_volume():
    """重点个股成交量放大检查: baostock 优先 + 东财兜底"""
    alerts = []
    vol_snapshot = []
    stocks = _load_alert_stocks()

    # 主进程登录检查 (修复裸登录 -> 静默无量比)
    logged_in = _bs_login_ok()
    if not logged_in:
        print("⚠️ baostock 主进程登录失败, 重点个股将走东财兜底")

    for code, name in stocks.items():
        sym6 = _six(code)
        vol = None
        src = None
        # 路径1: baostock
        if logged_in:
            try:
                start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
                rs = bs.query_history_k_data_plus(
                    code, "date,close,volume", start_date=start_date, adjustflag="2"
                )
                df = rs.get_data()
                if df is not None and not df.empty and len(df) >= 6:
                    v = pd.to_numeric(df['volume'], errors='coerce').dropna()
                    if len(v) >= 6:
                        vol = v; src = 'bs'
            except Exception as e:
                print(f"  {code} baostock 查询失败: {e}")
        # 路径2: 东财兜底
        if vol is None:
            v = _fetch_vol_em(sym6, days=15)
            if v is not None and len(v) >= 6:
                vol = v; src = 'em'

        if vol is None:
            vol_snapshot.append({"代码": code, "名称": name, "量比": None, "源": None, "triggered": False})
            print(f"⚠️ {name}({code}) 双源均无数据, 跳过")
            continue

        latest_vol = vol.iloc[-1]
        avg_vol = vol.iloc[-5:-1].mean()
        if avg_vol <= 0 or pd.isna(avg_vol):
            vol_snapshot.append({"代码": code, "名称": name, "量比": None, "源": src, "triggered": False})
            continue
        vol_ratio = latest_vol / avg_vol
        triggered = bool(vol_ratio > THRESHOLD_VOLUME)
        vol_snapshot.append({"代码": code, "名称": name, "量比": round(float(vol_ratio), 2),
                             "源": src, "triggered": triggered})
        print(f"{name}({code}) 量比: {vol_ratio:.2f} (源={src})")
        if triggered:
            alerts.append(f"📊 {name}({code}) 成交量放大: {vol_ratio:.2f}倍（阈值{THRESHOLD_VOLUME}倍，源{src}）")

    if logged_in:
        try:
            bs.logout()
        except Exception:
            pass

    return alerts, vol_snapshot


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== 资金异动检查 {now} ===")

    north_alerts, north_snap = check_north_flow()
    vol_alerts, vol_snap = check_stock_volume()
    all_alerts = north_alerts + vol_alerts
    triggered = bool(all_alerts)

    # 每次都存 json 留痕 (含"无异动"事实 + 北向原始快照 + 个股量比, 审计用)
    tag = datetime.now().strftime('%Y%m%d')
    record = {
        "check_time": now,
        "triggered": triggered,
        "alerts": all_alerts,
        "north": north_snap,
        "stock_volume": vol_snap,
    }
    json_path = f"{OUTPUT_DIR}/market_fund_alert_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 检查记录已保存: {json_path} (triggered={triggered})")

    if triggered:
        content = f"检查时间：{now}\n\n" + "\n".join(all_alerts)
        print("\n" + content)
        send_serverchan(f"资金异动预警 {now}", content)
    else:
        print("本次检查未触发任何异动阈值 (记录已存 json, 不推送)")
