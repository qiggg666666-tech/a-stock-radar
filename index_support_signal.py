# -*- coding: utf-8 -*-
"""
index_support_signal.py —— 指数 支撑位 + 量比自适应 买卖信号
对一组指数: 检测近期支撑点(find_peaks), 算量比自适应阈值, 结合"距支撑+站MA+量能"
给出 🟢强买入/🟢买入/🟡观察买入/🟡持有/🔴卖出 五级择时信号。

⚠️ 重要改造说明(为何不能"只修工程层"):
  原脚本绑死 Ashare(取数, 维护度低/Actions常装不上或数据源挂) + MyTT(仅用MA函数) +
  config.json(缺文件即崩)。这三者在 GitHub Actions 上几乎必导致脚本启动即失败,
  故必须换数据/工具层: Ashare->baostock+东财双源; MyTT.MA->pandas.rolling(数值等价);
  config.json->可选(有则读, 无则用内联默认)。
  但【信号策略本身一字未动】: 量比自适应阈值算法/支撑点find_peaks/五级信号判定全部保留;
  MA 换实现后数值与原 MyTT.MA 完全一致, 故支撑价/量比/信号与原版等价。

⚠️ 并发改造: 原 ThreadPoolExecutor 配 baostock 会串数据/崩(baostock 非线程安全);
  指数仅数个, 改串行(几秒), 无性能损失, 是正确性修复。

⚠️ 记号语义: 🟢/🟡/🔴 为【指数择时动作信号】(买/观/卖), 非选股层板块记号, 语境不同不混。
⚠️ 不加交易日判断: 支撑/量比基于历史K线, 非交易日看上一交易日信号仍有效(同 index_divergence 哲学)。
⚠️ config.json 的 output_file 项已废弃, 统一存 output/。
"""
import os
import sys
import json
import time
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import akshare as ak
import baostock as bs

# ------------------ 默认配置(无 config.json 时使用; 有 config.json 则覆盖) ------------------
DEFAULT_CONFIG = {
    "global": {
        "days": 250,                 # 回看日线根数
        "ma_period": 5,              # 均线周期(原变量名MA5, 故默认5)
        "prominence": 0.02,          # find_peaks 显著性系数(乘 lows.mean())
        "distance": 10,              # find_peaks 最小间距
        "near_ma_threshold": 0.02,   # 支撑点贴近MA的相对距离阈值
        "ma_band": 0.985,            # 买入条件: 收盘 >= MA*ma_band
        "hold_dist": 0.09,           # 持有条件: 距支撑 < hold_dist
    },
    "indices": {
        "上证指数": {"code": "sh.000001", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
        "深证成指": {"code": "sz.399001", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
        "创业板指": {"code": "sz.399006", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
        "科创50":   {"code": "sh.000688", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
        "沪深300":  {"code": "sh.000300", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
        "中证500":  {"code": "sh.000905", "buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0},
    },
}

# 每指数缺键时的兜底默认
_IDX_DEFAULT = {"buy_dist_threshold": 0.03, "vol_adjust_factor": 1.0, "ma_band": None, "hold_dist": None}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
PUSH_TOP = int(os.environ.get('PUSH_TOP', '10'))
AK_TIMEOUT = int(os.environ.get('AK_TIMEOUT', '20'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_BS_LOGGED = False


# ------------------ 配置加载(config.json 可选) ------------------
def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))   # 深拷贝默认
    path = os.environ.get('INDEX_SIGNAL_CONFIG', 'config.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                user = json.load(f)
            # 浅合并 global
            if isinstance(user.get('global'), dict):
                cfg['global'].update(user['global'])
            # indices: 用户提供的整体替换默认(保留默认里用户没列的指数? 这里用用户提供为准, 更符合"我的config")
            if isinstance(user.get('indices'), dict) and user['indices']:
                cfg['indices'] = user['indices']
            print(f"✅ 已加载外部配置 {path}")
        except Exception as e:
            print(f"⚠️ 读取 {path} 失败, 用内联默认: {e}")
    else:
        print(f"ℹ️ 未找到 {path}, 使用内联默认指数名单+参数")
    # 每指数补兜底键
    for name, ic in cfg['indices'].items():
        for k, v in _IDX_DEFAULT.items():
            ic.setdefault(k, v if v is not None else cfg['global'].get(k.replace('ma_band', 'ma_band').replace('hold_dist', 'hold_dist'), v))
        ic.setdefault('ma_band', cfg['global']['ma_band'])
        ic.setdefault('hold_dist', cfg['global']['hold_dist'])
    return cfg


# ------------------ 推送 / 登录 / 超时 ------------------
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


def _bs_q(code, fields, count, timeout=AK_TIMEOUT):
    """baostock 取最近 count 个交易日(用 start_date 近似: count*1.6 日历天)"""
    from datetime import datetime, timedelta
    sd = (datetime.now() - timedelta(days=int(count * 1.6))).strftime('%Y-%m-%d')
    def _do():
        return bs.query_history_k_data_plus(code, fields, start_date=sd, adjustflag="2").get_data()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_do).result(timeout=timeout)


def _call_with_timeout(fn, *a, timeout=AK_TIMEOUT, **kw):
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *a, **kw).result(timeout=timeout)


# ------------------ 指数日线(双源, 返回以 date 为索引的升序 df) ------------------
def _fetch_index(code, days):
    """返回 set_index('date') 升序 df, 含 high/low/close/volume; 双源+超时; 失败 None"""
    sym = code[3:] if len(code) > 3 and code[2] == '.' else code
    # 路径1: baostock
    if _BS_LOGGED:
        try:
            d = _bs_q(code, "date,high,low,close,volume", days)
            if d is not None and not d.empty:
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date')
                if len(d) >= 80:
                    return d.set_index('date')
        except Exception:
            pass
    # 路径2: 东财指数日线兜底
    from datetime import datetime, timedelta
    sy = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")
    for attempt in range(2):
        try:
            d = _call_with_timeout(ak.index_zh_a_hist, symbol=sym, period="daily",
                                   start_date=sy, end_date=datetime.now().strftime("%Y%m%d"), timeout=AK_TIMEOUT)
            if d is not None and not d.empty:
                d = d.rename(columns={'日期': 'date', '最高': 'high', '最低': 'low',
                                      '收盘': 'close', '成交量': 'volume'})
                for c in ['high', 'low', 'close', 'volume']:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
                d['date'] = pd.to_datetime(d['date'])
                d = d.dropna(subset=['close']).sort_values('date')
                if len(d) >= 80:
                    return d.set_index('date')
        except Exception:
            time.sleep(1 + attempt)
    return None


# ------------------ 单指数处理(信号策略一字未动, 仅取数/MA换实现) ------------------
def process_index(name, idx_cfg, g):
    code = idx_cfg['code']
    try:
        df = _fetch_index(code, g['days'])
        if df is None or len(df) < 80:
            print(f"{name}: 数据不足/获取失败, 跳过")
            return None

        # MA: 原 MyTT.MA(close, N) == rolling(N).mean(), 数值等价
        df = df.copy()
        df['MA5'] = df['close'].rolling(g['ma_period']).mean()
        df['Vol_MA20'] = df['volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['volume'] / df['Vol_MA20']

        # ================== 量比阈值优化算法(原封不动) ==================
        vol_series = df['Vol_Ratio'].tail(80)
        vol_p75 = vol_series.quantile(0.75)
        vol_p90 = vol_series.quantile(0.90)
        vol_mean = vol_series.mean()

        adjust = idx_cfg.get('vol_adjust_factor', 1.0)
        strong_vol_threshold = max(vol_p90 * adjust, 1.65)
        mild_vol_threshold   = max(vol_p75 * adjust, 1.20)
        low_vol_threshold    = vol_mean * 0.65

        # 支撑点检测(原封不动)
        lows = df['low'].values
        peaks, _ = find_peaks(-lows,
                              prominence=g['prominence'] * lows.mean(),
                              distance=g['distance'])

        supports = []
        for i in peaks:
            if i < 30 or i > len(df) - 10:
                continue
            row = df.iloc[i]
            ma5 = row.get('MA5', row['close'])
            if pd.notna(ma5) and abs(row['low'] - ma5) / ma5 < g['near_ma_threshold']:
                supports.append({'日期': df.index[i], '支撑价': round(row['low'], 2)})

        if not supports:
            print(f"{name}: 未检测到有效支撑点, 跳过")
            return None

        latest_support = pd.DataFrame(supports).iloc[-1]['支撑价']
        latest_close = df['close'].iloc[-1]
        latest_ma5 = df['MA5'].iloc[-1]
        latest_vol_ratio = df['Vol_Ratio'].iloc[-1]

        dist_to_support = (latest_close - latest_support) / latest_support

        ma_band = idx_cfg.get('ma_band') or g['ma_band']
        hold_dist = idx_cfg.get('hold_dist') or g['hold_dist']

        # ================== 信号判断(原封不动, 阈值改为可配) ==================
        if (dist_to_support <= idx_cfg['buy_dist_threshold'] and
                latest_close >= latest_ma5 * ma_band):
            if latest_vol_ratio >= strong_vol_threshold:
                signal = '🟢 强买入'
                reason = f'接近支撑 + 极强放量({latest_vol_ratio:.2f}x)'
            elif latest_vol_ratio >= mild_vol_threshold:
                signal = '🟢 买入'
                reason = f'接近支撑 + 温和放量({latest_vol_ratio:.2f}x)'
            else:
                signal = '🟡 观察买入'
                reason = f'接近支撑但量能偏弱'
        elif dist_to_support < hold_dist and latest_vol_ratio > low_vol_threshold:
            signal = '🟡 持有'
            reason = '支撑上方，量能可接受'
        else:
            signal = '🔴 卖出'
            reason = f'远离支撑或量能不足({latest_vol_ratio:.2f}x)'

        result = {
            '指数': name,
            '当前价': round(float(latest_close), 2),
            'MA5': round(float(latest_ma5), 2) if pd.notna(latest_ma5) else None,
            '最近支撑': round(float(latest_support), 2),
            '距离支撑%': round(float(dist_to_support) * 100, 2),
            '成交量比': round(float(latest_vol_ratio), 2),
            '信号': signal,
            '理由': reason,
        }
        print(f"{name:>8} | 价 {latest_close:.2f} | 支撑 {latest_support:.2f} | "
              f"量比 {latest_vol_ratio:.2f} | {signal}")
        return pd.DataFrame([result])

    except Exception as e:
        print(f"{name} 处理异常: {e}")
        traceback.print_exc()
        return None


# ------------------ 批量执行(串行: baostock 非线程安全) ------------------
def run_batch(CONFIG):
    g = CONFIG['global']
    print("\n🚀 开始处理指数(串行, baostock+东财双源)...\n")
    results = []
    for name, idx_cfg in CONFIG['indices'].items():
        res = process_index(name, idx_cfg, g)
        if res is not None:
            results.append(res)

    if not results:
        print("未生成有效结果")
        return None
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values(['信号', '距离支撑%']).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("📊 指数支撑+量比 择时信号")
    print("=" * 100)
    print(final_df.to_string(index=False))
    return final_df


def build_push(df):
    n_buy = int(df['信号'].str.contains('买入', na=False).sum())
    n_sell = int(df['信号'].str.contains('卖出', na=False).sum())
    L = [f"**📊 指数支撑+量比 择时信号** | 🟢买入{n_buy} 🔴卖出{n_sell}",
         "*(支撑位+量比自适应; 指数择时动作信号, 非个股选股; 滞后指标, 仅供参考)*", ""]
    for _, r in df.iterrows():
        L.append(f"- {r['信号']} **{r['指数']}** 现价{r['当前价']} | 支撑{r['最近支撑']} "
                 f"距{r['距离支撑%']}% | 量比{r['成交量比']}x | {r['理由']}")
    return "\n".join(L)


if __name__ == "__main__":
    print("=" * 70)
    print(f"📊 指数支撑+量比 择时信号 | {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print("数据源 baostock+东财双源; MA=pandas.rolling(等价MyTT.MA); 串行(线程安全)")
    print("=" * 70)
    CONFIG = load_config()
    # 主进程登录 baostock(失败也继续, 走东财兜底)
    if not _bs_login_ok():
        print("⚠️ baostock 登录失败, 各指数走东财指数日线兜底")
    df = run_batch(CONFIG)
    if _BS_LOGGED:
        try:
            bs.logout()
        except Exception:
            pass

    if df is None or df.empty:
        print("本次无有效指数信号"); sys.exit(0)
    # ---- 收尾全部包防护 ----
    tag = pd.Timestamp.now().strftime("%Y%m%d")
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"index_support_signal_{tag}.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, f"index_support_signal_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "indices": list(CONFIG['indices'].keys()),
                       "global": CONFIG['global'], "signals": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/index_support_signal_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘异常: {type(e).__name__}: {e}")
        traceback.print_exc()
    if SERVERCHAN_KEY:
        try:
            n_buy = int(df['信号'].str.contains('买入', na=False).sum())
            send_serverchan(f"📊 指数择时 | 🟢买入{n_buy} 🔴卖出{int(df['信号'].str.contains('卖出', na=False).sum())}",
                            build_push(df))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)
