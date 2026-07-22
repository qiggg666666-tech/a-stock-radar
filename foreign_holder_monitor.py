import os
import re
import json
import time
import random
import requests
from datetime import datetime

import pandas as pd
import akshare as ak

# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

"""
foreign_holder_monitor.py

外资 + 香港中央结算 十大流通股东监控。
⚠️ 单次运行架构（不是 while+schedule 常驻进程），靠 GitHub Actions 的 cron 每周定时触发一次
   （十大流通股东是季度更新数据，每周检查足够，没必要每天跑）。
⚠️ 正确函数是 stock_gdfx_free_top_10_em(symbol=股票代码)（不带sh./sz.前缀，直接6位数字），
   原代码里的 stock_gdfx_free_holding_statistics_em 在akshare里不存在。

【v2 升级说明】
- 修复 sc_send 硬导入(软导入+requests兜底); 结果每次存 output/ json 留痕
  (含"未检测到"事实 + 每只全十大股东快照 + 列名探测结果), 仅触发时推送。
- 列名健壮探测(股东名/持股比例/持股变动 各试多个候选列名), 取不到则降级显示, 不崩;
  关键词用 re.escape 包裹, 防 'J. P. Morgan' 的 '.' 当正则通配符误匹配。
- 在"有无外资"基础上, 零额外接口增强: ①外资/中央结算合计持股比例 ②外资动向方向
  (🆕新进/⬆️增持/⬇️减持/➖持平, 从持股变动列健壮提取)。注意: 季度数据, 变动为季度环比非实时。
- 本脚本是"持股结构监控"非选股/选板块, 故不并入统一emoji记号体系。
- 不加交易日判断: 十大流通股东为季度静态披露, 与是否交易日无关。
- 无 baostock 双源: baostock 无十大流通股东明细接口, akshare 失败则该只跳过(诚实降级)。
"""

# ------------------ 参数 (全部 env 可调) ------------------
FOREIGN_KEYWORDS = [
    "BARCLAYS BANK", "J. P. Morgan", "UBS AG", "高盛国际",
    "Morgan", "HSBC", "Citigroup", "BlackRock",
    "香港中央结算", "HKSCC", "Central Clearing"
]
WATCH_STOCKS = ["603619"]  # 默认监控名单（6位数字，不带前缀）

# env 追加: WATCH_STOCKS_EXTRA="600519,000858" (逗号分隔6位, 与默认合并去重)
# env 追加关键词: FOREIGN_KEYWORDS_EXTRA="Nomura,瑞银" (逗号分隔, 与默认合并)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_watch_stocks():
    base = [str(c).strip().split('.')[-1].zfill(6) for c in WATCH_STOCKS if str(c).strip()]
    extra = os.environ.get('WATCH_STOCKS_EXTRA', '').strip()
    if extra:
        base += [c.strip().split('.')[-1].zfill(6) for c in extra.split(',') if c.strip()]
    seen, out = set(), []
    for c in base:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out


def _load_keywords():
    kws = list(FOREIGN_KEYWORDS)
    extra = os.environ.get('FOREIGN_KEYWORDS_EXTRA', '').strip()
    if extra:
        kws += [k.strip() for k in extra.split(',') if k.strip()]
    return [k for k in kws if k]


# ------------------ 推送 (软导入) ------------------
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


# ------------------ 列名健壮探测 + 行提取 ------------------
def _find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _direction_emoji(change_val):
    """从持股变动原文映射方向 emoji; 取不到/无法判断返回 ''"""
    if change_val is None or (isinstance(change_val, float) and pd.isna(change_val)):
        return ""
    s = str(change_val).strip()
    if not s or s.lower() == 'nan':
        return ""
    sl = s.lower()
    if ('新进' in s) or ('new' in sl):
        return "🆕"
    if ('增' in s) or ('加' in s) or ('进' in s):
        return "⬆️"
    if ('减' in s) or ('退' in s):
        return "⬇️"
    if ('不变' in s) or ('持平' in s) or ('未变' in s):
        return "➖"
    return ""


def _extract_row(row, name_col, pct_col, chg_col):
    """从一行提取 股东名/比例/变动/方向; 比例转 float, 取不到为 None"""
    name = str(row[name_col]) if name_col else ""
    pct = None
    if pct_col:
        try:
            v = pd.to_numeric(row[pct_col], errors='coerce')
            pct = None if pd.isna(v) else round(float(v), 4)
        except Exception:
            pct = None
    chg = None
    if chg_col:
        cv = row[chg_col]
        chg = None if (cv is None or (isinstance(cv, float) and pd.isna(cv))) else str(cv)
    return {"股东名称": name, "持股比例%": pct, "持股变动": chg, "方向": _direction_emoji(chg)}


# ------------------ 单只检查 (列名健壮探测 + 动向增强) ------------------
def check_stock_foreign_holders(stock_code, keywords):
    """检查单只股票的十大流通股东里有没有外资/香港中央结算。
    返回 dict: {hit: bool, columns:[], top10:[rows], foreign:[extracted], foreign_total_pct: x|None, name_col, pct_col, chg_col}
    或 None(接口失败/空)。"""
    try:
        holders = ak.stock_gdfx_free_top_10_em(symbol=stock_code)
        if holders is None or holders.empty:
            print(f"{stock_code}: 未获取到股东数据")
            return None

        cols = list(holders.columns)
        print(f"{stock_code} 十大流通股东数据列名: {cols}")

        # 股东名称列
        name_col = _find_col(cols, ['股东名称', '名称', '股东', 'holder_name'])
        # 持股比例列
        pct_col = _find_col(cols, ['持股比例', '占流通股比例', '持股占总股本比例',
                                   '占总股本比例', '占流通A股比例', '持股比例(%)', '占比'])
        # 持股变动列
        chg_col = _find_col(cols, ['持股变动', '增减', '增减变动', '较上期变化',
                                   '变动', '持股变化', '增减仓', '变动比例'])

        if name_col is None:
            print(f"⚠️ {stock_code}: 未能识别出股东名称列，跳过匹配 (列名见 json 供核对)")
            return {"hit": False, "columns": cols, "top10": _safe_rows(holders),
                    "foreign": [], "foreign_total_pct": None,
                    "name_col": None, "pct_col": pct_col, "chg_col": chg_col}

        print(f"{stock_code} 前十大股东概览 (name={name_col}, pct={pct_col}, chg={chg_col}):")
        print(holders.head(10).to_string(index=False))

        # 关键词匹配 (re.escape 防 '.' 等正则特殊字符误匹配)
        pattern = '|'.join(re.escape(k) for k in keywords)
        foreign = holders[holders[name_col].astype(str).str.contains(pattern, na=False, case=False)]

        extracted = [_extract_row(row, name_col, pct_col, chg_col) for _, row in foreign.iterrows()]

        # 外资/中央结算合计持股比例 (取不到比例则为 None)
        pcts = [e["持股比例%"] for e in extracted if e["持股比例%"] is not None]
        foreign_total_pct = round(sum(pcts), 4) if pcts else None

        if extracted:
            print(f"\n【{stock_code} 检测到外资/中央结算】合计持股≈{foreign_total_pct if foreign_total_pct is not None else '比例列缺失'}%:")
            for e in extracted:
                print(f"  {e['方向']} {e['股东名称']} | 比例{e['持股比例%']} | 变动{e['持股变动']}")
            return {"hit": True, "columns": cols, "top10": _safe_rows(holders),
                    "foreign": extracted, "foreign_total_pct": foreign_total_pct,
                    "name_col": name_col, "pct_col": pct_col, "chg_col": chg_col}
        else:
            print(f"{stock_code}: 本次未检测到重点外资/中央结算")
            return {"hit": False, "columns": cols, "top10": _safe_rows(holders),
                    "foreign": [], "foreign_total_pct": None,
                    "name_col": name_col, "pct_col": pct_col, "chg_col": chg_col}

    except Exception as e:
        print(f"⚠️ {stock_code} 获取失败: {e}")
        return None


def _safe_rows(df):
    try:
        return df.head(10).to_dict('records')
    except Exception:
        return []


# ------------------ 推送内容 (只提取关键列, 不用整行 dict) ------------------
def build_alert_content(per_stock):
    lines = []
    for code, info in per_stock.items():
        tot = info.get("foreign_total_pct")
        tot_s = f"外资/中央结算合计持股≈{tot}%" if tot is not None else "外资/中央结算合计持股: 比例列缺失"
        lines.append(f"### 【{code}】{tot_s}")
        for e in info["foreign"]:
            pct_s = f" 比例{e['持股比例%']}%" if e["持股比例%"] is not None else ""
            chg_s = f" 变动:{e['持股变动']}" if e["持股变动"] else ""
            lines.append(f"  - {e['方向']} {e['股东名称']}{pct_s}{chg_s}")
    return "\n".join(lines)


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    watch = _load_watch_stocks()
    keywords = _load_keywords()
    print(f"=== 外资+香港中央结算监控 {now} ===")
    print(f"监控名单: {watch} | 关键词 {len(keywords)} 个 | 数据为季度披露(静态), 与交易日无关")

    per_stock = {}      # code -> info(仅命中的进 alert, 但全部进 json 留痕)
    all_info = {}       # code -> info(全部, 含未命中, 留痕)
    for code in watch:
        info = check_stock_foreign_holders(code, keywords)
        if info is None:
            all_info[code] = {"hit": None, "note": "接口失败/空"}   # 留痕: 该只没拿到数据
            continue
        all_info[code] = info
        if info["hit"]:
            per_stock[code] = info

    triggered = bool(per_stock)

    # 每次都存 json 留痕 (含"未检测到"事实 + 全十大快照 + 列名探测结果, 审计/核对用)
    tag = datetime.now().strftime('%Y%m%d')
    record = {
        "check_time": now,
        "triggered": triggered,
        "watch_stocks": watch,
        "keywords": keywords,
        "per_stock": all_info,
        "alerts_brief": [f"{c}: 外资合计{info.get('foreign_total_pct')}% / {len(info.get('foreign', []))}家"
                         for c, info in per_stock.items()],
    }
    json_path = f"{OUTPUT_DIR}/foreign_holder_{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 检查记录已保存: {json_path} (triggered={triggered})")

    if triggered:
        content = f"检查时间：{now}\n\n" + build_alert_content(per_stock) + \
                  "\n\n*注: 十大流通股东为季度披露, 持股变动为季度环比非实时; 列名/比例见 output json 核对。*"
        print("\n" + content)
        send_serverchan(f"外资/中央结算持股预警 {now} ({len(per_stock)}只)", content)
    else:
        print("\n本次检查未在监控名单中检测到外资/中央结算 (记录已存 json, 不推送)")
