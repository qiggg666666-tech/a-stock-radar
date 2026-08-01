# -*- coding: utf-8 -*-
"""
daily_summary.py —— 矩阵日报汇总(跨run拉取今日所有策略artifact, 合并成一条推送)
【为什么用API而非download-artifact】40策略错峰=一次run只跑1个job, download-artifact拿不到别的run产物;
  故用 GITHUB_TOKEN 调 GitHub API 拉"今天所有 *-results artifact"解压合并, 40个job零改动。
【鲁棒解析】各策略json结构不统一, 故不硬编码字段: json是list->命中=len; 是dict->找n/命中或hits/results/pool的len;
  样本名/代码/阶段用多候选键名容错提取; 全量分布只信顶层 stage_counts/n_resonance。csv兜底(行数-1=命中)。
【策略名】从文件名前缀推断(去末尾_YYYYMMDD), 映射中文+emoji, 未映射用key本身。
【容错】API/解析任何环节失败->降级推"汇总拉取失败"或跳过该文件, 绝不崩; 本地设 ARTIFACT_DIR 可绕过API直接读目录调试。
⚠️ 仅汇总今日已上传产物; 限流/跳过/未存json者不在此列(见各job), 非漏报。
"""
import os
import io
import re
import sys
import json
import time
import zipfile
import traceback
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

BJ = timezone(timedelta(hours=8))
TOKEN = os.environ.get('GITHUB_TOKEN', '')
REPO = os.environ.get('GITHUB_REPOSITORY', '')
ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', '')          # 本地调试: 设了且存在则读本地, 不走API
SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', '')
DATE_PAT = re.compile(r'^(.+)_\d{8}$')
EXCLUDE = {'calibration', 'backtest', 'cagr', 'report', 'picker_report'}
HEAD = {'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github+json'} if TOKEN else {}

# 策略 key -> (emoji, 中文名); 未列出的用 key 本身
KEY2LABEL = {
    'first_red_120': ('🔴', '120首红'), 'first_red_520': ('🔴', '520首红'),
    'first_red_macd_combo': ('🔴', '首红×MACD'), 'first_red_sector_combo': ('🔴', '首红×板块'),
    'bottom_acc': ('🪨', '坑底蓄势'), 'boll_wbottom': ('📈', 'W底突破'),
    'vegas_tunnel': ('📈', 'Vegas通道'), 'vagas_obv': ('🚀', 'Vegas+OBV'),
    'dip_buy': ('🛒', '回调买点'), 'trend_judge': ('📊', '走势判断'),
    'sector_flow_pcr': ('💹', '板块+PCR'), 'divergence': ('🔀', '多重背离'),
    'macd_zigzag_divergence': ('🔀', 'MACD背离'), 'kdj_macd': ('📉', 'KDJ+MACD'),
    'camarilla': ('📐', 'Camarilla'), 'ma_rsi_adx': ('📊', 'MA+RSI+ADX'),
    'bull_confirm': ('🐂', '共振翻多'), 'vcp': ('📦', 'VCP收缩'),
    'dupont_roe': ('💰', '杜邦ROE'), 'ma5_sideways': ('➡️', 'MA5横盘'),
    'ma520_bottom': ('🧱', '520筑底'), 'macd_resonance': ('📡', 'MACD共振'),
    'valuation_screen': ('💎', '估值'), 'strong_continuation': ('🔥', '强势延续'),
    'sector_momentum_valuation': ('📈', '板块动量'), 'cox_sector_bot': ('🌀', 'COX超卖'),
    'mtf_resonance': ('📡', 'MTF共振'), 'zt_pre': ('⚡', '涨停候选'),
    'quant_signal': ('📊', '超跌首板'), 'main': ('🎯', '主策略'),
    'sector_pipeline': ('🏭', '板块流水线'), 'market_fund_alert': ('💸', '资金异动'),
    'sector_rotation': ('🔄', '板块轮动'), 'index_divergence': ('📉', '指数背离'),
    'index_support_resistance': ('📏', '指数支撑'), 'index_support_signal': ('📏', '指数择时'),
    'daily_market_signal': ('🌡️', '大盘信号'), 'foreign_holder': ('🏦', '外资持股'),
}
ORDER = ['first_red_120', 'bottom_acc', 'boll_wbottom', 'vegas_tunnel', 'vagas_obv',
         'dip_buy', 'trend_judge', 'sector_flow_pcr', 'divergence', 'macd_zigzag_divergence',
         'ma520_bottom', 'ma5_sideways', 'vcp', 'bull_confirm', 'zt_pre', 'valuation_screen']


def today_bj():
    return datetime.now(BJ).strftime('%Y-%m-%d')


def label_of(key):
    e, n = KEY2LABEL.get(key, ('📌', key))
    return f"{e}{n}"


# ------------------ API 拉取今日 artifact ------------------
def _get_zip(url):
    """下载 artifact zip; 手动处理302(重定向到存储时不能带Authorization, 否则401)"""
    r = requests.get(url, headers=HEAD, allow_redirects=False, timeout=60)
    if r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get('Location')
        if not loc:
            r.raise_for_status()
        r2 = requests.get(loc, timeout=60)   # 不带 auth
        r2.raise_for_status()
        return r2.content
    r.raise_for_status()
    return r.content


def fetch_today_artifacts(dest):
    """拉今日所有 artifact 解压到 dest; 返回解压出的 json/csv 路径列表"""
    if not TOKEN or not REPO:
        print("  无 GITHUB_TOKEN/GITHUB_REPOSITORY, 跳过API(本地请设 ARTIFACT_DIR)")
        return []
    url = f'https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100'
    r = requests.get(url, headers=HEAD, timeout=30)
    r.raise_for_status()
    arts = r.json().get('artifacts', [])
    today = today_bj()
    files = []
    for a in arts:
        if a.get('expired'):
            continue
        ca = a.get('created_at', '')
        try:
            ca_bj = datetime.fromisoformat(ca.replace('Z', '+00:00')).astimezone(BJ).strftime('%Y-%m-%d')
        except Exception:
            ca_bj = ''
        if ca_bj != today:
            continue
        name = a.get('name', '')
        try:
            content = _get_zip(a['archive_download_url'])
            z = zipfile.ZipFile(io.BytesIO(content))
            sub = os.path.join(dest, re.sub(r'[^\w\-]', '_', name))
            os.makedirs(sub, exist_ok=True)
            for n in z.namelist():
                if n.endswith('.json') or n.endswith('.csv'):
                    z.extract(n, sub)
                    files.append(os.path.join(sub, n))
        except Exception as e:
            print(f"  下载artifact[{name}]失败, 跳过: {e}")
    return files


# ------------------ 鲁棒解析 ------------------
def _g(rec, *keys):
    for k in keys:
        if isinstance(rec, dict) and rec.get(k) not in (None, ''):
            return rec[k]
    return None


def _sample(rec):
    name = _g(rec, '名称', 'name', '股票名称')
    code = _g(rec, '代码', 'code', 'symbol')
    if code:
        code = str(code).split('.')[-1].zfill(6)
    if not name and not code:
        return None
    return f"{name or ''}({code or '?'})"


def _stage_text(d):
    """全量分布只信顶层统计字段"""
    parts = []
    sc = d.get('stage_counts')
    if isinstance(sc, dict) and sc:
        parts.append(' '.join(f"{k}{v}" for k, v in sc.items() if v))
    nr = d.get('n_resonance')
    if nr:
        parts.append(f"🎯{nr}")
    return (' 〔' + ' '.join(parts) + '〕') if parts else ''


def _key_of(stem):
    m = DATE_PAT.match(stem)
    return (m.group(1) if m else stem).lower()


def parse_json(path):
    stem = Path(path).stem
    key = _key_of(stem)
    if any(ex in key for ex in EXCLUDE):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  解析json失败[{Path(path).name}]: {e}")
        return None
    n, hits, d = 0, [], {}
    if isinstance(data, list):
        hits, n = data, len(data)
    elif isinstance(data, dict):
        d = data
        for hk in ('hits', 'results', 'pool', 'records', 'data'):
            if isinstance(data.get(hk), list):
                hits = data[hk]; break
        n = data.get('n')
        if n is None:
            n = data.get('命中')
        if n is None:
            n = len(hits)
    samples = [s for s in (_sample(r) for r in hits[:3]) if s]
    return {'key': key, 'n': int(n or 0), 'samples': samples, 'stage': _stage_text(d)}


def parse_csv(path):
    stem = Path(path).stem
    key = _key_of(stem)
    if any(ex in key for ex in EXCLUDE):
        return None
    try:
        df = pd.read_csv(path, nrows=50)
        if df.empty:
            return {'key': key, 'n': 0, 'samples': [], 'stage': ''}
        nc = next((c for c in df.columns if c in ('名称', 'name')), None)
        cc = next((c for c in df.columns if c in ('代码', 'code', 'symbol')), None)
        samples = []
        for _, r in df.head(3).iterrows():
            nm = r[nc] if nc else ''; cd = str(r[cc]).split('.')[-1].zfill(6) if cc else '?'
            if nm or cc:
                samples.append(f"{nm}({cd})")
        # 行数-1 不准(只读了50行), 故 csv 命中数用全文件行数
        full = pd.read_csv(path, usecols=[0])
        return {'key': key, 'n': max(0, len(full)), 'samples': samples, 'stage': ''}
    except Exception as e:
        print(f"  解析csv失败[{Path(path).name}]: {e}")
        return None


def collect(files):
    """json 优先; 同 key 有 json 就不用 csv"""
    by_key = {}
    csvs = []
    for p in files:
        if p.endswith('.json'):
            r = parse_json(p)
            if r:
                by_key[r['key']] = r
        else:
            csvs.append(p)
    for p in csvs:
        r = parse_csv(p)
        if r and r['key'] not in by_key:
            by_key[r['key']] = r
    return list(by_key.values())


# ------------------ 推送 ------------------
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
        return requests.post(f"https://sctapi.ftqq.com/{key}.send",
                             data={"title": title, "desp": content}, timeout=15).json().get('code') == 0
    except Exception as e:
        print(f"  requests推送失败: {e}"); return False


def send_serverchan(title, content, sendkey=""):
    key = sendkey or SERVERCHAN_KEY
    if not key:
        return False
    LIMIT = 3800
    lines = content.split("\n")
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
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
    print(f"📲 汇总推送 {'✅' if ok else '⚠️失败'} ({len(chunks)}条)")
    return ok


def build_push(rows):
    today = today_bj()
    has = [r for r in rows if r['n'] > 0]
    empty = [r for r in rows if r['n'] == 0]
    idx = {k: i for i, k in enumerate(ORDER)}
    has.sort(key=lambda r: (idx.get(r['key'], 999), -r['n']))
    L = [f"**📊 矩阵日报 {today[5:]}** | 有货 {len(has)} 策略 · 空跑 {len(empty)} 策略",
         "*(仅汇总今日已上传产物的策略; 限流/跳过/未存json者不在此列, 见各job; 名称为信号样本非全部)*", ""]
    if has:
        L.append("### ✅ 今日有命中")
        for r in has:
            samp = ' / '.join(r['samples'][:3]) if r['samples'] else '—'
            L.append(f"- {label_of(r['key'])}: **{r['n']}只**{r['stage']} → {samp}")
        L.append("")
    if empty:
        L.append("### ⚪ 空跑(0命中)")
        L.append('、'.join(label_of(r['key']) for r in sorted(empty, key=lambda r: idx.get(r['key'], 999))))
    return "\n".join(L)


# ------------------ 主程序 ------------------
def main():
    print("=" * 70)
    print(f"📊 矩阵日报汇总 | 北京 {datetime.now(BJ):%Y-%m-%d %H:%M} | 模式={'本地' if ARTIFACT_DIR else 'API'}")
    print("=" * 70)
    dest = ARTIFACT_DIR if (ARTIFACT_DIR and os.path.isdir(ARTIFACT_DIR)) else os.path.join(os.environ.get('OUTPUT_DIR', 'output'), '_summary_tmp')
    os.makedirs(dest, exist_ok=True)
    try:
        if ARTIFACT_DIR and os.path.isdir(ARTIFACT_DIR):
            files = [str(p) for p in Path(dest).rglob('*') if p.suffix in ('.json', '.csv')]
            print(f"  本地读取 {len(files)} 个文件")
        else:
            files = fetch_today_artifacts(dest)
            print(f"  API 拉取今日 artifact 解压得 {len(files)} 个文件")
    except Exception as e:
        print(f"⚠️ 拉取artifact异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        files = []

    rows = collect(files)
    if not rows:
        msg = f"**📊 矩阵日报 {today_bj()[5:]}** | ⚠️ 今日未汇总到任何策略产物\n\n*(可能: 各job尚未跑完/限流空跑未存json/API拉取失败; 详见各job运行状态)*"
        print("  无汇总数据")
        if SERVERCHAN_KEY:
            send_serverchan(f"📊 矩阵日报 {today_bj()[5:]} | 无产物", msg)
        sys.exit(0)

    has_n = sum(r['n'] for r in rows if r['n'] > 0)
    print(f"  汇总 {len(rows)} 策略, 有货 {sum(1 for r in rows if r['n']>0)} 个, 合计命中 {has_n}")
    content = build_push(rows)
    print("\n" + content)

    tag = today_bj().replace('-', '')
    try:
        with open(os.path.join(os.environ.get('OUTPUT_DIR', 'output'), f"daily_summary_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": today_bj(), "strategies": rows}, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"  存盘异常: {e}")

    if SERVERCHAN_KEY:
        send_serverchan(f"📊 矩阵日报 {today_bj()[5:]} | 有货{sum(1 for r in rows if r['n']>0)}策略", content)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"⚠️ 汇总异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(0)
# >>>FILE_END_summary<<<
