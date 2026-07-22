#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场 / 全板块量化选股系统 (akshare 版) — 超卖反弹型
====================================================================
本文件由原 COX 固定 8 股脚本升级而来, 保持文件名 cox_sector_bot.py 不变,
以无缝兼容现有 workflow 的 cox-sector-bot job (密钥/artifact/cron 全复用)。

定位: 自下而上选个股 —— 超卖反弹技术信号 + 支撑/阻力 + 买卖评分 + 行业热度榜
      (与 mtf_resonance_screener 的"趋势共振型"互补, 不重叠)

架构: 行业热度榜(1次接口) + 全A实时初筛(1次接口) + 候选K线精算(N次, 有上限)
依赖: 仓库现有 requirements.txt 即可 (akshare / serverchan-sdk / tqdm)
兼容: Python 3.10
====================================================================
"""
import os, sys, json, time, random, logging, warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List
from enum import Enum

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, 'append'):  # 兼容旧 baostock
    def _df_append(self, other, ignore_index=False, **kw):
        o = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, o], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import akshare as ak
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
class Config:
    UNIVERSE = os.environ.get('UNIVERSE', 'ACTIVE')        # ACTIVE/HS300/ZZ500/ALL
    MAX_CANDIDATES = int(os.environ.get('MAX_CANDIDATES', '500'))
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '6'))
    FETCH_INDUSTRY_FOR_TOP = os.environ.get('FETCH_INDUSTRY_FOR_TOP', 'False').lower() == 'true'
    TOP_N = int(os.environ.get('TOP_N', '20'))

    TURNOVER_MIN = 1.0
    AMOUNT_MIN = 1.0e8
    CAP_MIN = 2.0e9
    CAP_MAX = 3.0e11
    PRICE_MIN = 2.0
    LIMIT_PCT = 9.5

    RSI_BUY, RSI_SELL = 35, 70
    BIAS_BUY, BIAS_SELL = -8.0, 8.0
    VOL_CONFIRM, ATR_STOP_MULT = 1.2, 2.5

    RETRY, RETRY_DELAY = 3, 4
    QUERY_TIMEOUT_SEC = 25

    SERVERCHAN_KEY = os.environ.get('SERVERCHAN_KEY', '') or os.environ.get('SENDKEY', '')
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')


# ==================== 数据模型 ====================
class SignalType(Enum):
    STRONG_BUY = "强烈买入"; BUY = "买入"; HOLD = "持有"
    SELL = "卖出"; STRONG_SELL = "强烈卖出"; NO_DATA = "数据不足"

class PositionSize(Enum):
    FULL = 1.0; HEAVY = 0.7; MEDIUM = 0.5; LIGHT = 0.3; EMPTY = 0.0

@dataclass
class StockSignal:
    date: str; symbol: str; name: str; close: float; signal: str; action: str
    position_size: str; signal_score: int; buy_score: int; sell_score: int
    confidence: float; stop_loss: float; take_profit: float; risk_reward_ratio: float
    atr: float; atr_pct: float; indicators: Dict; reasons: List[str]
    action_reasons: List[str]; support_levels: List[Dict]; resistance_levels: List[Dict]
    sector: str
    def to_dict(self): return asdict(self)


# ==================== 技术指标 ====================
class TI:
    @staticmethod
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    @staticmethod
    def sma(s, n): return s.rolling(n).mean()
    @staticmethod
    def rsi(c, n=14):
        d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        ag = g.ewm(alpha=1/n, adjust=False).mean(); al = l.ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100/(1+ag/al)
    @staticmethod
    def atr(df, n=14):
        tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift()).abs(),
                        (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()
    @staticmethod
    def macd(c, f=12, s=26, sig=9):
        ml = TI.ema(c, f)-TI.ema(c, s); sl = TI.ema(ml, sig); return ml, sl, ml-sl
    @staticmethod
    def bb(c, w=20, k=2.0):
        m = c.rolling(w).mean(); sd = c.rolling(w).std(); return m+k*sd, m, m-k*sd
    @staticmethod
    def kdj(h, l, c, n=9, m1=3, m2=3):
        rsv = (c-l.rolling(n).min())/(h.rolling(n).max()-l.rolling(n).min())*100
        k = rsv.ewm(alpha=1/m1, adjust=False).mean(); d = k.ewm(alpha=1/m2, adjust=False).mean()
        return k, d, 3*k-2*d


# ==================== 成交量分布 ====================
class VolumeProfile:
    def __init__(self, df, bs=1):
        d = df[["close", "volume"]].dropna().copy()
        d["b"] = (d["close"]/bs).round()*bs
        self.p = d.groupby("b")["volume"].sum().reset_index().sort_values("b").reset_index(drop=True)
    def poc(self): return self.p.sort_values("volume", ascending=False).iloc[0]['b']
    def va(self, pct=0.7):
        tv = self.p["volume"].sum()*pct; sp = self.p.sort_values("volume", ascending=False)
        cs = 0; vp = []
        for _, r in sp.iterrows():
            cs += r["volume"]; vp.append(r['b'])
            if cs >= tv: break
        return min(vp), max(vp)


# ==================== akshare 容错调用 ====================
def ak_retry(fn, *a, desc="", **kw):
    for i in range(1, Config.RETRY+1):
        try:
            r = fn(*a, **kw)
            if r is not None and not (hasattr(r, 'empty') and r.empty):
                return r
            logger.warning(f"  [{desc}] 第{i}次返回空, 重试...")
        except Exception as e:
            logger.warning(f"  [{desc}] 第{i}次异常: {e}")
        time.sleep(Config.RETRY_DELAY + random.uniform(0, 2))
    return None


def is_trading_day():
    try:
        df = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d'))
        return datetime.now().strftime('%Y-%m-%d') in dates
    except Exception as e:
        logger.warning(f"交易日历获取失败, 默认继续: {e}")
        return True


# ==================== 股票池初筛 (akshare) ====================
class Universe:
    CN2EN = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
             '最低': 'low', '成交量': 'volume', '成交额': 'amount',
             '涨跌幅': 'pct_chg', '换手率': 'turnover'}

    @classmethod
    def industry_heat(cls):
        df = ak_retry(ak.stock_board_industry_name_em, desc="行业板块")
        return df if (df is not None and not df.empty) else pd.DataFrame()

    @classmethod
    def snapshot_filter(cls):
        df = ak_retry(ak.stock_zh_a_spot_em, desc="全A快照")
        if df is None or df.empty:
            logger.error("全A快照获取失败(东财可能限流), 无法选股")
            return []
        logger.info(f"  全A快照原始 {len(df)} 只")
        if len(df) < 1000:
            logger.warning(f"  快照行数异常偏少({len(df)}), 接口可能受限")

        df['代码'] = df['代码'].astype(str)
        for c in ['最新价', '涨跌幅', '换手率', '量比', '总市值', '流通市值', '成交额']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        m_code = df['代码'].str.match(r'^(60|00|30|68)')
        m_st = ~df['名称'].str.contains('ST|退', na=False, regex=True)
        m_act = (df['最新价'] >= Config.PRICE_MIN) & (df['成交额'] >= Config.AMOUNT_MIN)
        m_turn = df['换手率'] >= Config.TURNOVER_MIN
        m_cap = (df['总市值'] >= Config.CAP_MIN) & (df['总市值'] <= Config.CAP_MAX)
        m_lim = df['涨跌幅'].abs() < Config.LIMIT_PCT
        out = df[m_code & m_st & m_act & m_turn & m_cap & m_lim].copy()
        logger.info(f"  初筛后 {len(out)} 只")

        if Config.UNIVERSE in ("HS300", "ZZ500"):
            idx = "000300" if Config.UNIVERSE == "HS300" else "000905"
            cons = ak_retry(lambda: ak.index_stock_cons_csindex(symbol=idx), desc=f"成分{idx}")
            if cons is not None and not cons.empty:
                code_col = '成分券代码' if '成分券代码' in cons.columns else cons.columns[0]
                valid = set(cons[code_col].astype(str).str.zfill(6))
                out = out[out['代码'].isin(valid)]
                logger.info(f"  与{Config.UNIVERSE}成分交集后 {len(out)} 只")

        out = out.sort_values('成交额', ascending=False)
        if Config.UNIVERSE != "ALL" and len(out) > Config.MAX_CANDIDATES:
            logger.info(f"  截断至成交额 Top {Config.MAX_CANDIDATES}")
            out = out.head(Config.MAX_CANDIDATES)

        return out[['代码', '名称', '最新价', '涨跌幅', '换手率', '成交额']].to_dict('records')

    @classmethod
    def fetch_hist(cls, symbol):
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now()-timedelta(days=420)).strftime("%Y%m%d")
        def _q(): return ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start, end_date=end, adjust="qfq")
        df = ak_retry(_q, desc=f"K线{symbol}")
        if df is None or len(df) < 60:
            return None
        df = df.rename(columns=cls.CN2EN)
        if 'close' not in df.columns:
            return None
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    @classmethod
    def fetch_industry(cls, symbol):
        info = ak_retry(lambda: ak.stock_individual_info_em(symbol=symbol), desc=f"行业{symbol}")
        if info is None or info.empty:
            return "—"
        try:
            row = info[info['item'] == '行业']
            return row.iloc[0]['value'] if not row.empty else "—"
        except Exception:
            return "—"


# ==================== 支撑/阻力 ====================
class Support:
    @staticmethod
    def calc(df):
        o = df.copy()
        for w in (5, 20, 60, 120, 250):
            o[f"ma{w}"] = TI.sma(o["close"], w)
        o["h20"] = o["high"].rolling(20).max(); o["l20"] = o["low"].rolling(20).min()
        m20 = TI.sma(o["close"], 20); sd = o["close"].rolling(20).std()
        o["bbu"] = m20+2*sd; o["bbl"] = m20-2*sd
        atr = TI.atr(o, 14); L = o.iloc[-1]; cur = L["close"]
        vp = VolumeProfile(o.tail(120), max(1, cur*0.01)); poc = vp.poc(); val, vah = vp.va()
        h120 = o["high"].rolling(120).max().iloc[-1]; l120 = o["low"].rolling(120).min().iloc[-1]
        fr = h120-l120
        lv = [("3倍ATR阻力", cur+3*atr.iloc[-1], "resist"), ("2倍ATR阻力", cur+2*atr.iloc[-1], "resist"),
              ("布林上轨", L["bbu"], "resist"), ("近期高20日", L["h20"], "resist"), ("斐波0%", h120, "resist"),
              ("当前价", cur, "current"),
              ("MA20", L["ma20"], "support"), ("MA60", L["ma60"], "support"), ("MA250", L["ma250"], "support"),
              ("近期低20日", L["l20"], "support"), ("2倍ATR支撑", cur-2*atr.iloc[-1], "support"),
              ("布林下轨", L["bbl"], "support"), ("斐波61.8%", h120-0.618*fr, "support"),
              ("斐波100%", l120, "support"), ("成交量POC", poc, "support")]
        seen = set(); uniq = []
        for lab, pr, t in sorted(lv, key=lambda x: x[1], reverse=True):
            p = round(pr, 2)
            if p not in seen and p > 0:
                seen.add(p)
                uniq.append({"name": lab, "price": p, "type": t,
                             "distance_pct": round((p-cur)/cur*100, 2)})
        return {"resistance_levels": [x for x in uniq if x["type"] == "resist"],
                "support_levels": [x for x in uniq if x["type"] == "support"],
                "atr": round(atr.iloc[-1], 4)}


# ==================== 信号引擎 ====================
class Engine:
    def calc(self, df):
        o = df.copy()
        for w in (5, 20, 60, 120, 250):
            o[f"ma{w}"] = TI.sma(o["close"], w)
        o["bias120"] = (o["close"]-o["ma120"])/o["ma120"]*100
        o["bias250"] = (o["close"]-o["ma250"])/o["ma250"]*100
        o["rsi14"] = TI.rsi(o["close"], 14)
        o["macd"], o["msig"], o["mhist"] = TI.macd(o["close"]); o["mhist_p"] = o["mhist"].shift(1)
        o["bbu"], o["bbm"], o["bbl"] = TI.bb(o["close"])
        denom = (o["bbu"]-o["bbl"]).replace(0, np.nan)
        o["bbp"] = (o["close"]-o["bbl"])/denom
        o["k"], o["d"], o["j"] = TI.kdj(o["high"], o["low"], o["close"])
        o["vr"] = o["volume"]/o["volume"].rolling(20).mean()
        o["atr"] = TI.atr(o, 14); o["atrp"] = o["atr"]/o["close"]*100
        return o

    def signal(self, df, symbol, name, sup, sector):
        o = self.calc(df); L = o.iloc[-1].copy(); cur = L['close']
        if any(pd.isna(L[c]) for c in ["ma250", "rsi14", "mhist", "vr", "atr"]):
            return None
        b = s = 0; reasons = []
        if L["close"] > L["ma250"]: b += 1
        else: s += 1
        if L["ma5"] > L["ma20"] > L["ma60"]: b += 2
        elif L["ma5"] < L["ma20"] < L["ma60"]: s += 2
        elif L["ma5"] > L["ma20"]: b += 1
        if L["bias120"] <= Config.BIAS_BUY: b += 2; reasons.append(f"BIAS120超卖({L['bias120']:.1f}%)")
        elif L["bias120"] >= Config.BIAS_SELL: s += 2
        if L["rsi14"] < Config.RSI_BUY: b += 2; reasons.append(f"RSI超卖({L['rsi14']:.0f})")
        elif L["rsi14"] > Config.RSI_SELL: s += 2; reasons.append(f"RSI超买({L['rsi14']:.0f})")
        bull = L["mhist"] > 0 and L["mhist"] > L["mhist_p"]
        bear = L["mhist"] < 0 and L["mhist"] < L["mhist_p"]
        if bull: b += 2; reasons.append("MACD动能增强")
        elif bear: s += 2; reasons.append("MACD动能减弱")
        if L["vr"] >= Config.VOL_CONFIRM:
            if b > s: b += 1; reasons.append(f"放量({L['vr']:.1f}x)")
            elif s > b: s += 1
        if pd.notna(L["bbp"]) and L["bbp"] < 0.1: b += 1
        elif pd.notna(L["bbp"]) and L["bbp"] > 0.9: s += 1

        sc = b-s
        if b >= 6 and sc >= 4: sig, pos, act = SignalType.STRONG_BUY, PositionSize.HEAVY, "BUY"
        elif b >= 5 and sc >= 2: sig, pos, act = SignalType.BUY, PositionSize.MEDIUM, "BUY"
        elif s >= 6 and sc <= -4: sig, pos, act = SignalType.STRONG_SELL, PositionSize.EMPTY, "SELL"
        elif s >= 5 and sc <= -2: sig, pos, act = SignalType.SELL, PositionSize.LIGHT, "SELL"
        else: sig, pos, act = SignalType.HOLD, PositionSize.MEDIUM, "HOLD"

        atr = L["atr"]
        sl = cur-Config.ATR_STOP_MULT*atr if act != "SELL" else cur+Config.ATR_STOP_MULT*atr
        tp = cur+2.5*Config.ATR_STOP_MULT*atr if act != "SELL" else cur-2.5*Config.ATR_STOP_MULT*atr
        risk = abs(cur-sl); rr = abs(tp-cur)/risk if risk > 0 else 0
        conf = min(abs(sc)/10*100, 95)
        ind = {"ma20": round(L["ma20"], 2), "ma60": round(L["ma60"], 2), "ma250": round(L["ma250"], 2),
               "rsi14": round(L["rsi14"], 2), "macd_hist": round(L["mhist"], 4),
               "vol_ratio": round(L["vr"], 2), "atr_pct": round(L["atrp"], 2)}
        return StockSignal(str(L['date'].date()), symbol, name, round(cur, 2), sig.value, act, pos.name,
                           sc, b, s, round(conf, 1), round(sl, 2), round(tp, 2), round(rr, 2),
                           round(atr, 2), round(atr/cur*100, 2), ind, [], reasons,
                           sup["support_levels"], sup["resistance_levels"], sector)


# ==================== 推送 (serverchan-sdk 优先, requests 兜底) ====================
class Notifier:
    def __init__(self): self.key = Config.SERVERCHAN_KEY
    def send(self, title, content):
        if not self.key:
            logger.info("未配置 SERVERCHAN_KEY, 跳过推送"); return False
        if len(content) > 4000:
            content = content[:3900] + "\n\n...(已截断, 详见 output 报告)"
        try:  # 签名 sc_send(sendkey, title, desp); SDK 自动适配 key 版本
            from serverchan_sdk import sc_send
            sc_send(self.key, title, content)
            logger.info("serverchan-sdk 推送成功"); return True
        except Exception as e:
            logger.warning(f"serverchan-sdk 失败, 回退 requests: {e}")
        try:
            r = requests.post(f"https://sctapi.ftqq.com/{self.key}.send",
                              data={"title": title, "desp": content}, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"requests 推送也失败: {e}"); return False


# ==================== 主程序 ====================
class Picker:
    def __init__(self):
        self.eng = Engine(); self.notif = Notifier()
        self.results: List[StockSignal] = []; self.heat = pd.DataFrame()

    def _one(self, item):
        sym, name = item['代码'], item['名称']
        try:
            df = Universe.fetch_hist(sym)
            if df is None or len(df) < 250:
                return None
            sup = Support.calc(df)
            return self.eng.signal(df, sym, name, sup, "—")
        except Exception:
            return None

    def run(self):
        logger.info("="*70)
        logger.info(f"全板块量化选股启动 | 范围={Config.UNIVERSE} | 并发={Config.MAX_WORKERS}")
        logger.info("="*70)

        if not is_trading_day():
            logger.info("今日非A股交易日, 跳过选股"); return

        self.heat = Universe.industry_heat()
        logger.info(f"行业热度榜: {len(self.heat)} 个板块")

        cands = Universe.snapshot_filter()
        if not cands:
            logger.error("无候选股票, 退出"); return

        logger.info(f"并发精算 {len(cands)} 只 ...")
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
            futs = {ex.submit(self._one, c): c for c in cands}
            for f in tqdm(as_completed(futs), total=len(futs), desc="选股精算"):
                r = f.result()
                if r is not None:
                    self.results.append(r)

        if Config.FETCH_INDUSTRY_FOR_TOP:
            tops = [r for r in self.results if r.action == "BUY"][:Config.TOP_N]
            for r in tqdm(tops, desc="补行业标签"):
                r.sector = Universe.fetch_industry(r.symbol)

        self.results.sort(key=lambda x: x.signal_score, reverse=True)
        logger.info(f"✅ 完成, 有效 {len(self.results)} 只")
        self._save(); self._notify(); self._print()
        return self.results

    def _save(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        with open(f"{Config.OUTPUT_DIR}/picker_results.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        with open(f"{Config.OUTPUT_DIR}/picker_report.md", "w", encoding="utf-8") as f:
            f.write(self._md())
        logger.info(f"报告已保存: {Config.OUTPUT_DIR}/picker_report.md")

    def _md(self):
        L = [f"# 🎯 全板块量化选股日报 (超卖反弹型)", "",
             f"**时间**: {datetime.now():%Y-%m-%d %H:%M}  |  **范围**: {Config.UNIVERSE}  |  **精算**: {len(self.results)} 只", ""]
        buy = [r for r in self.results if r.action == "BUY"]
        L += ["## 🏆 核心买入标的", "",
              "| # | 代码 | 名称 | 现价 | 信号 | 评分 | 仓位 | 止损/止盈 | 理由 |",
              "|---|------|------|------|------|------|------|-----------|------|"]
        if not buy:
            L.append("| - | 今日无符合买入条件标的, 建议观望 | | | | | | | |")
        for i, r in enumerate(buy[:Config.TOP_N], 1):
            L.append(f"| {i} | {r.symbol} | {r.name} | {r.close} | {r.signal} | {r.signal_score:+d} | "
                     f"{r.position_size} | {r.stop_loss}/{r.take_profit} | {'; '.join(r.action_reasons[:2]) or '-'} |")
        L.append("")
        if not self.heat.empty:
            L += ["## 🔥 今日行业热度榜 (实时)", "",
                  "| 板块 | 涨跌幅 | 上涨家数 | 下跌家数 | 领涨股 |",
                  "|------|--------|----------|----------|--------|"]
            for _, row in self.heat.head(15).iterrows():
                L.append(f"| {row.get('板块名称','')} | {row.get('涨跌幅','')}% | "
                         f"{row.get('上涨家数','')} | {row.get('下跌家数','')} | {row.get('领涨股票','')} |")
            L.append("")
        L += ["## ⚠️ 风险提示",
              "1. 纯技术指标量化打分, 不构成投资建议。",
              "2. 东财实时数据偶有延迟/异常, 交易前请人工复核 K 线。",
              "3. 单票仓位建议 ≤ 总资金 10%, 严格执行止损。"]
        return "\n".join(L)

    def _notify(self):
        buy = [r for r in self.results if r.action == "BUY"]
        strong = [r for r in buy if "强烈" in r.signal]
        title = f"【反弹选股】{len(strong)}只强烈买入 / {len(buy)}只买入" if buy else "【反弹选股】今日市场偏弱"
        c = [f"范围 {Config.UNIVERSE} | {datetime.now():%H:%M}", ""]
        if strong:
            c.append("### 🚀 强烈买入")
            for r in strong[:5]:
                c.append(f"**{r.name}({r.symbol})** 现价{r.close} 评分{r.signal_score:+d} | {'; '.join(r.action_reasons[:2])}")
            c.append("")
        if not self.heat.empty:
            c.append("### 🔥 行业热度 Top5")
            for _, row in self.heat.head(5).iterrows():
                c.append(f"- {row.get('板块名称','')} {row.get('涨跌幅','')}% 领涨{row.get('领涨股票','')}")
        c.append(f"\n*共精算 {len(self.results)} 只, 买入 {len(buy)} 只。详见 output 报告。*")
        self.notif.send(title, "\n".join(c))

    def _print(self):
        buy = [r for r in self.results if r.action == "BUY"]
        print("\n" + "="*85)
        print(f"🎯 全板块选股结果 (超卖反弹型) | {datetime.now():%Y-%m-%d} | {Config.UNIVERSE}".center(85))
        print("="*85)
        print(f"\n🏆 买入标的 Top {min(len(buy), Config.TOP_N)}:")
        print("─"*85)
        if not buy:
            print("  今日无符合买入条件标的, 建议观望。")
        for i, r in enumerate(buy[:Config.TOP_N], 1):
            print(f"  {i:2d}. {r.name}({r.symbol}) 现价{r.close:>7} {r.signal}({r.signal_score:+d}) "
                  f"仓位{r.position_size} 止损{r.stop_loss} 止盈{r.take_profit}")
            if r.action_reasons:
                print(f"      理由: {'; '.join(r.action_reasons[:3])}")
        if not self.heat.empty:
            print(f"\n🔥 行业热度 Top5:")
            print("─"*85)
            for _, row in self.heat.head(5).iterrows():
                print(f"  {row.get('板块名称',''):<10} {row.get('涨跌幅','')}% "
                      f"涨{row.get('上涨家数','')}/跌{row.get('下跌家数','')} 领涨{row.get('领涨股票','')}")
        print("\n" + "="*85)
        print("⚠️ 量化模型仅供参考, 请结合基本面自主决策, 严格止损。".center(85))
        print("="*85 + "\n")


def main():
    Picker().run()

if __name__ == "__main__":
    main()
