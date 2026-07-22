#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场 / 全板块量化选股系统 (akshare + baostock 双源版) — 超卖反弹型
====================================================================
本版增强(在双源+扩大初筛+板块标注+推送精简 之上):
  5. 板块共振打星⭐: 当买入标的「所属行业」命中当日行业热度风口时,
     打 ⭐ 标记 + 排序温和加分(默认+1, 不改 signal 判定, 仅同档优先/边界提档)
     风口定义: 行业热度榜涨幅降序 Top N 且涨幅 > 阈值(普跌日不乱打星)
     匹配: 精确优先 + 双向子串兜底; 仅对买入类打星(卖出/持有不适用)
     注意: 共振依赖补标签, 故 LABEL_TOP 应 >= TOP_N (默认60>=30, 已满足),
           以保证展示Top内的票都带板块且共振判断准确
保持文件名 cox_sector_bot.py 不变, 兼容现有 workflow。
依赖: 仓库现有 requirements.txt (akshare / baostock / serverchan-sdk / tqdm)
兼容: Python 3.10
====================================================================
"""
import os, sys, json, time, random, logging, warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kw):
        o = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, o], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import akshare as ak
import baostock as bs
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
class Config:
    UNIVERSE = os.environ.get('UNIVERSE', 'ACTIVE')
    MAX_CANDIDATES = int(os.environ.get('MAX_CANDIDATES', '1500'))
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '3'))

    TURNOVER_MIN = float(os.environ.get('TURNOVER_MIN', '0.5'))
    AMOUNT_MIN = float(os.environ.get('AMOUNT_MIN', '5e7'))
    CAP_MIN = float(os.environ.get('CAP_MIN', '1e9'))
    CAP_MAX = float(os.environ.get('CAP_MAX', '3e11'))
    PRICE_MIN = float(os.environ.get('PRICE_MIN', '1.5'))
    LIMIT_PCT = 9.5

    TOP_N = int(os.environ.get('TOP_N', '30'))
    PUSH_TOP = int(os.environ.get('PUSH_TOP', '8'))
    LABEL_TOP = int(os.environ.get('LABEL_TOP', '60'))   # 应 >= TOP_N

    # 板块共振(打星+加分)
    HOT_SECTOR_TOP = int(os.environ.get('HOT_SECTOR_TOP', '10'))        # 热度榜取前N为风口
    HOT_SECTOR_MIN_PCT = float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0'))  # 涨幅门槛(普跌日防乱打星)
    RESONANCE_BONUS = int(os.environ.get('RESONANCE_BONUS', '1'))       # 共振排序加分(设0=纯打星不加分)

    RSI_BUY, RSI_SELL = 35, 70
    BIAS_BUY, BIAS_SELL = -8.0, 8.0
    VOL_CONFIRM, ATR_STOP_MULT = 1.2, 2.5

    RETRY, RETRY_DELAY = 5, 6
    BS_RETRY = 3

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
    resonance: bool = False          # 是否命中板块共振(风口+超卖)
    resonance_sector: str = ""       # 命中的风口板块名
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


# ==================== baostock 兜底客户端 (单线程) ====================
class BsClient:
    _logged = False
    @classmethod
    def login(cls):
        for i in range(1, Config.BS_RETRY+1):
            try:
                lg = bs.login()
                if getattr(lg, 'error_code', '1') == '0':
                    cls._logged = True
                    logger.info(f"  baostock 登录成功 (第{i}次)")
                    return True
                logger.warning(f"  baostock 登录失败: {getattr(lg,'error_msg','')}")
            except Exception as e:
                logger.warning(f"  baostock 登录异常: {e}")
            time.sleep(2 + random.uniform(0, 1))
        return False
    @classmethod
    def logout(cls):
        try:
            if cls._logged: bs.logout()
        except Exception: pass
        cls._logged = False
    @classmethod
    def fetch(cls, symbol, start_dash, end_dash, adjustflag):
        bs_code = f"sh.{symbol}" if symbol.startswith(('6', '9')) else f"sz.{symbol}"
        for i in range(2):
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume",
                    start_date=start_dash, end_date=end_dash, frequency="d", adjustflag=adjustflag)
                df = rs.get_data()
                if df is None or df.empty:
                    return None
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                return df if len(df) >= 60 else None
            except Exception as e:
                logger.debug(f"  baostock {symbol} 第{i+1}次失败: {e}")
                time.sleep(1 + random.uniform(0, 1))
        return None


# ==================== 股票池初筛 (东财) ====================
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
    def fetch_hist_em(cls, symbol):
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now()-timedelta(days=420)).strftime("%Y%m%d")
        def _q(): return ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start, end_date=end, adjust="qfq")
        df = ak_retry(_q, desc=f"东财K线{symbol}")
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
            row = info[info['item'].isin(['行业', '所属行业'])]
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


# ==================== 推送 ====================
class Notifier:
    def __init__(self): self.key = Config.SERVERCHAN_KEY
    def send(self, title, content):
        if not self.key:
            logger.info("未配置 SERVERCHAN_KEY, 跳过推送"); return False
        if len(content) > 4000:
            content = content[:3900] + "\n\n...(已截断, 详见 output 报告)"
        try:
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


# ==================== 主程序 (三阶段编排 + 板块标注 + 共振打星) ====================
class Picker:
    def __init__(self):
        self.eng = Engine(); self.notif = Notifier()
        self.results: List[StockSignal] = []
        self.heat = pd.DataFrame()
        self.hot: List[Tuple[str, float]] = []   # 当日风口 [(板块名, 涨幅%), ...]

    def _compute(self, symbol, name, df):
        try:
            sup = Support.calc(df)
            return self.eng.signal(df, symbol, name, sup, "—")
        except Exception:
            return None

    def _hot_sectors(self) -> List[Tuple[str, float]]:
        """当日风口板块: 热度榜按涨幅降序, 取 Top N 且涨幅 > 门槛"""
        if self.heat.empty or '板块名称' not in self.heat.columns or '涨跌幅' not in self.heat.columns:
            return []
        h = self.heat.copy()
        h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
        h = h[h['_chg'] >= Config.HOT_SECTOR_MIN_PCT].sort_values('_chg', ascending=False)
        return [(str(row['板块名称']), round(float(row['_chg']), 2))
                for _, row in h.head(Config.HOT_SECTOR_TOP).iterrows()]

    @staticmethod
    def _match_sector(sector: str, hot_names: List[str]) -> str:
        """个股行业 vs 风口板块: 精确优先 + 双向子串兜底; 命中返回风口名, 否则空串"""
        if not sector or sector == '—' or not hot_names:
            return ""
        s = sector.strip()
        for h in hot_names:
            if h and h == s:
                return h
        for h in hot_names:
            if h and (h in s or s in h):
                return h
        return ""

    def _label_sectors(self):
        """对排序后买入类标的并发补板块(行业)标签; 仅补会展示的, 控接口量"""
        buy_like = [r for r in self.results if r.action == "BUY"]
        targets = buy_like[:Config.LABEL_TOP]
        if not targets:
            return
        def _q(r):
            try:
                r.sector = Universe.fetch_industry(r.symbol)
            except Exception:
                r.sector = "—"
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
            list(tqdm(ex.map(_q, targets), total=len(targets), desc="补板块标签"))
        logger.info(f"  已为 {len(targets)} 只标的补板块标签 (其余标 —)")

    def _apply_resonance(self):
        """板块共振: 买入类且行业命中风口 -> 打星 + 排序加分; 然后终排序"""
        hot_names = [n for n, _ in self.hot]
        cnt = 0
        for r in self.results:
            if r.action == "BUY":   # 含强烈买入(act=BUY); 卖出/持有不打星
                matched = self._match_sector(r.sector, hot_names)
                if matched:
                    r.resonance = True
                    r.resonance_sector = matched
                    cnt += 1
        # 终排序: 信号分 + 共振加分(温和, 不改signal); 同分共振优先
        self.results.sort(key=lambda x: (x.signal_score + (Config.RESONANCE_BONUS if x.resonance else 0),
                                         x.signal_score), reverse=True)
        logger.info(f"  风口板块 {len(hot_names)} 个, 买入标的共振打星 {cnt} 只 (加分+{Config.RESONANCE_BONUS})")

    def run(self):
        logger.info("="*70)
        logger.info(f"全板块选股(双源+共振) 启动 | 范围={Config.UNIVERSE} | 精算上限={Config.MAX_CANDIDATES} | 并发={Config.MAX_WORKERS}")
        logger.info(f"初筛阈值: 换手≥{Config.TURNOVER_MIN} 成交额≥{Config.AMOUNT_MIN:.0e} 市值≥{Config.CAP_MIN:.0e} 价≥{Config.PRICE_MIN}")
        logger.info(f"共振: 风口Top{Config.HOT_SECTOR_TOP} 涨幅≥{Config.HOT_SECTOR_MIN_PCT}% 加分+{Config.RESONANCE_BONUS}")
        logger.info("="*70)

        if not is_trading_day():
            logger.info("今日非A股交易日, 跳过选股"); return

        self.heat = Universe.industry_heat()
        logger.info(f"行业热度榜: {len(self.heat)} 个板块")
        self.hot = self._hot_sectors()
        logger.info(f"当日风口: {', '.join(f'{n}({c}%)' for n, c in self.hot) or '(无, 普跌或数据缺失)'}")

        cands = Universe.snapshot_filter()
        if not cands:
            logger.error("无候选股票, 退出"); return
        name_map = {c['代码']: c['名称'] for c in cands}

        # 阶段A: 东财并发取 K 线
        em_ok, em_fail = {}, []
        logger.info(f"[阶段A] 东财并发取数 {len(cands)} 只 ...")
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
            futs = {ex.submit(Universe.fetch_hist_em, c['代码']): c['代码'] for c in cands}
            for f in tqdm(as_completed(futs), total=len(futs), desc="东财K线"):
                sym = futs[f]; df = f.result()
                if df is not None and len(df) >= 250:
                    em_ok[sym] = df
                else:
                    em_fail.append(sym)
        logger.info(f"  东财成功 {len(em_ok)} / 失败 {len(em_fail)}")

        # 阶段B: baostock 串行兜底 (前复权 adjustflag=2)
        if em_fail:
            logger.info(f"[阶段B] baostock 兜底补 {len(em_fail)} 只 ...")
            if BsClient.login():
                sd = (datetime.now()-timedelta(days=420)).strftime("%Y-%m-%d")
                ed = datetime.now().strftime("%Y-%m-%d")
                for sym in tqdm(em_fail, desc="baostock兜底", unit="只"):
                    df = BsClient.fetch(sym, sd, ed, "2")
                    if df is not None and len(df) >= 250:
                        em_ok[sym] = df
                BsClient.logout()
            else:
                logger.warning("  baostock 登录失败, 跳过兜底")
        logger.info(f"  双源合计有效 K 线 {len(em_ok)} 只")

        # 阶段C: 纯计算并发
        logger.info(f"[阶段C] 计算信号 {len(em_ok)} 只 ...")
        items = [(sym, name_map.get(sym, sym), df) for sym, df in em_ok.items()]
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
            futs = [ex.submit(self._compute, sym, name, df) for sym, name, df in items]
            for f in tqdm(as_completed(futs), total=len(futs), desc="算信号"):
                r = f.result()
                if r is not None:
                    self.results.append(r)

        self.results.sort(key=lambda x: x.signal_score, reverse=True)  # 初排序(决定补标签集合)

        # 阶段D: 补板块标签 (仅对会展示的买入类, 并发)
        logger.info(f"[阶段D] 补板块标签 ...")
        self._label_sectors()

        # 阶段E: 板块共振打星 + 终排序
        logger.info(f"[阶段E] 板块共振打星 ...")
        self._apply_resonance()

        logger.info(f"✅ 完成, 有效 {len(self.results)} 只")
        self._save(); self._notify(); self._print()
        return self.results

    @staticmethod
    def _sec_tag(r: StockSignal) -> str:
        """展示用板块标签: 共振加⭐"""
        return f"⭐{r.resonance_sector}" if r.resonance else (r.sector or "—")

    def _save(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        with open(f"{Config.OUTPUT_DIR}/picker_results.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        with open(f"{Config.OUTPUT_DIR}/picker_report.md", "w", encoding="utf-8") as f:
            f.write(self._md())
        logger.info(f"报告已保存: {Config.OUTPUT_DIR}/picker_report.md")

    def _md(self):
        buy = [r for r in self.results if r.action == "BUY"]
        reso = [r for r in buy if r.resonance]
        L = [f"# 🎯 全板块量化选股日报 (超卖反弹型·双源·共振)", "",
             f"**时间**: {datetime.now():%Y-%m-%d %H:%M}  |  **范围**: {Config.UNIVERSE}  |  **有效**: {len(self.results)} 只  |  **买入**: {len(buy)} 只  |  **⭐共振**: {len(reso)} 只", ""]
        if self.hot:
            L.append(f"> 🌪️ **当日风口板块**: " + "、".join(f"{n}({c}%)" for n, c in self.hot) +
                     f"  ｜ 其中 **{len(reso)}** 只买入标的与风口共振⭐ (超卖+风口=反弹助推, 优先关注)")
            L.append("")
        L += [f"## 🏆 核心买入标的 (Top {min(len(buy), Config.TOP_N)}, ⭐=风口共振)", "",
              "| # | 代码 | 名称 | 板块 | 现价 | 信号 | 评分 | 仓位 | 止损/止盈 | 理由 |",
              "|---|------|------|------|------|------|------|------|-----------|------|"]
        if not buy:
            L.append("| - | 今日无符合买入条件标的, 建议观望 | | | | | | | | |")
        for i, r in enumerate(buy[:Config.TOP_N], 1):
            L.append(f"| {i} | {r.symbol} | {r.name} | {self._sec_tag(r)} | {r.close} | {r.signal} | {r.signal_score:+d} | "
                     f"{r.position_size} | {r.stop_loss}/{r.take_profit} | {'; '.join(r.action_reasons[:2]) or '-'} |")
        if len(buy) > Config.TOP_N:
            L.append(f"\n*…另有 {len(buy)-Config.TOP_N} 只买入标的, 见 picker_results.json*")
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
              "2. ⭐共振=「超卖信号」+「所属行业在当日风口」的交叉提示, 非买卖改判; 板块名为东财「所属行业」(非概念板块)。",
              "3. 单票仓位建议 ≤ 总资金 10%, 严格执行止损。"]
        return "\n".join(L)

    def _notify(self):
        buy = [r for r in self.results if r.action == "BUY"]
        strong = [r for r in buy if "强烈" in r.signal]
        normal = [r for r in buy if "强烈" not in r.signal]
        reso = [r for r in buy if r.resonance]
        P = Config.PUSH_TOP
        title = (f"【反弹选股】⭐共振{len(reso)} 强烈{len(strong)}/买入{len(buy)}"
                 if buy else "【反弹选股】今日市场偏弱")
        c = [f"范围 {Config.UNIVERSE} | 有效{len(self.results)} | {datetime.now():%H:%M}", ""]
        if self.hot:
            c.append(f"🌪️ **风口**: " + "、".join(f"{n}({c}%)" for n, c in self.hot[:6]))
            c.append(f"⭐ **风口共振买入 {len(reso)} 只** (超卖+风口, 优先关注)")
            c.append("")
        if reso:
            c.append(f"### ⭐ 风口共振 Top{min(len(reso), P)}")
            for r in reso[:P]:
                c.append(f"**{r.name}({r.symbol})** [⭐{r.resonance_sector}] 现价{r.close} 评分{r.signal_score:+d} | {'; '.join(r.action_reasons[:2])}")
            if len(reso) > P:
                c.append(f"*…另有 {len(reso)-P} 只共振, 详见报告*")
            c.append("")
        if strong:
            c.append(f"### 🚀 强烈买入 Top{min(len(strong), P)}")
            for r in strong[:P]:
                c.append(f"**{r.name}({r.symbol})** [{self._sec_tag(r)}] 现价{r.close} 评分{r.signal_score:+d} | {'; '.join(r.action_reasons[:2])}")
            if len(strong) > P:
                c.append(f"*…另有 {len(strong)-P} 只强烈买入, 详见报告*")
            c.append("")
        if normal and not reso:  # 有共振时强烈/普通已含星, 普通买入可省略以免过长; 无共振时仍展示
            c.append(f"### 🟢 买入 Top{min(len(normal), P)}")
            for r in normal[:P]:
                c.append(f"**{r.name}({r.symbol})** [{self._sec_tag(r)}] 现价{r.close} 评分{r.signal_score:+d} | {'; '.join(r.action_reasons[:2])}")
            c.append("")
        if not self.heat.empty and not self.hot:
            c.append("### 🔥 行业热度 Top5")
            for _, row in self.heat.head(5).iterrows():
                c.append(f"- {row.get('板块名称','')} {row.get('涨跌幅','')}% 领涨{row.get('领涨股票','')}")
        c.append(f"\n*有效 {len(self.results)} 只, 买入 {len(buy)} 只, ⭐共振 {len(reso)} 只。完整列表见 output 报告。*")
        self.notif.send(title, "\n".join(c))

    def _print(self):
        buy = [r for r in self.results if r.action == "BUY"]
        reso = [r for r in buy if r.resonance]
        print("\n" + "="*92)
        print(f"🎯 全板块选股 (超卖反弹·双源·共振) | {datetime.now():%Y-%m-%d} | {Config.UNIVERSE}".center(92))
        print("="*92)
        if self.hot:
            print(f"\n🌪️ 当日风口: " + "、".join(f"{n}({c}%)" for n, c in self.hot))
            print(f"⭐ 风口共振买入 {len(reso)} 只 (超卖+风口, 优先关注)")
        print(f"\n🏆 买入标的 Top {min(len(buy), Config.TOP_N)}:")
        print("─"*92)
        if not buy:
            print("  今日无符合买入条件标的, 建议观望。")
        for i, r in enumerate(buy[:Config.TOP_N], 1):
            star = "⭐" if r.resonance else "  "
            print(f"  {i:2d}.{star}{r.name}({r.symbol}) [{self._sec_tag(r)}] 现价{r.close:>7} {r.signal}({r.signal_score:+d}) "
                  f"仓位{r.position_size} 止损{r.stop_loss} 止盈{r.take_profit}")
            if r.action_reasons:
                print(f"      理由: {'; '.join(r.action_reasons[:3])}")
        if len(buy) > Config.TOP_N:
            print(f"  …另有 {len(buy)-Config.TOP_N} 只买入标的, 见报告")
        if not self.heat.empty:
            print(f"\n🔥 行业热度 Top5:")
            print("─"*92)
            for _, row in self.heat.head(5).iterrows():
                print(f"  {row.get('板块名称',''):<10} {row.get('涨跌幅','')}% "
                      f"涨{row.get('上涨家数','')}/跌{row.get('下跌家数','')} 领涨{row.get('领涨股票','')}")
        print("\n" + "="*92)
        print("⚠️ 量化模型仅供参考, ⭐共振为交叉提示非改判, 请结合基本面自主决策, 严格止损。".center(92))
        print("="*92 + "\n")


def main():
    Picker().run()

if __name__ == "__main__":
    main()
