# -*- coding: utf-8 -*-
"""
mtf_full_screener.py —— 多周期双框架共振选股 · 增强版 v2.0
===============================================================
融合: easy_tdx捉妖大师 + Microsoft Qlib + vnpy + PandaFactor + QUANTAXIS
【本版修复】worker 子进程自动登录 baostock(原逻辑主进程取完列表即登出, 子进程 _bs_logged=False
导致全走 akshare, 在 Actions 上易失败); use_cache 默认关(Actions 环境缓存无用+多进程写SQLite易lock)。
"""
import os, re, sys, json, time, random, warnings, traceback, requests, sqlite3
import multiprocessing as mp
import concurrent.futures as cf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ====================== 0. 配置层 ======================
@dataclass
class Config:
    pair: str = "dw"
    short_score_min: float = 0.5
    long_score_min: float = 0.5
    short_bulls: int = 4
    long_bulls: int = 4
    resonance_weights: Dict[str, float] = field(default_factory=lambda: {
        "short_mom": 0.35, "mid_mom": 0.35, "long_mom": 0.20, "trend_strength": 0.10
    })
    adx_min: float = 20.0
    atr_max_pct: float = 0.06
    volume_confirm: bool = True
    use_rank: bool = True
    rank_top_pct: float = 0.15
    cache_dir: str = "cache"
    use_cache: bool = False
    cache_ttl_days: int = 1
    track_signals: bool = True
    track_days: List[int] = field(default_factory=lambda: [5, 10, 20])
    scan_limit: int = 0
    num_processes: int = 3
    sleep: float = 0.1
    fetch_timeout: int = 15
    ak_timeout: int = 25
    snapshot_pre: bool = True
    pre_amount_min: float = 5.0e7
    pre_turnover_min: float = 0.3
    keep_prefix: Tuple[str, ...] = ("0", "3", "6")
    exclude_name: Tuple[str, ...] = ("ST", "退")
    min_price: float = 3.0
    output_dir: str = "output"
    serverchan_key: str = ""
    push_top: int = 30
    cluster_top: int = 8
    hot_sector_top: int = 10
    hot_sector_min_pct: float = 1.0

# ====================== 1. 数据层 ======================
class DataManager:
    PAIR_CONFIG = {
        "dw": dict(short="daily", long="weekly", label="日+周", lookback_days=600,
                   min_s=80, min_l=40, short_ma=(20, 60), long_ma=(10, 30)),
        "mq": dict(short="monthly", long="quarterly", label="月+季", lookback_days=2100,
                   min_s=24, min_l=12, short_ma=(6, 12), long_ma=(4, 8)),
        "qy": dict(short="quarterly", long="yearly", label="季+年", lookback_days=4300,
                   min_s=12, min_l=8, short_ma=(4, 8), long_ma=(3, 5)),
    }
    RESAMPLE = {"daily": None, "weekly": "W-FRI", "monthly": "ME", 
                "quarterly": "QE", "yearly": "YE"}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg_dict = self.PAIR_CONFIG.get(cfg.pair, self.PAIR_CONFIG['dw'])
        self._bs_logged = False
        self._login_attempted = False
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self.db_path = os.path.join(cfg.cache_dir, "kline_cache.db")
        self._init_db()
        self._industry_map: Dict[str, str] = {}

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kline_cache (
                    code TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, updated_at TEXT,
                    PRIMARY KEY (code, date)
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_track (
                    code TEXT, signal_date TEXT, signal_price REAL,
                    check_date TEXT, return_5d REAL, return_10d REAL, return_20d REAL,
                    PRIMARY KEY (code, signal_date)
                )""")
            conn.commit()

    def _pref(self, c6: str) -> str:
        c = str(c6).split('.')[-1].zfill(6)
        return ('sh.' if c[:1] in ('6', '9') else 'sz.') + c

    def _clean_industry(self, s: str) -> str:
        if not s or not isinstance(s, str):
            return "—"
        return re.sub(r'^[A-Z]\d+\s*', '', s.strip()) or "—"

    def _bs_login_ok(self, retries=5):
        if self._bs_logged:
            return True
        try:
            import baostock as bs
        except ImportError:
            return False
        for i in range(retries):
            try:
                lg = bs.login()
                if getattr(lg, 'error_code', '1') == '0':
                    self._bs_logged = True
                    return True
            except Exception as e:
                print(f"  baostock登录异常: {e}")
            time.sleep(2 * (i + 1))
        return False

    def _ensure_login(self):
        """[修复] worker自动登录: 每个进程只尝试登录一次, 失败不反复重试, 回退akshare。"""
        if self._bs_logged:
            return True
        if self._login_attempted:
            return False
        self._login_attempted = True
        self._bs_logged = self._bs_login_ok()
        return self._bs_logged

    def _bs_logout(self):
        try:
            if self._bs_logged:
                import baostock as bs
                bs.logout()
        except Exception:
            pass
        finally:
            self._bs_logged = False

    def _call_with_timeout(self, fn, *a, timeout=25, **kw):
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(fn, *a, **kw).result(timeout=timeout)

    def _cache_read(self, code: str, days: int) -> Optional[pd.DataFrame]:
        if not self.cfg.use_cache:
            return None
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM kline_cache WHERE code=? AND date>=? ORDER BY date",
                conn, params=(code, cutoff))
        if df.empty or len(df) < 60:
            return None
        last_update = pd.to_datetime(df['updated_at'].max())
        if (datetime.now() - last_update).days > self.cfg.cache_ttl_days:
            return None
        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    def _cache_write(self, code: str, df: pd.DataFrame):
        if not self.cfg.use_cache or df is None or df.empty:
            return
        df = df.copy()
        df['code'] = code
        df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM kline_cache WHERE code=?", (code,))
            df[['code','date','open','high','low','close','volume','updated_at']].to_sql(
                'kline_cache', conn, if_exists='append', index=False)
            conn.commit()

    def fetch_hist(self, code: str) -> Optional[pd.DataFrame]:
        cached = self._cache_read(code, self.cfg_dict['lookback_days'])
        if cached is not None:
            return cached
        sd = (datetime.now() - timedelta(days=self.cfg_dict['lookback_days'])).strftime('%Y-%m-%d')
        ed = datetime.now().strftime('%Y-%m-%d')
        df = None
        self._ensure_login()
        if self._bs_logged:
            try:
                import baostock as bs
                d = bs.query_history_k_data_plus(
                    self._pref(code), "date,open,high,low,close,volume",
                    start_date=sd, end_date=ed, frequency="d", adjustflag="2").get_data()
                if d is not None and not d.empty and len(d) >= 60:
                    df = d
            except Exception:
                pass
        if df is None:
            try:
                import akshare as ak
                sym = code[3:] if len(code) > 3 and code[2] == '.' else code
                d = self._call_with_timeout(
                    ak.stock_zh_a_hist, symbol=sym, period="daily",
                    start_date=sd.replace('-', ''), end_date=ed.replace('-', ''),
                    adjust="qfq", timeout=self.cfg.ak_timeout)
                if d is not None and not d.empty:
                    d = d.rename(columns={
                        '日期':'date','开盘':'open','最高':'high',
                        '最低':'low','收盘':'close','成交量':'volume'})
                    df = d
            except Exception:
                pass
        if df is None or df.empty:
            return None
        for c in ['open','high','low','close','volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['close','volume'])
        df = df[df['volume'] > 0].sort_values('date').reset_index(drop=True)
        if len(df) < 60:
            return None
        df = df[['date','open','high','low','close','volume']]
        self._cache_write(code, df)
        return df

    def load_industry(self):
        if self._industry_map:
            return
        if self._bs_login_ok():
            try:
                import baostock as bs
                ind = bs.query_stock_industry().get_data()
                if ind is not None and not ind.empty:
                    for _, row in ind.iterrows():
                        self._industry_map[row['code']] = self._clean_industry(row['industry'])
            except Exception as e:
                print(f"  行业表异常: {e}")
            self._bs_logout()

    def get_industry(self, code: str) -> str:
        return self._industry_map.get(code, "—")

# ====================== 2. 因子层 ======================
class FactorEngine:
    @staticmethod
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @classmethod
    def calc_macd(cls, c: pd.Series):
        dif = cls.ema(c, 12) - cls.ema(c, 26)
        dea = cls.ema(dif, 9)
        return dif, dea, (dif - dea) * 2

    @classmethod
    def calc_rsi(cls, c: pd.Series, n=14):
        d = c.diff()
        g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1e-9))

    @classmethod
    def calc_kdj(cls, df: pd.DataFrame, n=9):
        ln = df['low'].rolling(n).min()
        hn = df['high'].rolling(n).max()
        rsv = (df['close'] - ln) / (hn - ln + 1e-12) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        return k, d, 3*k - 2*d

    @classmethod
    def calc_adx(cls, df: pd.DataFrame, n=14):
        h, l, c = df['high'], df['low'], df['close']
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        plus_dm = (h - h.shift(1)).clip(lower=0)
        minus_dm = (l.shift(1) - l).clip(lower=0)
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, 1e-9)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9) * 100
        adx = dx.ewm(alpha=1/n, adjust=False).mean()
        return adx, plus_di, minus_di

    @classmethod
    def calc_atr(cls, df: pd.DataFrame, n=14):
        h, l, c = df['high'], df['low'], df['close']
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()

    @classmethod
    def calc_momentum(cls, c: pd.Series, n: int) -> pd.Series:
        return (c / c.shift(n).replace(0, 1e-9) - 1) * 100

    @classmethod
    def calc_frame(cls, df: pd.DataFrame, ma_fast: int, ma_slow: int, cfg: Config) -> Optional[Dict]:
        if df is None or len(df) < max(ma_slow, 30):
            return None
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        dif, dea, hist = cls.calc_macd(c)
        macd_bull = bool(dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > 0)
        rsi_bull = bool(cls.calc_rsi(c).iloc[-1] > 50)
        k, d, j = cls.calc_kdj(df)
        kdj_bull = bool(k.iloc[-1] > d.iloc[-1] and j.iloc[-1] > 20)
        maf = c.rolling(ma_fast).mean()
        mas = c.rolling(ma_slow).mean()
        ma_bull = bool(c.iloc[-1] > maf.iloc[-1] and maf.iloc[-1] > mas.iloc[-1])
        bias_bull = bool(maf.iloc[-1] and (c.iloc[-1] - maf.iloc[-1]) / maf.iloc[-1] > 0)
        tp = (h + l + c) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-12)
        cci_bull = bool(cci.iloc[-1] > 0)
        hh = h.rolling(14).max()
        ll = l.rolling(14).min()
        wr = -100 * (hh - c) / (hh - ll + 1e-12)
        wr_bull = bool(wr.iloc[-1] > -50)
        fi = (c.diff() * v).ewm(span=13, adjust=False).mean()
        fi_bull = bool(fi.iloc[-1] > 0)
        bulls = sum([macd_bull, rsi_bull, kdj_bull, ma_bull, bias_bull, cci_bull, wr_bull, fi_bull])
        adx, plus_di, minus_di = cls.calc_adx(df)
        adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        trend_strong = adx_val >= cfg.adx_min
        atr = cls.calc_atr(df)
        atr_pct = atr.iloc[-1] / c.iloc[-1] if c.iloc[-1] > 0 else 1
        vol_ok = atr_pct < cfg.atr_max_pct
        mom_short = cls.calc_momentum(c, ma_fast).iloc[-1]
        mom_mid = cls.calc_momentum(c, ma_slow).iloc[-1]
        mom_long = cls.calc_momentum(c, ma_slow * 2).iloc[-1]
        mom_ok = mom_short > 0 and mom_mid > 0
        vol_confirm = True
        if cfg.volume_confirm:
            vol_ma = v.rolling(20).mean()
            vol_confirm = v.iloc[-1] > vol_ma.iloc[-1] * 0.8
        base_score = bulls / 8.0
        mom_bonus = 0.15 if mom_ok else 0
        trend_bonus = 0.1 if trend_strong else 0
        vol_bonus = 0.05 if vol_confirm else 0
        score = min(1.0, base_score + mom_bonus + trend_bonus + vol_bonus)
        trend = bool(c.iloc[-1] > maf.iloc[-1])
        return {
            "score": round(score, 4), "bulls": bulls, "trend": trend,
            "adx": round(adx_val, 2), "atr_pct": round(atr_pct, 4),
            "mom_short": round(mom_short, 2), "mom_mid": round(mom_mid, 2),
            "mom_long": round(mom_long, 2), "vol_confirm": vol_confirm,
            "trend_strong": trend_strong, "vol_ok": vol_ok,
        }

# ====================== 3. 信号层 ======================
class SignalEngine:
    def __init__(self, cfg: Config, dm: DataManager):
        self.cfg = cfg
        self.dm = dm
        self.cfg_dict = dm.cfg_dict
        self.factor = FactorEngine()
        self.resample = dm.RESAMPLE

    def resample_ohlcv(self, df: pd.DataFrame, rule) -> pd.DataFrame:
        if rule is None:
            return df.copy()
        return df.set_index('date').resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'}).dropna().reset_index()

    def evaluate_symbol(self, code: str) -> Optional[Dict]:
        df = self.dm.fetch_hist(code)
        if df is None or len(df) < 60:
            return {"__fail__": "数据不足"}
        time.sleep(self.cfg.sleep)
        short_df = self.resample_ohlcv(df, self.resample[self.cfg_dict['short']])
        long_df = self.resample_ohlcv(df, self.resample[self.cfg_dict['long']])
        if len(short_df) < self.cfg_dict['min_s'] or len(long_df) < self.cfg_dict['min_l']:
            return {"__fail__": "数据不足"}
        s = self.factor.calc_frame(short_df, *self.cfg_dict['short_ma'], self.cfg)
        l = self.factor.calc_frame(long_df, *self.cfg_dict['long_ma'], self.cfg)
        if s is None or l is None:
            return {"__fail__": "计算失败"}
        short_ok = s['score'] >= self.cfg.short_score_min and s['bulls'] >= self.cfg.short_bulls
        long_ok = l['score'] >= self.cfg.long_score_min and l['bulls'] >= self.cfg.long_bulls and l['trend']
        if not (short_ok and long_ok):
            return {"__fail__": "非双周期共振"}
        if not s['vol_ok'] or not l['vol_ok']:
            return {"__fail__": "波动率过高"}
        if not s['trend_strong'] and not l['trend_strong']:
            return {"__fail__": "趋势太弱"}
        resonance_score = (
            self.cfg.resonance_weights.get("short_mom", 0.35) * max(0, s['mom_short']) / 100 +
            self.cfg.resonance_weights.get("mid_mom", 0.35) * max(0, l['mom_mid']) / 100 +
            self.cfg.resonance_weights.get("long_mom", 0.20) * max(0, l['mom_long']) / 100 +
            self.cfg.resonance_weights.get("trend_strength", 0.10) * min(l['adx'], 50) / 50
        )
        final_score = (s['score'] + l['score']) / 2 * 100
        return {
            "代码": code, "名称": "", "行业": "",
            "最新价": round(float(df['close'].iloc[-1]), 2),
            "信号价": round(float(df['close'].iloc[-1]), 2),
            "信号日期": pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d'),
            "短分": round(s['score'] * 100, 1), "短多头": s['bulls'],
            "短ADX": s['adx'], "短ATR%": round(s['atr_pct'] * 100, 2), "短动量": s['mom_short'],
            "长分": round(l['score'] * 100, 1), "长多头": l['bulls'],
            "长ADX": l['adx'], "长ATR%": round(l['atr_pct'] * 100, 2), "长动量": l['mom_mid'],
            "共振分": round(resonance_score * 100, 1), "score": round(final_score, 1),
            "resonance": False, "resonance_sector": "",
        }

    def rank_filter(self, results: List[Dict]) -> List[Dict]:
        if not self.cfg.use_rank or len(results) < 10:
            return results
        df = pd.DataFrame(results)
        df['score_rank'] = df['score'].rank(pct=True)
        df['resonance_rank'] = df['共振分'].rank(pct=True)
        df['composite_rank'] = (df['score_rank'] * 0.6 + df['resonance_rank'] * 0.4)
        df['composite_pct'] = df['composite_rank'].rank(pct=True)
        keep = df[df['composite_pct'] >= (1 - self.cfg.rank_top_pct)]
        return keep.sort_values('composite_pct', ascending=False).to_dict('records')

# ====================== 4. 板块层 ======================
class SectorAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hot_sectors: List[Tuple[str, float]] = []

    def fetch_hot_sectors(self):
        try:
            import akshare as ak
            heat = None
            for i in range(3):
                try:
                    heat = ak.stock_board_industry_name_em()
                    if heat is not None and not heat.empty:
                        break
                except Exception:
                    time.sleep(2 + i)
            if heat is not None and not heat.empty and '板块名称' in heat.columns:
                h = heat.copy()
                h['_chg'] = pd.to_numeric(h['涨跌幅'], errors='coerce')
                h = h[h['_chg'] >= self.cfg.hot_sector_min_pct].sort_values('_chg', ascending=False)
                self.hot_sectors = [
                    (str(r['板块名称']), round(float(r['_chg']), 2))
                    for _, r in h.head(self.cfg.hot_sector_top).iterrows()]
        except Exception as e:
            print(f"  板块数据异常: {e}")

    def calc_sector_cluster(self, results: List[Dict]) -> List[Tuple[str, int]]:
        labeled = [r for r in results if r.get('行业') not in ('—', '未知', '', None)]
        if not labeled:
            return []
        counts = pd.Series([r['行业'] for r in labeled]).value_counts().head(self.cfg.cluster_top)
        return [(n, int(c)) for n, c in counts.items()]

    def enrich_resonance(self, results: List[Dict], dm: DataManager) -> List[Dict]:
        hot_names = [n for n, _ in self.hot_sectors]
        for r in results:
            sec = r.get('行业', '')
            r['resonance'] = False
            r['resonance_sector'] = ""
            if sec and sec not in ('—', '未知', '') and hot_names:
                s = sec.strip()
                for hh in hot_names:
                    if hh and (hh == s or hh in s or s in hh):
                        r['resonance'] = True
                        r['resonance_sector'] = hh
                        break
        return results

# ====================== 5. 实时价层 ======================
class PriceAligner:
    @staticmethod
    def fetch_realtime_tencent(codes: List[str]) -> Dict[str, float]:
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

    @staticmethod
    def align_suffix(r: Dict, spot_now: Dict) -> str:
        sig_price = r.get('信号价', r.get('最新价'))
        sig_date = r.get('信号日期')
        if sig_price is None or pd.isna(sig_price):
            return ""
        head = f"信号{sig_price}"
        if sig_date and not pd.isna(sig_date):
            head += f"@{str(sig_date)[:10][-5:]}"
        code6 = str(r.get('代码', '')).split('.')[-1].zfill(6)
        now = spot_now.get(code6)
        if now is not None:
            try:
                chg = (now - float(sig_price)) / float(sig_price) * 100
                return f" | {head} -> 现价{now}@run({chg:+.1f}%)"
            except Exception:
                return f" | {head}"
        return f" | {head}"

# ====================== 6. 信号追踪层 ======================
class SignalTracker:
    def __init__(self, cfg: Config, dm: DataManager):
        self.cfg = cfg
        self.dm = dm
        self.db_path = dm.db_path

    def record_signal(self, code: str, signal_date: str, signal_price: float):
        if not self.cfg.track_signals:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO signal_track (code, signal_date, signal_price, check_date) VALUES (?, ?, ?, ?)",
                    (code, signal_date, signal_price, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
        except Exception:
            pass

    def update_returns(self):
        if not self.cfg.track_signals:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM signal_track WHERE return_20d IS NULL", conn)
                if df.empty:
                    return
                for _, row in df.iterrows():
                    code = row['code']
                    sig_date = row['signal_date']
                    sig_price = row['signal_price']
                    hist = self.dm.fetch_hist(code)
                    if hist is None or hist.empty:
                        continue
                    hist = hist[hist['date'] >= sig_date].reset_index(drop=True)
                    if len(hist) < 2:
                        continue
                    returns = {}
                    for d in self.cfg.track_days:
                        if len(hist) > d:
                            ret = (hist['close'].iloc[d] - sig_price) / sig_price * 100
                            returns[f'return_{d}d'] = round(ret, 2)
                    if returns:
                        set_clause = ", ".join([f"{k}=?" for k in returns.keys()])
                        vals = list(returns.values()) + [code, sig_date]
                        conn.execute(f"UPDATE signal_track SET {set_clause} WHERE code=? AND signal_date=?", vals)
                conn.commit()
        except Exception as e:
            print(f"  信号追踪更新异常: {e}")

    def get_signal_stats(self) -> Dict:
        if not self.cfg.track_signals:
            return {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM signal_track WHERE return_5d IS NOT NULL", conn)
                if df.empty:
                    return {}
                stats = {}
                for d in self.cfg.track_days:
                    col = f'return_{d}d'
                    if col in df.columns:
                        stats[f'{d}日胜率'] = round((df[col] > 0).mean() * 100, 1)
                        stats[f'{d}日均收益'] = round(df[col].mean(), 2)
                        stats[f'{d}日最大回撤'] = round(df[col].min(), 2)
                return stats
        except Exception:
            return {}

# ====================== 7. 推送层 ======================
class PushNotifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _send_one(self, title: str, content: str, key: str) -> bool:
        try:
            from serverchan_sdk import sc_send
            ret = sc_send(key, title, content)
            ok = (ret.get('code', ret.get('errno', -1)) == 0) if isinstance(ret, dict) else (ret if isinstance(ret, bool) else ret not in (None, False))
            if ok:
                return True
        except Exception as e:
            print(f"  sdk失败回退requests: {e}")
        try:
            j = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                            data={"title": title, "desp": content}, timeout=15).json()
            return j.get('code') == 0
        except Exception as e:
            print(f"  requests推送失败: {e}")
            return False

    def send(self, title: str, content: str):
        key = self.cfg.serverchan_key
        if not key:
            return False
        LIMIT = 3800
        chunks, cur, cur_len = [], [], 0
        newline = chr(10)
        for ln in content.split(newline):
            lnlen = len(ln) + 1
            if cur_len + lnlen > LIMIT and cur:
                chunks.append(newline.join(cur))
                cur, cur_len = [], 0
            cur.append(ln)
            cur_len += lnlen
        if cur:
            chunks.append(newline.join(cur))
        chunks = chunks or [""]
        ok = True
        for i, ch in enumerate(chunks):
            t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
            ok = self._send_one(t, ch, key) and ok
            if i < len(chunks) - 1:
                time.sleep(1)
        print("推送完成" + (" ✅" if ok else " ⚠️存在失败"))
        return ok

# ====================== 8. 主控层 ======================
class ScreenerMaster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dm = DataManager(cfg)
        self.signal_engine = SignalEngine(cfg, self.dm)
        self.sector_analyzer = SectorAnalyzer(cfg)
        self.price_aligner = PriceAligner()
        self.tracker = SignalTracker(cfg, self.dm)
        self.pusher = PushNotifier(cfg)
        self.fail_stats = {"抓取失败": 0, "数据不足": 0, "非双周期共振": 0,
                          "波动率过高": 0, "趋势太弱": 0, "计算失败": 0}
        os.makedirs(cfg.output_dir, exist_ok=True)

    def snapshot_prefilter(self, codes_with_prefix: List[str]) -> List[str]:
        if not self.cfg.snapshot_pre:
            return codes_with_prefix
        try:
            import akshare as ak
            spot = self.dm._call_with_timeout(ak.stock_zh_a_spot_em, timeout=25)
            if spot is None or spot.empty or '代码' not in spot.columns:
                return codes_with_prefix
            spot['代码'] = spot['代码'].astype(str).str.zfill(6)
            for col in ['最新价', '成交额', '换手率']:
                if col in spot.columns:
                    spot[col] = pd.to_numeric(spot[col], errors='coerce')
            m = (spot['代码'].str.startswith(self.cfg.keep_prefix)
                 & ~spot['名称'].astype(str).str.contains("|".join(self.cfg.exclude_name), na=False, regex=True)
                 & (spot['最新价'] >= self.cfg.min_price))
            if '成交额' in spot.columns:
                m &= (spot['成交额'] >= self.cfg.pre_amount_min)
            if '换手率' in spot.columns:
                m &= (spot['换手率'] >= self.cfg.pre_turnover_min)
            keep = set(spot.loc[m, '代码'])
            out = [c for c in codes_with_prefix if c[3:] in keep]
            print(f"  快照预筛: {len(codes_with_prefix)} -> {len(out)} 只")
            return out if out else codes_with_prefix
        except Exception as e:
            print(f"  快照预筛失败, 退化全扫: {e}")
            return codes_with_prefix

    def load_stock_list(self) -> Tuple[List[str], Dict[str, str]]:
        import akshare as ak
        stock_df = pd.DataFrame()
        if self.dm._bs_login_ok():
            try:
                self.dm.load_industry()
                import baostock as bs
                stock_df = bs.query_stock_basic().get_data()
            except Exception as e:
                print(f"  baostock取列表异常: {e}")
            self.dm._bs_logout()
        if stock_df is None or stock_df.empty or 'code' not in stock_df.columns:
            for attempt in range(3):
                try:
                    d = ak.stock_info_a_code_name()
                    if d is not None and not d.empty and 'code' in d.columns:
                        nc = 'name' if 'name' in d.columns else d.columns[1]
                        d = d[['code', nc]].copy()
                        d.columns = ['code', 'code_name']
                        d['code'] = d['code'].astype(str).str.zfill(6)
                        d['code'] = d['code'].apply(lambda c: ('sh.' if c[:1] in ('6', '9') else 'sz.') + c)
                        d['type'] = '1'
                        d['status'] = '1'
                        stock_df = d
                        break
                except Exception as e:
                    print(f"  akshare列表第{attempt+1}次失败: {e}")
                time.sleep(2 + attempt)
        if stock_df is None or stock_df.empty:
            print("无股票列表")
            return [], {}
        stock_df = stock_df[
            stock_df['code'].str.startswith(('sh.', 'sz.')) &
            (stock_df['type'] == '1') &
            (stock_df['status'] == '1')
        ].copy()
        stock_df = stock_df[~stock_df['code_name'].astype(str).str.contains('ST|退', na=False, regex=True)]
        codes = self.snapshot_prefilter(stock_df['code'].tolist())
        if self.cfg.scan_limit and len(codes) > self.cfg.scan_limit:
            codes = codes[:self.cfg.scan_limit]
        name_map = dict(zip(stock_df['code'], stock_df['code_name']))
        return codes, name_map

    def _process_one(self, args) -> Optional[Dict]:
        code, name = args
        try:
            res = self.signal_engine.evaluate_symbol(code)
            if res is None:
                return None
            if "__fail__" in res:
                self.fail_stats[res["__fail__"]] = self.fail_stats.get(res["__fail__"], 0) + 1
                return None
            res['名称'] = name
            res['行业'] = self.dm.get_industry(code)
            return res
        except cf.TimeoutError:
            self.fail_stats["抓取失败"] += 1
            return None
        except Exception:
            self.fail_stats["抓取失败"] += 1
            return None

    def run_scan(self) -> pd.DataFrame:
        self.fail_stats = {k: 0 for k in self.fail_stats}
        codes, name_map = self.load_stock_list()
        if not codes:
            return pd.DataFrame()
        tasks = [(c, name_map.get(c, "")) for c in codes]
        results = []
        print(f"开始多周期双共振[{self.dm.cfg_dict['label']}]扫描 {len(tasks)} 只"
              f"（{self.cfg.num_processes}进程, 回看{self.dm.cfg_dict['lookback_days']}天）...")
        def init_worker():
            time.sleep(random.uniform(0, 2))
        with mp.Pool(processes=self.cfg.num_processes, initializer=init_worker) as pool:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="双共振", unit="只")
            for res in pool.imap_unordered(self._process_one, tasks):
                if res and "__fail__" not in res:
                    results.append(res)
                    pbar.write(f"  √ {res['代码']} {res['名称']} "
                               f"短{res['短分']}%({res['短多头']}/8) "
                               f"长{res['长分']}%({res['长多头']}/8) "
                               f"ADX{res.get('短ADX', 0)}/{res.get('长ADX', 0)}")
                pbar.update(1)
                pbar.set_postfix(命中=len(results), 失败=sum(self.fail_stats.values()))
            pbar.close()
        if self.cfg.use_rank:
            print(f"\n截面RANK过滤 (取前{self.cfg.rank_top_pct*100:.0f}%)...")
            results = self.signal_engine.rank_filter(results)
            print(f"  RANK后保留 {len(results)} 只")
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
        return df

    def enrich_and_push(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        self.sector_analyzer.fetch_hot_sectors()
        cluster = self.sector_analyzer.calc_sector_cluster(df.to_dict('records'))
        targets = df.to_dict('records')
        targets = self.sector_analyzer.enrich_resonance(targets, self.dm)
        df = pd.DataFrame(targets)
        df = df.copy()
        if '信号价' not in df.columns:
            df['信号价'] = df['最新价']
        codes6 = [str(c).split('.')[-1].zfill(6) for c in df['代码']]
        rt = self.price_aligner.fetch_realtime_tencent(codes6)
        if rt:
            df['实时价'] = [rt.get(c) for c in codes6]
            df['最新价'] = df['实时价'].where(df['实时价'].notna(), df['最新价'])
        if self.cfg.track_signals:
            for _, r in df.iterrows():
                self.tracker.record_signal(r['代码'], r['信号日期'], r['信号价'])
            self.tracker.update_returns()
            sig_stats = self.tracker.get_signal_stats()
            if sig_stats:
                print("\n信号历史统计:")
                for k, v in sig_stats.items():
                    print(f"  {k}: {v}")
        df = df.sort_values(['resonance', 'score'], ascending=[False, False]).reset_index(drop=True)
        tag = datetime.now().strftime("%Y%m%d")
        pair = self.cfg.pair
        try:
            df.to_csv(os.path.join(self.cfg.output_dir, f"mtf_full_{pair}_{tag}.csv"),
                     index=False, encoding="utf-8-sig")
            with open(os.path.join(self.cfg.output_dir, f"mtf_full_{pair}_{tag}.json"),
                     'w', encoding='utf-8') as f:
                json.dump({
                    "date": tag, "pair": pair, "cluster": cluster,
                    "n": int(len(df)), "fail_stats": self.fail_stats,
                    "signal_stats": self.tracker.get_signal_stats() if self.cfg.track_signals else {},
                    "hits": df.to_dict('records')
                }, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n已存 {self.cfg.output_dir}/mtf_full_{pair}_{tag}.*")
        except Exception as e:
            print(f"\n存盘异常: {e}")
        try:
            disp = df.copy()
            disp.insert(2, '板块', [
                ('->' + r.get('resonance_sector', '')) if r.get('resonance') else (r.get('行业') or '—')
                for r in disp.to_dict('records')
            ])
            drop_cols = ['行业', 'resonance', 'resonance_sector', '实时价', '信号价', 'score']
            disp = disp.drop(columns=[c for c in drop_cols if c in disp.columns], errors='ignore')
            print("\n" + disp.head(self.cfg.push_top).to_string(index=False))
        except Exception as e:
            print(f"展示异常: {e}")
        if self.cfg.serverchan_key:
            try:
                n_reso = int(df['resonance'].sum()) if 'resonance' in df.columns else 0
                title = f"增强双共振[{self.dm.cfg_dict['label']}] 命中{len(df)}只 风口{n_reso}"
                content = self._build_push(df, cluster, self.sector_analyzer.hot_sectors, rt)
                self.pusher.send(title, content)
            except Exception as e:
                print(f"推送异常: {e}")

    def _build_push(self, df: pd.DataFrame, cluster: List, hot: List, spot_now: Dict) -> str:
        cfg_dict = self.dm.cfg_dict
        reso = df[df['resonance'] == True] if 'resonance' in df.columns else pd.DataFrame()
        newline = chr(10)
        L = [
            f"**增强多周期双共振({cfg_dict['label']})** | 命中{len(df)}只 风口{len(reso)} (现价=实时价)",
            f"*(短>={self.cfg.short_score_min*100:.0f}%且>={self.cfg.short_bulls}/8 × "
            f"长>={self.cfg.long_score_min*100:.0f}%且>={self.cfg.long_bulls}/8+ADX>={self.cfg.adx_min}+ATR过滤; "
            f"趋势确认非预测, 必止损)*",
            ""
        ]
        if hot:
            L.append("风口: " + "、".join(f"{n}({c}%)" for n, c in hot[:6]))
            L.append("")
        if cluster:
            L.append("共振板块: " + "、".join(f"{n}({c})" for n, c in cluster))
            L.append("")
        if self.cfg.track_signals:
            stats = self.tracker.get_signal_stats()
            if stats:
                L.append("历史信号表现: " + " | ".join(f"{k}:{v}" for k, v in list(stats.items())[:4]))
                L.append("")
        def line(r):
            sec_tag = ('->' + r['resonance_sector']) if r.get('resonance') else (r.get('行业') or '—')
            return (f"- **{r['名称']}({r['代码']})** [{sec_tag}] 现价{r['最新价']} | "
                    f"短{r['短分']}%({r['短多头']}/8) 长{r['长分']}%({r['长多头']}/8) "
                    f"ADX{r.get('短ADX', 0)}/{r.get('长ADX', 0)} 均分{r['score']}"
                    f"{self.price_aligner.align_suffix(r, spot_now)}")
        if not reso.empty:
            L.append(f"### 共振遇风口 共{len(reso)}只")
            L += [line(r) for _, r in reso.head(self.cfg.push_top).iterrows()]
            L.append("")
        L.append(f"### 全部双共振 共{len(df)}只")
        L += [line(r) for _, r in df.head(self.cfg.push_top).iterrows()]
        if len(df) > self.cfg.push_top:
            L.append(f"{newline}*…另有 {len(df)-self.cfg.push_top} 只, 详见 output*")
        return newline.join(L)

    def run(self):
        print("=" * 70)
        print(f"增强多周期双共振[{self.dm.cfg_dict['label']}] | "
              f"{datetime.now():%Y-%m-%d %H:%M} | "
              f"回看{self.dm.cfg_dict['lookback_days']}天 | "
              f"进程{self.cfg.num_processes}")
        print(f"短>={self.cfg.short_score_min*100:.0f}%且>={self.cfg.short_bulls}/8 × "
              f"长>={self.cfg.long_score_min*100:.0f}%且>={self.cfg.long_bulls}/8+ADX>={self.cfg.adx_min}+ATR过滤")
        print(f"截面RANK: {'开' if self.cfg.use_rank else '关'} | "
              f"信号追踪: {'开' if self.cfg.track_signals else '关'} | "
              f"缓存: {'开' if self.cfg.use_cache else '关'}")
        print("=" * 70)
        df = self.run_scan()
        print("\n各失败原因统计：")
        for k, v in self.fail_stats.items():
            if v:
                print(f"  {k}: {v}")
        if df is None or df.empty:
            print(f"\n本次未发现 [{self.dm.cfg_dict['label']}] 双周期共振票(门槛严, 0命中属正常)。")
            if self.cfg.serverchan_key:
                self.pusher.send(
                    f"增强双共振[{self.dm.cfg_dict['label']}] | 0命中",
                    f"**增强多周期双共振[{self.dm.cfg_dict['label']}]** | 本次无同时满足短+长双框架的票。")
            return
        self.enrich_and_push(df)

# ====================== 9. 入口 ======================
def main():
    cfg = Config(
        pair=os.environ.get('PAIR', 'dw'),
        short_score_min=float(os.environ.get('SHORT_SCORE_MIN', '0.5')),
        long_score_min=float(os.environ.get('LONG_SCORE_MIN', '0.5')),
        short_bulls=int(os.environ.get('SHORT_BULLS', '4')),
        long_bulls=int(os.environ.get('LONG_BULLS', '4')),
        scan_limit=int(os.environ.get('SCAN_LIMIT', '0')),
        num_processes=int(os.environ.get('NUM_PROCESSES', '3')),
        sleep=float(os.environ.get('SLEEP', '0.1')),
        fetch_timeout=int(os.environ.get('FETCH_TIMEOUT', '15')),
        ak_timeout=int(os.environ.get('AK_TIMEOUT', '25')),
        snapshot_pre=os.environ.get('SNAPSHOT_PRE', '1').strip() in ('1', 'true', 'True'),
        pre_amount_min=float(os.environ.get('PRE_AMOUNT_MIN', '5.0e7')),
        pre_turnover_min=float(os.environ.get('PRE_TURNOVER_MIN', '0.3')),
        min_price=float(os.environ.get('MIN_PRICE', '3.0')),
        output_dir=os.environ.get('OUTPUT_DIR', 'output'),
        serverchan_key=os.environ.get('SERVERCHAN_KEY') or os.environ.get('SENDKEY', ''),
        push_top=int(os.environ.get('PUSH_TOP', '30')),
        cluster_top=int(os.environ.get('CLUSTER_TOP', '8')),
        hot_sector_top=int(os.environ.get('HOT_SECTOR_TOP', '10')),
        hot_sector_min_pct=float(os.environ.get('HOT_SECTOR_MIN_PCT', '1.0')),
        adx_min=float(os.environ.get('ADX_MIN', '20.0')),
        atr_max_pct=float(os.environ.get('ATR_MAX_PCT', '0.06')),
        use_rank=os.environ.get('USE_RANK', '1').strip() in ('1', 'true', 'True'),
        rank_top_pct=float(os.environ.get('RANK_TOP_PCT', '0.15')),
        track_signals=os.environ.get('TRACK_SIGNALS', '1').strip() in ('1', 'true', 'True'),
        use_cache=os.environ.get('USE_CACHE', '0').strip() in ('1', 'true', 'True'),
    )
    master = ScreenerMaster(cfg)
    master.run()
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_mtf_full<<<
