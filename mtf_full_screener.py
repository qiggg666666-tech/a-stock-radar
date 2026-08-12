# -*- coding: utf-8 -*-
"""
MTF Resonance Screener Pro v3.1 (矩阵修复版)
修复: ①补sqlite3/requests导入 ②去Pydantic依赖(改dataclass) ③baostock查询超时 ④多进程PicklingError ⑤run_scan重写+会话复用
矩阵接入: 新增 SCAN_OFFSET 环境变量支持，配合 SCAN_LIMIT 实现分段扫描。
依赖: numpy pandas pyarrow akshare baostock requests tqdm (无需pydantic)
"""
from __future__ import annotations
import os, re, sys, json, time, random, warnings, logging, functools, sqlite3, requests
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Protocol, runtime_checkable, Union
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

# ====================== 0. 配置层 (dataclass, 免pydantic) ======================
class PairEnum(str, Enum):
    DW = "dw"; MQ = "mq"; QY = "qy"
class LogLevel(str, Enum):
    DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"

@dataclass(frozen=True)
class Settings:
    pair: PairEnum = PairEnum.DW
    short_score_min: float = 0.5
    long_score_min: float = 0.5
    short_bulls: int = 4
    long_bulls: int = 4
    weight_short_mom: float = 0.35
    weight_mid_mom: float = 0.35
    weight_long_mom: float = 0.20
    weight_trend: float = 0.10
    adx_min: float = 20.0
    atr_max_pct: float = 0.06
    volume_confirm: bool = True
    use_rank: bool = True
    rank_top_pct: float = 0.15
    cache_dir: str = "cache"
    use_cache: bool = False
    cache_ttl_days: int = 1
    track_signals: bool = True
    track_days: Tuple[int, ...] = (5, 10, 20)
    scan_limit: int = 0
    scan_offset: int = 0  # 矩阵新增: 分段扫描偏移量
    num_processes: int = max(1, mp.cpu_count() - 1)
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
    log_level: LogLevel = LogLevel.INFO
    serverchan_key: str = ""
    push_top: int = 30
    cluster_top: int = 8
    hot_sector_top: int = 10
    hot_sector_min_pct: float = 1.0
    @property
    def resonance_weights(self):
        return {"short_mom": self.weight_short_mom, "mid_mom": self.weight_mid_mom,
                "long_mom": self.weight_long_mom, "trend_strength": self.weight_trend}
    @classmethod
    def from_env(cls):
        def _s(k, d): return os.environ.get(k, d)
        def _f(k, d):
            v = os.environ.get(k); return d if v is None else float(v)
        def _i(k, d):
            v = os.environ.get(k); return d if v is None else int(v)
        def _b(k, d):
            v = os.environ.get(k); return d if v is None else v.strip().lower() in ("1", "true", "yes")
        wsum = _f("W_SHORT_MOM", 0.35) + _f("W_MID_MOM", 0.35) + _f("W_LONG_MOM", 0.20) + _f("W_TREND", 0.10)
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"共振权重之和必须等于1.0, 当前={wsum}")
        return cls(pair=PairEnum(_s("PAIR", "dw")), short_score_min=_f("SHORT_SCORE_MIN", 0.5),
            long_score_min=_f("LONG_SCORE_MIN", 0.5), short_bulls=_i("SHORT_BULLS", 4), long_bulls=_i("LONG_BULLS", 4),
            weight_short_mom=_f("W_SHORT_MOM", 0.35), weight_mid_mom=_f("W_MID_MOM", 0.35),
            weight_long_mom=_f("W_LONG_MOM", 0.20), weight_trend=_f("W_TREND", 0.10),
            adx_min=_f("ADX_MIN", 20.0), atr_max_pct=_f("ATR_MAX_PCT", 0.06),
            volume_confirm=_b("VOLUME_CONFIRM", True), use_rank=_b("USE_RANK", True), rank_top_pct=_f("RANK_TOP_PCT", 0.15),
            cache_dir=_s("CACHE_DIR", "cache"), use_cache=_b("USE_CACHE", False), cache_ttl_days=_i("CACHE_TTL_DAYS", 1),
            track_signals=_b("TRACK_SIGNALS", True), scan_limit=_i("SCAN_LIMIT", 0),
            scan_offset=_i("SCAN_OFFSET", 0),
            num_processes=_i("NUM_PROCESSES", max(1, mp.cpu_count() - 1)), sleep=_f("SLEEP", 0.1),
            fetch_timeout=_i("FETCH_TIMEOUT", 15), ak_timeout=_i("AK_TIMEOUT", 25),
            snapshot_pre=_b("SNAPSHOT_PRE", True), pre_amount_min=_f("PRE_AMOUNT_MIN", 5.0e7),
            pre_turnover_min=_f("PRE_TURNOVER_MIN", 0.3), min_price=_f("MIN_PRICE", 3.0),
            output_dir=_s("OUTPUT_DIR", "output"), log_level=LogLevel(_s("LOG_LEVEL", "INFO").upper()),
            serverchan_key=_s("SERVERCHAN_KEY", ""), push_top=_i("PUSH_TOP", 30),
            cluster_top=_i("CLUSTER_TOP", 8), hot_sector_top=_i("HOT_SECTOR_TOP", 10),
            hot_sector_min_pct=_f("HOT_SECTOR_MIN_PCT", 1.0))

def setup_logging(level):
    logger = logging.getLogger("mtf_screener"); logger.setLevel(level.value)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger

# ====================== 2. 数据模型 ======================
@dataclass(frozen=True)
class FrameMetrics:
    score: float; bulls: int; trend: bool; adx: float; atr_pct: float
    mom_short: float; mom_mid: float; mom_long: float
    vol_confirm: bool; trend_strong: bool; vol_ok: bool
@dataclass(frozen=True)
class SignalResult:
    code: str; name: str; industry: str
    latest_price: float; signal_price: float; signal_date: str
    short_score: float; short_bulls: int; short_adx: float; short_atr_pct: float; short_mom: float
    long_score: float; long_bulls: int; long_adx: float; long_atr_pct: float; long_mom: float
    resonance_score: float; composite_score: float
    resonance: bool = False; resonance_sector: str = ""
    def to_dict(self): return asdict(self)
@dataclass
class ScanFailure:
    code: str; reason: str

# ====================== 3. 数据层 ======================
class ParquetCache:
    def __init__(self, cache_dir, ttl_days):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True); self.ttl_days = ttl_days
    def _path(self, code):
        subdir = self.cache_dir / f"{hash(code) % 256:02x}"; subdir.mkdir(exist_ok=True)
        return subdir / f"{code.replace('.', '_')}.parquet"
    def read(self, code, min_rows=60):
        path = self._path(code)
        if not path.exists(): return None
        if (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days > self.ttl_days: return None
        try: df = pd.read_parquet(path)
        except Exception: return None
        if len(df) < min_rows: return None
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close", "volume"]).query("volume > 0").sort_values("date").reset_index(drop=True)
    def write(self, code, df):
        if df is None or df.empty: return
        d = df.copy(); d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        try: d.to_parquet(self._path(code), index=False)
        except Exception: pass

class BaostockProvider:
    def __init__(self, logger, fetch_timeout=15):
        self.logger = logger; self.fetch_timeout = fetch_timeout; self._logged_in = False; self._bs = None
        try:
            import baostock as bs; self._bs = bs
        except ImportError: self.logger.warning("baostock 未安装，跳过")
    def login(self):
        if self._bs is None or self._logged_in: return self._logged_in
        for attempt in range(1, 4):
            try:
                lg = self._bs.login()
                if getattr(lg, "error_code", "1") == "0": self._logged_in = True; return True
            except Exception as e: self.logger.warning(f"baostock 登录第{attempt}次失败: {e}")
            time.sleep(2 * attempt)
        return False
    def logout(self):
        if self._bs and self._logged_in:
            try: self._bs.logout()
            except Exception: pass
            finally: self._logged_in = False
    def _pref(self, code):
        c6 = str(code).split(".")[-1].zfill(6)
        return ("sh." if c6[:1] in ("6", "9") else "sz.") + c6
    def fetch(self, code, start, end):
        if self._bs is None or not self.login(): return None
        def _do():
            rs = self._bs.query_history_k_data_plus(self._pref(code), "date,open,high,low,close,volume",
                start_date=start, end_date=end, frequency="d", adjustflag="2")
            return rs.get_data()
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do)
            try: data = fut.result(timeout=self.fetch_timeout)
            except Exception:
                self._logged_in = False; return None
        if data is None or data.empty or len(data) < 60: return None
        return data

class AkshareProvider:
    def __init__(self, logger, timeout=25):
        self.logger = logger; self.timeout = timeout; self._ak = None
        try:
            import akshare as ak; self._ak = ak
        except ImportError: self.logger.warning("akshare 未安装，跳过")
    def login(self): return self._ak is not None
    def logout(self): pass
    def fetch(self, code, start, end):
        if self._ak is None: return None
        sym = code[3:] if len(code) > 3 and code[2] == "." else code
        try:
            df = self._ak.stock_zh_a_hist(symbol=sym, period="daily",
                start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
        except Exception: return None
        if df is None or df.empty: return None
        rm = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        return df.rename(columns={k: v for k, v in rm.items() if k in df.columns})

class DataManager:
    PAIR_CONFIG = {
        "dw": dict(short="daily", long="weekly", label="日+周", lookback_days=600, min_s=80, min_l=40, short_ma=(20, 60), long_ma=(10, 30)),
        "mq": dict(short="monthly", long="quarterly", label="月+季", lookback_days=2100, min_s=24, min_l=12, short_ma=(6, 12), long_ma=(4, 8)),
        "qy": dict(short="quarterly", long="yearly", label="季+年", lookback_days=4300, min_s=12, min_l=8, short_ma=(4, 8), long_ma=(3, 5)),
    }
    RESAMPLE_RULE = {"daily": None, "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "yearly": "YE"}
    def __init__(self, cfg, logger):
        self.cfg = cfg; self.logger = logger; self.cfg_dict = self.PAIR_CONFIG[cfg.pair.value]
        self.cache = ParquetCache(cfg.cache_dir, cfg.cache_ttl_days) if cfg.use_cache else None
        self.bs_provider = BaostockProvider(logger, cfg.fetch_timeout)
        self.ak_provider = AkshareProvider(logger, cfg.ak_timeout)
        self._industry_map: Dict[str, str] = {}
    def _clean_industry(self, s):
        if not s or not isinstance(s, str): return "—"
        return re.sub(r"^[A-Z]\d+\s*", "", s.strip()) or "—"
    def load_industry(self):
        if self._industry_map: return
        if self.bs_provider.login():
            try:
                import baostock as bs
                ind = bs.query_stock_industry().get_data()
                if ind is not None and not ind.empty:
                    self._industry_map = {row["code"]: self._clean_industry(row.get("industry", "")) for _, row in ind.iterrows()}
            except Exception as e: self.logger.warning(f"行业表获取异常: {e}")
            finally: self.bs_provider.logout()
    def get_industry(self, code): return self._industry_map.get(code, "—")
    def fetch_hist(self, code):
        sd = (datetime.now() - timedelta(days=self.cfg_dict["lookback_days"])).strftime("%Y-%m-%d")
        ed = datetime.now().strftime("%Y-%m-%d")
        if self.cache:
            cached = self.cache.read(code, min_rows=60)
            if cached is not None: return cached
        df = None
        try: df = self.bs_provider.fetch(code, sd, ed)
        except Exception: pass
        if df is None:
            try: df = self.ak_provider.fetch(code, sd, ed)
            except Exception: pass
        if df is None or df.empty: return None
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["close", "volume"]).query("volume > 0").sort_values("date").reset_index(drop=True)
        if len(df) < 60: return None
        df = df[["date", "open", "high", "low", "close", "volume"]]
        if self.cache: self.cache.write(code, df)
        return df
    def resample_ohlcv(self, df, rule):
        if rule is None: return df.copy()
        return (df.set_index("date").resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index())

# ====================== 4. 因子层 ======================
class FactorEngine:
    @staticmethod
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    @classmethod
    def calc_macd(cls, c):
        dif = cls.ema(c, 12) - cls.ema(c, 26); dea = cls.ema(dif, 9); return dif, dea, (dif - dea) * 2
    @classmethod
    def calc_rsi(cls, c, n=14):
        d = c.diff(); g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1e-9))
    @classmethod
    def calc_kdj(cls, df, n=9):
        ln = df["low"].rolling(n).min(); hn = df["high"].rolling(n).max()
        rsv = (df["close"] - ln) / (hn - ln + 1e-12) * 100
        k = rsv.ewm(com=2, adjust=False).mean(); d = k.ewm(com=2, adjust=False).mean(); return k, d, 3*k - 2*d
    @classmethod
    def _calc_tr(cls, df):
        h, l, c = df["high"], df["low"], df["close"]
        return pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    @classmethod
    def calc_adx(cls, df, n=14):
        tr = cls._calc_tr(df); h, l = df["high"], df["low"]
        pdm = (h - h.shift(1)).clip(lower=0); mdm = (l.shift(1) - l).clip(lower=0)
        pdm = pdm.where(pdm > mdm, 0); mdm = mdm.where(mdm > pdm, 0)
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, 1e-9)
        mdi = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, 1e-9)
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9) * 100
        return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, mdi
    @classmethod
    def calc_atr(cls, df, n=14): return cls._calc_tr(df).ewm(alpha=1/n, adjust=False).mean()
    @classmethod
    def calc_momentum(cls, c, n): return (c / c.shift(n).replace(0, 1e-9) - 1) * 100
    @classmethod
    def calc_frame(cls, df, ma_fast, ma_slow, cfg):
        if df is None or len(df) < max(ma_slow, 30): return None
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        dif, dea, hist = cls.calc_macd(c)
        macd_bull = bool(dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > 0)
        rsi_bull = bool(cls.calc_rsi(c).iloc[-1] > 50)
        k, d, j = cls.calc_kdj(df); kdj_bull = bool(k.iloc[-1] > d.iloc[-1] and j.iloc[-1] > 20)
        maf = c.rolling(ma_fast).mean(); mas = c.rolling(ma_slow).mean()
        ma_bull = bool(c.iloc[-1] > maf.iloc[-1] > mas.iloc[-1])
        bias = (c.iloc[-1] - maf.iloc[-1]) / maf.iloc[-1] if maf.iloc[-1] != 0 else -1
        bias_bull = bool(bias > 0)
        tp = (h + l + c) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-12); cci_bull = bool(cci.iloc[-1] > 0)
        hh, ll = h.rolling(14).max(), l.rolling(14).min()
        wr = -100 * (hh - c) / (hh - ll + 1e-12); wr_bull = bool(wr.iloc[-1] > -50)
        fi = (c.diff() * v).ewm(span=13, adjust=False).mean(); fi_bull = bool(fi.iloc[-1] > 0)
        bulls = sum([macd_bull, rsi_bull, kdj_bull, ma_bull, bias_bull, cci_bull, wr_bull, fi_bull])
        adx, _, _ = cls.calc_adx(df)
        adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        trend_strong = adx_val >= cfg.adx_min
        atr = cls.calc_atr(df); atr_pct = float(atr.iloc[-1] / c.iloc[-1]) if c.iloc[-1] > 0 else 1.0
        vol_ok = atr_pct < cfg.atr_max_pct
        mom_s = float(cls.calc_momentum(c, ma_fast).iloc[-1]); mom_m = float(cls.calc_momentum(c, ma_slow).iloc[-1]); mom_l = float(cls.calc_momentum(c, ma_slow*2).iloc[-1])
        mom_ok = mom_s > 0 and mom_m > 0
        vol_confirm = True
        if cfg.volume_confirm: vol_confirm = bool(v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 0.8)
        score = min(1.0, bulls/8.0 + (0.15 if mom_ok else 0) + (0.1 if trend_strong else 0) + (0.05 if vol_confirm else 0))
        return FrameMetrics(score=round(score, 4), bulls=bulls, trend=bool(c.iloc[-1] > maf.iloc[-1]),
            adx=round(adx_val, 2), atr_pct=round(atr_pct, 4), mom_short=round(mom_s, 2), mom_mid=round(mom_m, 2),
            mom_long=round(mom_l, 2), vol_confirm=vol_confirm, trend_strong=trend_strong, vol_ok=vol_ok)

# ====================== 5. 信号层 ======================
class SignalEngine:
    def __init__(self, cfg, dm, logger):
        self.cfg = cfg; self.dm = dm; self.logger = logger; self.cfg_dict = dm.cfg_dict; self.factor = FactorEngine()
    def evaluate_symbol(self, code):
        df = self.dm.fetch_hist(code)
        if df is None or len(df) < 60: return ScanFailure(code, "数据不足")
        time.sleep(self.cfg.sleep)
        short_df = self.dm.resample_ohlcv(df, self.dm.RESAMPLE_RULE[self.cfg_dict["short"]])
        long_df = self.dm.resample_ohlcv(df, self.dm.RESAMPLE_RULE[self.cfg_dict["long"]])
        if len(short_df) < self.cfg_dict["min_s"] or len(long_df) < self.cfg_dict["min_l"]: return ScanFailure(code, "数据不足")
        s = self.factor.calc_frame(short_df, *self.cfg_dict["short_ma"], self.cfg)
        l = self.factor.calc_frame(long_df, *self.cfg_dict["long_ma"], self.cfg)
        if s is None or l is None: return ScanFailure(code, "计算失败")
        short_ok = s.score >= self.cfg.short_score_min and s.bulls >= self.cfg.short_bulls
        long_ok = l.score >= self.cfg.long_score_min and l.bulls >= self.cfg.long_bulls and l.trend
        if not (short_ok and long_ok): return ScanFailure(code, "非双周期共振")
        if not s.vol_ok or not l.vol_ok: return ScanFailure(code, "波动率过高")
        if not s.trend_strong and not l.trend_strong: return ScanFailure(code, "趋势太弱")
        w = self.cfg.resonance_weights
        reso = (w.get("short_mom", 0.35)*max(0, s.mom_short)/100 + w.get("mid_mom", 0.35)*max(0, l.mom_mid)/100
                + w.get("long_mom", 0.20)*max(0, l.mom_long)/100 + w.get("trend_strength", 0.10)*min(l.adx, 50)/50)
        final = (s.score + l.score) / 2 * 100
        latest = round(float(df["close"].iloc[-1]), 2)
        sig_date = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")
        return SignalResult(code=code, name="", industry="", latest_price=latest, signal_price=latest, signal_date=sig_date,
            short_score=round(s.score*100, 1), short_bulls=s.bulls, short_adx=s.adx, short_atr_pct=round(s.atr_pct*100, 2), short_mom=s.mom_short,
            long_score=round(l.score*100, 1), long_bulls=l.bulls, long_adx=l.adx, long_atr_pct=round(l.atr_pct*100, 2), long_mom=l.mom_mid,
            resonance_score=round(reso*100, 1), composite_score=round(final, 1))
    def rank_filter(self, results):
        if not self.cfg.use_rank or len(results) < 10: return results
        df = pd.DataFrame([r.to_dict() for r in results])
        df["score_rank"] = df["composite_score"].rank(pct=True); df["resonance_rank"] = df["resonance_score"].rank(pct=True)
        df["composite_rank"] = df["score_rank"]*0.6 + df["resonance_rank"]*0.4
        df["composite_pct"] = df["composite_rank"].rank(pct=True)
        keep_codes = set(df[df["composite_pct"] >= (1 - self.cfg.rank_top_pct)]["code"].tolist())
        return [r for r in results if r.code in keep_codes]

# ====================== 6. 板块与实时价 ======================
class SectorAnalyzer:
    def __init__(self, cfg, logger):
        self.cfg = cfg; self.logger = logger; self.hot_sectors = []
    def fetch_hot_sectors(self):
        try: import akshare as ak
        except ImportError: return
        heat = None
        for i in range(3):
            try:
                heat = ak.stock_board_industry_name_em()
                if heat is not None and not heat.empty: break
            except Exception: time.sleep(2 + i)
        if heat is None or heat.empty or "板块名称" not in heat.columns: return
        heat["_chg"] = pd.to_numeric(heat.get("涨跌幅"), errors="coerce")
        heat = heat[heat["_chg"] >= self.cfg.hot_sector_min_pct].sort_values("_chg", ascending=False)
        self.hot_sectors = [(str(r["板块名称"]), round(float(r["_chg"]), 2)) for _, r in heat.head(self.cfg.hot_sector_top).iterrows()]
    def calc_sector_cluster(self, results):
        labeled = [r for r in results if r.industry not in ("—", "未知", "", None)]
        if not labeled: return []
        counts = pd.Series([r.industry for r in labeled]).value_counts().head(self.cfg.cluster_top)
        return [(str(n), int(c)) for n, c in counts.items()]
    def enrich_resonance(self, results):
        hot_names = [n for n, _ in self.hot_sectors]; out = []
        for r in results:
            sec = r.industry; resonance = False; resonance_sector = ""
            if sec and sec not in ("—", "未知", "") and hot_names:
                s = sec.strip()
                for hh in hot_names:
                    if hh and (hh == s or hh in s or s in hh): resonance = True; resonance_sector = hh; break
            out.append(SignalResult(code=r.code, name=r.name, industry=r.industry, latest_price=r.latest_price,
                signal_price=r.signal_price, signal_date=r.signal_date, short_score=r.short_score, short_bulls=r.short_bulls,
                short_adx=r.short_adx, short_atr_pct=r.short_atr_pct, short_mom=r.short_mom, long_score=r.long_score,
                long_bulls=r.long_bulls, long_adx=r.long_adx, long_atr_pct=r.long_atr_pct, long_mom=r.long_mom,
                resonance_score=r.resonance_score, composite_score=r.composite_score,
                resonance=resonance, resonance_sector=resonance_sector))
        return out

class PriceAligner:
    @staticmethod
    def fetch_realtime_tencent(codes, logger):
        out = {}
        if not codes: return out
        syms = []
        for c in codes:
            c6 = str(c).split(".")[-1].zfill(6)
            pref = "sh" if c6[:1] in ("6", "9") else ("bj" if c6[:1] in ("4", "8") else "sz")
            syms.append(f"{pref}{c6}")
        for i in range(0, len(syms), 50):
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(syms[i:i+50]), timeout=10); r.encoding = "gbk"
                for line in r.text.strip().split(";"):
                    if "=" not in line: continue
                    parts = line.split("=", 1)[1].strip().strip('"').split("~")
                    if len(parts) > 4 and parts[2]:
                        try:
                            px = float(parts[3])
                            if px > 0: out[parts[2].zfill(6)] = px
                        except Exception: pass
            except Exception: pass
            time.sleep(0.3)
        return out
    @staticmethod
    def align_suffix(r, spot_now):
        head = f"信号{r.signal_price}"
        if r.signal_date: head += f"@{r.signal_date[5:10]}"
        code6 = str(r.code).split(".")[-1].zfill(6); now = spot_now.get(code6)
        if now is not None and r.signal_price:
            try:
                chg = (now - float(r.signal_price)) / float(r.signal_price) * 100
                return f" | {head} -> 现价{now}@run({chg:+.1f}%)"
            except Exception: return f" | {head}"
        return f" | {head}"

# ====================== 7. 信号追踪 ======================
class SignalTracker:
    def __init__(self, cfg, logger):
        self.cfg = cfg; self.logger = logger; self.db_path = Path(cfg.cache_dir) / "signal_track.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS signal_track (code TEXT, signal_date TEXT, signal_price REAL, check_date TEXT, return_5d REAL, return_10d REAL, return_20d REAL, PRIMARY KEY (code, signal_date))")
            conn.commit()
    def record_signals(self, signals):
        if not self.cfg.track_signals or not signals: return
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.executemany("INSERT OR REPLACE INTO signal_track (code, signal_date, signal_price, check_date) VALUES (?,?,?,?)",
                    [(c, d, p, today) for c, d, p in signals]); conn.commit()
        except Exception as e: self.logger.warning(f"信号批量写入异常: {e}")
    def update_returns(self, dm):
        if not self.cfg.track_signals: return
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                df = pd.read_sql_query("SELECT * FROM signal_track WHERE return_20d IS NULL", conn)
                if df.empty: return
                for _, row in df.iterrows():
                    hist = dm.fetch_hist(row["code"])
                    if hist is None or hist.empty: continue
                    hist = hist[hist["date"] >= row["signal_date"]].reset_index(drop=True)
                    if len(hist) < 2: continue
                    vals = {}
                    for d in self.cfg.track_days:
                        if len(hist) > d:
                            vals[f"return_{d}d"] = round((hist["close"].iloc[d] - row["signal_price"]) / row["signal_price"] * 100, 2)
                    if vals:
                        conn.execute(f"UPDATE signal_track SET {', '.join(f'{k}=?' for k in vals)} WHERE code=? AND signal_date=?",
                            list(vals.values()) + [row["code"], row["signal_date"]])
                conn.commit()
        except Exception as e: self.logger.warning(f"信号追踪更新异常: {e}")
    def get_signal_stats(self):
        if not self.cfg.track_signals: return {}
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                df = pd.read_sql_query("SELECT * FROM signal_track WHERE return_5d IS NOT NULL", conn)
                if df.empty: return {}
                stats = {}
                for d in self.cfg.track_days:
                    col = f"return_{d}d"
                    if col in df.columns:
                        stats[f"{d}日胜率"] = round((df[col] > 0).mean() * 100, 1)
                        stats[f"{d}日均收益"] = round(df[col].mean(), 2)
                        stats[f"{d}日最大回撤"] = round(df[col].min(), 2)
                return stats
        except Exception: return {}

# ====================== 8. 推送层 ======================
class ServerChanPusher:
    def __init__(self, key, logger):
        self.key = key; self.logger = logger
    def send(self, title, content):
        if not self.key: return False
        chunks = self._chunk(content, 3800); ok = True
        for i, ch in enumerate(chunks):
            t = title if i == 0 else f"{title}（续{i+1}/{len(chunks)}）"
            ok = self._send_one(t, ch) and ok
            if i < len(chunks) - 1: time.sleep(1)
        return ok
    def _send_one(self, title, content):
        try:
            from serverchan_sdk import sc_send
            ret = sc_send(self.key, title, content)
            if (ret.get("code", ret.get("errno", -1)) == 0) if isinstance(ret, dict) else bool(ret): return True
        except Exception: pass
        try:
            return requests.post(f"https://sctapi.ftqq.com/{self.key}.send", data={"title": title, "desp": content}, timeout=15).json().get("code") == 0
        except Exception: return False
    @staticmethod
    def _chunk(text, limit):
        chunks, cur, cur_len = [], [], 0
        for ln in text.split("\n"):
            lnlen = len(ln) + 1
            if cur_len + lnlen > limit and cur: chunks.append("\n".join(cur)); cur, cur_len = [], 0
            cur.append(ln); cur_len += lnlen
        if cur: chunks.append("\n".join(cur))
        return chunks or [""]

# ====================== 8.5 多进程 worker（模块级, 会话复用+超时） ======================
_W_CFG: Optional[Settings] = None
_W_DM: Optional[DataManager] = None
_W_SE: Optional[SignalEngine] = None

def _mtf_worker_init(cfg, industry_map):
    global _W_CFG, _W_DM, _W_SE
    time.sleep(random.uniform(0, 2))
    _W_CFG = cfg
    logger = logging.getLogger("mtf_worker")
    _W_DM = DataManager(cfg, logger)
    _W_DM._industry_map = dict(industry_map or {})
    _W_DM.bs_provider.login()
    _W_SE = SignalEngine(cfg, _W_DM, logger)
    import atexit; atexit.register(_W_DM.bs_provider.logout)

def _mtf_worker_eval(task):
    code, name = task
    global _W_CFG, _W_DM, _W_SE
    try:
        if _W_SE is None:
            logger = logging.getLogger("mtf_worker")
            _W_DM = DataManager(_W_CFG, logger); _W_DM.bs_provider.login()
            _W_SE = SignalEngine(_W_CFG, _W_DM, logger)
        res = _W_SE.evaluate_symbol(code)
        if isinstance(res, ScanFailure): return ("fail", res.reason, None)
        if res is None: return ("none", None, None)
        final = SignalResult(code=res.code, name=name, industry=_W_DM.get_industry(code),
            latest_price=res.latest_price, signal_price=res.signal_price, signal_date=res.signal_date,
            short_score=res.short_score, short_bulls=res.short_bulls, short_adx=res.short_adx,
            short_atr_pct=res.short_atr_pct, short_mom=res.short_mom, long_score=res.long_score,
            long_bulls=res.long_bulls, long_adx=res.long_adx, long_atr_pct=res.long_atr_pct, long_mom=res.long_mom,
            resonance_score=res.resonance_score, composite_score=res.composite_score,
            resonance=res.resonance, resonance_sector=res.resonance_sector)
        return ("ok", None, final)
    except Exception:
        return ("fail", "抓取失败", None)

# ====================== 9. 主控层 ======================
class ScreenerMaster:
    def __init__(self, cfg):
        self.cfg = cfg; self.logger = setup_logging(cfg.log_level)
        self.dm = DataManager(cfg, self.logger)
        self.signal_engine = SignalEngine(cfg, self.dm, self.logger)
        self.sector_analyzer = SectorAnalyzer(cfg, self.logger)
        self.price_aligner = PriceAligner()
        self.tracker = SignalTracker(cfg, self.logger)
        self.pusher = ServerChanPusher(cfg.serverchan_key, self.logger)
        self.fail_stats = defaultdict(int)
        self.output_dir = Path(cfg.output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
    def snapshot_prefilter(self, codes):
        if not self.cfg.snapshot_pre: return codes
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is None or spot.empty or "代码" not in spot.columns: return codes
            spot["代码"] = spot["代码"].astype(str).str.zfill(6)
            for col in ("最新价", "成交额", "换手率"):
                if col in spot.columns: spot[col] = pd.to_numeric(spot[col], errors="coerce")
            m = (spot["代码"].str.startswith(self.cfg.keep_prefix)
                 & ~spot["名称"].astype(str).str.contains("|".join(self.cfg.exclude_name), na=False, regex=True)
                 & (spot["最新价"] >= self.cfg.min_price))
            if "成交额" in spot.columns: m &= spot["成交额"] >= self.cfg.pre_amount_min
            if "换手率" in spot.columns: m &= spot["换手率"] >= self.cfg.pre_turnover_min
            keep = set(spot.loc[m, "代码"])
            out = [c for c in codes if c[3:] in keep]
            self.logger.info(f"快照预筛: {len(codes)} -> {len(out)} 只")
            return out if out else codes
        except Exception as e:
            self.logger.warning(f"快照预筛失败, 退化全扫: {e}"); return codes
    def load_stock_list(self):
        import akshare as ak
        stock_df = pd.DataFrame()
        if self.dm.bs_provider.login():
            try:
                import baostock as bs
                stock_df = bs.query_stock_basic().get_data()
            except Exception as e: self.logger.warning(f"baostock 取列表异常: {e}")
            finally: self.dm.bs_provider.logout()
        self.dm.load_industry()
        if stock_df is None or stock_df.empty or "code" not in stock_df.columns:
            for attempt in range(3):
                try:
                    d = ak.stock_info_a_code_name()
                    if d is not None and not d.empty and "code" in d.columns:
                        nc = "name" if "name" in d.columns else d.columns[1]
                        d = d[["code", nc]].copy(); d.columns = ["code", "code_name"]
                        d["code"] = d["code"].astype(str).str.zfill(6)
                        d["code"] = d["code"].apply(lambda c: ("sh." if c[:1] in ("6", "9") else "sz.") + c)
                        d["type"] = "1"; d["status"] = "1"; stock_df = d; break
                except Exception as e: self.logger.warning(f"akshare 列表第{attempt+1}次失败: {e}")
                time.sleep(2 + attempt)
        if stock_df is None or stock_df.empty:
            self.logger.error("无法获取股票列表"); return [], {}
        stock_df = stock_df[stock_df["code"].str.startswith(("sh.", "sz.")) & (stock_df["type"] == "1") & (stock_df["status"] == "1")].copy()
        stock_df = stock_df[~stock_df["code_name"].astype(str).str.contains("ST|退", na=False, regex=True)]
        codes = self.snapshot_prefilter(stock_df["code"].tolist())
        
        # 矩阵新增: 分段扫描逻辑 (先 offset，再 limit)
        if self.cfg.scan_offset and len(codes) > self.cfg.scan_offset:
            codes = codes[self.cfg.scan_offset:]
            self.logger.info(f"分段扫描: 跳过前 {self.cfg.scan_offset} 只, 本段剩余 {len(codes)} 只")
        if self.cfg.scan_limit and len(codes) > self.cfg.scan_limit: 
            codes = codes[:self.cfg.scan_limit]
            
        return codes, dict(zip(stock_df["code"], stock_df["code_name"]))
    def run_scan(self):
        self.fail_stats.clear()
        codes, name_map = self.load_stock_list()
        if not codes: return []
        industry_map = dict(self.dm._industry_map)
        tasks = [(c, name_map.get(c, "")) for c in codes]
        self.logger.info(f"开始扫描 [{self.dm.cfg_dict['label']}] {len(tasks)} 只 ({self.cfg.num_processes} 进程, 会话复用+超时) ...")
        results = []
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(processes=self.cfg.num_processes, initializer=_mtf_worker_init, initargs=(self.cfg, industry_map)) as pool:
                for status, reason, res in pool.imap_unordered(_mtf_worker_eval, tasks):
                    if status == "ok" and res is not None: results.append(res)
                    elif status == "fail": self.fail_stats[reason] += 1
        except KeyboardInterrupt:
            self.logger.warning("用户中断扫描")
        if self.cfg.use_rank:
            results = self.signal_engine.rank_filter(results)
            self.logger.info(f"RANK 后保留 {len(results)} 只")
        return sorted(results, key=lambda x: x.composite_score, reverse=True)
    def enrich_and_push(self, results):
        if not results: return
        self.sector_analyzer.fetch_hot_sectors()
        cluster = self.sector_analyzer.calc_sector_cluster(results)
        results = self.sector_analyzer.enrich_resonance(results)
        codes6 = [str(r.code).split(".")[-1].zfill(6) for r in results]
        rt = self.price_aligner.fetch_realtime_tencent(codes6, self.logger)
        if rt:
            enriched = []
            for r in results:
                now = rt.get(str(r.code).split(".")[-1].zfill(6))
                enriched.append(r.__class__(**{**r.to_dict(), "latest_price": now}) if now is not None else r)
            results = enriched
        if self.cfg.track_signals:
            self.tracker.record_signals([(r.code, r.signal_date, r.signal_price) for r in results])
            self.tracker.update_returns(self.dm)
        results = sorted(results, key=lambda x: (x.resonance, x.composite_score), reverse=True)
        tag = datetime.now().strftime("%Y%m%d"); pair = self.cfg.pair.value
        df_out = pd.DataFrame([r.to_dict() for r in results])
        csv_path = self.output_dir / f"mtf_full_{pair}_{tag}.csv"
        json_path = self.output_dir / f"mtf_full_{pair}_{tag}.json"
        try:
            df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"date": tag, "pair": pair, "cluster": cluster, "n": len(results),
                    "fail_stats": dict(self.fail_stats), "hits": [r.to_dict() for r in results]}, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"已存盘: {csv_path} / {json_path}")
        except Exception as e: self.logger.error(f"存盘异常: {e}")
        try:
            disp = df_out.copy()
            disp.insert(2, "板块", [("-->" + r.resonance_sector) if r.resonance else (r.industry or "—") for r in results])
            disp = disp.drop(columns=[c for c in ["industry", "resonance", "resonance_sector", "signal_price"] if c in disp.columns], errors="ignore")
            print("\n" + disp.head(self.cfg.push_top).to_string(index=False))
        except Exception as e: self.logger.warning(f"展示异常: {e}")
        if self.cfg.serverchan_key:
            try:
                n_reso = sum(1 for r in results if r.resonance)
                self.pusher.send(f"增强双共振[{self.dm.cfg_dict['label']}] 命中{len(results)}只 风口{n_reso}",
                    self._build_push(results, cluster, self.sector_analyzer.hot_sectors, rt))
            except Exception as e: self.logger.error(f"推送异常: {e}")
    def _build_push(self, results, cluster, hot, spot_now):
        cfg_dict = self.dm.cfg_dict
        reso = [r for r in results if r.resonance]
        lines = [f"**增强多周期双共振({cfg_dict['label']})** | 命中{len(results)}只 风口{len(reso)} (现价=实时价)",
            f"*(短>={self.cfg.short_score_min*100:.0f}%且>={self.cfg.short_bulls}/8 × 长>={self.cfg.long_score_min*100:.0f}%且>={self.cfg.long_bulls}/8+ADX>={self.cfg.adx_min}+ATR过滤; 必止损)*", ""]
        if hot: lines.append("风口: " + "、".join(f"{n}({c}%)" for n, c in hot[:6])); lines.append("")
        if cluster: lines.append("共振板块: " + "、".join(f"{n}({c})" for n, c in cluster)); lines.append("")
        if self.cfg.track_signals:
            stats = self.tracker.get_signal_stats()
            if stats: lines.append("历史信号表现: " + " | ".join(f"{k}:{v}" for k, v in list(stats.items())[:4])); lines.append("")
        def line(r):
            sec = ("-->" + r.resonance_sector) if r.resonance else (r.industry or "—")
            return (f"- **{r.name}({r.code})** [{sec}] 现价{r.latest_price} | 短{r.short_score}%({r.short_bulls}/8) "
                f"长{r.long_score}%({r.long_bulls}/8) ADX{r.short_adx}/{r.long_adx} 均分{r.composite_score}"
                f"{self.price_aligner.align_suffix(r, spot_now)}")
        if reso:
            lines.append(f"### 共振遇风口 共{len(reso)}只"); lines += [line(r) for r in reso[:self.cfg.push_top]]; lines.append("")
        lines.append(f"### 全部双共振 共{len(results)}只"); lines += [line(r) for r in results[:self.cfg.push_top]]
        if len(results) > self.cfg.push_top: lines.append(f"\n*…另有 {len(results)-self.cfg.push_top} 只, 详见 output*")
        return "\n".join(lines)
    def run(self):
        self.logger.info("=" * 70)
        self.logger.info(f"增强多周期双共振[{self.dm.cfg_dict['label']}] | {datetime.now():%Y-%m-%d %H:%M} | 回看{self.dm.cfg_dict['lookback_days']}天 | 进程{self.cfg.num_processes} | Offset={self.cfg.scan_offset}")
        self.logger.info("=" * 70)
        results = self.run_scan()
        self.logger.info("各失败原因统计：")
        for k, v in sorted(self.fail_stats.items(), key=lambda x: -x[1]):
            if v: self.logger.info(f"  {k}: {v}")
        if not results:
            self.logger.warning(f"本次未发现 [{self.dm.cfg_dict['label']}] 双周期共振票(门槛严, 0命中属正常)。")
            if self.cfg.serverchan_key:
                self.pusher.send(f"增强双共振[{self.dm.cfg_dict['label']}] | 0命中",
                    f"**增强多周期双共振[{self.dm.cfg_dict['label']}]** | 本次无同时满足短+长双框架的票。")
            return
        self.enrich_and_push(results)

# ====================== 10. 入口 ======================
def main():
    cfg = Settings.from_env()
    ScreenerMaster(cfg).run()
    sys.exit(0)

if __name__ == "__main__":
    main()
# >>>FILE_END_mtf_pro_v31_offset<<<
