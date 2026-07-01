"""
Sequoia-X 策略选股模块 (BaoStock 版)
适配 GitHub Actions 环境，规避 IP 封锁
"""
import baostock as bs
import pandas as pd
import random
import time
import akshare as ak
from datetime import datetime, timedelta
import concurrent.futures as cf

# 配置参数
MAX_WORKERS = 4  # 降低并发，减轻服务器压力
RETRY = 2        # 请求重试次数
_bs_logged_in = False

def _ensure_bs_login():
    """BaoStock登录管理"""
    global _bs_logged_in
    if not _bs_logged_in:
        result = bs.login()
        if result.error_code != '0':
            raise RuntimeError(f"baostock登录失败: {result.error_msg}")
        _bs_logged_in = True

def _to_bs_code(code: str) -> str:
    """股票代码转baostock格式"""
    if code.startswith("6"): return f"sh.{code}"
    elif code.startswith(("8", "4")): return f"bj.{code}"
    else: return f"sz.{code}"

def _fetch_daily(code: str, days: int) -> pd.DataFrame | None:
    """统一日线数据获取（带重试与随机延时）"""
    _ensure_bs_login()
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.6))
    
    for attempt in range(RETRY + 1):
        try:
            time.sleep(random.uniform(0.5, 1.0))
            rs = bs.query_history_k_data_plus(
                _to_bs_code(code), 
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start.strftime("%Y-%m-%d"), 
                end_date=end.strftime("%Y-%m-%d"), 
                frequency="d", 
                adjustflag="2"
            )
            rows = []
            while rs.next(): rows.append(rs.get_row_data())
            if not rows: return None
            
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df[df["close"] != ""].copy()
            for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.dropna(subset=["close"])
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "high": "最高", "low": "最低", 
                "close": "收盘", "volume": "成交量", "amount": "成交额", 
                "turn": "换手率", "pctChg": "涨跌幅"
            })
            return df.sort_values("日期").reset_index(drop=True)
        except Exception as e:
            if attempt == RETRY: print(f"[BaoStock] {code} 获取失败: {e}")
            time.sleep(random.uniform(2.0, 4.0))
    return None

def _default_candidates(limit: int | None = None) -> pd.DataFrame:
    """获取候选股票列表，增加兜底逻辑"""
    try:
        stock_list = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
        # 过滤掉 ST 股，减少计算量，避开低质量数据
        stock_list = stock_list[~stock_list['名称'].str.contains('ST')]
        return stock_list.head(limit) if limit else stock_list
    except Exception:
        return pd.DataFrame({"代码": ["600519", "000001"], "名称": ["贵州茅台", "平安银行"]})

def _run_strategy(check_fn, min_bars: int, candidates: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """优化后的并发框架：增加节奏控制"""
    results = []
    print(f"[{strategy_name}] 开始扫描，共 {len(candidates)} 只候选股...")
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_fn, row["代码"], row["名称"], min_bars): row["代码"] 
                   for _, row in candidates.iterrows()}
        
        for i, future in enumerate(cf.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            
            # 节奏控制：每处理50只强制停顿
            if (i + 1) % 50 == 0:
                time.sleep(random.uniform(1.0, 2.0))
                print(f"[{strategy_name}] 进度: {i + 1}/{len(candidates)}")
                
    print(f"[{strategy_name}] 完成，选出 {len(results)} 只股票")
    return pd.DataFrame(results)

# ============================================================
# 策略逻辑部分 (保持原有策略 check 函数不变，直接调用上述函数即可)
# ============================================================
# (此处省略具体策略函数 _check_turtle 等，逻辑与你原版一致)

def screen_turtle_breakout(limit: int | None = None) -> pd.DataFrame:
    """海龟策略"""
    candidates = _default_candidates(limit)
    df = _run_strategy(_check_turtle, 21, candidates, "海龟突破")
    return df
