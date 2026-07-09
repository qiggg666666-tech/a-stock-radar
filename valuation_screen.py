import pandas as pd
# 补丁：解决 baostock 调用已废弃的 DataFrame.append 报错的问题
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
from serverchan_sdk import sc_send
import os
import time
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# ------------------ 阈值参数 ------------------
PE_MIN, PE_MAX = 0, 40
PB_MIN, PB_MAX = 0, 8
MIN_PRICE = 5
SLEEP_PER_STOCK = 0.15
NUM_PROCESSES = 3  # 与仓库其他脚本保持一致，避免 GitHub Actions 上并发过高触发限流


def _init_worker():
    """每个子进程启动时独立登录baostock，带重试+错开延迟"""
    import random
    time.sleep(random.uniform(0, 2))
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                return
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("⚠️ 子进程登录多次重试后仍失败，该进程后续请求可能持续报错")


def _process_one(args):
    """单只股票的估值抓取+判断逻辑，运行在子进程里"""
    code, name = args
    try:
        rs = bs.query_history_k_data_plus(
            code, "date,close,peTTM,pbMRQ",
            start_date=datetime.now().strftime('%Y-%m-01'),  # 只需最近一期，取当月即可
            adjustflag="2"
        )
        df = rs.get_data()
        time.sleep(SLEEP_PER_STOCK)

        if df.empty:
            return None

        latest = df.iloc[-1]
        close = float(latest['close']) if latest['close'] else None
        pe = float(latest['peTTM']) if latest['peTTM'] else None
        pb = float(latest['pbMRQ']) if latest['pbMRQ'] else None

        if not close or close < MIN_PRICE or pe is None or pb is None:
            return None
        if not (PE_MIN < pe < PE_MAX):
            return None
        if not (PB_MIN < pb < PB_MAX):
            return None

        return {"代码": code, "名称": name, "最新价": round(close, 2), "PE": round(pe, 2), "PB": round(pb, 2)}
    except Exception as e:
        return {"__error__": f"{code} 处理失败: {e}"}


def run_valuation_screen(limit=None):
    print("正在连接 Baostock（主进程，用于取股票列表）...")
    bs.login()
    rs = bs.query_stock_basic()
    stock_df = rs.get_data()
    stock_df = stock_df[
        stock_df['code'].str.startswith(('sh.', 'sz.')) &
        (stock_df['type'] == '1') &
        (stock_df['status'] == '1')
    ]
    bs.logout()

    target_stocks = stock_df['code'].tolist()[:limit] if limit else stock_df['code'].tolist()
    code_to_name = dict(zip(stock_df['code'], stock_df['code_name']))
    tasks = [(code, code_to_name.get(code, "")) for code in target_stocks]

    results = []
    fail_count = 0
    print(f"开始估值筛选 {len(tasks)} 只股票（{NUM_PROCESSES} 个进程并行）...")

    with mp.Pool(processes=NUM_PROCESSES, initializer=_init_worker) as pool:
        pbar = tqdm(total=len(tasks), desc="扫描进度", unit="只")
        for res in pool.imap_unordered(_process_one, tasks):
            if res:
                if "__error__" in res:
                    fail_count += 1
                    pbar.write(f"⚠️ {res['__error__']}")
                else:
                    results.append(res)
                    pbar.write(f"✅ 命中: {res['代码']} {res['名称']}（PE {res['PE']} / PB {res['PB']}）")
            pbar.update(1)
            pbar.set_postfix(命中=len(results), 失败=fail_count)

    print(f"扫描完成，共失败 {fail_count} 只")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("PE", ascending=True).reset_index(drop=True)
    return result_df


def build_push_content(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(f"- {row['名称']}（{row['代码']}）最新价 {row['最新价']} | PE {row['PE']} | PB {row['PB']}")
    return "\n".join(lines)


def send_to_serverchan(sendkey, title, desp):
    try:
        response = sc_send(sendkey, title, desp)
        print(f"推送结果: {response}")
        if isinstance(response, dict) and response.get("code") not in (0, None):
            print(f"⚠️ 推送未成功，code={response.get('code')}，message={response.get('message')}")
        return response
    except Exception as e:
        print(f"推送失败（抛出异常）: {e}")
        return None


if __name__ == "__main__":
    df = run_valuation_screen(limit=500)
    if not df.empty:
        sendkey = os.getenv("SENDKEY")
        if sendkey:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"估值筛选 命中 {len(df)} 只（PE/PB合理）"
            content = f"扫描时间：{now}\n\n" + build_push_content(df)
            send_to_serverchan(sendkey, title, content)
        print(df)
    else:
        print("本次未找到符合条件的股票")
