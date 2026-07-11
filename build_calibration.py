import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
from datetime import datetime, timedelta
from market_signal_utils import INDEX_CODE, BANK_STOCKS, calculate_score

HISTORY_DAYS = 730  # 约2年历史，用于统计校准
BINS = 10
OUTPUT_CSV = "calibration_table.csv"


def fetch_history(code, start_date):
    """通用历史数据获取：baostock自带pctChg字段，直接是涨跌幅(%)，不用手动算"""
    rs = bs.query_history_k_data_plus(
        code, "date,close,pctChg",
        start_date=start_date,
        frequency="d", adjustflag="2"
    )
    df = rs.get_data()
    df['date'] = pd.to_datetime(df['date'])
    df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna(subset=['pctChg']).sort_values('date').reset_index(drop=True)


def build_calibration_table():
    start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')

    bs.login()
    idx_df = fetch_history(INDEX_CODE, start_date)
    bank_dfs = {name: fetch_history(code, start_date) for name, code in BANK_STOCKS.items()}
    bs.logout()

    merged = idx_df[['date', 'pctChg']].rename(columns={'pctChg': 'sz_chg'})
    for name, df in bank_dfs.items():
        merged = merged.merge(
            df[['date', 'pctChg']].rename(columns={'pctChg': f'{name}_chg'}),
            on='date', how='inner'
        )

    merged = merged.sort_values('date').reset_index(drop=True)
    merged['next_day_up'] = (merged['sz_chg'].shift(-1) > 0).astype(int)
    merged = merged.iloc[:-1]

    scores = []
    for _, row in merged.iterrows():
        s, _ = calculate_score(row['sz_chg'], row['建设银行_chg'], row['工商银行_chg'], row['招商银行_chg'])
        scores.append(s)
    merged['score'] = scores

    if len(merged) < BINS * 5:
        print(f"⚠️ 历史样本量偏少（{len(merged)}条），分箱校准结果参考价值有限")

    merged['score_bin'] = pd.qcut(merged['score'], q=BINS, duplicates='drop')
    calibration = merged.groupby('score_bin', observed=True).agg(
        样本数=('next_day_up', 'count'),
        次日上涨频率=('next_day_up', 'mean')
    ).reset_index()

    calibration['bin_left'] = calibration['score_bin'].apply(lambda x: x.left)
    calibration['bin_right'] = calibration['score_bin'].apply(lambda x: x.right)
    calibration = calibration.drop(columns=['score_bin'])

    calibration.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 校准表已生成: {OUTPUT_CSV}（{len(merged)}个历史交易日样本，{len(calibration)}个分箱）")
    print(calibration.to_string(index=False))

    return calibration


if __name__ == "__main__":
    build_calibration_table()
