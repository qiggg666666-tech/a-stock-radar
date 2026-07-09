import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs


def check_cagr_below_5(code='sz.002436', threshold=0.05):
    """code 需为 baostock 格式，如 'sz.002436' / 'sh.600000'"""
    bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,close",
        start_date="1990-01-01", adjustflag="2"
    )
    df = rs.get_data()
    bs.logout()

    if df.empty:
        print("无法获取历史数据")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)

    start_date = df['date'].iloc[0].date()
    end_date = df['date'].iloc[-1].date()
    initial_price = df['close'].iloc[0]
    current_price = df['close'].iloc[-1]

    days_held = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    years_held = days_held / 365.25

    if years_held <= 0:
        raise ValueError(
            f"持有年限计算异常（years_held={years_held}），"
            f"起始日={start_date}，截止日={end_date}，请检查数据源"
        )

    total_multiplier = current_price / initial_price
    cagr = total_multiplier ** (1 / years_held) - 1
    is_below_5 = cagr < threshold

    print(f"股票代码: {code}")
    print(f"数据起始日期: {start_date}  （注意核对是否接近真实上市日）")
    print(f"计算截至日期: {end_date}")
    print(f"持有年限: {years_held:.2f} 年")
    print(f"起始价格（前复权）: {initial_price:.4f} 元")
    print(f"当前价格（前复权）: {current_price:.4f} 元")
    print(f"总复利回报倍数: {total_multiplier:.2f} 倍")
    print(f"年化复利回报 (CAGR): {cagr * 100:.2f}%")
    print(f"是否低于 {threshold * 100:.0f}%？: {'是' if is_below_5 else '否'}")

    return {
        'cagr': cagr, 'years': years_held, 'total_mult': total_multiplier,
        'start_date': start_date, 'end_date': end_date, 'below_threshold': is_below_5,
    }


if __name__ == "__main__":
    check_cagr_below_5()
