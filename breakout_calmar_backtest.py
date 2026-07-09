import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs


def calculate_calmar_ratio(returns, periods_per_year=252):
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_return = cum_returns.iloc[-1] - 1
    num_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    return calmar, cagr * 100, abs(max_drawdown) * 100


def get_daily_history(code, start_date="1990-01-01"):
    """code 需为 baostock 格式，如 'sh.600000'"""
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date=start_date, adjustflag="2"
    )
    df = rs.get_data()
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)


def backtest_breakout_strategy(code="sh.600000", ma_period=250, vol_multiplier=1.8, initial_capital=100000):
    bs.login()
    df = get_daily_history(code)
    bs.logout()

    if len(df) < ma_period + 10:
        print(f"{code} 数据不足，跳过")
        return None, None

    df['MA250'] = df['close'].rolling(ma_period).mean()
    df['Avg_Vol'] = df['volume'].rolling(20).mean()

    position = 0.0
    equity = [initial_capital]
    dates = [df['date'].iloc[ma_period]]
    trades = []

    for i in range(ma_period + 1, len(df)):
        price_today = df['close'].iloc[i]

        if position == 0:
            price_break = (price_today > df['MA250'].iloc[i]) and \
                          (df['close'].iloc[i - 1] <= df['MA250'].iloc[i - 1])
            vol_confirm = (
                (df['volume'].iloc[i] / df['Avg_Vol'].iloc[i]) >= vol_multiplier
                if df['Avg_Vol'].iloc[i] > 0 else False
            )
            if price_break and vol_confirm:
                position = equity[-1] / price_today
                trades.append({'入场日期': df['date'].iloc[i].date(), '入场价': price_today})
                equity.append(position * price_today)
            else:
                equity.append(equity[-1])
        else:
            if price_today < df['MA250'].iloc[i]:
                equity.append(position * price_today)
                trades[-1]['出场日期'] = df['date'].iloc[i].date()
                trades[-1]['出场价'] = price_today
                position = 0.0
            else:
                equity.append(position * price_today)

        dates.append(df['date'].iloc[i])

    equity_curve = pd.Series(equity, index=pd.to_datetime(dates))
    strategy_returns = equity_curve.pct_change().dropna()
    calmar, cagr_pct, mdd_pct = calculate_calmar_ratio(strategy_returns)

    print(f"\n=== {code} 突破策略回测结果 ===")
    print(f"回测天数: {len(equity_curve)}")
    print(f"年化收益率 (CAGR): {cagr_pct:.2f}%")
    print(f"最大回撤 (MDD): {mdd_pct:.2f}%")
    print(f"卡尔马比率 (Calmar): {calmar:.2f}")
    print(f"交易次数: {len(trades)}")
    if trades:
        print("\n部分交易记录:")
        print(pd.DataFrame(trades).head(10))

    return calmar, equity_curve


if __name__ == "__main__":
    backtest_breakout_strategy("sh.600000")
