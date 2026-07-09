import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

import baostock as bs
from first_red_520 import detect_first_red_to_520_low, LOW_WINDOW


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
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date=start_date, adjustflag="2"
    )
    df = rs.get_data()
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)


def backtest_first_red_520(code="sh.600000", hold_days=20, initial_capital=100000):
    bs.login()
    df = get_daily_history(code)
    bs.logout()

    if len(df) < LOW_WINDOW + hold_days + 1:
        print(f"{code} 数据不足，跳过")
        return None, None

    signals = detect_first_red_to_520_low(df)
    if signals.empty:
        print(f"{code} 无历史信号")
        return None, None

    trades = []
    for _, sig in signals.iterrows():
        sig_idx = df.index[df['date'] == sig['date']]
        if len(sig_idx) == 0:
            continue
        sig_idx = sig_idx[0]
        entry_idx = sig_idx + 1  # 信号次日开盘买入，避免收盘价信号+收盘价成交的前视偏差
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(df):
            continue

        entry_price = df['open'].iloc[entry_idx]
        exit_price = df['close'].iloc[exit_idx]
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            '信号日': sig['date'].date() if hasattr(sig['date'], 'date') else sig['date'],
            '入场日': df['date'].iloc[entry_idx].date(),
            '出场日': df['date'].iloc[exit_idx].date(),
            '入场价': round(entry_price, 2),
            '出场价': round(exit_price, 2),
            '收益率%': round(ret * 100, 2),
        })

    if not trades:
        print(f"{code} 无有效可回测交易")
        return None, None

    trades_df = pd.DataFrame(trades)

    equity = [initial_capital]
    for ret in trades_df['收益率%'] / 100:
        equity.append(equity[-1] * (1 + ret))
    equity_curve = pd.Series(equity)
    trade_returns = equity_curve.pct_change().dropna()

    calmar, cagr_pct, mdd_pct = calculate_calmar_ratio(trade_returns, periods_per_year=len(trades_df))

    print(f"\n=== {code} 520首红信号回测（持有{hold_days}天）===")
    print(f"信号次数: {len(trades_df)}")
    print(f"平均单次收益率: {trades_df['收益率%'].mean():.2f}%")
    print(f"胜率: {(trades_df['收益率%'] > 0).mean() * 100:.1f}%")
    print(f"累计收益率: {(equity_curve.iloc[-1] / initial_capital - 1) * 100:.2f}%")
    print(f"最大回撤: {mdd_pct:.2f}%")
    print(f"卡尔马比率: {calmar:.2f}")
    print("\n交易明细:")
    print(trades_df.to_string(index=False))

    return trades_df, equity_curve


if __name__ == "__main__":
    backtest_first_red_520("sh.600000", hold_days=20)
