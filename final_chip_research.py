# ====================== 改进版盘中确认（新浪优先 + 东财兜底） ======================
INTRADAY_MA_PERIOD = 5
INTRADAY_TURN_UP_THRESHOLD = 0.0008


def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一不同源的字段名为 datetime / close"""
    if df is None or df.empty:
        raise ValueError("empty_frame")
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("时间", "datetime", "day", "date", "time"):
            col_map[c] = "datetime"
        elif cl in ("收盘", "close"):
            col_map[c] = "close"
    df = df.rename(columns=col_map)
    if "datetime" not in df.columns or "close" not in df.columns:
        if len(df.columns) >= 5:
            df = df.copy()
            df.columns = ["datetime", "open", "high", "low", "close"] + list(df.columns[5:])
        else:
            raise ValueError(f"cannot_normalize_columns:{list(df.columns)}")
    df = df[["datetime", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
    if df.empty:
        raise ValueError("empty_after_normalize")
    return df


def _fetch_sina_5min(code: str) -> pd.DataFrame:
    import akshare as ak
    code6 = str(code).zfill(6)
    symbol = f"sh{code6}" if code6.startswith(("5", "6", "9")) else f"sz{code6}"
    df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
    return _normalize_minute_df(df)


def _fetch_em_5min(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    code6 = str(code).zfill(6)
    df = ak.stock_zh_a_hist_min_em(
        symbol=code6,
        period="5",
        adjust="qfq",
        start_date=start,
        end_date=end,
    )
    return _normalize_minute_df(df)


def fetch_intraday_ma5_turnup(
    code: str,
    timeout_seconds: float = 25.0,
    retries: int = 3,
) -> tuple[bool, float, str, list[str]]:
    """
    改进版：新浪优先 → 东财兜底，带限速和指数退避。
    返回 (is_turn_up, slope_now, note, errors)
    """
    import time
    import random

    errors: list[str] = []
    start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d 09:30:00")
    end = datetime.now().strftime("%Y-%m-%d 15:00:00")

    sources = [
        ("sina", lambda: _fetch_sina_5min(code)),
        ("em",   lambda: _fetch_em_5min(code, start, end)),
    ]

    for source_name, fetcher in sources:
        for attempt in range(1, max(retries, 1) + 1):
            sleep_t = 1.6 + random.uniform(0.3, 0.9)
            if attempt > 1:
                sleep_t += (2 ** (attempt - 2)) * 0.9
            time.sleep(sleep_t)

            try:
                frame = fetcher()
                frame = frame.set_index("datetime").sort_index()
                close_10 = (
                    pd.to_numeric(frame["close"], errors="coerce")
                    .resample("10min")
                    .last()
                    .dropna()
                )
                if len(close_10) < INTRADAY_MA_PERIOD + 3:
                    errors.append(f"{source_name}:{attempt}:10min_data_insufficient:{len(close_10)}")
                    continue

                ma5 = close_10.rolling(INTRADAY_MA_PERIOD).mean()
                recent = ma5.iloc[-3:].values
                if np.any(np.isnan(recent)):
                    errors.append(f"{source_name}:{attempt}:ma5_nan")
                    continue

                slope_now = float(recent[-1] - recent[-2])
                slope_prev = float(recent[-2] - recent[-3])
                is_up = bool(
                    slope_prev <= INTRADAY_TURN_UP_THRESHOLD
                    and slope_now > INTRADAY_TURN_UP_THRESHOLD
                )
                note = f"{source_name}|前={slope_prev:.5f} 现={slope_now:.5f}"
                return is_up, slope_now, note, errors

            except Exception as exc:
                errors.append(
                    f"{source_name}:{attempt}:{type(exc).__name__}:{str(exc)[:180]}"
                )
                continue

    return False, 0.0, "盘中数据获取失败(多源)", errors
