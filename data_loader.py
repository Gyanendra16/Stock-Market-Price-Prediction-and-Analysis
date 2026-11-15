import pandas as pd
import numpy as np
import yfinance as yf


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None

        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        print(f"[ERROR] download_stock_data: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26       = df["Close"].ewm(span=26, adjust=False).mean()


    df["MACD"]        = df["EMA_12"] - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI_14"] = _compute_rsi(df["Close"], period=14)

    df["Daily_Return"] = df["Close"].pct_change()

    df.dropna(inplace=True)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
