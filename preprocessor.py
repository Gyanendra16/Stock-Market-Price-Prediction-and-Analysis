import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_12",
    "RSI_14", "MACD", "Daily_Return"
]


def prepare_features(df: pd.DataFrame):

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    X = df[feature_cols].copy()

    y = df["Close"].shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]

    
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X, y, feature_cols


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):

    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)     
    X_test  = scaler.transform(X_test)           

    return X_train, X_test, y_train, y_test, scaler
