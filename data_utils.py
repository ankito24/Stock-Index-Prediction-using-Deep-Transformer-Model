"""
data_utils.py
Data downloading, preprocessing and moving-window dataset construction,
following the methodology in:
"Stock market index prediction using deep Transformer model" (Wang et al., 2022)

Steps implemented (Section 5.1 of the paper):
1. Download daily closing prices for an index.
2. Split into training (first 80%) and testing (last 20%) sets.
3. Normalize using training-set mean/std:  x_hat = (x - mu) / sigma
4. Build (X, y) pairs with a moving window of length T:
   use previous T closing prices to predict the next day's close.
"""

import numpy as np
import pandas as pd


# Ticker symbols for the four indices used in the paper
TICKERS = {
    "CSI300": "000300.SS",   # Shanghai/Shenzhen CSI 300
    "SP500": "^GSPC",        # S&P 500
    "HSI": "^HSI",           # Hang Seng Index
    "N225": "^N225",         # Nikkei 225
}


def download_index(name: str, start="2010-01-01", end="2020-12-31") -> pd.Series:
    """Download daily closing price series for a given index name."""
    import yfinance as yf  # imported lazily so the rest of the module works offline/without yfinance

    if name not in TICKERS:
        raise ValueError(f"Unknown index '{name}'. Choose from {list(TICKERS)}")
    ticker = TICKERS[name]
    # auto_adjust=True set explicitly to match current yfinance default and silence the FutureWarning
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {name} ({ticker}). Check your internet connection.")

    # Newer yfinance versions return MultiIndex columns like ('Close', '^GSPC')
    # even when downloading a single ticker. If we don't unwrap this, df["Close"]
    # stays a 1-column DataFrame (not a Series), which silently adds an extra
    # dimension all the way through to the model input (causing shape errors
    # like "query should be unbatched 2D or batched 3D but received 4-D").
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]

    close = close.dropna()
    close = pd.Series(np.asarray(close).ravel(), index=close.index)  # guarantee 1D Series
    close.name = name
    return close


def train_test_split_series(series: pd.Series, train_ratio: float = 0.8):
    """First train_ratio of the data -> train, remainder -> test (paper: 80/20)."""
    n = len(series)
    split = int(n * train_ratio)
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test


def normalize(train: pd.Series, test: pd.Series):
    """Normalize using train mean/std (Eq. 6 in the paper)."""
    mu = train.mean()
    sigma = train.std()
    train_n = (train - mu) / sigma
    test_n = (test - mu) / sigma
    return train_n, test_n, mu, sigma


def make_windows(series: np.ndarray, T: int):
    """
    Moving window construction (Fig. 5 in the paper).
    Uses the previous T closing prices to predict the next day's close.

    Returns:
        X: shape (num_samples, T)
        y: shape (num_samples,)
    """
    X, y = [], []
    for i in range(len(series) - T):
        X.append(series[i:i + T])
        y.append(series[i + T])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_dataset(name: str, T: int = 9, start="2010-01-01", end="2020-12-31",
                     train_ratio: float = 0.8):
    """
    Full pipeline: download -> split -> normalize -> window.
    Returns a dict with everything needed for training/evaluation/backtesting.
    """
    series = download_index(name, start=start, end=end)
    assert series.values.ndim == 1, (
        f"Expected a 1D price series, got shape {series.values.shape}. "
        "This usually means yfinance returned multi-level columns."
    )
    train_raw, test_raw = train_test_split_series(series, train_ratio)
    train_n, test_n, mu, sigma = normalize(train_raw, test_raw)

    X_train, y_train = make_windows(train_n.values, T)
    X_test, y_test = make_windows(test_n.values, T)
    assert X_train.ndim == 2, f"Expected X_train to be 2D (samples, T), got shape {X_train.shape}"
    assert X_test.ndim == 2, f"Expected X_test to be 2D (samples, T), got shape {X_test.shape}"

    return {
        "name": name,
        "series_raw": series,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "mu": mu,
        "sigma": sigma,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "T": T,
    }


def denormalize(x, mu, sigma):
    return x * sigma + mu
