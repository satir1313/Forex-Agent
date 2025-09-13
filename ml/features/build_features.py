from __future__ import annotations

import math
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, pd.NA)


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = _safe_div(gain, loss)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std()
    upper = m + k * s
    lower = m - k * s
    return m, upper, lower


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def make_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute basic, leakage-safe features from an OHLCV dataframe.

    Expects columns: time_utc, open, high, low, close, volume
    Returns a dataframe keyed by time_utc with engineered columns.
    """
    need = ["time_utc", "open", "high", "low", "close", "volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    out = df.copy()
    out = out.sort_values("time_utc").reset_index(drop=True)

    # Basic returns
    close = out["close"].astype(float)
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    # Volatility
    out["vol_10"] = close.pct_change().rolling(10, min_periods=10).std()
    out["vol_20"] = close.pct_change().rolling(20, min_periods=20).std()

    # RSI / ATR / Bollinger / MACD
    out["rsi_14"] = _rsi(close, 14)
    out["atr_14"] = _atr(out["high"].astype(float), out["low"].astype(float), close, 14)
    mid, up, lo = _bollinger(close, 20, 2.0)
    out["bb_mid_20"] = mid
    out["bb_up_20"] = up
    out["bb_lo_20"] = lo
    macd, macd_sig, macd_hist = _macd(close)
    out["macd"] = macd
    out["macd_sig"] = macd_sig
    out["macd_hist"] = macd_hist

    # Candle/range features
    rng = (out["high"] - out["low"]).astype(float)
    body = (out["close"] - out["open"]).astype(float)
    out["body_rel"] = _safe_div(body, rng)
    out["rng_rel"] = _safe_div(rng, close)

    # Time features
    ts = pd.to_datetime(out["time_utc"], utc=True)
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek

    # No forward fill; keep NaNs at the start, caller may drop/trim if needed
    return out

