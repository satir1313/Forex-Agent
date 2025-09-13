import os
from typing import Optional
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parquet_path(base_dir: str, symbol: str, timeframe: str, kind: str = "raw") -> str:
    """Return a canonical parquet path like ml/data/{kind}/{symbol}/{timeframe}.parquet"""
    safe_tf = timeframe.upper()
    out_dir = os.path.join(base_dir, kind, symbol)
    ensure_dir(out_dir)
    return os.path.join(out_dir, f"{safe_tf}.parquet")


def append_dedupe_parquet(df: pd.DataFrame, path: str, key: str = "time_utc") -> pd.DataFrame:
    """Append to an existing Parquet file and drop duplicates by key.

    Returns the combined dataframe actually written.
    """
    df_new = df.copy()
    # Ensure a simple RangeIndex and unique columns
    try:
        df_new = df_new.reset_index(drop=True)
    except Exception:
        pass
    try:
        if hasattr(df_new.columns, "duplicated") and df_new.columns.duplicated().any():
            df_new = df_new.loc[:, ~df_new.columns.duplicated()].copy()
    except Exception:
        pass
    if key in df_new.columns:
        df_new = df_new.drop_duplicates(subset=[key])
    if os.path.isfile(path):
        try:
            old = pd.read_parquet(path)
            try:
                old = old.reset_index(drop=True)
            except Exception:
                pass
            try:
                if hasattr(old.columns, "duplicated") and old.columns.duplicated().any():
                    old = old.loc[:, ~old.columns.duplicated()].copy()
            except Exception:
                pass
        except Exception:
            old = pd.DataFrame()
        if not old.empty:
            combined = pd.concat([old, df_new], axis=0, ignore_index=True)
            if key in combined.columns:
                combined = combined.drop_duplicates(subset=[key]).sort_values(key)
            else:
                combined = combined.drop_duplicates().reset_index(drop=True)
        else:
            combined = df_new
    else:
        combined = df_new
    # Ensure directory exists then write
    ensure_dir(os.path.dirname(path))
    combined.to_parquet(path, index=False)
    return combined
