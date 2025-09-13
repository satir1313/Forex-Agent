"""
Data Agent: Fetch historical bars from MT5 for configured symbols/timeframes
and persist them to Parquet as the foundation for the ML pipeline.

Usage:
  python -m ml.agents.data_agent \
    --symbols USDJPY.a EURAUD.a \
    --timeframes M5 M15 H1 \
    --lookback-days 365 \
    --data-dir ml/data

Defaults are read from ml/config/train.yaml if not provided.

Logs progress to stdout; safe to run daily (idempotent append + dedupe).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import yaml

# Ensure project root on path so we can import gpt_agent
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gpt_agent as ga  # type: ignore
from ml.utils.io import parquet_path, append_dedupe_parquet, ensure_dir


def load_defaults() -> Dict[str, Any]:
    cfg_path = os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


def connect_mt5() -> None:
    res = ga.mt5_connect()
    if not res.get("ok"):
        # Try relaunch once
        res = ga.mt5_connect(relaunch=True)
    if not res.get("ok"):
        raise RuntimeError(f"MT5 connect failed: {res.get('error','unknown')}")


def fetch_df(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    # Reuse engine fetcher; it already ensures UTC and standard columns
    try:
        df = ga._copy_rates_recent_df(symbol, timeframe, lookback_days)  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: use copy_rates_range via engine timeframe map if private helper missing
        raise RuntimeError("Engine function _copy_rates_recent_df not available.")
    if df is None or df.empty:
        return pd.DataFrame()
    # Reset index if engine set time_utc as index to avoid ambiguity
    try:
        if getattr(df.index, "name", None) == "time_utc":
            df = df.reset_index(drop=False)
    except Exception:
        pass
    # Drop duplicate columns that can arise from reset_index when time_utc exists both
    try:
        if hasattr(df, "columns") and hasattr(df.columns, "duplicated"):
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        pass
    # Ensure time_utc exists as a column
    if "time_utc" not in df.columns:
        if "time" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        elif getattr(df.index, "name", None) == "time_utc":
            df["time_utc"] = df.index
        else:
            raise RuntimeError("Missing time_utc in MT5 data and cannot derive from index/time column")
    # Standardize columns and add metadata
    cols = [c for c in ["time_utc", "open", "high", "low", "close", "volume"] if c in df.columns]
    out = df.loc[:, cols].copy()
    out["symbol"] = symbol
    out["timeframe"] = timeframe.upper()
    # Ensure index not named time_utc and enforce stable ordering
    try:
        idx_names = list(out.index.names) if getattr(out.index, "names", None) is not None else []
        if getattr(out.index, "name", None) == "time_utc" or ("time_utc" in idx_names):
            import pandas as pd
            out.index = pd.RangeIndex(start=0, stop=len(out), step=1)
    except Exception:
        pass
    if "time_utc" in out.columns:
        out = out.sort_values("time_utc").drop_duplicates(subset=["time_utc"]) 
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = out[c].astype(float)
    return out.reset_index(drop=True)


def run(symbols: List[str], timeframes: List[str], lookback_days: int, data_dir: str) -> None:
    print(f"[data_agent] Starting. symbols={symbols} tfs={timeframes} lookback_days={lookback_days} data_dir={data_dir}")
    ensure_dir(data_dir)

    connect_mt5()
    # Enable symbols
    for s in symbols:
        try:
            _ = ga.mt5_symbol_enable(s)
        except Exception as e:
            print(f"[data_agent] warn: enable symbol {s} failed: {e}")

    for s in symbols:
        for tf in timeframes:
            try:
                df = fetch_df(s, tf, lookback_days)
                if df.empty:
                    print(f"[data_agent] {s} {tf}: no data returned")
                    continue
                raw_path = parquet_path(data_dir, s, tf, kind="raw")
                combined = append_dedupe_parquet(df, raw_path, key="time_utc")
                # For now, interim mirrors raw (cleaning step is minimal)
                interim_path = parquet_path(data_dir, s, tf, kind="interim")
                append_dedupe_parquet(combined, interim_path, key="time_utc")
                tmin = str(combined["time_utc"].min()) if "time_utc" in combined.columns else "?"
                tmax = str(combined["time_utc"].max()) if "time_utc" in combined.columns else "?"
                print(f"[data_agent] {s} {tf}: wrote {len(combined)} rows [{tmin} â†’ {tmax}] -> {raw_path}")
            except Exception as e:
                print(f"[data_agent] error: {s} {tf}: {e}")


def main(argv: List[str] | None = None) -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="MT5 Data Agent")
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []), help="Symbols to fetch")
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []), help="Timeframes to fetch")
    parser.add_argument("--lookback-days", type=int, default=int(defaults.get("lookback_days", 180)))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    timeframes = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    run(symbols=symbols, timeframes=timeframes, lookback_days=args.lookback_days, data_dir=args.data_dir)


if __name__ == "__main__":
    main()

