from __future__ import annotations

"""
Feature Agent: Read interim OHLCV parquet, compute engineered features,
and persist to ml/data/features, per symbol and timeframe.

Usage:
  python -m ml.agents.feature_agent \
    --symbols USDJPY.a EURAUD.a \
    --timeframes M5 M15 H1 \
    --data-dir ml/data
"""

import argparse
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.features.build_features import make_features
from ml.utils.io import parquet_path, append_dedupe_parquet, ensure_dir


def load_defaults() -> Dict[str, Any]:
    cfg_path = os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


def run(symbols: List[str], timeframes: List[str], data_dir: str) -> None:
    print(f"[feature_agent] Starting. symbols={symbols} tfs={timeframes} data_dir={data_dir}")
    ensure_dir(data_dir)

    for s in symbols:
        for tf in timeframes:
            try:
                interim_path = parquet_path(data_dir, s, tf, kind="interim")
                if not os.path.isfile(interim_path):
                    print(f"[feature_agent] {s} {tf}: no interim parquet at {interim_path}; skip")
                    continue
                df = pd.read_parquet(interim_path)
                if df is None or len(df) == 0:
                    print(f"[feature_agent] {s} {tf}: interim empty; skip")
                    continue
                # Ensure required columns exist and types are consistent
                if "time_utc" not in df.columns:
                    if getattr(df.index, "name", None) == "time_utc":
                        df = df.reset_index(drop=False)
                    else:
                        raise RuntimeError("interim parquet missing time_utc column")
                # Build features
                feats = make_features(df[["time_utc", "open", "high", "low", "close", "volume"]].copy(), tf)
                # Persist
                features_path = parquet_path(data_dir, s, tf, kind="features")
                combined = append_dedupe_parquet(feats, features_path, key="time_utc")
                tmin = str(pd.to_datetime(combined["time_utc"]).min()) if "time_utc" in combined.columns else "?"
                tmax = str(pd.to_datetime(combined["time_utc"]).max()) if "time_utc" in combined.columns else "?"
                print(f"[feature_agent] {s} {tf}: wrote {len(combined)} rows [{tmin} -> {tmax}] -> {features_path}")
            except Exception as e:
                print(f"[feature_agent] error: {s} {tf}: {e}")


def main(argv: List[str] | None = None) -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Feature Agent")
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    timeframes = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    run(symbols=symbols, timeframes=timeframes, data_dir=args.data_dir)


if __name__ == "__main__":
    main()

