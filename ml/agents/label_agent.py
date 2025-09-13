from __future__ import annotations

"""
Label Agent: Read features parquet, compute k-bar labels per TF/horizon,
and persist to ml/data/labels in long format (time_utc × h).

Usage:
  python -m ml.agents.label_agent \
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

from ml.labeling.kbar_labels import compute_kbar_labels
from ml.utils.io import parquet_path, ensure_dir


def load_defaults() -> Dict[str, Any]:
    cfg_path = os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


def _k_for_tf(cfg: Dict[str, Any], tf: str, default: float = 0.15) -> float:
    m = (cfg or {}).get("threshold_k", {}) or {}
    return float(m.get(tf.upper(), default))


def _horizons_for_tf(cfg: Dict[str, Any], tf: str) -> List[int]:
    hmap = (cfg or {}).get("horizons", {}) or {}
    hs = hmap.get(tf.upper(), [])
    return [int(x) for x in hs]


def _write_labels_long(path: str, df_new: pd.DataFrame) -> None:
    """Append/dedupe by (time_utc, h) without changing io helper signature."""
    ensure_dir(os.path.dirname(path))
    if os.path.isfile(path):
        try:
            old = pd.read_parquet(path)
        except Exception:
            old = pd.DataFrame(columns=["time_utc", "h", "fwd_ret", "theta", "label"])
        combined = pd.concat([old, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time_utc", "h"]).sort_values(["time_utc", "h"]).reset_index(drop=True)
    else:
        combined = df_new.sort_values(["time_utc", "h"]).reset_index(drop=True)
    combined.to_parquet(path, index=False)


def run(symbols: List[str], timeframes: List[str], data_dir: str) -> None:
    print(f"[label_agent] Starting. symbols={symbols} tfs={timeframes} data_dir={data_dir}")
    ensure_dir(data_dir)
    cfg = load_defaults()

    for s in symbols:
        for tf in timeframes:
            try:
                feats_path = parquet_path(data_dir, s, tf, kind="features")
                if not os.path.isfile(feats_path):
                    print(f"[label_agent] {s} {tf}: no features parquet at {feats_path}; skip")
                    continue
                feats = pd.read_parquet(feats_path)
                if feats is None or len(feats) == 0:
                    print(f"[label_agent] {s} {tf}: features empty; skip")
                    continue
                hs = _horizons_for_tf(cfg, tf)
                if not hs:
                    print(f"[label_agent] {s} {tf}: no horizons configured; skip")
                    continue
                k = _k_for_tf(cfg, tf, default=0.15)
                # Compute labels (requires time_utc, close, atr_14)
                src = feats[["time_utc", "close", "atr_14"]].copy()
                labels = compute_kbar_labels(src, horizons=hs, k=float(k))
                if labels.empty:
                    print(f"[label_agent] {s} {tf}: labels empty; skip")
                    continue
                out_path = parquet_path(data_dir, s, tf, kind="labels")
                _write_labels_long(out_path, labels)
                # Stats
                counts = labels["label"].value_counts().to_dict()
                print(f"[label_agent] {s} {tf}: wrote {len(labels)} label rows across horizons {hs} (counts={counts}) -> {out_path}")
            except Exception as e:
                print(f"[label_agent] error: {s} {tf}: {e}")


def main(argv: List[str] | None = None) -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Label Agent")
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    timeframes = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    run(symbols=symbols, timeframes=timeframes, data_dir=args.data_dir)


if __name__ == "__main__":
    main()

