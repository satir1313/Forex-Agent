from __future__ import annotations

"""
Quick integrity check for data -> features pipeline.
Runs a minimal pass over M5 with 7-day lookback to keep it fast.
Prints a concise summary and basic assertions (no exceptions on failure).
"""

import os
import sys
from typing import List
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.agents.data_agent import run as data_run
from ml.agents.feature_agent import run as feat_run
from ml.agents.label_agent import run as label_run
from ml.agents.train_agent import run as train_run
from ml.agents.eval_agent import run as eval_run
from ml.utils.io import parquet_path


SYMBOLS = ["USDJPY.a", "EURAUD.a"]
TFS = ["M5"]
LOOKBACK_DAYS = 7
DATA_DIR = os.path.join("ml", "data")


def _ok(msg: str):
    print(f"[OK] {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


def _fail(msg: str):
    print(f"[FAIL] {msg}")


def main() -> None:
    print("[integrity] Running data agent...")
    try:
        data_run(symbols=SYMBOLS, timeframes=TFS, lookback_days=LOOKBACK_DAYS, data_dir=DATA_DIR)
        _ok("data_agent completed")
    except Exception as e:
        _fail(f"data_agent failed: {e}")
        return

    print("[integrity] Running feature agent...")
    try:
        feat_run(symbols=SYMBOLS, timeframes=TFS, data_dir=DATA_DIR)
        _ok("feature_agent completed")
    except Exception as e:
        _fail(f"feature_agent failed: {e}")
        return

    print("[integrity] Running label agent...")
    try:
        label_run(symbols=SYMBOLS, timeframes=TFS, data_dir=DATA_DIR)
        _ok("label_agent completed")
    except Exception as e:
        _fail(f"label_agent failed: {e}")
        return

    print("[integrity] Running train agent...")
    try:
        train_run(symbols=SYMBOLS, timeframes=TFS, data_dir=DATA_DIR, registry_dir=os.path.join("ml","models","registry"))
        _ok("train_agent completed")
    except Exception as e:
        _fail(f"train_agent failed: {e}")
        return

    print("[integrity] Running eval agent...")
    try:
        eval_run(symbols=SYMBOLS, timeframes=TFS, data_dir=DATA_DIR, registry_dir=os.path.join("ml","models","registry"), eval_frac=0.15)
        _ok("eval_agent completed")
    except Exception as e:
        _fail(f"eval_agent failed: {e}")
        return

    print("[integrity] Verifying outputs...")
    for s in SYMBOLS:
        for tf in TFS:
            raw_p = parquet_path(DATA_DIR, s, tf, kind="raw")
            interim_p = parquet_path(DATA_DIR, s, tf, kind="interim")
            feats_p = parquet_path(DATA_DIR, s, tf, kind="features")
            for p, kind in [(raw_p, "raw"), (interim_p, "interim"), (feats_p, "features")]:
                if not os.path.isfile(p):
                    _fail(f"{s} {tf}: missing {kind} parquet at {p}")
                    return
            # Load and sanity check
            raw = pd.read_parquet(raw_p)
            feats = pd.read_parquet(feats_p)
            labels_p = parquet_path(DATA_DIR, s, tf, kind="labels")
            if not os.path.isfile(labels_p):
                _fail(f"{s} {tf}: missing labels parquet at {labels_p}")
                return
            labs = pd.read_parquet(labels_p)
            if "time_utc" not in raw.columns:
                _fail(f"{s} {tf}: raw missing time_utc column")
                return
            if not raw["time_utc"].is_monotonic_increasing:
                _warn(f"{s} {tf}: raw time_utc not monotonic (may be benign after concat)")
            if raw["time_utc"].duplicated().any():
                _fail(f"{s} {tf}: raw has duplicate time_utc values")
                return
            # Feature columns spot-check
            need_cols = ["ret_1", "rsi_14", "atr_14", "macd", "hour", "dow"]
            missing = [c for c in need_cols if c not in feats.columns]
            if missing:
                _fail(f"{s} {tf}: features missing columns {missing}")
                return
            # Labels spot-check
            if not {"time_utc", "h", "label"}.issubset(set(labs.columns)):
                _fail(f"{s} {tf}: labels missing required columns")
                return
            if labs.empty:
                _fail(f"{s} {tf}: labels empty")
                return
            uniq = labs["label"].value_counts().to_dict()
            # Verify a model registry entry exists for first horizon
            # (Integrity test uses M5 with horizons from config)
            from ml.agents.train_agent import horizons_for_tf, load_defaults
            cfg = load_defaults()
            hs = horizons_for_tf(cfg, tf)
            if hs:
                h0 = hs[0]
                reg_dir = os.path.join("ml","models","registry", s, tf, str(h0))
                active = os.path.join(reg_dir, "active.json")
                if not os.path.isfile(active):
                    _fail(f"{s} {tf}: missing active.json in registry {reg_dir}")
                    return
            _ok(f"{s} {tf}: raw={raw.shape} features={feats.shape} labels={labs.shape} label_counts={uniq} registry_ok={bool(hs)}")

    print("[integrity] All checks passed.")


if __name__ == "__main__":
    main()
