from __future__ import annotations

"""
Train Agent: Train a 3-class classifier (buy/sell/none) per (symbol, timeframe, horizon)
using engineered features and k-bar labels. Saves artifacts and metrics to a
versioned registry under ml/models/registry.

Algorithm: sklearn HistGradientBoostingClassifier with class weights +
isotonic calibration on a time-based validation split.

Usage:
  python -m ml.agents.train_agent \
    --symbols USDJPY.a EURAUD.a \
    --timeframes M5 M15 H1 \
    --data-dir ml/data \
    --registry-dir ml/models/registry
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

# Prefer FrozenEstimator on sklearn >=1.6 to avoid cv='prefit' deprecation
try:
    from sklearn.frozen import FrozenEstimator  # correct for sklearn >= 1.6
    _HAVE_FROZEN = True
except Exception:  # sklearn < 1.6
    _HAVE_FROZEN = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.utils.io import parquet_path, ensure_dir


def load_defaults() -> Dict[str, Any]:
    cfg_path = os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


def horizons_for_tf(cfg: Dict[str, Any], tf: str) -> List[int]:
    hmap = (cfg or {}).get("horizons", {}) or {}
    return [int(x) for x in hmap.get(tf.upper(), [])]


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Exclude non-feature columns
    drop_cols = {"time_utc", "symbol", "timeframe"}
    cols = [c for c in df.columns if c not in drop_cols]
    X = df[cols].copy()
    # Ensure numeric dtype (coerce where needed)
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Replace inf/NaN with zeros for tree models
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, cols


def _time_split(n: int, valid_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n_valid = max(1, int(n * valid_frac))
    idx = np.arange(n)
    return idx[:-n_valid], idx[-n_valid:]


def _train_one(symbol: str, tf: str, h: int, feats: pd.DataFrame, labels_long: pd.DataFrame,
               registry_dir: str) -> Dict[str, Any]:
    # Prepare dataset by joining features with labels for specific horizon
    feats = feats.copy()
    labels_long = labels_long.copy()
    # Normalize timezones to UTC
    try:
        feats["time_utc"] = pd.to_datetime(feats["time_utc"], utc=True)
    except Exception:
        pass
    try:
        labels_long["time_utc"] = pd.to_datetime(labels_long["time_utc"], utc=True)
    except Exception:
        pass
    labs = labels_long[labels_long["h"] == int(h)][["time_utc", "label"]].copy()
    data = pd.merge(feats, labs, on="time_utc", how="inner")
    if len(data) < 200:
        return {"ok": False, "error": f"not enough samples after join: {len(data)}"}

    X, feature_cols = _select_features(data.drop(columns=["label"]))
    y = data["label"].astype(str).values

    # Encode classes consistently (string labels are fine for sklearn)
    classes = np.array(["buy", "sell", "none"], dtype=object)

    # Time-based split
    train_idx, valid_idx = _time_split(len(X), valid_frac=0.2)
    Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
    ytr, yva = y[train_idx], y[valid_idx]

    # Base classifier
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    # ---- Calibration on validation split (no leakage) ----
    # sklearn >=1.6: use FrozenEstimator to avoid the cv='prefit' deprecation
    if _HAVE_FROZEN:
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    else:  # fallback for older sklearn
        cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)

    # Metrics on validation
    yhat = cal.predict(Xva)
    bal_acc = float(balanced_accuracy_score(yva, yhat))
    report = classification_report(yva, yhat, labels=list(classes), output_dict=True, zero_division=0)

    # Persist artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(registry_dir, symbol, tf.upper(), str(int(h)), f"v{ts}")
    ensure_dir(base_dir)
    joblib.dump(clf, os.path.join(base_dir, "model_base.joblib"))
    joblib.dump(cal, os.path.join(base_dir, "model_calibrated.joblib"))
    with open(os.path.join(base_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    metrics = {
        "n_train": int(len(Xtr)),
        "n_valid": int(len(Xva)),
        "balanced_accuracy": bal_acc,
        "report": report,
        "classes": list(classes),
    }
    with open(os.path.join(base_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Record last_trained pointer (promotion is handled by eval_agent)
    last_ptr = {
        "symbol": symbol,
        "timeframe": tf.upper(),
        "horizon": int(h),
        "version": f"v{ts}",
        "created_utc": ts,
    }
    h_dir = os.path.join(registry_dir, symbol, tf.upper(), str(int(h)))
    ensure_dir(h_dir)
    with open(os.path.join(h_dir, "last_trained.json"), "w", encoding="utf-8") as f:
        json.dump(last_ptr, f, indent=2)

    return {"ok": True, "base_dir": base_dir, "metrics": metrics}


def run(symbols: List[str], timeframes: List[str], data_dir: str, registry_dir: str) -> None:
    print(f"[train_agent] Starting. symbols={symbols} tfs={timeframes} data_dir={data_dir} registry_dir={registry_dir}")
    cfg = load_defaults()
    ensure_dir(registry_dir)

    for s in symbols:
        for tf in timeframes:
            feats_path = parquet_path(data_dir, s, tf, kind="features")
            labels_path = parquet_path(data_dir, s, tf, kind="labels")
            if not os.path.isfile(feats_path) or not os.path.isfile(labels_path):
                print(f"[train_agent] {s} {tf}: missing features or labels; skip")
                continue
            feats = pd.read_parquet(feats_path)
            labels = pd.read_parquet(labels_path)
            hs = horizons_for_tf(cfg, tf)
            if not hs:
                print(f"[train_agent] {s} {tf}: no horizons configured; skip")
                continue
            for h in hs:
                try:
                    res = _train_one(s, tf, h, feats, labels, registry_dir)
                    if not res.get("ok"):
                        print(f"[train_agent] {s} {tf} h={h}: failed: {res.get('error')}")
                    else:
                        m = res.get("metrics", {})
                        print(f"[train_agent] {s} {tf} h={h}: ok, bal_acc={m.get('balanced_accuracy'):.3f}")
                except Exception as e:
                    print(f"[train_agent] error: {s} {tf} h={h}: {e}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Agent")
    defaults = load_defaults()
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    parser.add_argument("--registry-dir", default=os.path.join("ml", "models", "registry"))
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    tfs = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    run(symbols=symbols, timeframes=tfs, data_dir=args.data_dir, registry_dir=args.registry_dir)


if __name__ == "__main__":
    main()
