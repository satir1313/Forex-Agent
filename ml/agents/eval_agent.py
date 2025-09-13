from __future__ import annotations

"""
Eval Agent: Load the active calibrated model for each (symbol, timeframe, horizon),
evaluate on a recent out-of-sample tail slice from the features+labels join,
and save eval.json under the model's version directory in the registry.

Usage:
  python -m ml.agents.eval_agent \
    --symbols USDJPY.a EURAUD.a \
    --timeframes M5 M15 H1 \
    --data-dir ml/data \
    --registry-dir ml/models/registry \
    --eval-frac 0.1
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.utils.io import parquet_path
from ml.agents.train_agent import load_defaults, horizons_for_tf  # reuse config helpers


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = {"time_utc", "symbol", "timeframe"}
    cols = [c for c in df.columns if c not in drop_cols]
    X = df[cols].copy()
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, cols


def _load_active_model(registry_dir: str, symbol: str, tf: str, h: int):
    base = os.path.join(registry_dir, symbol, tf.upper(), str(int(h)))
    active_path = os.path.join(base, "active.json")
    if not os.path.isfile(active_path):
        return None, None
    with open(active_path, "r", encoding="utf-8") as f:
        ptr = json.load(f)
    ver = ptr.get("version")
    if not ver:
        return None, None
    dir_ver = os.path.join(base, ver)
    model_path = os.path.join(dir_ver, "model_calibrated.joblib")
    if not os.path.isfile(model_path):
        return None, None
    model = joblib.load(model_path)
    return model, dir_ver


def _latest_version_dir(registry_dir: str, symbol: str, tf: str, h: int):
    h_dir = os.path.join(registry_dir, symbol, tf.upper(), str(int(h)))
    if not os.path.isdir(h_dir):
        return None
    try:
        dirs = [d for d in os.listdir(h_dir) if d.startswith("v") and os.path.isdir(os.path.join(h_dir, d))]
        if not dirs:
            return None
        latest = sorted(dirs)[-1]
        return os.path.join(h_dir, latest)
    except Exception:
        return None


def _evaluate_model(model, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=["buy", "sell", "none"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["buy", "sell", "none"]).tolist()
    return {"balanced_accuracy": bal_acc, "report": report, "confusion_matrix": cm}


def run(symbols: List[str], timeframes: List[str], data_dir: str, registry_dir: str, eval_frac: float = 0.1,
        promote: bool = True, min_delta: float = 0.01, metric: str = "balanced_accuracy") -> None:
    print(f"[eval_agent] Starting. symbols={symbols} tfs={timeframes} eval_frac={eval_frac} data_dir={data_dir} registry_dir={registry_dir}")
    cfg = load_defaults()
    eval_frac = float(eval_frac)

    for s in symbols:
        for tf in timeframes:
            feats_path = parquet_path(data_dir, s, tf, kind="features")
            labels_path = parquet_path(data_dir, s, tf, kind="labels")
            if not os.path.isfile(feats_path) or not os.path.isfile(labels_path):
                print(f"[eval_agent] {s} {tf}: missing features or labels; skip")
                continue
            feats = pd.read_parquet(feats_path)
            labels_long = pd.read_parquet(labels_path)
            # Normalize timestamps to UTC
            feats["time_utc"] = pd.to_datetime(feats["time_utc"], utc=True)
            labels_long["time_utc"] = pd.to_datetime(labels_long["time_utc"], utc=True)

            hs = horizons_for_tf(cfg, tf)
            if not hs:
                print(f"[eval_agent] {s} {tf}: no horizons configured; skip")
                continue
            for h in hs:
                try:
                    # Active model (if any)
                    active_model, active_dir = _load_active_model(registry_dir, s, tf, h)
                    candidate_dir = _latest_version_dir(registry_dir, s, tf, h)
                    candidate_model = None
                    if candidate_dir is not None:
                        cand_path = os.path.join(candidate_dir, "model_calibrated.joblib")
                        if os.path.isfile(cand_path):
                            candidate_model = joblib.load(cand_path)
                    # Prepare OOS data from the tail
                    labs = labels_long[labels_long["h"] == int(h)][["time_utc", "label"]]
                    data = pd.merge(feats, labs, on="time_utc", how="inner")
                    if len(data) < 200:
                        print(f"[eval_agent] {s} {tf} h={h}: too few samples ({len(data)})")
                        continue
                    n = len(data)
                    n_oos = max(50, int(n * eval_frac))
                    oos = data.iloc[-n_oos:]
                    X, _ = _select_features(oos.drop(columns=["label"]))
                    y = oos["label"].astype(str).values
                    # Evaluate candidate and active (if present)
                    cand_metrics = None
                    act_metrics = None
                    if candidate_model is not None and candidate_dir is not None:
                        cand_metrics = _evaluate_model(candidate_model, X, y)
                        with open(os.path.join(candidate_dir, "eval.json"), "w", encoding="utf-8") as f:
                            json.dump({
                                "symbol": s,
                                "timeframe": tf,
                                "horizon": int(h),
                                "n_oos": int(n_oos),
                                "metrics": cand_metrics,
                                "kind": "candidate",
                            }, f, indent=2)
                        print(f"[eval_agent] {s} {tf} h={h}: candidate bal_acc_oos={cand_metrics['balanced_accuracy']:.3f} -> {os.path.join(candidate_dir,'eval.json')}")
                    if active_model is not None and active_dir is not None:
                        act_metrics = _evaluate_model(active_model, X, y)
                        with open(os.path.join(active_dir, "eval.json"), "w", encoding="utf-8") as f:
                            json.dump({
                                "symbol": s,
                                "timeframe": tf,
                                "horizon": int(h),
                                "n_oos": int(n_oos),
                                "metrics": act_metrics,
                                "kind": "active",
                            }, f, indent=2)
                        print(f"[eval_agent] {s} {tf} h={h}: active bal_acc_oos={act_metrics['balanced_accuracy']:.3f} -> {os.path.join(active_dir,'eval.json')}")

                    # Promotion logic
                    def _metric(m: Dict[str, Any]) -> float:
                        if not m:
                            return float('-inf')
                        return float(m.get(metric, m.get('balanced_accuracy', 0.0)))

                    promoted = False
                    if promote and candidate_dir is not None and candidate_model is not None:
                        cand_val = _metric(cand_metrics)
                        act_val = _metric(act_metrics)
                        # If no active, promote; else require improvement >= min_delta
                        if active_model is None or (cand_val >= act_val + float(min_delta)):
                            # Write active.json pointing to candidate
                            ptr = {
                                "symbol": s,
                                "timeframe": tf.upper(),
                                "horizon": int(h),
                                "version": os.path.basename(candidate_dir),
                                "promoted_by": "eval_agent",
                            }
                            h_dir = os.path.dirname(candidate_dir)
                            with open(os.path.join(h_dir, "active.json"), "w", encoding="utf-8") as f:
                                json.dump(ptr, f, indent=2)
                            promoted = True
                    if promoted:
                        print(f"[eval_agent] {s} {tf} h={h}: promoted active -> {os.path.basename(candidate_dir)}")
                except Exception as e:
                    print(f"[eval_agent] error: {s} {tf} h={h}: {e}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Eval Agent")
    defaults = load_defaults()
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    parser.add_argument("--registry-dir", default=os.path.join("ml", "models", "registry"))
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--metric", default="balanced_accuracy")
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    tfs = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    run(symbols=symbols, timeframes=tfs, data_dir=args.data_dir, registry_dir=args.registry_dir,
        eval_frac=args.eval_frac, promote=bool(args.promote), min_delta=args.min_delta, metric=args.metric)


if __name__ == "__main__":
    main()
