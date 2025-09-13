from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Ensure project root for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gpt_agent as ga  # engine for MT5 and feature prep helpers
from ml.features.build_features import make_features


def _active_dir(registry_dir: str, symbol: str, timeframe: str, h: int) -> Optional[str]:
    base = os.path.join(registry_dir, symbol, timeframe.upper(), str(int(h)))
    ap = os.path.join(base, "active.json")
    if not os.path.isfile(ap):
        return None
    try:
        with open(ap, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ver = meta.get("version")
        if not ver:
            return None
        d = os.path.join(base, ver)
        return d if os.path.isdir(d) else None
    except Exception:
        return None


def _load_model_and_schema(version_dir: str):
    model_path = os.path.join(version_dir, "model_calibrated.joblib")
    schema_path = os.path.join(version_dir, "feature_schema.json")
    if not os.path.isfile(model_path) or not os.path.isfile(schema_path):
        return None, None
    model = joblib.load(model_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema.get("features") or []
    return model, features


def _latest_features_row(symbol: str, timeframe: str, lookback_days: int = 90) -> Optional[pd.Series]:
    # Reuse engine helper to fetch bars, then build features and return latest completed bar features
    try:
        df = ga._copy_rates_recent_df(symbol, timeframe, int(lookback_days))  # type: ignore[attr-defined]
        if df is None or df.empty:
            return None
        feats = make_features(df[["time_utc", "open", "high", "low", "close", "volume"]].copy(), timeframe)
        if feats is None or len(feats) == 0:
            return None
        return feats.iloc[-1]
    except Exception:
        return None


def _active_eval_metric(registry_dir: str, symbol: str, timeframe: str, h: int, metric: str = "balanced_accuracy") -> float:
    """Read eval.json for the active model of a horizon and return the metric value.
    Returns -inf if unavailable.
    """
    ver_dir = _active_dir(registry_dir, symbol, timeframe, h)
    if not ver_dir:
        return float("-inf")
    path = os.path.join(ver_dir, "eval.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        m = obj.get("metrics", {}) or {}
        return float(m.get(metric, m.get("balanced_accuracy", float("-inf"))))
    except Exception:
        return float("-inf")


def _proba(model, X: pd.DataFrame) -> Dict[str, float]:
    # Map probabilities to labels regardless of class_ ordering
    try:
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        out = {str(c): float(proba[i]) for i, c in enumerate(classes)}
        # Ensure keys for buy/sell/none present
        for k in ("buy", "sell", "none"):
            out.setdefault(k, 0.0)
        return out
    except Exception:
        return {"buy": 0.0, "sell": 0.0, "none": 1.0}


def infer_once(symbol: str, timeframe: str, h: int,
               registry_dir: str = os.path.join("ml", "models", "registry"),
               lookback_days: int = 180) -> Optional[Dict[str, Any]]:
    """Run a single (symbol, timeframe, horizon) inference and return a strategy-like row.

    Returns a dict: {strategy, decision, confidence, timeframe, as_of_utc, extras}
    or None if model/inputs unavailable.
    """
    ver_dir = _active_dir(registry_dir, symbol, timeframe, h)
    if not ver_dir:
        return None
    model, feature_cols = _load_model_and_schema(ver_dir)
    if model is None or not feature_cols:
        return None
    last = _latest_features_row(symbol, timeframe, lookback_days=lookback_days)
    if last is None:
        return None
    # Build input row using the saved schema
    row = pd.DataFrame([[pd.to_numeric(last.get(c), errors="coerce") for c in feature_cols]], columns=feature_cols)
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probs = _proba(model, row)
    # Choose decision between buy/sell; ignore 'none' for decision to allow downstream min_conf filter
    buy_p = probs.get("buy", 0.0); sell_p = probs.get("sell", 0.0)
    decision = "buy" if buy_p >= sell_p else "sell"
    confidence = float(max(buy_p, sell_p))
    as_of = str(last.get("time_utc"))
    return {
        "strategy": "Arvid v1",
        "decision": decision,
        "confidence": confidence,
        "timeframe": timeframe,
        "as_of_utc": as_of,
        "extras": {
            "model": "Arvid v1",
            "horizon": int(h),
            "probs": probs,
            "version_dir": ver_dir,
        },
    }


def infer_timeframes(symbol: str, timeframes: List[str],
                     horizons_map: Dict[str, List[int]],
                     registry_dir: str = os.path.join("ml", "models", "registry"),
                     lookback_days: int = 180,
                     select_policy: str = "best_eval",
                     metric: str = "balanced_accuracy") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tf in timeframes:
        hs_all = [int(h) for h in horizons_map.get(tf.upper(), [])]
        if not hs_all:
            continue
        hs: List[int]
        policy = (select_policy or "best_eval").lower()
        if policy == "all":
            hs = hs_all
        elif policy == "longest":
            hs = [max(hs_all)]
        elif policy == "shortest":
            hs = [min(hs_all)]
        elif policy == "best_eval":
            # Pick horizon with best active eval metric; fallback to longest if none
            best_h = None
            best_val = float("-inf")
            for h in hs_all:
                val = _active_eval_metric(registry_dir, symbol, tf, h, metric=metric)
                if val > best_val:
                    best_val, best_h = val, h
            hs = [best_h if best_h is not None else max(hs_all)]
        else:
            # Unknown policy: default to all
            hs = hs_all

        for h in hs:
            r = infer_once(symbol, tf, h, registry_dir=registry_dir, lookback_days=lookback_days)
            if r:
                out.append(r)
    return out
