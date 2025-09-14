# ml/agents/orchestrator.py
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

# Ensure project root for relative imports when run from anywhere
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Agents (we call their run() directly)
from ml.agents import data_agent, feature_agent, label_agent, train_agent, eval_agent

DEFAULT_STEPS = ("data", "features", "labels", "train", "eval")

# ---------- helpers ----------
def _cfg_path() -> str:
    return os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")

def load_defaults() -> Dict[str, Any]:
    path = _cfg_path()
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def parse_steps(steps_csv: str) -> List[str]:
    raw = [s.strip().lower() for s in (steps_csv or "").split(",") if s.strip()]
    return [s for s in raw if s in DEFAULT_STEPS] or list(DEFAULT_STEPS)

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _deploy_active(registry_dir: str, symbol: str, timeframe: str, horizons: List[int]) -> None:
    """
    For each horizon, pick the most recently modified version folder and write active.json.
    Layout: {registry_dir}/{symbol}/{TF}/{h}/{version}/...
    """
    base_tf = os.path.join(registry_dir, symbol, timeframe.upper())
    for h in horizons:
        h_dir = os.path.join(base_tf, str(int(h)))
        if not os.path.isdir(h_dir):
            continue
        # find version subdirs
        versions = [d for d in os.listdir(h_dir) if os.path.isdir(os.path.join(h_dir, d))]
        if not versions:
            continue
        # pick latest by mtime
        versions.sort(key=lambda v: os.path.getmtime(os.path.join(h_dir, v)), reverse=True)
        chosen = versions[0]
        active_path = os.path.join(h_dir, "active.json")
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump({"version": chosen, "updated_utc": _now_utc_iso()}, f, indent=2)
        print(f"[orchestrator] deployed active.json → {symbol}/{timeframe}/{h} -> {chosen}")

# ---------- main run ----------
def run(symbols: List[str],
        timeframes: List[str],
        lookback_days: int,
        data_dir: str,
        registry_dir: str,
        steps: List[str],
        eval_frac: float = 0.1) -> None:
    cfg = load_defaults()
    horizons_map = cfg.get("horizons", {}) or {}
    print(f"[orchestrator] start steps={steps} symbols={symbols} tfs={timeframes} lookback={lookback_days}")
    print(f"[orchestrator] data_dir={data_dir} registry_dir={registry_dir}")

    if "data" in steps:
        try:
            data_agent.run(symbols=symbols, timeframes=timeframes, lookback_days=int(lookback_days), data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] data step failed: {e}")

    if "features" in steps:
        try:
            feature_agent.run(symbols=symbols, timeframes=timeframes, data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] features step failed: {e}")

    if "labels" in steps:
        try:
            label_agent.run(symbols=symbols, timeframes=timeframes, data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] labels step failed: {e}")

    if "train" in steps:
        try:
            train_agent.run(symbols=symbols, timeframes=timeframes, data_dir=data_dir, registry_dir=registry_dir)
        except Exception as e:
            print(f"[orchestrator] train step failed: {e}")

    if "eval" in steps:
        try:
            eval_agent.run(symbols=symbols, timeframes=timeframes, data_dir=data_dir, registry_dir=registry_dir, eval_frac=float(eval_frac))
        except Exception as e:
            print(f"[orchestrator] eval step failed: {e}")

    # Light deploy (write active.json) so serving can find models
    for s in symbols:
        for tf in timeframes:
            hs = [int(h) for h in horizons_map.get(tf.upper(), [])] or []
            if hs:
                _deploy_active(registry_dir, s, tf, hs)

    print("[orchestrator] Done.")

# ---------- for UI: one-call train+deploy ----------
def train_run(symbols: List[str],
              timeframes: List[str],
              lookback_days: int,
              data_dir: str,
              registry_dir: str,
              eval_frac: float = 0.1) -> None:
    steps = list(DEFAULT_STEPS)  # data -> features -> labels -> train -> eval
    run(symbols=symbols,
        timeframes=timeframes,
        lookback_days=int(lookback_days),
        data_dir=data_dir,
        registry_dir=registry_dir,
        steps=steps,
        eval_frac=float(eval_frac))


def main(argv: List[str] | None = None) -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--lookback-days", type=int, default=int(defaults.get("lookback_days", 180)))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    parser.add_argument("--registry-dir", default=os.path.join("ml", "models", "registry"))
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS), help="Comma-separated: data,features,labels,train,eval")
    parser.add_argument("--eval-frac", type=float, default=0.1)
    args = parser.parse_args(argv)

    symbols = [*args.symbols] if args.symbols else (defaults.get("symbols") or [])
    timeframes = [tf.upper() for tf in (args.timeframes or (defaults.get("timeframes") or []))]

    run(symbols=symbols,
        timeframes=timeframes,
        lookback_days=int(args.lookback_days),
        data_dir=args.data_dir,
        registry_dir=args.registry_dir,
        steps=parse_steps(args.steps),
        eval_frac=float(args.eval_frac))

if __name__ == "__main__":
    main()
