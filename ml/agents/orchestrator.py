from __future__ import annotations

"""
Orchestrator Agent: Delegates the end-to-end pipeline in the right order.

Default steps: data -> features -> labels -> train -> eval

Usage examples:
  # Full run with config defaults
  python -m ml.agents.orchestrator

  # Limit to M5 and quick lookback
  python -m ml.agents.orchestrator --timeframes M5 --lookback-days 30

  # Run a subset of steps
  python -m ml.agents.orchestrator --steps data,features,labels
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.agents.data_agent import run as data_run
from ml.agents.feature_agent import run as feat_run
from ml.agents.label_agent import run as label_run
from ml.agents.train_agent import run as train_run
from ml.agents.eval_agent import run as eval_run


def load_defaults() -> Dict[str, Any]:
    cfg_path = os.path.join(_PROJECT_ROOT, "ml", "config", "train.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


DEFAULT_STEPS = ["data", "features", "labels", "train", "eval"]


def parse_steps(s: str | None) -> List[str]:
    if not s:
        return list(DEFAULT_STEPS)
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    return [x for x in items if x in set(DEFAULT_STEPS)]


def run(
    symbols: List[str],
    timeframes: List[str],
    lookback_days: int,
    data_dir: str,
    registry_dir: str,
    steps: List[str],
    eval_frac: float,
) -> None:
    print(
        f"[orchestrator] Start symbols={symbols} tfs={timeframes} lookback_days={lookback_days} "
        f"steps={steps} data_dir={data_dir} registry_dir={registry_dir} eval_frac={eval_frac}"
    )

    if "data" in steps:
        try:
            data_run(symbols=symbols, timeframes=timeframes, lookback_days=lookback_days, data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] data step failed: {e}")

    if "features" in steps:
        try:
            feat_run(symbols=symbols, timeframes=timeframes, data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] features step failed: {e}")

    if "labels" in steps:
        try:
            label_run(symbols=symbols, timeframes=timeframes, data_dir=data_dir)
        except Exception as e:
            print(f"[orchestrator] labels step failed: {e}")

    if "train" in steps:
        try:
            train_run(symbols=symbols, timeframes=timeframes, data_dir=data_dir, registry_dir=registry_dir)
        except Exception as e:
            print(f"[orchestrator] train step failed: {e}")

    if "eval" in steps:
        try:
            # Promote on improvement by default
            eval_run(symbols=symbols, timeframes=timeframes, data_dir=data_dir, registry_dir=registry_dir,
                     eval_frac=eval_frac, promote=True, min_delta=0.01, metric="balanced_accuracy")
        except Exception as e:
            print(f"[orchestrator] eval step failed: {e}")

    print("[orchestrator] Done.")


def main(argv: List[str] | None = None) -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--symbols", nargs="*", default=defaults.get("symbols", []))
    parser.add_argument("--timeframes", nargs="*", default=defaults.get("timeframes", []))
    parser.add_argument("--lookback-days", type=int, default=int(defaults.get("lookback_days", 180)))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", os.path.join("ml", "data")))
    parser.add_argument("--registry-dir", default=os.path.join("ml", "models", "registry"))
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS), help="Comma-separated: data,features,labels,train,eval")
    parser.add_argument("--eval-frac", type=float, default=0.15)
    args = parser.parse_args(argv)

    symbols = list(args.symbols) if args.symbols else ["USDJPY.a", "EURAUD.a"]
    timeframes = [tf.upper() for tf in (args.timeframes or ["M5", "M15", "H1"])]
    steps = parse_steps(args.steps)
    run(
        symbols=symbols,
        timeframes=timeframes,
        lookback_days=int(args.lookback_days),
        data_dir=args.data_dir,
        registry_dir=args.registry_dir,
        steps=steps,
        eval_frac=float(args.eval_frac),
    )


if __name__ == "__main__":
    main()
