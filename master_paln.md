# Master Plan: ML “Model-Indicator” for FX Agent

This document outlines a pragmatic, incremental plan to add a daily‑trained ML model as a new indicator alongside the existing strategies. We’ll start with two symbols and three timeframes, implement a clean backend‑only pipeline, and orchestrate it with small, focused agents. No frontend work until we validate the signal quality.

Targets and constraints
- Symbols: USDJPY.a, EURAUD.a
- Timeframes: M5, M15, H1
- Prediction horizons:
  - M5 → 7 bars
  - M15 → 4–6 bars (three parallel horizons)
  - H1 → 2–3 bars (two parallel horizons)
- Hardware: local Intel i9, 16 GB RAM; Colab optional for sweeps
- Cadence: train daily (rolling window), deploy best model if it beats baseline

High‑level approach
- Treat the model as another strategy (“ML Model v1”), producing buy/sell/no‑trade + confidence per timeframe.
- Strict time awareness: features computed from past bars only; labels aligned strictly after features; use purged walk‑forward evaluation.
- Continuous learning: rolling window retrains daily; save versioned artifacts and calibration.

Proposed repository layout (backend only)
- `ml/`
  - `config/` — YAML/JSON configs for symbols, horizons, features, training params
  - `data/` — Parquet datasets
    - `raw/` (bars direct from MT5, partitioned by `symbol/timeframe/date`)
    - `interim/` (aligned OHLCV with cleaned timestamps, deduped)
    - `features/` (feature tables; schema versioned)
    - `labels/` (supervised targets per horizon)
  - `features/` — feature engineering library
    - `build_features.py` (OHLCV returns, RSI/MACD/ATR/Bollinger/ADX, volatility/seasonality, candle/structure, multi‑TF joins)
  - `labeling/`
    - `kbar_labels.py` (forward‑return classifiers with thresholds)
    - `triple_barrier.py` (optional; PT/SL/time barrier)
  - `models/`
    - `lightgbm/` (baseline classifier + calibration)
    - `registry/` (versioned artifacts: model.bin, feature_schema.json, calibrator.pkl, metrics.json)
  - `evaluation/`
    - `metrics.py`, `walk_forward.py` (purged rolling CV, daily promotion rules)
  - `serving/`
    - `infer.py` (load model, compute features on latest bars, emit decision/confidence)
  - `agents/`
    - `data_agent.py` (fetch bars → raw/interim)
    - `feature_agent.py` (interim → features)
    - `label_agent.py` (features → labels)
    - `train_agent.py` (labels → models; sweeps optional)
    - `eval_agent.py` (walk‑forward, compare vs baseline; decide promotion)
    - `deploy_agent.py` (atomic update of active model registry)
  - `utils/` (I/O helpers, time handling, Parquet partitioning, logging)
  - `README.md` (pipeline usage)

Data pipeline (MVP)
1) Ingest bars
   - Use existing MT5 helpers (`gpt_agent._copy_rates_recent_df`) or range‑looped MT5 calls to build a simple fetcher.
   - Store as Parquet partitioned by `symbol=timeframe=date` with UTC timestamps and a stable `bar_index`.
   - Daily incremental append (idempotent; dedupe by timestamp).

2) Clean & align
   - Ensure columns: `time_utc, open, high, low, close, volume` (float), monotonic time, no gaps within exchange calendar.
   - Persist to `ml/data/interim/`.

3) Feature engineering
   - Start set: returns (1/3/5/10), rolling z‑scores, RSI(14), MACD(12,26,9), ATR(14), Bollinger(20,2), ADX(14), momentum(10), volatility bands, hour‑of‑day, day‑of‑week.
   - Optional structure: pivot levels, swing hi/lo distance, candle bodies/wicks, range ratio.
   - Multi‑TF: for each TF sample, include a few higher‑TF aggregates (e.g., H1 RSI for M15 rows) using only completed higher‑TF bars.
   - Write to `ml/data/features/` with `feature_schema.json`.

4) Labeling
   - Forward‑return classifier per TF/horizon (no lookahead):
     - Compute `fwd_ret = close[t+h] / close[t] - 1` (using shifted indices; drop samples without full horizon).
     - Decision rules (initial):
       - buy if `fwd_ret > +θ`; sell if `fwd_ret < -θ`; else no‑trade.
       - θ scaled by volatility (e.g., `θ = 0.15 × ATR[t]/close[t]`) and includes spread/commission.
     - Store as categorical label and retain the continuous fwd_ret for analysis.
   - Optionally add triple‑barrier labels later.

5) Training (baseline)
   - Model: LightGBM classifier with class weights and early stopping; probability calibration (isotonic/Platt) post‑fit.
   - Inputs: feature set; Target: 3‑class (buy/sell/none).
   - Splits: walk‑forward or purged K‑fold (purge ≥ horizon bars; gap to reduce leakage).
   - Metrics: class precision/recall, AUC/PR, calibration Brier, expected value after costs, turnover; report per horizon and per TF.
   - Save artifacts to `ml/models/registry/{symbol}/{tf}/{horizon}/{version}/`.

6) Evaluation & promotion
   - Walk‑forward OOS performance vs baseline heuristics (e.g., majority class or a simple momentum rule).
   - Promote a model version only if it beats baseline by a margin (EV after costs, drawdown constraints).
   - Record `metrics.json` and `promotion_decision.json`.

7) Serving (backend stub)
   - `serving/infer.py` loads the active model for `(symbol, tf, horizon)` from the registry, computes features on the latest fully closed bar(s), and returns `{decision, confidence, timeframe, as_of_utc, extras}`.
   - Later, we will expose a thin adapter in `agent_bridge.py` to include “ML Model vX” as a strategy without touching the UI for now.

Agents (orchestration)
- `data_agent.py`:
  - Inputs: symbols, tfs, lookback_days
  - Tasks: connect MT5 → fetch → Parquet raw/interim (idempotent daily update)
- `feature_agent.py`:
  - Inputs: new interim partitions
  - Tasks: compute features → write features parquet
- `label_agent.py`:
  - Inputs: features parquet, horizons per TF
  - Tasks: build labels with volatility‑scaled thresholds; write labels parquet
- `train_agent.py`:
  - Inputs: features + labels
  - Tasks: train LightGBM per `(symbol, tf, horizon)`; calibrate; persist artifacts
- `eval_agent.py`:
  - Inputs: latest artifacts; OOS slice
  - Tasks: walk‑forward; compare vs baselines; produce decision
- `deploy_agent.py`:
  - Tasks: mark active model in registry (atomic symlink or `active.json` pointer)

Daily run (initial cadence)
- 00:10 UTC: Data agent updates last N days (e.g., 180–365 days rolling window per TF).
- 00:20 UTC: Feature + Label agents update affected partitions.
- 00:30 UTC: Train + Eval agents run; if promoted, update registry.
- Inference uses the latest `active` model.

Storage & performance notes
- Two symbols × three TFs × small horizons are light enough for CPU.
- LightGBM on i9/16GB easily handles months of minute data; use Colab for larger sweeps.
- Use Parquet with snappy; partition by `symbol`, `tf`, `date` to keep I/O fast.

Success criteria (MVP)
- End‑to‑end pipeline runs locally and produces a model‑indicator row for each TF/horizon.
- OOS metrics logged and versioned; daily retrains stable and reproducible.
- Inference returns calibrated confidence aligned with existing UI thresholds (min_conf).

Incremental roadmap (backend‑only)
1) Scaffold `ml/` folders, configs, and utils.
2) Implement `data_agent.py` (MT5→Parquet), unit test with 2 symbols × 3 TFs × 180 days.
3) Implement `features/build_features.py` + `feature_agent.py`.
4) Implement `labeling/kbar_labels.py` + `label_agent.py` (horizons per TF as specified).
5) Implement LightGBM baseline + calibration + `train_agent.py`.
6) Implement `evaluation/walk_forward.py` + `eval_agent.py`; define promotion rules.
7) Implement `serving/infer.py` and a minimal backend adapter (no UI change yet).
8) Add a simple CLI entry to run agents sequentially; cron/Task Scheduler later.
9) Iterate thresholds, features, and horizons based on OOS metrics.

Configuration sketch
- `ml/config/train.yaml`:
  - symbols: [USDJPY.a, EURAUD.a]
  - timeframes: [M5, M15, H1]
  - horizons: { M5: [7], M15: [4,6], H1: [2,3] }
  - lookback_days: 365 (start)  
  - costs: { spread_pips: per‑symbol, commission_per_lot: … }
  - thresholds: ATR_multiple: 0.15 (initial)
  - model: lightgbm params (num_leaves, max_depth, learning_rate, n_estimators, early_stopping_rounds)
  - cv: { type: walk_forward, n_splits: 5, purge_bars: horizon, gap_bars: horizon }

Open questions to finalize before coding
- Exact cost model (spread/commission/slippage) per symbol to bake into labels/EV.
- Minimum data horizon per TF (180 vs 365+ days) given available history.
- Multi‑TF features scope for MVP (which aggregates to include without heavy complexity).
- Promotion criterion (e.g., EV improvement ≥ X% and drawdown ≤ Y%).

Next step (per your request)
- Confirm this plan. Once approved, I will scaffold the `ml/` folders and the first agent (`data_agent.py`) without touching the frontend, and wire a simple CLI command to run it.
