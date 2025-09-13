## FX Agent – Chat Console + ML Pipeline (Arvid v1)

Local chat-style operator console on top of your MT5 agent, now with a backend machine‑learning pipeline that produces an additional strategy signal called “Arvid v1”. This README covers setup, running the UI, and the end‑to‑end ML workflow (data → features → labels → train → eval → serve).

## Quick checklist

- Python 3.10+ installed
- MetaTrader 5 installed and logged in with the same Windows user
- Node.js (16+) and npm installed for the MCP server
- Create and activate a Python virtualenv for the project
- Install Python dependencies (Gradio + Parquet + ML)
- Start the MCP server in a separate PowerShell
- Run the UI with the project venv active

## 1) Create & activate a Python virtual environment (recommended)

Open PowerShell in the project root (C:\development\mt5) and run:

```powershell
python -m venv .venv
# Option A: activate for this PowerShell session (preferred)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1

# Option B: do not change execution policy — use the venv python directly
# .\.venv\Scripts\python -m <command>
```

If your system blocks script execution, either use Option B above or temporarily allow activation for the session with the Set-ExecutionPolicy command shown.

## 2) Install Python dependencies

With the venv activated (or by using the venv python), install required Python packages:

```powershell
# recommended (when venv active)
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt

# Engines for Parquet IO and ML (required for the ML pipeline)
python -m pip install pyarrow fastparquet scikit-learn joblib gradio
```

If you prefer not to activate the venv, prefix pip/python with the venv path:

```powershell
.\.venv\Scripts\python -m pip install -r .\requirements.txt
.\.venv\Scripts\python -m pip install pyarrow fastparquet scikit-learn joblib gradio
```

## 3) Start the MCP (local Node) server (separate PowerShell)

The project includes an MCP server under `mcp-local` used by the agent. Start it in a separate PowerShell window so it runs in the background while you use the UI.

```powershell
cd C:\development\mt5\mcp-local
npm install
npm run start
```

This runs `tsx src/server.ts` (see `mcp-local/package.json`). Keep this PowerShell window open — the server should listen on its configured port.

## 4) Run the UI (Console)

Back in your main project PowerShell (with the venv active) run the UI. Two recommended ways:

- Preferred (module-run, resolves package imports):

```powershell
python -m ui.chat_console
```

- Alternative (script-run) — the repository includes a small sys.path fix so running the file directly works too:

```powershell
python ui/chat_console.py
```

If you did not activate the venv, run using the venv python explicitly:

```powershell
.\.venv\Scripts\python -m ui.chat_console
```

When the UI starts, open the displayed Gradio link in a browser.

### Arvid v1 in the UI
- Once you have trained models and promoted an active version (see ML workflow below), the backend automatically appends an “Arvid v1” row per timeframe to the normal strategy results returned by `agent_bridge.evaluate`. No UI change is required. The row includes:
  - strategy: `Arvid v1`
  - decision: `buy`/`sell` (argmax of calibrated probs)
  - confidence: calibrated probability of the chosen side
  - timeframe, as_of_utc
  - extras: horizon used, per‑class probabilities, and model version path

## Troubleshooting

- ModuleNotFoundError: No module named 'agent'
  - Fix: run using `python -m ui.chat_console` or ensure the project root is on PYTHONPATH. The repository includes `__init__.py` and `ui/chat_console.py` adds the project root to sys.path when executed directly.

- Execution policy blocks activation of venv in PowerShell
  - Workaround: use the venv python directly (`.\.venv\Scripts\python -m ...`) or temporarily set policy for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

- Parquet engine errors
  - Install `pyarrow` or `fastparquet` in your venv (see step 2). The pipeline reads/writes Parquet files.

- “time_utc is both an index level and a column label”
  - This is handled in the agents, but if you see it elsewhere, reset the index and ensure `time_utc` is a plain column before sorting/deduping.

- M5 returns no data
  - MT5 often needs you to open the symbol’s M5 chart and scroll back to force history download. Also verify the exact symbol suffix in Market Watch (e.g., `USDJPY.a`).

## Developer notes

- You can run the UI and MCP server on separate machines — update configuration in `config/settings.py` to point at the MCP server host/port.
- The UI uses `agent.agent_bridge` to talk to the local engine (`gpt_agent.py`) which in turn connects to MetaTrader5. Make sure MT5 is running and logged in as the same Windows user.

---

## ML Pipeline Overview (Arvid v1)

The ML pipeline produces an additional indicator “Arvid v1” that learns from historical OHLCV and technical features. It is organized under `ml/` and runs as a sequence of small agents.

### Configuration
- `ml/config/train.yaml`
  - `symbols`: e.g., `[USDJPY.a, EURAUD.a]`
  - `timeframes`: `[M5, M15, H1]`
  - `horizons`: `{ M5: [7], M15: [4,6], H1: [2,3] }`
  - `lookback_days`: default for data fetching
  - `data_dir`: default `ml/data`
  - `threshold_k`: ATR multiple per TF (labeling), default: `M5:0.12, M15:0.14, H1:0.16`

### Data layout
- `ml/data/raw/{symbol}/{TF}.parquet`: OHLCV with `time_utc, open, high, low, close, volume`
- `ml/data/interim/...`: cleaned/aligned (initially mirrors raw)
- `ml/data/features/...`: engineered features (returns, RSI, ATR, Bollinger, MACD, candle, hour, dow)
- `ml/data/labels/...`: k‑bar forward labels in long format (`time_utc, h, fwd_ret, theta, label`)
- `ml/models/registry/{symbol}/{TF}/{h}/vYYYYMMDD_HHMMSS/`: trained models + metrics
  - `model_base.joblib`, `model_calibrated.joblib`, `feature_schema.json`, `metrics.json`, `eval.json`
  - `active.json` (written by eval agent on promotion)
  - `last_trained.json` (latest trained candidate)

### Agents
- `ml/agents/data_agent.py`: fetch MT5 bars → raw/interim (Parquet)
- `ml/agents/feature_agent.py`: interim → engineered features
- `ml/agents/label_agent.py`: features → k‑bar labels (no lookahead)
- `ml/agents/train_agent.py`: fit 3‑class classifier (buy/sell/none) with calibration; save artifacts
- `ml/agents/eval_agent.py`: evaluate candidate vs active on OOS slice; promote on improvement
- `ml/agents/orchestrator.py`: run the steps in order with one command

### Run the full pipeline

Using defaults from config:

```powershell
# end-to-end with promotion
.\.venv\Scripts\python.exe -m ml.agents.orchestrator

# limit to M5 and shorter lookback for quick iteration
.\.venv\Scripts\python.exe -m ml.agents.orchestrator --symbols USDJPY.a EURAUD.a --timeframes M5 --lookback-days 7
```

Run steps individually if needed:

```powershell
#.\.venv\Scripts\python.exe -m ml.agents.data_agent --symbols USDJPY.a EURAUD.a --timeframes M15 H1 --lookback-days 90
#.\.venv\Scripts\python.exe -m ml.agents.feature_agent --symbols USDJPY.a EURAUD.a --timeframes M15 H1
#.\.venv\Scripts\python.exe -m ml.agents.label_agent   --symbols USDJPY.a EURAUD.a --timeframes M15 H1
.\.venv\Scripts\python.exe -m ml.agents.train_agent   --symbols USDJPY.a EURAUD.a --timeframes M5 M15 H1
.\.venv\Scripts\python.exe -m ml.agents.eval_agent    --symbols USDJPY.a EURAUD.a --timeframes M5 M15 H1 --promote --eval-frac 0.15
```

### Serving (Arvid v1)
- Backend code automatically appends Arvid v1 rows to `agent_bridge.evaluate()` if an active model exists.
- Horizon selection: by default, the best evaluated horizon per TF is chosen (based on `balanced_accuracy`). The chosen horizon is shown in `extras.horizon`.

### Integrity test

Quick end‑to‑end validation on M5:

```powershell
.\.venv\Scripts\python.exe -m ml.tests.integrity_test
```

This runs data → features → labels → train → eval and checks files, shapes, and registry pointers.

### Promotion rules
- `train_agent` only writes versioned models and `last_trained.json`.
- `eval_agent` evaluates both the latest candidate and the active model and promotes the candidate to `active.json` when it improves the chosen metric by at least `min_delta` (default 0.01).

### Notes
- Calibration uses isotonic regression on a validation split; sklearn warns about `cv='prefit'` deprecation — acceptable for MVP.
- M5 weekend: historical bars are still retrievable; “no data returned” typically means MT5 has not downloaded history yet — open the chart and scroll back.

