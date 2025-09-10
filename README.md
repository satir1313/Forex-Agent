## FX Agent – Chat Console

Local chat-style operator console on top of your MT5 agent. This README documents all steps to get the project running on Windows using PowerShell, including running the local MCP (Model Context Protocol) Node server that the agent expects.

## Quick checklist

- Python 3.10+ installed
- MetaTrader 5 installed and logged in with the same Windows user
- Node.js (16+ or compatible) and npm installed for the MCP server
- Create and activate a Python virtualenv for the project
- Install Python dependencies (including Gradio for the UI)
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

# Gradio is required for the web UI but may not be present in requirements.txt; ensure it's installed
python -m pip install gradio
```

Note: If you prefer not to activate the venv, prefix pip/python with the venv path:

```powershell
.\.venv\Scripts\python -m pip install -r .\requirements.txt
.\.venv\Scripts\python -m pip install gradio
```

## 3) Start the MCP (local Node) server (separate PowerShell)

The project includes an MCP server under `mcp-local` used by the agent. Start it in a separate PowerShell window so it runs in the background while you use the UI.

Open a new PowerShell and run:

```powershell
cd C:\development\mt5\mcp-local
npm install
npm run start
```

This runs `tsx src/server.ts` (see `mcp-local/package.json`). Keep this PowerShell window open — the server should listen on its configured port.

If you use `pnpm`/`yarn` change the commands accordingly (or use `npx tsx src/server.ts`).

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

## Troubleshooting

- ModuleNotFoundError: No module named 'agent'
  - Fix: run using `python -m ui.chat_console` or ensure the project root is on PYTHONPATH. The repository now contains a small `__init__.py` in `agent/` and `ui/chat_console.py` adds the project root to sys.path when executed directly.

- Execution policy blocks activation of venv in PowerShell
  - Workaround: use the venv python directly (`.\.venv\Scripts\python -m ...`) or temporarily set policy for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

- Gradio import errors
  - Ensure `gradio` is installed into the same Python environment you are running. If you activated the venv then `pip install gradio` will install it into the correct venv.

- MCP server not running / agent can't connect
  - Ensure you started the server in `mcp-local` (see step 3) and keep that PowerShell open. Check its console for errors and that it reports listening on a port.

## Developer notes

- If you prefer, you can run the UI and MCP server on separate machines — update the configuration in `config/settings.py` to point at the MCP server host/port.
- The UI uses `agent.agent_bridge` to talk to the local engine (`gpt_agent.py`) which in turn connects to MetaTrader5. Make sure MT5 is running and logged in as the same Windows user.

---

If you'd like, I can also add `gradio` to `requirements.txt` and a small `scripts` section to the root (PowerShell friendly) to simplify the run steps. Tell me which you'd prefer and I'll update the repo.
