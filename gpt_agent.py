# gpt_agent.py
import os
import sys
import json
import shutil
import glob
import platform
import subprocess
import webbrowser
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# MT5 / time / data
import MetaTrader5 as mt5
import pytz
import pandas as pd
from datetime import datetime, timedelta

# ====== CONFIG ======
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"  # or gpt-4o, etc.
REQUIRE_APPROVAL = True      # set False if you want auto-execution (NOT RECOMMENDED)
WORKDIR_DEFAULT = os.path.expanduser("~")
MAX_READ_BYTES = 200_000

# ====== MT5 CONFIG (env-friendly) ======
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")
MT5_TIMEOUT_MS = int(os.getenv("MT5_TIMEOUT_MS") or "60000")

UTC = pytz.timezone("Etc/UTC")

# Map friendly strings to MT5 constants
_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}

_FILLING_MAP = {"FOK": mt5.ORDER_FILLING_FOK, "IOC": mt5.ORDER_FILLING_IOC, "RETURN": mt5.ORDER_FILLING_RETURN}

client = OpenAI()

SYSTEM_PROMPT = """You are a careful local-automation agent.
You have tools to list directories, search files, read files, write files,
run commands, and open files/apps. IMPORTANT RULES:
- Never exfiltrate secrets or environment variables unless explicitly asked.
- Prefer read-only actions first. Use the least powerful tool that can do the job.
- Before destructive actions (delete/overwrite/run command), explain what will happen.
- Keep outputs concise; summarize big outputs and offer to save to a file if large.
- For any command, use safe arguments (no shell metacharacters) when possible.
- DO NOT guess file paths; ask the user if uncertain.
- If a task could be dangerous, ask for confirmation.
"""
SYSTEM_PROMPT += """
MT5 Agent Skills & Safety:
- For any MT5 task (quote, data, trade, history), do:
  1) mt5_connect (relaunch=false). If initialize fails, auto-find and open terminal, then retry connect.
  2) mt5_symbol_enable(symbol) before reading ticks or trading.
- Time handling: All MT5 copy_* history endpoints require UTC times; pass UTC ISO strings.
- For trades: Summarize (symbol, side, volume, SL/TP points, deviation, filling) before sending mt5_place_order.
- For recent bars without explicit times, use mt5_copy_rates_recent_to_csv (defaults: last 30 days).
- For huge tick exports, use chunked tools to avoid memory blowups.
"""

# ----------------- Generic Tool implementations -----------------

def norm_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def list_dir(path: str) -> Dict[str, Any]:
    p = norm_path(path or WORKDIR_DEFAULT)
    if not os.path.isdir(p):
        return {"ok": False, "error": f"Not a directory: {p}"}
    try:
        entries = []
        for name in os.listdir(p):
            full = os.path.join(p, name)
            entries.append({"name": name, "is_dir": os.path.isdir(full),
                            "size": (os.path.getsize(full) if os.path.isfile(full) else None)})
        return {"ok": True, "path": p, "entries": entries}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def search_files(root: str, pattern: str, max_results: int = 200) -> Dict[str, Any]:
    r = norm_path(root or WORKDIR_DEFAULT)
    if not os.path.isdir(r):
        return {"ok": False, "error": f"Not a directory: {r}"}
    try:
        hits = []
        for path in glob.iglob(os.path.join(r, "**", pattern), recursive=True):
            hits.append(path)
            if len(hits) >= max_results:
                break
        return {"ok": True, "root": r, "pattern": pattern, "results": hits}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def read_file(path: str, max_bytes: Optional[int] = None, encoding: str = "utf-8") -> Dict[str, Any]:
    p = norm_path(path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"Not a file: {p}"}
    limit = max_bytes if max_bytes is not None else MAX_READ_BYTES
    try:
        with open(p, "rb") as f:
            data = f.read(limit + 1)
        truncated = len(data) > limit
        text = data[:limit].decode(encoding, errors="replace")
        return {"ok": True, "path": p, "truncated": truncated, "content": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def write_file(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> Dict[str, Any]:
    p = norm_path(path)
    parent = os.path.dirname(p)
    try:
        os.makedirs(parent, exist_ok=True)
        with open(p, mode, encoding=encoding) as f:
            f.write(content)
        return {"ok": True, "path": p, "bytes": len(content.encode(encoding))}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def open_path(path: str) -> Dict[str, Any]:
    try:
        if path.startswith(("http://", "https://")):
            webbrowser.open(path)
            return {"ok": True, "opened": path, "kind": "url"}
        p = norm_path(path)
        if platform.system() == "Windows":
            os.startfile(p)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
        return {"ok": True, "opened": p}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_command(cmd: List[str], cwd: Optional[str] = None, timeout_sec: Optional[int] = 60) -> Dict[str, Any]:
    if not isinstance(cmd, list) or not cmd:
        return {"ok": False, "error": "cmd must be a non-empty list of strings"}
    try:
        proc = subprocess.run(cmd, cwd=(norm_path(cwd) if cwd else None),
                              capture_output=True, text=True, timeout=timeout_sec, shell=False)
        return {"ok": True, "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Command timed out after {timeout_sec}s"}
    except FileNotFoundError:
        return {"ok": False, "error": f"Executable not found: {cmd[0]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def execute_exe(path: str, args: Optional[List[str]] = None, cwd: Optional[str] = None, timeout_sec: Optional[int] = 60) -> Dict[str, Any]:
    p = norm_path(path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"Not a file: {p}"}
    cmd = [p] + (args or [])
    try:
        proc = subprocess.run(cmd, cwd=(norm_path(cwd) if cwd else None),
                              capture_output=True, text=True, timeout=timeout_sec, shell=False)
        return {"ok": True, "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Executable timed out after {timeout_sec}s"}
    except FileNotFoundError:
        return {"ok": False, "error": f"Executable not found: {cmd[0]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ----------------- MT5 helpers -----------------

def _iso_to_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt

# Replace your existing helper with this one (same name to avoid changing callers)
def _namedtuples_to_df(records) -> pd.DataFrame:
    """
    Robustly convert MT5 results to a pandas DataFrame.
    - Bars/Ticks: MT5 returns a NumPy structured ndarray -> use from_records.
    - Positions/Orders/Deals: tuple/list of namedtuples -> convert via _asdict.
    - Dicts: pass through.
    - Anything else: best-effort DataFrame().
    """
    import numpy as np
    # None -> empty
    if records is None:
        return pd.DataFrame()

    # NumPy structured array (bars/ticks)
    if hasattr(records, "dtype") and getattr(records.dtype, "names", None):
        # This is a numpy ndarray with named fields, e.g. time, open, high, low, close...
        try:
            return pd.DataFrame.from_records(records)
        except Exception:
            # fallback if something odd
            return pd.DataFrame(records)

    # Sequences (positions_get/orders_get/deals_get often return tuple/list of namedtuples)
    if isinstance(records, (list, tuple)):
        if len(records) == 0:
            return pd.DataFrame()
        rows = []
        for r in records:
            if hasattr(r, "_asdict"):
                rows.append(r._asdict())
            elif isinstance(r, dict):
                rows.append(r)
            else:
                # very defensive fallback for arbitrary simple objects
                rows.append({k: getattr(r, k) for k in dir(r)
                             if not k.startswith("_") and not callable(getattr(r, k))})
        return pd.DataFrame(rows)

    # Last resort
    try:
        # Works for many iterables / generators
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def _ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Cannot select symbol: {symbol}")

def _discover_mt5_terminal_candidates() -> List[str]:
    candidates = []
    # Env var first
    if MT5_TERMINAL_PATH and os.path.isfile(MT5_TERMINAL_PATH):
        candidates.append(MT5_TERMINAL_PATH)

    if platform.system() == "Windows":
        # Common Program Files locations
        patterns = [
            r"C:\Program Files\MetaTrader*\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader*\terminal64.exe",
        ]
        # Per-user installed terminals under %APPDATA%
        appdata = os.getenv("APPDATA")
        if appdata:
            patterns.append(os.path.join(appdata, "MetaQuotes", "Terminal", "**", "terminal64.exe"))

        for pat in patterns:
            for p in glob.iglob(pat, recursive=True):
                if os.path.isfile(p):
                    candidates.append(p)

    # Dedup while preserving order
    seen = set(); unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c); seen.add(c)
    return unique

# ----------------- MT5 Tool implementations -----------------

def mt5_open_terminal(path: Optional[str] = None) -> Dict[str, Any]:
    """Open MT5 terminal executable explicitly."""
    p = path or (MT5_TERMINAL_PATH if MT5_TERMINAL_PATH else None)
    if not p:
        cands = _discover_mt5_terminal_candidates()
        if not cands:
            return {"ok": False, "error": "MT5 terminal not found. Set MT5_TERMINAL_PATH or install MetaTrader 5."}
        p = cands[0]
    return execute_exe(p, args=[])

def mt5_connect(path: Optional[str] = None,
                login: Optional[int] = None,
                password: Optional[str] = None,
                server: Optional[str] = None,
                timeout_ms: Optional[int] = None,
                relaunch: bool = False) -> Dict[str, Any]:
    """Initialize connection; auto-open terminal if needed; retry once."""
    try:
        if relaunch:
            try:
                mt5.shutdown()
            except Exception:
                pass

        def _try_init(_path=None):
            kwargs = {}
            if _path:
                kwargs["path"] = _path
            elif MT5_TERMINAL_PATH:
                kwargs["path"] = MT5_TERMINAL_PATH
            if login or MT5_LOGIN:
                kwargs["login"] = int(login or MT5_LOGIN)
            if password or MT5_PASSWORD:
                kwargs["password"] = password or MT5_PASSWORD
            if server or MT5_SERVER:
                kwargs["server"] = server or MT5_SERVER
            kwargs["timeout"] = int(timeout_ms or MT5_TIMEOUT_MS)
            return mt5.initialize(**kwargs) if kwargs else mt5.initialize()

        # 1) first attempt (may autostart)
        if not _try_init():
            # 2) open terminal explicitly then retry
            term_path = path or (MT5_TERMINAL_PATH if MT5_TERMINAL_PATH else None)
            if not term_path:
                cands = _discover_mt5_terminal_candidates()
                term_path = cands[0] if cands else None
            if not term_path:
                return {"ok": False, "error": f"initialize failed: {mt5.last_error()} and no terminal found"}
            _ = execute_exe(term_path, args=[])
            time.sleep(2.5)  # give terminal a moment to boot up
            if not _try_init(term_path):
                return {"ok": False, "error": f"initialize after launch failed: {mt5.last_error()}"}

        # Optional explicit login/switch
        if login and password and server:
            if not mt5.login(login=int(login), password=password, server=server, timeout=int(timeout_ms or MT5_TIMEOUT_MS)):
                return {"ok": False, "error": f"login failed: {mt5.last_error()}"}

        acc = mt5.account_info()
        term = mt5.terminal_info()
        ver = mt5.version()
        return {
            "ok": True,
            "version": ver,
            "account_info": (acc._asdict() if acc else None),
            "terminal_info": (term._asdict() if term else None)
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_shutdown() -> Dict[str, Any]:
    try:
        mt5.shutdown(); return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_account_info() -> Dict[str, Any]:
    try:
        acc = mt5.account_info()
        if acc is None:
            return {"ok": False, "error": f"account_info unavailable: {mt5.last_error()}"}
        return {"ok": True, "account_info": acc._asdict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_symbol_enable(symbol: str) -> Dict[str, Any]:
    try:
        _ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        return {"ok": True, "symbol_info": (info._asdict() if info else None)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_tick(symbol: str) -> Dict[str, Any]:
    try:
        _ensure_symbol(symbol)
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            return {"ok": False, "error": f"tick unavailable: {mt5.last_error()}"}
        return {"ok": True, "tick": t._asdict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_place_order(symbol: str, side: str, volume: float,
                    sl_points: Optional[int] = None, tp_points: Optional[int] = None,
                    deviation: int = 20, filling: str = "RETURN",
                    comment: str = "agent order", magic: int = 990001) -> Dict[str, Any]:
    try:
        _ensure_symbol(symbol)
        sinfo = mt5.symbol_info(symbol)
        if sinfo is None:
            return {"ok": False, "error": f"symbol_info None for {symbol}"}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "error": f"symbol_info_tick None for {symbol}"}
        point = sinfo.point
        if side.lower() == "buy":
            order_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        elif side.lower() == "sell":
            order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            return {"ok": False, "error": "side must be 'buy' or 'sell'"}

        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
               "type": order_type, "price": price, "deviation": int(deviation),
               "magic": int(magic), "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": _FILLING_MAP.get(filling.upper(), mt5.ORDER_FILLING_RETURN)}
        if sl_points:
            req["sl"] = (price - sl_points*point) if order_type == mt5.ORDER_TYPE_BUY else (price + sl_points*point)
        if tp_points:
            req["tp"] = (price + tp_points*point) if order_type == mt5.ORDER_TYPE_BUY else (price - tp_points*point)

        check = mt5.order_check(req)
        if check is None: return {"ok": False, "error": f"order_check failed: {mt5.last_error()}"}
        result = mt5.order_send(req)
        if result is None: return {"ok": False, "error": f"order_send failed: {mt5.last_error()}"}
        return {"ok": result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED),
                "request": req, "check": getattr(check, "_asdict", lambda: str(check))(),
                "result": getattr(result, "_asdict", lambda: str(result))()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_positions_get(symbol: Optional[str] = None) -> Dict[str, Any]:
    try:
        kw = {"symbol": symbol} if symbol else {}
        pos = mt5.positions_get(**kw)
        df = _namedtuples_to_df(pos)
        return {"ok": True, "count": len(df), "positions": df.to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_orders_get(symbol: Optional[str] = None, group: str = "") -> Dict[str, Any]:
    try:
        kw = {}
        if symbol: kw["symbol"] = symbol
        if group: kw["group"] = group
        orders = mt5.orders_get(**kw)
        df = _namedtuples_to_df(orders)
        return {"ok": True, "count": len(df), "orders": df.to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_copy_rates_range(symbol: str, timeframe: str, start_utc: str, end_utc: str) -> Dict[str, Any]:
    try:
        tf = _TIMEFRAME_MAP[timeframe.upper()]
        _ensure_symbol(symbol)
        ts = _iso_to_utc(start_utc); te = _iso_to_utc(end_utc)
        rates = mt5.copy_rates_range(symbol, tf, ts, te)
        df = _namedtuples_to_df(rates)
        if not df.empty and "time" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return {"ok": True, "count": len(df), "rates": df.head(5).to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_copy_rates_range_to_csv(symbol: str, timeframe: str, start_utc: str, end_utc: str, out_csv_path: str) -> Dict[str, Any]:
    try:
        tf = _TIMEFRAME_MAP[timeframe.upper()]
        _ensure_symbol(symbol)
        ts = _iso_to_utc(start_utc); te = _iso_to_utc(end_utc)
        rates = mt5.copy_rates_range(symbol, tf, ts, te)
        df = _namedtuples_to_df(rates)
        os.makedirs(os.path.dirname(norm_path(out_csv_path)), exist_ok=True)
        if df.empty:
            pd.DataFrame().to_csv(norm_path(out_csv_path), index=False)
            return {"ok": True, "path": norm_path(out_csv_path), "rows": 0}
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.to_csv(norm_path(out_csv_path), index=False)
        return {"ok": True, "path": norm_path(out_csv_path), "rows": len(df)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_copy_rates_recent_to_csv(symbol: str, timeframe: str, lookback_days: int = 30, out_csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience: export last N days of bars to CSV (UTC)."""
    try:
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=int(lookback_days))
        if not out_csv_path:
            out_csv_path = os.path.join(WORKDIR_DEFAULT, "mt5_exports",
                                        f"{symbol}_{timeframe}_{start.date()}_{end.date()}.csv")
        return mt5_copy_rates_range_to_csv(symbol, timeframe, start.isoformat(), end.isoformat(), out_csv_path)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_copy_ticks_range_to_csv(symbol: str, start_utc: str, end_utc: str, out_csv_path: str,
                                flags: str = "ALL", chunk_days: int = 7) -> Dict[str, Any]:
    try:
        _ensure_symbol(symbol)
        ts = _iso_to_utc(start_utc); te = _iso_to_utc(end_utc)
        flag_map = {"ALL": mt5.COPY_TICKS_ALL, "INFO": mt5.COPY_TICKS_INFO, "TRADE": mt5.COPY_TICKS_TRADE}
        fl = flag_map.get(flags.upper(), mt5.COPY_TICKS_ALL)
        rows = []; cur = ts
        while cur < te:
            nxt = min(cur + timedelta(days=chunk_days), te)
            ticks = mt5.copy_ticks_range(symbol, cur, nxt, fl)
            if ticks is not None and len(ticks) > 0:
                rows.append(_namedtuples_to_df(ticks))
            cur = nxt
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out["time_utc"] = pd.to_datetime(out["time"], unit="s", utc=True) if not out.empty else out
        os.makedirs(os.path.dirname(norm_path(out_csv_path)), exist_ok=True)
        out.to_csv(norm_path(out_csv_path), index=False)
        return {"ok": True, "path": norm_path(out_csv_path), "rows": len(out)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_copy_ticks_last_year_to_csv(symbol: str, out_csv_path: str, flags: str = "ALL", chunk_days: int = 7) -> Dict[str, Any]:
    try:
        now = datetime.now(tz=UTC); ts = now - timedelta(days=365)
        return mt5_copy_ticks_range_to_csv(symbol, ts.isoformat(), now.isoformat(), out_csv_path, flags, chunk_days)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mt5_history_deals_to_csv(start_utc: str, end_utc: str, out_csv_path: str, group: str = "") -> Dict[str, Any]:
    try:
        ts = _iso_to_utc(start_utc); te = _iso_to_utc(end_utc)
        deals = mt5.history_deals_get(ts, te, group=group)
        df = _namedtuples_to_df(deals)
        os.makedirs(os.path.dirname(norm_path(out_csv_path)), exist_ok=True)
        df.to_csv(norm_path(out_csv_path), index=False)
        return {"ok": True, "path": norm_path(out_csv_path), "rows": len(df)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ----------------- Tool schema -----------------

TOOLS = [
    {"type":"function","function":{
        "name":"list_dir","description":"List files/folders in a directory",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":[]}
    }},
    {"type":"function","function":{
        "name":"search_files","description":"Recursively search for files matching a pattern",
        "parameters":{"type":"object","properties":{
            "root":{"type":"string"},"pattern":{"type":"string"},"max_results":{"type":"integer"}},
            "required":["root","pattern"]}
    }},
    {"type":"function","function":{
        "name":"read_file","description":"Read a file (truncated)",
        "parameters":{"type":"object","properties":{
            "path":{"type":"string"},"max_bytes":{"type":"integer"},"encoding":{"type":"string"}},
            "required":["path"]}
    }},
    {"type":"function","function":{
        "name":"write_file","description":"Write text to a file",
        "parameters":{"type":"object","properties":{
            "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"]},"encoding":{"type":"string"}},
            "required":["path","content"]}
    }},
    {"type":"function","function":{
        "name":"open_path","description":"Open a path or URL with default app",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    }},
    {"type":"function","function":{
        "name":"run_command","description":"Run a system command safely (no shell)",
        "parameters":{"type":"object","properties":{
            "cmd":{"type":"array","items":{"type":"string"}},"cwd":{"type":"string"},"timeout_sec":{"type":"integer"}},
            "required":["cmd"]}
    }},
    {"type":"function","function":{
        "name":"execute_exe","description":"Execute a local .exe with optional args",
        "parameters":{"type":"object","properties":{
            "path":{"type":"string"},"args":{"type":"array","items":{"type":"string"}},"cwd":{"type":"string"},"timeout_sec":{"type":"integer"}},
            "required":["path"]}
    }},
]

# ---------- MT5 tool schema additions ----------
TOOLS += [
    {"type":"function","function":{
        "name":"mt5_open_terminal","description":"Open the MT5 terminal executable explicitly",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_connect","description":"Initialize MT5 connection; auto-open terminal and retry if needed",
        "parameters":{"type":"object","properties":{
            "path":{"type":"string"},"login":{"type":"integer"},"password":{"type":"string"},
            "server":{"type":"string"},"timeout_ms":{"type":"integer"},"relaunch":{"type":"boolean"}},
            "required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_shutdown","description":"Shutdown MT5 IPC connection",
        "parameters":{"type":"object","properties":{},"required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_account_info","description":"Return current account information",
        "parameters":{"type":"object","properties":{},"required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_symbol_enable","description":"Ensure a symbol is visible/ready in MarketWatch",
        "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}
    }},
    {"type":"function","function":{
        "name":"mt5_tick","description":"Get best bid/ask tick snapshot",
        "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}
    }},
    {"type":"function","function":{
        "name":"mt5_place_order","description":"Place a market order with optional SL/TP points",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"side":{"type":"string","enum":["buy","sell"]},"volume":{"type":"number"},
            "sl_points":{"type":"integer"},"tp_points":{"type":"integer"},
            "deviation":{"type":"integer"},"filling":{"type":"string","enum":["FOK","IOC","RETURN"]},
            "comment":{"type":"string"},"magic":{"type":"integer"}},
            "required":["symbol","side","volume"]}
    }},
    {"type":"function","function":{
        "name":"mt5_positions_get","description":"List open positions (optionally filter by symbol)",
        "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_orders_get","description":"List active orders (optionally filter by symbol/group)",
        "parameters":{"type":"object","properties":{"symbol":{"type":"string"},"group":{"type":"string"}},"required":[]}
    }},
    {"type":"function","function":{
        "name":"mt5_copy_rates_range","description":"Fetch bars in [start,end) UTC and return preview",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"timeframe":{"type":"string","enum":["M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30","H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"]},
            "start_utc":{"type":"string"},"end_utc":{"type":"string"}},
            "required":["symbol","timeframe","start_utc","end_utc"]}
    }},
    {"type":"function","function":{
        "name":"mt5_copy_rates_range_to_csv","description":"Export bars in UTC range to CSV",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"timeframe":{"type":"string"},
            "start_utc":{"type":"string"},"end_utc":{"type":"string"},"out_csv_path":{"type":"string"}},
            "required":["symbol","timeframe","start_utc","end_utc","out_csv_path"]}
    }},
    {"type":"function","function":{
        "name":"mt5_copy_rates_recent_to_csv","description":"Export last N days of bars (UTC) to CSV (defaults: N=30)",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"timeframe":{"type":"string"},
            "lookback_days":{"type":"integer"},"out_csv_path":{"type":"string"}},
            "required":["symbol","timeframe"]}
    }},
    {"type":"function","function":{
        "name":"mt5_copy_ticks_range_to_csv","description":"Export ticks in UTC range to CSV (chunked)",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"start_utc":{"type":"string"},"end_utc":{"type":"string"},
            "out_csv_path":{"type":"string"},"flags":{"type":"string","enum":["ALL","INFO","TRADE"]},
            "chunk_days":{"type":"integer"}},
            "required":["symbol","start_utc","end_utc","out_csv_path"]}
    }},
    {"type":"function","function":{
        "name":"mt5_copy_ticks_last_year_to_csv","description":"Export last 365 days of ticks to CSV (UTC, chunked)",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"out_csv_path":{"type":"string"},
            "flags":{"type":"string","enum":["ALL","INFO","TRADE"]},"chunk_days":{"type":"integer"}},
            "required":["symbol","out_csv_path"]}
    }},
    {"type":"function","function":{
        "name":"mt5_history_deals_to_csv","description":"Export account deals history [start,end) UTC to CSV",
        "parameters":{"type":"object","properties":{
            "start_utc":{"type":"string"},"end_utc":{"type":"string"},
            "out_csv_path":{"type":"string"},"group":{"type":"string"}},
            "required":["start_utc","end_utc","out_csv_path"]}
    }},
]

def approve_or_deny(tool_name: str, args: Dict[str, Any]) -> bool:
    if not REQUIRE_APPROVAL:
        return True
    print("\n--- TOOL REQUEST --------------------------------")
    print(f"Tool: {tool_name}")
    print("Args:", json.dumps(args, indent=2))
    print("-------------------------------------------------")
    resp = input("Approve? [y/N]: ").strip().lower()
    return resp == "y"

def call_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        func = tool_call.get("function", {})
        name = func.get("name")
        args_str = func.get("arguments") or "{}"
    else:
        name = getattr(tool_call.function, "name", None)
        args_str = getattr(tool_call.function, "arguments", None) or "{}"
    try:
        args = json.loads(args_str)
    except Exception:
        args = {}

    if not approve_or_deny(name, args):
        return {"ok": False, "error": "User denied execution."}

    # Generic tools
    if name == "list_dir":                      return list_dir(**args)
    if name == "search_files":                  return search_files(**args)
    if name == "read_file":                     return read_file(**args)
    if name == "write_file":                    return write_file(**args)
    if name == "open_path":                     return open_path(**args)
    if name == "run_command":                   return run_command(**args)
    if name == "execute_exe":                   return execute_exe(**args)

    # MT5 tools
    if name == "mt5_open_terminal":             return mt5_open_terminal(**args)
    if name == "mt5_connect":                   return mt5_connect(**args)
    if name == "mt5_shutdown":                  return mt5_shutdown()
    if name == "mt5_account_info":              return mt5_account_info()
    if name == "mt5_symbol_enable":             return mt5_symbol_enable(**args)
    if name == "mt5_tick":                      return mt5_tick(**args)
    if name == "mt5_place_order":               return mt5_place_order(**args)
    if name == "mt5_positions_get":             return mt5_positions_get(**args)
    if name == "mt5_orders_get":                return mt5_orders_get(**args)
    if name == "mt5_copy_rates_range":          return mt5_copy_rates_range(**args)
    if name == "mt5_copy_rates_range_to_csv":   return mt5_copy_rates_range_to_csv(**args)
    if name == "mt5_copy_rates_recent_to_csv":  return mt5_copy_rates_recent_to_csv(**args)
    if name == "mt5_copy_ticks_range_to_csv":   return mt5_copy_ticks_range_to_csv(**args)
    if name == "mt5_copy_ticks_last_year_to_csv": return mt5_copy_ticks_last_year_to_csv(**args)
    if name == "mt5_history_deals_to_csv":      return mt5_history_deals_to_csv(**args)

    return {"ok": False, "error": f"Unknown tool: {name}"}

def _assistant_msg_with_tool_calls(msg):
    if not getattr(msg, "tool_calls", None):
        return {"role": "assistant", "content": msg.content or ""}
    tool_calls_payload = []
    for tc in msg.tool_calls:
        tool_calls_payload.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
        })
    return {"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls_payload}

def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Local agent ready. Model: {MODEL}")
    print("Type your request (or 'exit'): ")
    while True:
        user = input("> ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            print("Bye.")
            break
        messages.append({"role": "user", "content": user})

        # Keep looping tool calls until model is done
        while True:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2,
            )
            msg = response.choices[0].message
            if getattr(msg, "tool_calls", None):
                messages.append(_assistant_msg_with_tool_calls(msg))
                for tc in msg.tool_calls:
                    result = call_tool({"function": {"name": tc.function.name, "arguments": tc.function.arguments}})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)[:120000]})
                continue
            text = (msg.content or "").strip()
            if text:
                print("\n" + text + "\n")
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

if __name__ == "__main__":
    try:
        chat_loop()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
