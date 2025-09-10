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
from typing import List, Dict, Any, Optional, Callable, Tuple
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# MT5 / time / data
import MetaTrader5 as mt5
import pytz
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Optional technical indicators
try:
    from ta import add_all_ta_features
    from ta.utils import dropna
    _TA_AVAILABLE = True
except Exception:
    _TA_AVAILABLE = False

# ====== CONFIG ======
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
REQUIRE_APPROVAL = True
WORKDIR_DEFAULT = os.path.expanduser("~")
MAX_READ_BYTES = 200_000

# ====== MT5 CONFIG (env-friendly) ======
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")
MT5_TIMEOUT_MS = int(os.getenv("MT5_TIMEOUT_MS") or "60000")

UTC = pytz.timezone("Etc/UTC")

import MetaTrader5 as mt5

# --- Safe retcode mapping (handles older MetaTrader5 wheels) ---
def _build_retcode_labels():
    import MetaTrader5 as mt5
    name_to_label = {
        "TRADE_RETCODE_DONE": "DONE",
        "TRADE_RETCODE_PLACED": "PLACED",
        "TRADE_RETCODE_REJECT": "REJECT",
        "TRADE_RETCODE_CANCEL": "CANCEL",
        "TRADE_RETCODE_INVALID": "INVALID",
        "TRADE_RETCODE_INVALID_VOLUME": "INVALID_VOLUME",
        "TRADE_RETCODE_INVALID_PRICE": "INVALID_PRICE",
        "TRADE_RETCODE_INVALID_STOPS": "INVALID_STOPS",
        "TRADE_RETCODE_MARKET_CLOSED": "MARKET_CLOSED",
        "TRADE_RETCODE_NO_MONEY": "NO_MONEY",
        "TRADE_RETCODE_PRICE_CHANGED": "PRICE_CHANGED",
        "TRADE_RETCODE_OFFQUOTES": "OFFQUOTES",
        "TRADE_RETCODE_TRADE_DISABLED": "TRADE_DISABLED",
        "TRADE_RETCODE_TRADE_TIMEOUT": "TRADE_TIMEOUT",
        "TRADE_RETCODE_REQUOTE": "REQUOTE",
        "TRADE_RETCODE_TOO_MANY_REQUESTS": "TOO_MANY_REQUESTS",
        # Optional in some builds:
        "TRADE_RETCODE_PLACED_PARTIAL": "PLACED_PARTIAL",
    }
    labels = {}
    for attr_name, label in name_to_label.items():
        code = getattr(mt5, attr_name, None)
        if code is not None:
            labels[code] = label
    # Known numeric that may not exist as a constant
    labels.setdefault(10030, "UNSUPPORTED_FILLING")
    return labels

_RETCODE_LABELS = _build_retcode_labels()

def _retcode_label(code: int) -> str:
    return _RETCODE_LABELS.get(code, f"CODE_{code}")

# Symbol execution modes (safe getattr)
_EXEC_MARKET  = getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", 0)
_EXEC_INSTANT = getattr(mt5, "SYMBOL_TRADE_EXECUTION_INSTANT", 1)
_EXEC_REQUEST = getattr(mt5, "SYMBOL_TRADE_EXECUTION_REQUEST", 2)
_EXEC_EXCHANGE= getattr(mt5, "SYMBOL_TRADE_EXECUTION_EXCHANGE", 3)

# Allowed-filling flags exposed on the symbol (bit flags)
_SYM_FILL_IOC = getattr(mt5, "SYMBOL_FILLING_IOC", 1 << 1)
_SYM_FILL_FOK = getattr(mt5, "SYMBOL_FILLING_FOK", 1 << 0)
# (MetaQuotes may add more later; we only need these two)

# Order filling constants
_FILLING_IOC    = getattr(mt5, "ORDER_FILLING_IOC", None)
_FILLING_FOK    = getattr(mt5, "ORDER_FILLING_FOK", None)
_FILLING_RETURN = getattr(mt5, "ORDER_FILLING_RETURN", None)

def _allowed_fillings_for_symbol(sinfo) -> list:
    """
    Compute allowed ORDER_FILLING_* constants for this symbol,
    based on execution mode and symbol filling flags per docs.
    """
    exec_mode = int(getattr(sinfo, "trade_exemode", _EXEC_MARKET))
    flags = int(getattr(sinfo, "filling_mode", 0))

    allowed = []

    # IOC & FOK depend on symbol flags for Market/Exchange; always allowed for Instant/Request
    def flag_has(mask): return (flags & mask) == mask

    if exec_mode in (_EXEC_INSTANT, _EXEC_REQUEST):
        if _FILLING_IOC is not None: allowed.append(_FILLING_IOC)
        if _FILLING_FOK is not None: allowed.append(_FILLING_FOK)
        # RETURN is always allowed in Instant/Request
        if _FILLING_RETURN is not None: allowed.append(_FILLING_RETURN)
    elif exec_mode == _EXEC_MARKET:
        # RETURN is disabled regardless of symbol settings (doc table)
        if flag_has(_SYM_FILL_IOC) and _FILLING_IOC is not None:
            allowed.append(_FILLING_IOC)
        if flag_has(_SYM_FILL_FOK) and _FILLING_FOK is not None:
            allowed.append(_FILLING_FOK)
    elif exec_mode == _EXEC_EXCHANGE:
        # RETURN always allowed; IOC/FOK if flags set
        if flag_has(_SYM_FILL_IOC) and _FILLING_IOC is not None:
            allowed.append(_FILLING_IOC)
        if flag_has(_SYM_FILL_FOK) and _FILLING_FOK is not None:
            allowed.append(_FILLING_FOK)
        if _FILLING_RETURN is not None:
            allowed.append(_FILLING_RETURN)
    else:
        # Unknown: be conservative — try IOC/FOK if flags say so, else RETURN
        if flag_has(_SYM_FILL_IOC) and _FILLING_IOC is not None:
            allowed.append(_FILLING_IOC)
        if flag_has(_SYM_FILL_FOK) and _FILLING_FOK is not None:
            allowed.append(_FILLING_FOK)
        if _FILLING_RETURN is not None:
            allowed.append(_FILLING_RETURN)

    # Keep unique order, prefer IOC → FOK → RETURN for market-taker behavior
    pref = [_FILLING_IOC, _FILLING_FOK, _FILLING_RETURN]
    allowed_sorted = [x for x in pref if x in allowed] + [x for x in allowed if x not in pref]
    return allowed_sorted

_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
}

# --- Safe filling mode lookup (defaults gracefully) ---
_ORDER_FILLING_FOK = getattr(mt5, "ORDER_FILLING_FOK", None)
_ORDER_FILLING_IOC = getattr(mt5, "ORDER_FILLING_IOC", None)
_ORDER_FILLING_RETURN = getattr(mt5, "ORDER_FILLING_RETURN", None)

_FILLING_MAP = {}
if _ORDER_FILLING_IOC is not None:    _FILLING_MAP["IOC"] = _ORDER_FILLING_IOC
if _ORDER_FILLING_FOK is not None:    _FILLING_MAP["FOK"] = _ORDER_FILLING_FOK
if _ORDER_FILLING_RETURN is not None: _FILLING_MAP["RETURN"] = _ORDER_FILLING_RETURN

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

def _namedtuples_to_df(records) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if hasattr(records, "dtype") and getattr(records.dtype, "names", None):
        try:
            return pd.DataFrame.from_records(records)
        except Exception:
            return pd.DataFrame(records)
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
                rows.append({k: getattr(r, k) for k in dir(r) if not k.startswith("_") and not callable(getattr(r, k))})
        return pd.DataFrame(rows)
    try:
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
    if MT5_TERMINAL_PATH and os.path.isfile(MT5_TERMINAL_PATH):
        candidates.append(MT5_TERMINAL_PATH)
    if platform.system() == "Windows":
        patterns = [
            r"C:\Program Files\MetaTrader*\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader*\terminal64.exe",
        ]
        appdata = os.getenv("APPDATA")
        if appdata:
            patterns.append(os.path.join(appdata, "MetaQuotes", "Terminal", "**", "terminal64.exe"))
        for pat in patterns:
            for p in glob.iglob(pat, recursive=True):
                if os.path.isfile(p):
                    candidates.append(p)
    seen = set(); unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c); seen.add(c)
    return unique

def _copy_rates_recent_df(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    tf = _TIMEFRAME_MAP[timeframe.upper()]
    _ensure_symbol(symbol)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=int(lookback_days))
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    df = _namedtuples_to_df(rates)
    if df.empty:
        return df
    if "time" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time_utc", inplace=True, drop=False)
    if "volume" not in df.columns:
        rv = df["real_volume"] if "real_volume" in df.columns else None
        tv = df["tick_volume"] if "tick_volume" in df.columns else None
        if rv is not None and rv.sum() > 0:
            df["volume"] = rv
        elif tv is not None:
            df["volume"] = tv
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df

# ----------------- Indicator helpers (fallbacks if ta missing) -----------------

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std()
    upper = m + k*s
    lower = m - k*s
    return m, upper, lower

def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3, smooth_k=3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    raw_k = 100 * (close - ll) / (hh - ll + 1e-9)
    k_s = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d_s = k_s.rolling(d, min_periods=d).mean()
    return k_s, d_s

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = _ema(tr, n)
    plus_di = 100 * (_ema(pd.Series(plus_dm, index=high.index), n) / (atr + 1e-9))
    minus_di = 100 * (_ema(pd.Series(minus_dm, index=high.index), n) / (atr + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = _ema(dx, n)
    return adx, plus_di, minus_di

def _keltner(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20, m: float = 2.0):
    mid = _ema(close, n)
    atr = _atr(high, low, close, n)
    upper = mid + m * atr
    lower = mid - m * atr
    return mid, upper, lower

def _pivot_points_daily(df: pd.DataFrame) -> Dict[str, float]:
    # Use previous full UTC day
    if df.empty:
        return {}
    ddf = df.tz_convert(UTC) if df.index.tz is not None else df
    daily = ddf.resample("1D", label="right", closed="right").agg({"high":"max","low":"min","close":"last"})
    if len(daily) < 2:
        return {}
    prev = daily.iloc[-2]
    P = (prev.high + prev.low + prev.close) / 3.0
    R1 = 2*P - prev.low; S1 = 2*P - prev.high
    R2 = P + (prev.high - prev.low); S2 = P - (prev.high - prev.low)
    R3 = prev.high + 2*(P - prev.low); S3 = prev.low - 2*(prev.high - P)
    return {"P":float(P),"R1":float(R1),"S1":float(S1),"R2":float(R2),"S2":float(S2),"R3":float(R3),"S3":float(S3)}

def _swing_hilo(df: pd.DataFrame, left=5, right=5) -> Tuple[float, float]:
    # crude recent swing hi/lo across last N bars
    lastN = df.tail(max(left+right+5, 30))
    return float(lastN["high"].max()), float(lastN["low"].min())

def _fib_levels_from_swing(hi: float, lo: float) -> Dict[str, float]:
    rng = hi - lo
    return {
        "23.6%": hi - 0.236*rng,
        "38.2%": hi - 0.382*rng,
        "50.0%": hi - 0.5*rng,
        "61.8%": hi - 0.618*rng,
        "78.6%": hi - 0.786*rng,
    }

def _near(px: float, lvl: float, tol_abs: float) -> bool:
    return abs(px - lvl) <= tol_abs

def _ensure_min_rows(df: pd.DataFrame, min_rows: int) -> None:
    if len(df) < min_rows:
        raise ValueError(f"Not enough bars ({len(df)}) for calculation; need >= {min_rows}.")

# ----------------- MT5 Tool implementations -----------------

def mt5_open_terminal(path: Optional[str] = None) -> Dict[str, Any]:
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

        if not _try_init():
            term_path = path or (MT5_TERMINAL_PATH if MT5_TERMINAL_PATH else None)
            if not term_path:
                cands = _discover_mt5_terminal_candidates()
                term_path = cands[0] if cands else None
            if not term_path:
                return {"ok": False, "error": f"initialize failed: {mt5.last_error()} and no terminal found"}
            _ = execute_exe(term_path, args=[])
            time.sleep(2.5)
            if not _try_init(term_path):
                return {"ok": False, "error": f"initialize after launch failed: {mt5.last_error()}"}

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

# --- Safe retcode mapping (handles older MetaTrader5 modules) ---
def _build_retcode_labels():
    name_to_label = {
        "TRADE_RETCODE_DONE": "DONE",
        "TRADE_RETCODE_PLACED": "PLACED",
        "TRADE_RETCODE_REJECT": "REJECT",
        "TRADE_RETCODE_CANCEL": "CANCEL",
        "TRADE_RETCODE_INVALID": "INVALID",
        "TRADE_RETCODE_INVALID_VOLUME": "INVALID_VOLUME",
        "TRADE_RETCODE_INVALID_PRICE": "INVALID_PRICE",
        "TRADE_RETCODE_INVALID_STOPS": "INVALID_STOPS",
        "TRADE_RETCODE_MARKET_CLOSED": "MARKET_CLOSED",
        "TRADE_RETCODE_NO_MONEY": "NO_MONEY",
        "TRADE_RETCODE_PRICE_CHANGED": "PRICE_CHANGED",
        "TRADE_RETCODE_OFFQUOTES": "OFFQUOTES",
        "TRADE_RETCODE_TRADE_DISABLED": "TRADE_DISABLED",
        "TRADE_RETCODE_TRADE_TIMEOUT": "TRADE_TIMEOUT",
        "TRADE_RETCODE_REQUOTE": "REQUOTE",
        "TRADE_RETCODE_TOO_MANY_REQUESTS": "TOO_MANY_REQUESTS",
        # Optional in some builds:
        "TRADE_RETCODE_PLACED_PARTIAL": "PLACED_PARTIAL",
    }
    labels = {}
    for attr_name, label in name_to_label.items():
        code = getattr(mt5, attr_name, None)
        if code is not None:
            labels[code] = label
    # Manual known codes that might not exist as attrs in some wheels:
    labels.setdefault(10030, "UNSUPPORTED_FILLING")   # broker rejects chosen filling mode
    return labels

_RETCODE_LABELS = _build_retcode_labels()

def _retcode_label(code: int) -> str:
    return _RETCODE_LABELS.get(code, f"CODE_{code}")


from typing import Optional, Dict, Any

def mt5_place_order(symbol: str, side: str, volume: float,
                    sl_points: Optional[int] = None, tp_points: Optional[int] = None,
                    deviation: int = 20, filling: Optional[str] = None,
                    comment: str = "agent order", magic: int = 990001) -> Dict[str, Any]:
    """
    Sends a market order. Chooses a valid fill policy based on the symbol,
    trying user-preferred first (if any), then IOC → FOK → RETURN when allowed.
    Surfaces full diagnostics on failure.
    """
    try:
        # ensure symbol visible/enabled (use your own helper if you have one)
        if not mt5.symbol_select(symbol, True):
            return {"ok": False, "error": f"symbol_select failed for {symbol}", "last_error": mt5.last_error()}

        sinfo = mt5.symbol_info(symbol)
        if sinfo is None:
            return {"ok": False, "error": f"symbol_info None for {symbol}", "last_error": mt5.last_error()}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "error": f"symbol_info_tick None for {symbol}", "last_error": mt5.last_error()}

        point = sinfo.point or 0.0
        side_l = (side or "").lower()
        if side_l == "buy":
            order_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        elif side_l == "sell":
            order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            return {"ok": False, "error": "side must be 'buy' or 'sell'", "last_error": mt5.last_error()}

        # Build a base request (we’ll just swap type_filling)
        base_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if sl_points:
            base_req["sl"] = (price - sl_points*point) if order_type == mt5.ORDER_TYPE_BUY else (price + sl_points*point)
        if tp_points:
            base_req["tp"] = (price + tp_points*point) if order_type == mt5.ORDER_TYPE_BUY else (price - tp_points*point)

        # Compute allowed fillings for this symbol/mode from symbol_info
        allowed = _allowed_fillings_for_symbol(sinfo)

        # If user passed a preferred 'filling' string, try it first (if present)
        preferred_map = {
            "IOC": _FILLING_IOC,
            "FOK": _FILLING_FOK,
            "RETURN": _FILLING_RETURN,
        }
        trial_list = []
        if filling:
            fconst = preferred_map.get(str(filling).upper())
            if fconst in allowed:
                trial_list.append(fconst)
        # Then add all allowed (de-dup)
        for f in allowed:
            if f not in trial_list:
                trial_list.append(f)

        attempts = []
        for fconst in trial_list or [None]:
            req = dict(base_req)
            if fconst is not None:
                req["type_filling"] = fconst

            check = mt5.order_check(req)
            check_dict = getattr(check, "_asdict", lambda: {})()

            result = mt5.order_send(req)
            result_dict = getattr(result, "_asdict", lambda: {})()

            retcode = int(result_dict.get("retcode", -1)) if result_dict else -1
            ok = retcode in (getattr(mt5, "TRADE_RETCODE_DONE", 10009),
                             getattr(mt5, "TRADE_RETCODE_PLACED", 10008))

            # Success path
            if ok:
                return {
                    "ok": True,
                    "used_filling": fconst,
                    "request": req,
                    "check": check_dict,
                    "result": result_dict,
                    "retcode": retcode,
                    "retcode_label": _retcode_label(retcode),
                }

            # Failure, collect diagnostics and decide if we should try next filling
            attempts.append({
                "used_filling": fconst,
                "check": check_dict,
                "result": result_dict,
                "retcode": retcode,
                "retcode_label": _retcode_label(retcode),
                "comment": result_dict.get("comment") if isinstance(result_dict, dict) else None,
            })

            # If error is not unsupported filling (10030), break early
            if retcode != 10030:  # UNSUPPORTED_FILLING
                break

        # All attempts failed
        parts = []
        for a in attempts:
            used = a["used_filling"]
            used_name = ("IOC" if used == _FILLING_IOC else
                         "FOK" if used == _FILLING_FOK else
                         "RETURN" if used == _FILLING_RETURN else str(used))
            parts.append(
                f"[{used_name}] retcode={a['retcode']} ({a['retcode_label']})"
                + (f", comment={a['comment']}" if a.get("comment") else "")
            )
        return {
            "ok": False,
            "error": " ; ".join(parts) or "unknown",
            "attempts": attempts,
            "last_error": mt5.last_error(),
            "exec_mode": int(getattr(sinfo, "trade_exemode", -1)),
            "symbol_flags": int(getattr(sinfo, "filling_mode", -1)),
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "last_error": mt5.last_error()}

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

def mt5_close_position(ticket: int, volume: Optional[float] = None, deviation: int = 20,
                       comment: str = "agent close", magic: int = 990001) -> Dict[str, Any]:
    """
    Close an existing position by ticket. If volume is None, closes full position volume.
    Tries valid type_filling policies for the symbol similarly to mt5_place_order.
    """
    try:
        # Find the position
        all_pos = mt5.positions_get()
        if not all_pos:
            return {"ok": False, "error": f"No open positions found"}
        pos_match = None
        for p in all_pos:
            try:
                if int(getattr(p, "ticket", -1)) == int(ticket):
                    pos_match = p
                    break
            except Exception:
                continue
        if pos_match is None:
            return {"ok": False, "error": f"Position ticket {ticket} not found"}

        symbol = getattr(pos_match, "symbol", None)
        pos_type = int(getattr(pos_match, "type", -1))
        pos_volume = float(getattr(pos_match, "volume", 0.0))
        if not symbol:
            return {"ok": False, "error": "Position has no symbol"}
        if pos_volume <= 0:
            return {"ok": False, "error": "Position volume is zero"}

        vol_to_close = float(volume) if volume not in (None, "") else pos_volume
        if vol_to_close <= 0:
            return {"ok": False, "error": "Close volume must be > 0"}

        # Ensure symbol and get tick
        if not mt5.symbol_select(symbol, True):
            return {"ok": False, "error": f"symbol_select failed for {symbol}", "last_error": mt5.last_error()}
        sinfo = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if sinfo is None or tick is None:
            return {"ok": False, "error": f"symbol info/tick missing for {symbol}", "last_error": mt5.last_error()}

        # Determine opposite order type and price
        if pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0):
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        elif pos_type == getattr(mt5, "POSITION_TYPE_SELL", 1):
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            return {"ok": False, "error": f"Unknown position type: {pos_type}"}

        # Base request
        base_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": symbol,
            "volume": float(vol_to_close),
            "type": order_type,
            "price": price,
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        # Try allowed fillings
        allowed = _allowed_fillings_for_symbol(sinfo)
        attempts = []
        for fconst in allowed or [None]:
            req = dict(base_req)
            if fconst is not None:
                req["type_filling"] = fconst

            check = mt5.order_check(req)
            check_dict = getattr(check, "_asdict", lambda: {})()
            result = mt5.order_send(req)
            result_dict = getattr(result, "_asdict", lambda: {})()

            retcode = int(result_dict.get("retcode", -1)) if isinstance(result_dict, dict) else -1
            ok = retcode in (getattr(mt5, "TRADE_RETCODE_DONE", 10009),
                             getattr(mt5, "TRADE_RETCODE_PLACED", 10008))
            if ok:
                return {
                    "ok": True,
                    "request": req,
                    "check": check_dict,
                    "result": result_dict,
                    "retcode": retcode,
                    "retcode_label": _retcode_label(retcode),
                }

            attempts.append({
                "used_filling": fconst,
                "check": check_dict,
                "result": result_dict,
                "retcode": retcode,
                "retcode_label": _retcode_label(retcode),
                "comment": result_dict.get("comment") if isinstance(result_dict, dict) else None,
            })

            # If error is not unsupported filling (10030), break
            if retcode != 10030:
                break

        # All failed
        parts = []
        for a in attempts:
            used = a["used_filling"]
            used_name = ("IOC" if used == _FILLING_IOC else
                         "FOK" if used == _FILLING_FOK else
                         "RETURN" if used == _FILLING_RETURN else str(used))
            parts.append(
                f"[{used_name}] retcode={a['retcode']} ({a['retcode_label']})"
                + (f", comment={a['comment']}" if a.get("comment") else "")
            )
        return {"ok": False, "error": " ; ".join(parts) or "unknown", "last_error": mt5.last_error()}

    except Exception as e:
        return {"ok": False, "error": str(e), "last_error": mt5.last_error()}

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
        if not out.empty and "time" in out.columns:
            out["time_utc"] = pd.to_datetime(out["time"], unit="s", utc=True)
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

# ----------------- Analytics / TA tools -----------------

def calculate_indicators(symbol: str, timeframe: str, lookback_days: int = 90,
                         out_csv_path: Optional[str] = None) -> Dict[str, Any]:
    if not _TA_AVAILABLE:
        return {"ok": False, "error": "Python package 'ta' not installed. `pip install ta`"}
    try:
        df = _copy_rates_recent_df(symbol, timeframe, lookback_days)
        if df.empty:
            return {"ok": False, "error": "No data returned from MT5 for given inputs."}
        need = {"open","high","low","close","volume"}
        if not need.issubset(set(df.columns)):
            missing = sorted(list(need - set(df.columns)))
            return {"ok": False, "error": f"Missing required columns for TA: {missing}"}
        df2 = dropna(df) if _TA_AVAILABLE else df.dropna()
        df_ta = add_all_ta_features(
            df2, open="open", high="high", low="low", close="close", volume="volume", fillna=True
        )
        if out_csv_path is None:
            end = datetime.now(tz=UTC).date()
            out_csv_path = os.path.join(WORKDIR_DEFAULT, "mt5_exports", f"{symbol}_{timeframe}_ta_{end}.csv")
        os.makedirs(os.path.dirname(norm_path(out_csv_path)), exist_ok=True)
        df_ta.to_csv(norm_path(out_csv_path), index=False)
        return {"ok": True, "path": norm_path(out_csv_path), "rows": len(df_ta),
                "preview": df_ta.tail(3).to_dict(orient="records")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def analyze_market_status(symbol: str, timeframe: str = "H1", lookback_days: int = 120) -> Dict[str, Any]:
    try:
        df = _copy_rates_recent_df(symbol, timeframe, lookback_days)
        if df.empty:
            return {"ok": False, "error": "No data returned from MT5 for given inputs."}
        px = df["close"].astype(float)
        if len(px) < 210:
            return {"ok": False, "error": f"Not enough data for analysis ({len(px)} bars), need >= 210."}
        sma50 = px.rolling(50).mean()
        sma200 = px.rolling(200).mean()
        trend = "uptrend" if sma50.iloc[-1] > sma200.iloc[-1] else "downtrend"
        delta = px.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])
        momentum = "overbought" if rsi_val >= 70 else "oversold" if rsi_val <= 30 else "neutral"
        high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
        tr = pd.concat([(high - low),(high - close.shift()).abs(),(low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        status = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_used": len(df),
            "trend": trend,
            "sma50": float(sma50.iloc[-1]),
            "sma200": float(sma200.iloc[-1]),
            "rsi14": rsi_val,
            "momentum": momentum,
            "atr14": atr_val,
            "last_close": float(close.iloc[-1]),
            "as_of_utc": df.index[-1].isoformat(),
        }
        return {"ok": True, "status": status}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def list_forex_strategies() -> Dict[str, Any]:
    strategies = [{"name": n} for n in sorted(_STRATEGY_FUNCS.keys())]
    return {"ok": True, "strategies": strategies}

# ----------------- Strategy Engine -----------------

@dataclass
class StratDecision:
    strategy: str
    decision: str           # "buy" | "sell" | "no-trade"
    confidence: float       # 0..1
    timeframe: str
    as_of_utc: str
    extras: Optional[Dict[str, Any]] = None

def _nz01(x: float, lo: float, hi: float) -> float:
    if hi == lo: return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

def _prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    px = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    out = df.copy()

    out["sma50"] = _sma(px, 50)
    out["sma200"] = _sma(px, 200)
    out["ema21"] = _ema(px, 21)

    out["rsi14"] = _rsi(px, 14)
    out["atr14"] = _atr(high, low, px, 14)

    bb_mid, bb_upper, bb_lower = _bollinger(px, 20, 2.0)
    out["bb_mid"] = bb_mid; out["bb_upper"] = bb_upper; out["bb_lower"] = bb_lower

    out["don_high20"] = high.rolling(20, min_periods=20).max()
    out["don_low20"]  = low.rolling(20, min_periods=20).min()

    macd, macd_sig, macd_hist = _macd(px)
    out["macd"] = macd; out["macd_sig"] = macd_sig; out["macd_hist"] = macd_hist

    k, d = _stoch(high, low, px)
    out["stoch_k"] = k; out["stoch_d"] = d

    adx, pdi, mdi = _adx(high, low, px)
    out["adx14"] = adx; out["+di14"] = pdi; out["-di14"] = mdi

    kmid, kup, klo = _keltner(px, high, low)
    out["kel_mid"] = kmid; out["kel_upper"] = kup; out["kel_lower"] = klo

    # wick ratios for sweep heuristic
    last = out.iloc[-1]
    body = abs(last["close"] - last["open"]) if "open" in out.columns else abs(px.iloc[-1] - px.iloc[-2])
    up_wick = last["high"] - max(last["close"], last["open"]) if "open" in out.columns else last["high"] - max(px.iloc[-1], px.iloc[-2])
    dn_wick = min(last["close"], last["open"]) - last["low"] if "open" in out.columns else min(px.iloc[-1], px.iloc[-2]) - last["low"]
    out.at[out.index[-1], "wick_up_ratio"] = float(up_wick / (body + 1e-9))
    out.at[out.index[-1], "wick_dn_ratio"] = float(abs(dn_wick) / (body + 1e-9))

    return out

# --------- Core strategies (existing + extended) ---------

def _strat_trend_following(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 200)
    d = df.iloc[-1]
    sma50, sma200, ema21, px = d["sma50"], d["sma200"], d["ema21"], d["close"]
    sep = abs(sma50 - sma200) / max(px, 1e-9)
    conf = _nz01(sep, 0.0005, 0.01)
    aligned = (px > ema21 and sma50 > sma200) or (px < ema21 and sma50 < sma200)
    if aligned: conf = min(1.0, conf + 0.1)
    decision = "buy" if sma50 > sma200 else "sell"
    return StratDecision("Trend Following", decision, conf, timeframe, df.index[-1].isoformat(),
                         extras={"sep": float(sep)})

def _strat_counter_trend(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 200)
    d = df.iloc[-1]
    uptrend = d["sma50"] > d["sma200"]
    rsi = float(d["rsi14"])
    conf = 0.0; decision = "no-trade"
    if uptrend and rsi >= 70:
        decision = "sell"; conf = _nz01(rsi, 70, 85)
    elif (not uptrend) and rsi <= 30:
        decision = "buy"; conf = _nz01(30 - rsi, 0, 15)
    return StratDecision("Counter-Trend", decision, conf, timeframe, df.index[-1].isoformat(),
                         extras={"rsi14": rsi})

def _strat_breakout(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 20)
    d = df.iloc[-1]
    px = d["close"]; hi20 = d["don_high20"]; lo20 = d["don_low20"]; atr = max(d["atr14"], 1e-9)
    if px > hi20:
        mag = (px - hi20) / atr
        conf = _nz01(mag, 0.25, 2.0)
        return StratDecision("Breakout Trading", "buy", conf, timeframe, df.index[-1].isoformat(), extras={"atr_mag": float(mag)})
    elif px < lo20:
        mag = (lo20 - px) / atr
        conf = _nz01(mag, 0.25, 2.0)
        return StratDecision("Breakout Trading", "sell", conf, timeframe, df.index[-1].isoformat(), extras={"atr_mag": float(mag)})
    return StratDecision("Breakout Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_range(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 20)
    d = df.iloc[-1]
    px = d["close"]; upper = d["bb_upper"]; lower = d["bb_lower"]; mid = d["bb_mid"]
    width = (upper - lower) / max(mid, 1e-9)
    conf_base = _nz01(width, 0.002, 0.03)
    tol = (upper - lower) * 0.05
    if px >= upper - tol:
        return StratDecision("Range Trading", "sell", conf_base, timeframe, df.index[-1].isoformat(),
                             extras={"band_width": float(width)})
    elif px <= lower + tol:
        return StratDecision("Range Trading", "buy", conf_base, timeframe, df.index[-1].isoformat(),
                             extras={"band_width": float(width)})
    return StratDecision("Range Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_swing(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 50)
    d = df.iloc[-1]
    k = float(d["stoch_k"]); kd = float(d["stoch_d"])
    trend_up = d["sma50"] > d["sma200"]
    cross_up = (df["stoch_k"].iloc[-2] < df["stoch_d"].iloc[-2]) and (k > kd) and k < 40
    cross_dn = (df["stoch_k"].iloc[-2] > df["stoch_d"].iloc[-2]) and (k < kd) and k > 60
    if trend_up and cross_up:
        conf = _nz01(40 - k, 0, 40)
        return StratDecision("Swing Trading", "buy", conf, timeframe, df.index[-1].isoformat(), extras={"stoch_k": k})
    if (not trend_up) and cross_dn:
        conf = _nz01(k - 60, 0, 40)
        return StratDecision("Swing Trading", "sell", conf, timeframe, df.index[-1].isoformat(), extras={"stoch_k": k})
    return StratDecision("Swing Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_scalping(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 50)
    rsi7 = _rsi(df["close"].astype(float), 7)
    val = float(rsi7.iloc[-1])
    dv = float(rsi7.diff().iloc[-1])
    conf = _nz01(abs(dv), 0.1, 2.0)
    if val < 30 and dv > 0:
        return StratDecision("Scalping", "buy", conf, timeframe, df.index[-1].isoformat(), extras={"rsi7": val, "drsi": dv})
    if val > 70 and dv < 0:
        return StratDecision("Scalping", "sell", conf, timeframe, df.index[-1].isoformat(), extras={"rsi7": val, "drsi": dv})
    return StratDecision("Scalping", "no-trade", 0.0, timeframe, df.index[-1].isoformat(), extras={"rsi7": val, "drsi": dv})

def _strat_day_trading(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Intraday range breakout (today's initial range = first 30 bars or 2 hours, whichever smaller)
    if len(df) < 60:
        return StratDecision("Day Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(), extras={"reason":"insufficient bars"})
    df_today = df[df.index.date == df.index[-1].date()]
    if len(df_today) < 30:
        return StratDecision("Day Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(), extras={"reason":"not enough today bars"})
    window = min(30, len(df_today)//2)
    init = df_today.head(window)
    hi = init["high"].max(); lo = init["low"].min()
    px = df.iloc[-1]["close"]; atr = max(df.iloc[-1]["atr14"], 1e-9)
    if px > hi:
        conf = _nz01((px-hi)/atr, 0.25, 1.5)
        return StratDecision("Day Trading", "buy", conf, timeframe, df.index[-1].isoformat())
    if px < lo:
        conf = _nz01((lo-px)/atr, 0.25, 1.5)
        return StratDecision("Day Trading", "sell", conf, timeframe, df.index[-1].isoformat())
    return StratDecision("Day Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_position_trading(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Long-horizon bias: SMA200 trend + MACD sign
    _ensure_min_rows(df, 200)
    d = df.iloc[-1]
    trend_up = d["sma50"] > d["sma200"]
    macd_pos = d["macd"] > d["macd_sig"]
    if trend_up and macd_pos:
        conf = 0.6 + 0.4*_nz01(float(d["adx14"]), 20, 40)
        return StratDecision("Position Trading", "buy", conf, timeframe, df.index[-1].isoformat())
    if (not trend_up) and (not macd_pos):
        conf = 0.6 + 0.4*_nz01(float(d["adx14"]), 20, 40)
        return StratDecision("Position Trading", "sell", conf, timeframe, df.index[-1].isoformat())
    return StratDecision("Position Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_momentum(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 50)
    px = df["close"].astype(float)
    roc = (px / px.shift(10) - 1.0) * 100.0
    d = df.iloc[-1]
    adx = float(d["adx14"])
    if roc.iloc[-1] > 0 and adx > 20:
        conf = _nz01(roc.iloc[-1], 0.1, 1.5) * _nz01(adx, 20, 40)
        return StratDecision("Momentum Trading", "buy", conf, timeframe, df.index[-1].isoformat(), extras={"roc10": float(roc.iloc[-1])})
    if roc.iloc[-1] < 0 and adx > 20:
        conf = _nz01(-roc.iloc[-1], 0.1, 1.5) * _nz01(adx, 20, 40)
        return StratDecision("Momentum Trading", "sell", conf, timeframe, df.index[-1].isoformat(), extras={"roc10": float(roc.iloc[-1])})
    return StratDecision("Momentum Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_price_action(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Simple engulfing signal with trend filter
    if len(df) < 5: return StratDecision("Price Action Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())
    o = df["open"]; c = df["close"]
    bull_engulf = (c.shift(1) < o.shift(1)) & (c > o) & (c >= c.shift(1)) & (o <= o.shift(1))
    bear_engulf = (c.shift(1) > o.shift(1)) & (c < o) & (c <= c.shift(1)) & (o >= o.shift(1))
    trend_up = df["sma50"].iloc[-1] > df["sma200"].iloc[-1]
    if bull_engulf.iloc[-1] and trend_up:
        return StratDecision("Price Action Trading", "buy", 0.55, timeframe, df.index[-1].isoformat())
    if bear_engulf.iloc[-1] and (not trend_up):
        return StratDecision("Price Action Trading", "sell", 0.55, timeframe, df.index[-1].isoformat())
    return StratDecision("Price Action Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_support_resistance(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 60)
    hi, lo = _swing_hilo(df, 10, 10)
    px = float(df.iloc[-1]["close"])
    atr = float(df.iloc[-1]["atr14"])
    tol = max(atr*0.25, 0.0001*px)
    if _near(px, hi, tol):
        return StratDecision("Support and Resistance Trading", "sell", _nz01((hi-px)/max(atr,1e-9),0,1.0), timeframe, df.index[-1].isoformat(), extras={"level":hi})
    if _near(px, lo, tol):
        return StratDecision("Support and Resistance Trading", "buy", _nz01((px-lo)/max(atr,1e-9),0,1.0), timeframe, df.index[-1].isoformat(), extras={"level":lo})
    return StratDecision("Support and Resistance Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_ma_crossover(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 200)
    s1 = df["sma50"]; s2 = df["sma200"]
    cross_up = s1.iloc[-2] <= s2.iloc[-2] and s1.iloc[-1] > s2.iloc[-1]
    cross_dn = s1.iloc[-2] >= s2.iloc[-2] and s1.iloc[-1] < s2.iloc[-1]
    if cross_up:
        conf = 0.6 + 0.4*_nz01(float(df.iloc[-1]["adx14"]), 20, 40)
        return StratDecision("Moving Average Crossover", "buy", conf, timeframe, df.index[-1].isoformat())
    if cross_dn:
        conf = 0.6 + 0.4*_nz01(float(df.iloc[-1]["adx14"]), 20, 40)
        return StratDecision("Moving Average Crossover", "sell", conf, timeframe, df.index[-1].isoformat())
    return StratDecision("Moving Average Crossover", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_bollinger(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return _strat_range(df, timeframe)._replace(strategy="Bollinger Band Strategy") if hasattr(StratDecision, "_replace") else \
        StratDecision("Bollinger Band Strategy", *_strat_range(df, timeframe).__dict__.values())

def _strat_ichimoku(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 52)
    high = df["high"]; low = df["low"]; close = df["close"]
    conv = (high.rolling(9).max() + low.rolling(9).min())/2
    base = (high.rolling(26).max() + low.rolling(26).min())/2
    spanA = (conv + base)/2
    spanB = (high.rolling(52).max() + low.rolling(52).min())/2
    d = df.iloc[-1]
    px = d["close"]; sA = spanA.iloc[-1]; sB = spanB.iloc[-1]; c = conv.iloc[-1]; b = base.iloc[-1]
    bull = px > max(sA, sB) and c > b
    bear = px < min(sA, sB) and c < b
    if bull:
        conf = 0.6 + 0.4*_nz01(float(df.iloc[-1]["adx14"]), 20, 40)
        return StratDecision("Ichimoku Cloud Strategy", "buy", conf, timeframe, df.index[-1].isoformat())
    if bear:
        conf = 0.6 + 0.4*_nz01(float(df.iloc[-1]["adx14"]), 20, 40)
        return StratDecision("Ichimoku Cloud Strategy", "sell", conf, timeframe, df.index[-1].isoformat())
    return StratDecision("Ichimoku Cloud Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_fibonacci(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 60)
    hi, lo = _swing_hilo(df, 10, 10)
    levels = _fib_levels_from_swing(hi, lo)
    px = float(df.iloc[-1]["close"]); atr = float(df.iloc[-1]["atr14"]); tol = max(atr*0.25, 0.0001*px)
    # Buy near 61.8/78.6 retracement in uptrend; Sell near 61.8/78.6 in downtrend
    uptrend = df["sma50"].iloc[-1] > df["sma200"].iloc[-1]
    near = None
    for name in ["61.8%","78.6%","50.0%","38.2%","23.6%"]:
        lvl = levels[name]
        if _near(px, lvl, tol):
            near = (name, lvl); break
    if near:
        name, lvl = near
        if uptrend and lvl >= (lo + (hi-lo)*0.5) and px >= lvl:
            return StratDecision("Fibonacci Retracement Strategy", "buy", 0.55, timeframe, df.index[-1].isoformat(), extras={"level":name})
        if (not uptrend) and lvl <= (lo + (hi-lo)*0.5) and px <= lvl:
            return StratDecision("Fibonacci Retracement Strategy", "sell", 0.55, timeframe, df.index[-1].isoformat(), extras={"level":name})
    return StratDecision("Fibonacci Retracement Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_pivot_points(df: pd.DataFrame, timeframe: str) -> StratDecision:
    piv = _pivot_points_daily(df)
    if not piv:
        return StratDecision("Pivot Point Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())
    px = float(df.iloc[-1]["close"]); atr = float(df.iloc[-1]["atr14"]); tol = max(atr*0.25, 0.0001*px)
    # Fade near R1/S1; breakout beyond R2/S2
    if _near(px, piv["R1"], tol): return StratDecision("Pivot Point Strategy", "sell", 0.5, timeframe, df.index[-1].isoformat(), extras={"level":"R1"})
    if _near(px, piv["S1"], tol): return StratDecision("Pivot Point Strategy", "buy", 0.5, timeframe, df.index[-1].isoformat(), extras={"level":"S1"})
    if px > piv["R2"]: return StratDecision("Pivot Point Strategy", "buy", 0.6, timeframe, df.index[-1].isoformat(), extras={"level":"R2+"})
    if px < piv["S2"]: return StratDecision("Pivot Point Strategy", "sell", 0.6, timeframe, df.index[-1].isoformat(), extras={"level":"S2-"})
    return StratDecision("Pivot Point Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_carry(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Use swap info as carry proxy
    sinfo = mt5.symbol_info(df.iloc[-1]["symbol"] if "symbol" in df.columns else None)  # may be None
    # Fallback: fetch via mt5 on last symbol name; we can't infer reliably from df
    decision = "no-trade"; conf = 0.0; extras={}
    try:
        # User should ensure mt5_symbol_enable called
        # Assume we can get symbol name from extras; otherwise skip
        pass
    except Exception:
        pass
    # Simpler: ask MT5 directly using last-known symbol from index if available
    return StratDecision("Carry Trade", decision, conf, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Provide symbol_info().swap_long/short to enable."})

def _strat_grid(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Range proxy via low ADX: suggest "no-trade" signal (grid is execution style)
    adx = float(df.iloc[-1]["adx14"])
    if adx < 15:
        return StratDecision("Grid Trading", "no-trade", 0.4, timeframe, df.index[-1].isoformat(), extras={"adx14": adx})
    return StratDecision("Grid Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_hedging(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Hedging Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Requires multi-symbol correlation; not auto-fired."})

def _strat_news(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("News Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Needs calendar feed; not available here."})

def _strat_arbitrage(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Arbitrage Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Requires cross-venue latency/price feed."})

def _strat_mean_reversion(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Use z-score of price vs SMA50
    _ensure_min_rows(df, 60)
    px = df["close"].astype(float)
    sma = df["sma50"]; std = px.rolling(50).std()
    z = (px - sma) / (std + 1e-9)
    val = float(z.iloc[-1])
    if val > 2:
        return StratDecision("Mean Reversion", "sell", _nz01(val, 2, 4), timeframe, df.index[-1].isoformat())
    if val < -2:
        return StratDecision("Mean Reversion", "buy", _nz01(-val, 2, 4), timeframe, df.index[-1].isoformat())
    return StratDecision("Mean Reversion", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_algo(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Algorithmic Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Umbrella category; use other specific strategies."})

def _strat_hft(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("High-Frequency Trading (HFT)", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Needs tick microstructure & colocation."})

# --------- Institutional / SMC family ---------

def _strat_market_making(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Market Making", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Requires order book & quotes; not available."})

def _strat_order_flow(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Order Flow Trading", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Needs L2/footprint/vol profiles."})

def _strat_liquidity_grab(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 30)
    win = 10
    highs = df["high"].rolling(win, min_periods=win).max()
    lows = df["low"].rolling(win, min_periods=win).min()
    d = df.iloc[-1]
    prev_hi = float(highs.iloc[-2]) if not np.isnan(highs.iloc[-2]) else float(df["high"].iloc[-2])
    prev_lo = float(lows.iloc[-2]) if not np.isnan(lows.iloc[-2]) else float(df["low"].iloc[-2])
    px = float(d["close"])
    hi = float(d["high"]); lo = float(d["low"])
    wick_up = float(d.get("wick_up_ratio", 0.0)); wick_dn = float(d.get("wick_dn_ratio", 0.0))
    atr = float(max(d.get("atr14", 0.0), 1e-9))
    if hi > prev_hi and px < prev_hi:
        overshoot = (hi - prev_hi) / atr
        conf = min(1.0, 0.4 + _nz01(overshoot, 0.1, 1.5) + _nz01(wick_up, 0.5, 5.0)*0.3)
        return StratDecision("Liquidity Grab / Stop Hunt", "sell", conf, timeframe, df.index[-1].isoformat(),
                             extras={"overshoot_atr": overshoot, "wick_up_ratio": wick_up})
    if lo < prev_lo and px > prev_lo:
        overshoot = (prev_lo - lo) / atr
        conf = min(1.0, 0.4 + _nz01(overshoot, 0.1, 1.5) + _nz01(wick_dn, 0.5, 5.0)*0.3)
        return StratDecision("Liquidity Grab / Stop Hunt", "buy", conf, timeframe, df.index[-1].isoformat(),
                             extras={"overshoot_atr": overshoot, "wick_dn_ratio": wick_dn})
    return StratDecision("Liquidity Grab / Stop Hunt", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_wyckoff(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Wyckoff Method", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Full phase detection out of scope here."})

def _strat_supply_demand(df: pd.DataFrame, timeframe: str) -> StratDecision:
    _ensure_min_rows(df, 60)
    hi, lo = _swing_hilo(df, 15, 15)
    px = float(df.iloc[-1]["close"])
    atr = float(df.iloc[-1]["atr14"]); tol = max(atr*0.5, 0.00015*px)
    if _near(px, hi, tol):  # supply
        return StratDecision("Supply and Demand Zones", "sell", _nz01((hi-px)/max(atr,1e-9),0,1.0), timeframe, df.index[-1].isoformat(), extras={"zone":"supply"})
    if _near(px, lo, tol):  # demand
        return StratDecision("Supply and Demand Zones", "buy", _nz01((px-lo)/max(atr,1e-9),0,1.0), timeframe, df.index[-1].isoformat(), extras={"zone":"demand"})
    return StratDecision("Supply and Demand Zones", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_fvg(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Detect last FVG: candle i-2 high < i low (bullish gap) or i-2 low > i high (bearish gap)
    if len(df) < 3: return StratDecision("Fair Value Gap (FVG) Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())
    h = df["high"]; l = df["low"]; px = float(df.iloc[-1]["close"])
    bull_gap = h.shift(2) < l  # bullish imbalance
    bear_gap = l.shift(2) > h  # bearish imbalance
    atr = float(df.iloc[-1]["atr14"]); tol = atr*0.25
    if bull_gap.iloc[-1]:
        # if price pulls back into gap low..high and bounces above gap high → buy
        gap_low = float(h.iloc[-3]); gap_high = float(l.iloc[-1])
        if px >= gap_low - tol and px <= gap_high + tol:
            return StratDecision("Fair Value Gap (FVG) Strategy", "buy", 0.55, timeframe, df.index[-1].isoformat())
    if bear_gap.iloc[-1]:
        gap_high = float(l.iloc[-3]); gap_low = float(h.iloc[-1])
        if px <= gap_high + tol and px >= gap_low - tol:
            return StratDecision("Fair Value Gap (FVG) Strategy", "sell", 0.55, timeframe, df.index[-1].isoformat())
    return StratDecision("Fair Value Gap (FVG) Strategy", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_order_blocks(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Institutional Order Blocks", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Robust BOS+OB detection omitted."})

def _strat_accum_dist(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Low ADX indicates potential accumulation/distribution (no immediate bias)
    adx = float(df.iloc[-1]["adx14"])
    if adx < 15:
        return StratDecision("Accumulation / Distribution Phases", "no-trade", 0.4, timeframe, df.index[-1].isoformat(), extras={"adx14": adx})
    return StratDecision("Accumulation / Distribution Phases", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

def _strat_breaker_mitigation(df: pd.DataFrame, timeframe: str) -> StratDecision:
    return StratDecision("Breaker Blocks & Mitigation Blocks", "no-trade", 0.0, timeframe, df.index[-1].isoformat(),
                         extras={"note":"Pattern mining not implemented."})

def _strat_dealer_range_asian(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # Asian session range (00:00–06:00 UTC) sweep and return
    ddf = df.copy()
    idx = ddf.index
    today = idx[-1].date()
    overnight = ddf[(idx >= datetime(today.year,today.month,today.day,tzinfo=UTC)) &
                    (idx < datetime(today.year,today.month,today.day,6,tzinfo=UTC))]
    if len(overnight) < 5:
        return StratDecision("Dealer Range / Asian Session Manipulation", "no-trade", 0.0, timeframe, idx[-1].isoformat())
    a_hi = float(overnight["high"].max()); a_lo = float(overnight["low"].min())
    px = float(ddf.iloc[-1]["close"]); atr = float(ddf.iloc[-1]["atr14"])
    if px < a_lo and ddf.iloc[-1]["close"] > a_lo:
        return StratDecision("Dealer Range / Asian Session Manipulation", "buy", _nz01((a_lo - px)/max(atr,1e-9),0.1,1.0), timeframe, idx[-1].isoformat())
    if px > a_hi and ddf.iloc[-1]["close"] < a_hi:
        return StratDecision("Dealer Range / Asian Session Manipulation", "sell", _nz01((px - a_hi)/max(atr,1e-9),0.1,1.0), timeframe, idx[-1].isoformat())
    return StratDecision("Dealer Range / Asian Session Manipulation", "no-trade", 0.0, timeframe, idx[-1].isoformat())

def _strat_smc_divergence(df: pd.DataFrame, timeframe: str) -> StratDecision:
    # RSI vs price divergence
    _ensure_min_rows(df, 60)
    px = df["close"].astype(float)
    rsi = df["rsi14"].astype(float)
    # recent swing points
    hi1 = px.iloc[-1]; hi2 = px.iloc[-10]
    r1 = rsi.iloc[-1]; r2 = rsi.iloc[-10]
    # simplistic check
    if hi1 > hi2 and r1 < r2 and r1 > 50:
        return StratDecision("Smart Money Divergence", "sell", 0.55, timeframe, df.index[-1].isoformat())
    lo1 = px.iloc[-1]; lo2 = px.iloc[-10]
    if lo1 < lo2 and r1 > r2 and r1 < 50:
        return StratDecision("Smart Money Divergence", "buy", 0.55, timeframe, df.index[-1].isoformat())
    return StratDecision("Smart Money Divergence", "no-trade", 0.0, timeframe, df.index[-1].isoformat())

# ---- Registry ----
_STRATEGY_FUNCS: Dict[str, Callable[[pd.DataFrame, str], StratDecision]] = {
    # Retail / classic
    "Trend Following": _strat_trend_following,
    "Counter-Trend": _strat_counter_trend,
    "Breakout Trading": _strat_breakout,
    "Range Trading": _strat_range,
    "Swing Trading": _strat_swing,
    "Scalping": _strat_scalping,
    "Day Trading": _strat_day_trading,
    "Position Trading": _strat_position_trading,
    "Momentum Trading": _strat_momentum,
    "Price Action Trading": _strat_price_action,
    "Support and Resistance Trading": _strat_support_resistance,
    "Moving Average Crossover": _strat_ma_crossover,
    "Bollinger Band Strategy": _strat_bollinger,
    "Ichimoku Cloud Strategy": _strat_ichimoku,
    "Fibonacci Retracement Strategy": _strat_fibonacci,
    "Pivot Point Strategy": _strat_pivot_points,
    "Carry Trade": _strat_carry,
    "Grid Trading": _strat_grid,
    "Hedging Strategy": _strat_hedging,
    "News Trading": _strat_news,
    "Arbitrage Trading": _strat_arbitrage,
    "Mean Reversion": _strat_mean_reversion,
    "Algorithmic Trading": _strat_algo,
    "High-Frequency Trading (HFT)": _strat_hft,

    # Institutional / SMC
    "Market Making": _strat_market_making,
    "Order Flow Trading": _strat_order_flow,
    "Liquidity Grab / Stop Hunt": _strat_liquidity_grab,
    "Wyckoff Method": _strat_wyckoff,
    "Supply and Demand Zones": _strat_supply_demand,
    "Fair Value Gap (FVG) Strategy": _strat_fvg,
    "Institutional Order Blocks": _strat_order_blocks,
    "Accumulation / Distribution Phases": _strat_accum_dist,
    "Breaker Blocks & Mitigation Blocks": _strat_breaker_mitigation,
    "Dealer Range / Asian Session Manipulation": _strat_dealer_range_asian,
    "Smart Money Divergence": _strat_smc_divergence,
}

def _fetch_and_prepare(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    df = _copy_rates_recent_df(symbol, timeframe, lookback_days)
    if df.empty:
        raise ValueError("No data returned from MT5 for given inputs.")
    df["symbol"] = symbol
    df = _prepare_indicators(df)
    return df

def evaluate_strategies(symbol: str, timeframe: str, lookback_days: int = 90, which: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        df = _fetch_and_prepare(symbol, timeframe, lookback_days)
        names = which if which else list(_STRATEGY_FUNCS.keys())
        out: List[Dict[str, Any]] = []
        for name in names:
            fn = _STRATEGY_FUNCS.get(name)
            if not fn:
                out.append({"strategy": name, "error": "unknown strategy"})
                continue
            try:
                dec = fn(df, timeframe)
            except Exception as e:
                out.append({"strategy": name, "decision":"no-trade","confidence":0.0,"timeframe":timeframe,"as_of_utc":df.index[-1].isoformat(),"error":str(e)})
                continue
            out.append({
                "strategy": dec.strategy,
                "decision": dec.decision,
                "confidence": round(float(dec.confidence), 4),
                "timeframe": dec.timeframe,
                "as_of_utc": dec.as_of_utc,
                "extras": dec.extras or {}
            })
        return {"ok": True, "symbol": symbol, "timeframe": timeframe, "results": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def evaluate_strategies_batch(symbols: List[str], timeframes: List[str], lookback_days: int = 90,
                              which: Optional[List[str]] = None) -> Dict[str, Any]:
    all_results = []
    for s in symbols:
        for tf in timeframes:
            res = evaluate_strategies(s, tf, lookback_days, which)
            all_results.append(res)
    return {"ok": True, "batch": all_results}

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
    # --- Analytics & Strategies ---
    {"type":"function","function":{
        "name":"calculate_indicators","description":"Compute TA features from recent MT5 bars and save to CSV.",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"timeframe":{"type":"string"},
            "lookback_days":{"type":"integer"},"out_csv_path":{"type":"string"}},
            "required":["symbol","timeframe"]}
    }},
    {"type":"function","function":{
        "name":"analyze_market_status","description":"Lightweight market analysis (trend, RSI, ATR) on recent bars.",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},"timeframe":{"type":"string"},"lookback_days":{"type":"integer"}},
            "required":["symbol"]}
    }},
    {"type":"function","function":{
        "name":"list_forex_strategies","description":"List supported strategies.",
        "parameters":{"type":"object","properties":{},"required":[]}
    }},
    {"type":"function","function":{
        "name":"evaluate_strategies","description":"Apply strategies and return buy/sell/no-trade + confidence.",
        "parameters":{"type":"object","properties":{
            "symbol":{"type":"string"},
            "timeframe":{"type":"string"},
            "lookback_days":{"type":"integer"},
            "which":{"type":"array","items":{"type":"string"}}},
            "required":["symbol","timeframe"]}
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

    if name == "list_dir":                      return list_dir(**args)
    if name == "search_files":                  return search_files(**args)
    if name == "read_file":                     return read_file(**args)
    if name == "write_file":                    return write_file(**args)
    if name == "open_path":                     return open_path(**args)
    if name == "run_command":                   return run_command(**args)
    if name == "execute_exe":                   return execute_exe(**args)

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

    if name == "calculate_indicators":          return calculate_indicators(**args)
    if name == "analyze_market_status":         return analyze_market_status(**args)
    if name == "list_forex_strategies":         return list_forex_strategies()
    if name == "evaluate_strategies":           return evaluate_strategies(**args)

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
