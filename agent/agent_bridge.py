import traceback
from typing import List, Dict, Any, Optional
from config.settings import SETTINGS

# Import your engine
import importlib
ga = importlib.import_module("gpt_agent")

# Disable CLI approvals so the web UI won't block
try:
    if SETTINGS.force_disable_tool_approval and hasattr(ga, "REQUIRE_APPROVAL"):
        ga.REQUIRE_APPROVAL = False  # runtime override
except Exception:
    pass

def mt5_connect_safe() -> Dict[str, Any]:
    try:
        res = ga.mt5_connect()
        if not res.get("ok"):
            res = ga.mt5_connect(relaunch=True)
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

def ensure_symbol(symbol: str) -> Dict[str, Any]:
    try:
        return ga.mt5_symbol_enable(symbol)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def list_strategies() -> List[str]:
    try:
        res = ga.list_forex_strategies()
        if res.get("ok"):
            items = res.get("strategies", [])
            if items and isinstance(items[0], dict) and "name" in items[0]:
                return [x["name"] for x in items]
            if items and isinstance(items[0], str):
                return items
        if hasattr(ga, "_STRATEGY_FUNCS"):
            return list(ga._STRATEGY_FUNCS.keys())  # type: ignore
    except Exception:
        pass
    return []

def evaluate(
    symbol: str,
    timeframes: List[str],
    which: Optional[List[str]] = None,
    lookback_days: int = SETTINGS.default_lookback_days,
    min_confidence: float = SETTINGS.default_min_confidence,
) -> Dict[str, Any]:
    """
    Run evaluate_strategies on each timeframe, flatten results,
    filter by min_confidence and drop 'no-trade'.
    """
    out_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    c = mt5_connect_safe()
    if not c.get("ok"):
        return {"ok": False, "error": f"MT5 connect failed: {c.get('error', 'unknown')}"}
    s = ensure_symbol(symbol)
    if not s.get("ok"):
        return {"ok": False, "error": f"Symbol enable failed: {s.get('error', 'unknown')}"}

    for tf in timeframes:
        try:
            res = ga.evaluate_strategies(symbol=symbol, timeframe=tf, lookback_days=lookback_days, which=which)
            if not res.get("ok"):
                errors.append(f"{tf}: {res.get('error','unknown')}")
                continue
            for r in res.get("results", []):
                if r.get("decision") == "no-trade":
                    continue
                if float(r.get("confidence", 0.0)) < float(min_confidence):
                    continue
                out_rows.append({
                    "strategy": r.get("strategy"),
                    "decision": r.get("decision"),
                    "confidence": float(r.get("confidence", 0.0)),
                    "timeframe": r.get("timeframe", tf),
                    "as_of_utc": r.get("as_of_utc"),
                    "extras": r.get("extras", {}),
                })
        except Exception:
            errors.append(f"{tf}: {traceback.format_exc(limit=1)}")

    out_rows.sort(key=lambda x: x["confidence"], reverse=True)
    return {"ok": True, "rows": out_rows, "errors": errors}

def place_order(symbol: str, side: str, volume: float,
                sl_points: int | None = None, tp_points: int | None = None) -> Dict[str, Any]:
    """
    Guarded passthrough to engine mt5_place_order.
    """
    try:
        c = mt5_connect_safe()
        if not c.get("ok"):
            return {"ok": False, "error": f"MT5 connect failed: {c.get('error', 'unknown')}"}
        s = ensure_symbol(symbol)
        if not s.get("ok"):
            return {"ok": False, "error": f"Symbol enable failed: {s.get('error', 'unknown')}"}
        res = ga.mt5_place_order(
            symbol=symbol,
            side=side,
            volume=float(volume),
            sl_points=(None if sl_points in ("", None) else int(sl_points)),
            tp_points=(None if tp_points in ("", None) else int(tp_points)),
        )
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

def list_positions(symbol: Optional[str] = None) -> Dict[str, Any]:
    try:
        c = mt5_connect_safe()
        if not c.get("ok"):
            return {"ok": False, "error": f"MT5 connect failed: {c.get('error', 'unknown')}"}
        res = ga.mt5_positions_get(symbol=symbol)
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

def close_position(ticket: int, volume: Optional[float] = None) -> Dict[str, Any]:
    try:
        c = mt5_connect_safe()
        if not c.get("ok"):
            return {"ok": False, "error": f"MT5 connect failed: {c.get('error', 'unknown')}"}
        res = ga.mt5_close_position(ticket=int(ticket), volume=(None if volume in (None, "") else float(volume)))
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}
