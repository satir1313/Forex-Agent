"""
auto_trader.py - Automated trading logic for MT5 Agent

- Accepts `timeframes` (normalized to M5, M15, H1)
- One-cycle AutoTrader.run() (works with current Gradio button)
- Optional MultiAutoTrader for multi-symbol infinite loop
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from agent.agent_bridge import evaluate, train_pipeline, list_strategies  # noqa: F401


# ---------- Constants ----------
ALLOWED_TIMEFRAMES: tuple[str, ...] = ("M5", "M15", "H1")
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
AI_CONFIDENCE_THRESHOLD = 80  # percent
LOOKBACK_DAYS_DEFAULT = 30
MIN_CONFIDENCE_DEFAULT = 0.25
CYCLE_SLEEP_SECONDS = 60  # used by MultiAutoTrader


# ---------- OpenAI client (robust init) ----------
def _build_openai_client() -> OpenAI:
    # Prefer existing env var; only load .env if missing
    if not os.environ.get("OPENAI_API_KEY"):
        load_dotenv(find_dotenv(usecwd=True), override=False)

    key = os.environ.get("OPENAI_API_KEY", "") or ""
    key = key.strip().strip('"').strip("'").replace("\n", "").replace("\r", "").replace("\t", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your shell env or .env (OPENAI_API_KEY=sk-...).")

    print(f"[AutoTrader] Using OpenAI key suffix: ...{key[-4:]} (len={len(key)})")
    return OpenAI(api_key=key)


_client = _build_openai_client()


# ---------- Single-symbol worker ----------
class AutoTrader:
    def __init__(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,   # <— added
        min_confidence: float = MIN_CONFIDENCE_DEFAULT,
        lookback_days: int = LOOKBACK_DAYS_DEFAULT,
        openai_model: str = OPENAI_MODEL_DEFAULT,
    ):
        self.symbol = symbol

        # Normalize timeframes to only the allowed set (M5, M15, H1).
        norm = [tf.strip().upper() for tf in (timeframes or []) if isinstance(tf, str)]
        if not norm:
            norm = list(ALLOWED_TIMEFRAMES)
        self.timeframes = [tf for tf in norm if tf in ALLOWED_TIMEFRAMES] or list(ALLOWED_TIMEFRAMES)

        self.min_confidence = float(min_confidence)
        self.lookback_days = int(lookback_days)
        self.openai_model = openai_model

    def train_arvid(self) -> bool:
        """Train Arvid ML model for the symbol/timeframes."""
        print(f"[AutoTrader] Training Arvid for {self.symbol} {self.timeframes}")
        result = train_pipeline([self.symbol], self.timeframes, self.lookback_days)
        if not result.get("ok"):
            print(f"[AutoTrader] Training failed for {self.symbol}: {result.get('error')}")
            return False
        print(f"[AutoTrader] Training complete for {self.symbol}.")
        return True

    def fetch_signals(self) -> List[Dict]:
        """Fetch all strategy signals (including Arvid) only for allowed TFs."""
        print(f"[AutoTrader] Fetching strategy signals for {self.symbol} (TFs: {self.timeframes})")
        result = evaluate(
            symbol=self.symbol,
            timeframes=self.timeframes,
            lookback_days=self.lookback_days,
            min_confidence=self.min_confidence,
        )
        if not result.get("ok"):
            print(f"[AutoTrader] Evaluation failed for {self.symbol}: {result.get('error')}")
            return []

        rows = result.get("rows", []) or []
        # Extra guard
        filtered = [r for r in rows if r.get("timeframe") in ALLOWED_TIMEFRAMES]
        return filtered

    def ask_ai_confidence(self, signals: List[Dict]) -> Tuple[int, str]:
        """Send signals to OpenAI and get confidence in order profitability."""
        prompt_lines = []
        prompt_lines += [
            "",
            f"If you are more than {AI_CONFIDENCE_THRESHOLD}% confident, say so explicitly.",
            "At the very end, output exactly one line in this format: CONFIDENCE: <0-100>",
        ]
        for s in signals:
            conf_pct = round(float(s.get("confidence", 0.0)) * 100.0, 2)
            prompt_lines.append(
                f"- Strategy: {s.get('strategy')} | Decision: {s.get('decision')} | "
                f"Confidence: {conf_pct}% | Timeframe: {s.get('timeframe')}"
            )
        prompt_lines += ["", f"If you are more than {AI_CONFIDENCE_THRESHOLD}% confident, say so explicitly."]
        prompt = "\n".join(prompt_lines)

        print(f"[AutoTrader] Sending to OpenAI for {self.symbol}:\n{prompt}")

        try:
            resp = _client.responses.create(model=self.openai_model, input=prompt)
            reply = "AI ANALYSIS: " + (resp.output_text or "")
        except Exception as e:
            msg = f"[AutoTrader] OpenAI error for {self.symbol}: {e}"
            print(msg)
            return 0, msg

        print(f"[AutoTrader] OpenAI reply ({self.symbol}): {reply}")
        m = re.search(r"CONFIDENCE\s*:\s*(100|[1-9]?\d)\b", reply)
        if m:
            confidence = int(m.group(1))
        else:
            # Fallback: accept decimals and pick the LAST percentage in the reply (usually the summary)
            # This matches e.g. 77%, 79.94%, 100%, 0.0%
            nums = re.findall(r"(?:100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)\s*%", reply)
            confidence = int(float(nums[-1])) if nums else 0

        confidence = max(0, min(confidence, 100))
        return confidence, reply

    def run_once(self) -> None:
        """One full cycle for this symbol."""
        if not self.train_arvid():
            print(f"[AutoTrader] Aborting {self.symbol}: training failed.")
            return

        signals = self.fetch_signals()
        if not signals:
            print(f"[AutoTrader] No signals for {self.symbol}.")
            return

        confidence, _ = self.ask_ai_confidence(signals)
        print(f"[AutoTrader] AI confidence for {self.symbol}: {confidence}%")

        top_signal = next((s for s in signals if s.get("decision") in ("buy", "sell")), None)
        if confidence >= AI_CONFIDENCE_THRESHOLD and top_signal:
            price = (top_signal.get("extras") or {}).get("price", "N/A")
            print(
                f"[AutoTrader] ✅ Would place {top_signal['decision']} on {self.symbol} "
                f"({top_signal.get('timeframe')}) at price {price}"
            )
        else:
            print(f"[AutoTrader] ❌ Confidence too low or no actionable signal for {self.symbol}.")

    # Keep `run()` as a single cycle so your Gradio button doesn't block forever
    def run(self) -> None:
        self.run_once()


# ---------- Optional: Multi-symbol infinite runner ----------
class MultiAutoTrader:
    """
    Run many symbols sequentially, forever (Ctrl+C to stop).
    Not used by the UI button unless you wire it there.
    """
    def __init__(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        openai_model: str = OPENAI_MODEL_DEFAULT,
        min_confidence: float = MIN_CONFIDENCE_DEFAULT,
        lookback_days: int = LOOKBACK_DAYS_DEFAULT,
        cycle_sleep_seconds: int = CYCLE_SLEEP_SECONDS,
    ):
        self.symbols = list(dict.fromkeys(symbols))  # de-dup, preserve order
        norm = [tf.strip().upper() for tf in (timeframes or []) if isinstance(tf, str)]
        if not norm:
            norm = list(ALLOWED_TIMEFRAMES)
        self.timeframes = [tf for tf in norm if tf in ALLOWED_TIMEFRAMES] or list(ALLOWED_TIMEFRAMES)

        self.openai_model = openai_model
        self.min_confidence = float(min_confidence)
        self.lookback_days = int(lookback_days)
        self.cycle_sleep_seconds = int(cycle_sleep_seconds)

    def run_forever(self) -> None:
        print(
            "[AutoTrader] Starting infinite loop.\n"
            f"  Symbols: {self.symbols}\n"
            f"  Timeframes: {self.timeframes}\n"
            f"  OpenAI model: {self.openai_model}\n"
            f"  Cycle sleep: {self.cycle_sleep_seconds}s\n"
        )
        try:
            while True:
                cycle_start = time.time()
                for sym in self.symbols:
                    print("\n" + "=" * 80)
                    print(f"[AutoTrader] Processing symbol: {sym}")
                    try:
                        worker = AutoTrader(
                            symbol=sym,
                            timeframes=self.timeframes,
                            min_confidence=self.min_confidence,
                            lookback_days=self.lookback_days,
                            openai_model=self.openai_model,
                        )
                        worker.run_once()
                    except Exception as e:
                        print(f"[AutoTrader] Unhandled error for {sym}: {e}")
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.cycle_sleep_seconds - int(elapsed))
                print(f"\n[AutoTrader] Cycle complete in {elapsed:.1f}s. Sleeping {sleep_time}s...\n")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[AutoTrader] Stopped by user. Goodbye.")


# Example (manual, not via Gradio):
# if __name__ == "__main__":
#     symbols = ["USDJPY.a", "EURUSD.a", "GBPUSD.a"]
#     runner = MultiAutoTrader(symbols=symbols, timeframes=["M5","M15","H1"], cycle_sleep_seconds=60)
#     runner.run_forever()
