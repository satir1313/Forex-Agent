from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    # UI defaults
    default_timeframes: tuple = ("M5","M15","H1","H4","D1")
    default_lookback_days: int = 365
    default_min_confidence: float = 0.15

    # Behavior
    # We’ll flip the engine’s interactive tool approvals OFF from the bridge.
    force_disable_tool_approval: bool = True

    # Optional: limit results shown
    max_rows_display: int = 200

SETTINGS = Settings()
