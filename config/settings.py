from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

@dataclass
class Settings:
    # UI defaults
    default_timeframes: tuple = ("M5", "M15", "H1", "H4", "D1")
    default_lookback_days: int = 365
    default_min_confidence: float = 0.15

    # Trading defaults (only used if you enable Place Order)
    default_volume: float = 0.01
    default_sl_points: int | None = None
    default_tp_points: int | None = None

    # Behavior: the web UI disables agent CLI approvals
    force_disable_tool_approval: bool = True

    # Persistence
    export_dir: str = _expand("./exports")
    sessions_dir: str = _expand("./sessions")

    # Optional: limit results shown
    max_rows_display: int = 200

SETTINGS = Settings()
os.makedirs(SETTINGS.export_dir, exist_ok=True)
os.makedirs(SETTINGS.sessions_dir, exist_ok=True)
