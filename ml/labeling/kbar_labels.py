from __future__ import annotations

import pandas as pd
from typing import List, Dict


def compute_kbar_labels(
    df: pd.DataFrame,
    horizons: List[int],
    k: float,
) -> pd.DataFrame:
    """Compute k-bar forward-return labels with volatility scaling.

    Inputs:
      - df: DataFrame with columns time_utc, close, atr_14 (no lookahead).
      - horizons: list of horizons (bars ahead) to label, e.g., [4,6].
      - k: ATR multiple threshold per TF.

    Outputs (long format):
      columns: time_utc, h, fwd_ret, theta, label in {buy,sell,none}
    """
    need = ["time_utc", "close", "atr_14"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    out_rows = []
    base = df[["time_utc", "close", "atr_14"]].copy()
    base = base.sort_values("time_utc").reset_index(drop=True)
    close = base["close"].astype(float)
    atr = base["atr_14"].astype(float)
    theta_series = k * (atr / close)

    for h in sorted(set(int(x) for x in horizons)):
        fwd_close = close.shift(-h)
        fwd_ret = (fwd_close - close) / close
        theta = theta_series
        label = pd.Series("none", index=fwd_ret.index, dtype="object")
        label = label.mask(fwd_ret > theta, "buy")
        label = label.mask(fwd_ret < -theta, "sell")
        # Drop rows where forward data is unavailable (tail h rows)
        valid = fwd_ret.notna()
        frame = pd.DataFrame(
            {
                "time_utc": base.loc[valid, "time_utc"].values,
                "h": h,
                "fwd_ret": fwd_ret.loc[valid].values,
                "theta": theta.loc[valid].values,
                "label": label.loc[valid].values,
            }
        )
        out_rows.append(frame)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=["time_utc", "h", "fwd_ret", "theta", "label"]
    )
    return out

