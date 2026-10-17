"""Post-hoc calibration.

Applies a per-asset-class spread multiplier learned from the walk-forward backtest
(`validation.py` writes `configs/calibration.json`; see `docs/VALIDATION.md` for how it was
produced and its honestly-reported coverage numbers). If no calibration file is present or an
asset class is unseen, the multiplier defaults to 1.0 -- calibration only ever adjusts spread
based on a MEASURED miscalibration, never invents one.

Also enforces a numerical floor so a degenerate (near-zero) scale can never reach the sampler --
a floor, not a correction: CRPS punishes an overconfident/near-delta forecast severely, and this
exists purely to stop a divide-by-zero-adjacent scale from a thin-history asset, not to change a
well-estimated forecast.
"""

from __future__ import annotations

import json
import pathlib

MIN_SCALE_FLOOR_RELATIVE = 1e-4  # relative to the anchor level's own magnitude


def load_calibration_table(path: pathlib.Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): float(v) for k, v in raw.get("spread_multiplier_by_asset_class", {}).items()}


def asset_class_of(asset: str) -> str:
    a = asset.upper()
    if a.startswith("UST") or "YIELD" in a or a.startswith("RATE"):
        return "rates"
    if len(a) == 6 and a.isalpha():  # e.g. EURUSD-style
        return "fx"
    if a in {"MKT", "SMB", "HML", "MOM", "BAB", "QMJ"}:
        return "factors"
    if "CPI" in a or "PCE" in a or "NFP" in a or "GDP" in a:
        return "macro"
    return "fx" if len(a) <= 3 else "other"


def calibrate_scale(
    asset: str, anchor_level: float, scale: float, table: dict[str, float],
) -> float:
    mult = table.get(asset_class_of(asset), 1.0)
    calibrated = scale * mult
    floor = max(abs(anchor_level) * MIN_SCALE_FLOOR_RELATIVE, 1e-8)
    return max(calibrated, floor)
