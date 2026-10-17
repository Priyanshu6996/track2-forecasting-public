"""Learn the text/numeric fusion weights from historical walk-forward data, instead of hand
picking them ("hawkish -> add X sigma").

## What is learned

Two scalars, both consumed by `fusion.fuse` as `learned_calibration` and both dimensionless
blend/sensitivity weights (never a step past `FusionConfig`'s hard rails, see fusion.py):

  bias_blend_weight   how much weight the per-asset `directional_bias` gets vs. the broad
                       `policy_stance` when they disagree (fusion.py's old code hard-coded 0.5).
  vol_sensitivity      how strongly `expected_vol_change` should actually move the forecast's
                       scale, relative to the `FusionConfig.max_vol_mult` rail (old code assumed
                       this was exactly 1.0, i.e. that the LLM's stated vol view is exactly as
                       reliable as the rail was designed to tolerate).

## How, and how leakage is avoided

For every pseudo as-of cutoff inside a unit's own pre-official-as-of window (the exact same
cutoffs `validation.py` uses for its ablation folds — see that module's docstring for why this
is leakage-safe: it only uses a unit's own, earlier, self-contained history), this module:

  1. extracts text signals using only documents timestamped <= that pseudo as-of (`extract_text_signals`, unchanged),
  2. fits the numeric model using only panel rows <= that pseudo as-of (`fit_numeric_forecast`, unchanged),
  3. reads the REALIZED value at pseudo_asof + horizon from the unit's own later panel rows
     (exactly as `validation.evaluate_unit` does for scoring),
  4. records the standardized surprise `(realized - numeric.point) / numeric.resid_std` and the
     text signal fields active at that cutoff.

Step 3 is the only place "future" information appears, and it only appears as the REGRESSION
TARGET for the coefficient fit, at fit time — precisely mirroring how the official scorer will
later use a sealed realized outcome as a target Claude never sees at inference time. The fitted
coefficients themselves are then applied identically to every future forecast, so no forecast
ever uses its own realized outcome.

## Aggregation and shrinkage

Rows are pooled across every practice unit passed in (more data, better-conditioned fit; still
never crosses from one unit's TEXT into another unit's FORECAST — only the fitted scalar weights
themselves are shared, which is what "learn a general calibration" has to mean here). Given how
few text-bearing cutoffs a handful of practice units actually offer, `fit_calibration` shrinks
both coefficients toward `fusion.DEFAULT_CALIBRATION` in proportion to sample size (a simple
James-Stein-style shrinkage, `n / (n + PRIOR_STRENGTH)`), and refuses to move the coefficient at
all below `MIN_ROWS` observations — i.e. this explicitly follows the review's "do not tune
against a handful of practice cards" instruction rather than fitting an confident-looking but
noise-driven number from three data points.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from .config import AgentConfig
from .fusion import DEFAULT_CALIBRATION
from .numeric_model import fit_numeric_forecast
from .panels import all_asset_ids, asset_series, read_panels
from .text_signals import extract_text_signals

# `validation.py` imports `cli.py` (for ADD_ANCHOR_BY_TARGET_TYPE / _to_feature_level) and
# `cli.py` imports this module (for load_text_calibration), so importing validation.py at
# MODULE level here would be a circular import. `_pseudo_cutoffs` / `_shift_business_days` are
# imported lazily, inside `collect_rows`, instead -- by the time this module's functions are
# actually called, all three modules have finished initializing.

MIN_ROWS = 8            # below this many pooled observations, do not deviate from the prior at all
PRIOR_STRENGTH = 12.0     # shrinkage denominator; larger = more conservative toward the prior


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    unit_id: str
    asset: str
    horizon: int
    bias: float
    policy_stance: float
    expected_vol_change: float
    confidence: float
    standardized_surprise: float  # (realized - numeric.point) / numeric.resid_std


def collect_rows(unit_dir: pathlib.Path, n_folds: int = 4) -> list[CalibrationRow]:
    """One unit's worth of calibration rows, built the same way `validation.evaluate_unit` builds
    its scoring folds (see module docstring for why this is leakage-safe)."""
    import tomllib

    from .features import asset_feature_frame
    from .validation import _pseudo_cutoffs, _shift_business_days

    card_path = unit_dir / "card.toml"
    if not card_path.exists():
        return []
    card = tomllib.loads(card_path.read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    unit_id = card["task"]["id"]
    text_dir = unit_dir / "text"
    cfg = AgentConfig.from_name("full")

    panels_full = read_panels(unit_dir)
    available = set(all_asset_ids(panels_full))
    if any(a not in available for a in assets):
        return []

    all_dates = sorted({
        d for df in panels_full.values() for d in df["date"].astype(str).str.slice(0, 10).unique()
    })
    cutoffs = _pseudo_cutoffs(all_dates, max(horizons), n_folds)

    rows: list[CalibrationRow] = []
    for pseudo_asof in cutoffs:
        text_signals = extract_text_signals(text_dir, pseudo_asof, assets, cfg.text)
        if text_signals.method == "none" or text_signals.confidence <= 0:
            continue  # nothing to calibrate against at this cutoff
        for a in assets:
            series_upto = asset_series(
                {k: df[df["date"].astype(str).str.slice(0, 10) <= pseudo_asof]
                 for k, df in panels_full.items()},
                a, pseudo_asof,
            ).values
            if len(series_upto) < 40:
                continue
            feats = asset_feature_frame(series_upto)
            for h in horizons:
                target_date = _shift_business_days(all_dates, pseudo_asof, h)
                if target_date is None:
                    continue
                realized_series = asset_series(panels_full, a, target_date).values
                if str(realized_series.index[-1].date()) != target_date:
                    continue
                realized = float(realized_series.iloc[-1])
                nf = fit_numeric_forecast(feats, series_upto, h, pseudo_asof, cfg.model)
                if nf.resid_std <= 0 or not np.isfinite(nf.resid_std):
                    continue
                surprise = (realized - nf.point) / nf.resid_std
                rows.append(CalibrationRow(
                    unit_id=unit_id, asset=a, horizon=h,
                    bias=text_signals.directional_bias.get(a, 0.0),
                    policy_stance=text_signals.policy_stance,
                    expected_vol_change=text_signals.expected_vol_change,
                    confidence=text_signals.confidence,
                    standardized_surprise=surprise,
                ))
    return rows


def _weighted_ols_two_predictor(x1, x2, y, w):
    """Minimal weighted least squares for y ~ b1*x1 + b2*x2 (no intercept — all three signals are
    already zero-centered by construction: bias/policy_stance in [-1,1] around 0, surprise around
    0 under a well-calibrated numeric model)."""
    W = np.diag(w)
    X = np.column_stack([x1, x2])
    XtW = X.T @ W
    A = XtW @ X + 1e-6 * np.eye(2)  # tiny ridge for numerical stability, not for bias
    b = XtW @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0.5, 0.5
    return float(coef[0]), float(coef[1])


def fit_calibration(rows: list[CalibrationRow]) -> dict[str, float]:
    """Fits `bias_blend_weight` and `vol_sensitivity`, shrunk toward `DEFAULT_CALIBRATION` by
    sample size (see module docstring). Returns a dict with the fitted values PLUS diagnostic
    fields (`n_rows`, `shrinkage`) so callers/reports can see how much the fit actually moved."""
    n = len(rows)
    prior = dict(DEFAULT_CALIBRATION)
    if n < MIN_ROWS:
        return {**prior, "n_rows": n, "shrinkage": 1.0}

    bias = np.array([r.bias for r in rows])
    stance = np.array([r.policy_stance for r in rows])
    conf = np.array([max(r.confidence, 1e-3) for r in rows])
    surprise = np.array([r.standardized_surprise for r in rows])

    b1, b2 = _weighted_ols_two_predictor(bias, stance, surprise, conf)
    denom = abs(b1) + abs(b2)
    fitted_blend = 0.5 if denom < 1e-9 else float(np.clip(abs(b1) / denom, 0.0, 1.0))

    vol_signal = np.array([r.expected_vol_change for r in rows]) * conf
    abs_surprise = np.abs(surprise)
    var = float(np.var(vol_signal))
    fitted_vol_sensitivity = 1.0 if var < 1e-9 else float(
        np.clip(np.cov(vol_signal, abs_surprise)[0, 1] / var, 0.0, 3.0)
    )

    shrink = PRIOR_STRENGTH / (n + PRIOR_STRENGTH)  # weight on the PRIOR; -> 0 as n grows
    blend = shrink * prior["bias_blend_weight"] + (1 - shrink) * fitted_blend
    vol_sens = shrink * prior["vol_sensitivity"] + (1 - shrink) * fitted_vol_sensitivity

    return {
        "bias_blend_weight": round(blend, 4),
        "vol_sensitivity": round(vol_sens, 4),
        "n_rows": n,
        "shrinkage": round(shrink, 4),
        "fitted_before_shrinkage": {
            "bias_blend_weight": round(fitted_blend, 4),
            "vol_sensitivity": round(fitted_vol_sensitivity, 4),
        },
    }


def learn_from_units(unit_dirs: list[pathlib.Path], n_folds: int = 4) -> dict:
    all_rows: list[CalibrationRow] = []
    for ud in unit_dirs:
        try:
            all_rows.extend(collect_rows(ud, n_folds=n_folds))
        except Exception:  # noqa: BLE001 - a broken/unusable unit just contributes no rows
            continue
    calibration = fit_calibration(all_rows)
    return {
        "spread_multiplier_by_asset_class": {},  # reserved for calibration.py's own table
        "text_fusion_calibration": calibration,
        "n_units": len(unit_dirs),
    }


def write_calibration_file(path: pathlib.Path, unit_dirs: list[pathlib.Path], n_folds: int = 4) -> dict:
    result = learn_from_units(unit_dirs, n_folds=n_folds)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_text_calibration(path: pathlib.Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    calib = raw.get("text_fusion_calibration")
    if not isinstance(calib, dict):
        return None
    return {
        "bias_blend_weight": float(calib.get("bias_blend_weight", DEFAULT_CALIBRATION["bias_blend_weight"])),
        "vol_sensitivity": float(calib.get("vol_sensitivity", DEFAULT_CALIBRATION["vol_sensitivity"])),
    }
