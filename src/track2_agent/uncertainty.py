"""Residual-scale estimation for cases the walk-forward numeric model can't validate directly.

## The bug this replaced

`fit_numeric_forecast`'s small-sample fallback (used whenever a unit has fewer than
`ModelConfig.min_train_rows` usable training rows for a given asset/horizon — 15-23 rows for
`t2-F1-sahm-watch-2024` / `t2-F1-cpi-glidepath-2023` in the diagnostics that motivated this
module) used to compute:

    resid_std = values.diff().std() * sqrt(horizon)

This has two independent problems, and both get WORSE exactly when the sample is smallest:

1. `values.diff().std()` is not robust — a single outlier day (a release-day spike, a COVID-era
   print) can dominate a 15-23 observation sample's standard deviation.
2. `sqrt(horizon)` assumes the series behaves like a random walk with i.i.d. increments, so its
   implied dispersion at horizon h scales EXACTLY with sqrt(h) no matter what. That's a
   defensible default for a freely floating asset price; it is systematically wrong for a slow,
   bounded, mean-reverting macro release (an unemployment rate, a CPI level, a yield-curve-
   control target), and applying it anyway with a large horizon (145-189 business days in the
   cases above) and a noisy small-sample base turns a modest per-day estimate into an
   economically nonsensical one (the diagnostics that motivated this fix: UNRATE forecast
   distributions spanning roughly -41 to +45 percentage points).

## The fix

`estimate_fallback_resid_std` still uses only information available up to `asof` (no leakage —
see its body), but replaces both problems with something that adapts to whatever evidence the
series itself provides, rather than assuming a fixed model:

- Uses a MAD-based robust scale (resistant to a handful of outlier days) instead of raw std,
  wherever a "dispersion of a sample of diffs" needs computing.
- Estimates how the series' OWN dispersion actually scales with lag (a log-log regression of
  robust dispersion vs. lag, over every lag from 1 up to `horizon` that has enough overlapping
  observations to fit) instead of assuming the random-walk exponent (0.5) applies universally.
  This directly distinguishes a trending/compounding series (exponent > 0.5) from a
  mean-reverting/bounded one (exponent < 0.5) FROM THE SERIES' OWN HISTORY — it never
  special-cases a named target (no `if target == "UNRATE"`), so it generalizes to any level,
  rate, or price series the competition throws at it.
- Shrinks that estimated exponent toward the random-walk prior (0.5) in proportion to how many
  lags actually supported the fit, so a noisy 3-point regression cannot produce an extreme,
  overconfident exponent.
- Falls back to the robust-scale-times-sqrt(horizon) random-walk assumption only when there
  truly isn't enough overlapping data to estimate a lag-scaling exponent at all.
- Applies a final, generic ceiling: the estimated dispersion is capped at `range_ceiling_mult`
  times the full historical range the series has actually shown up to `asof`. This does not
  forbid genuinely wide/fat-tailed forecasts when the series' own history supports them (e.g. a
  target whose training window spans a real historical shock keeps a wide ceiling, because its
  observed range is itself wide); it only stops a short-sample extrapolation from asserting more
  dispersion than the series has ever actually exhibited.

This function is also used (via `apply_range_ceiling`) as a light-touch sanity check on the
resid_std produced by the FULLY validated (non-fallback) path too — see `numeric_model.py` — on
the theory that a generic historical-range ceiling is a reasonable safety net regardless of which
branch produced the estimate, and should never bind for a genuinely well-evidenced wide
distribution (it does not bind, for example, on the F4 COVID/NFP diagnostic case, whose observed
training-window range already reflects the 2020 shock).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _robust_scale(x: np.ndarray) -> float:
    """MAD-based robust standard-deviation estimate: resistant to a small number of outliers in
    a way a raw sample std is not, at the cost of some efficiency on a truly clean sample (an
    acceptable trade here, since these estimators are only used precisely when the sample is
    too small for that efficiency loss to matter much anyway)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad * 1.4826)  # 1.4826 makes MAD a consistent estimator of std under normality


def _lag_scaling_exponent(values: pd.Series, max_lag: int) -> tuple[float | None, int, float | None]:
    """Fits dispersion(lag) ~ lag^alpha via log-log regression using only lags 1..max_lag that
    have at least 4 overlapping historical observations available. Returns
    (alpha, n_lags_used, dispersion_implied_at_lag_1); alpha and the implied dispersion are None
    if fewer than 3 lags had enough data to fit at all."""
    lags, disps = [], []
    for lag in range(1, max_lag + 1):
        d = values.diff(lag).dropna().to_numpy()
        if d.size < 4:
            continue
        s = _robust_scale(d)
        if np.isfinite(s) and s > 0:
            lags.append(lag)
            disps.append(s)
    if len(lags) < 3:
        return None, len(lags), None
    log_lag = np.log(np.asarray(lags, dtype=float))
    log_disp = np.log(np.asarray(disps, dtype=float))
    slope, intercept = np.polyfit(log_lag, log_disp, 1)
    alpha = float(np.clip(slope, 0.05, 1.0))
    base_from_fit = float(np.exp(intercept))  # implied dispersion at lag=1
    return alpha, len(lags), base_from_fit


def estimate_fallback_resid_std(values: pd.Series, horizon: int) -> float:
    """Scale-aware, small-sample-aware residual-scale estimate for the insufficient-data
    fallback (see module docstring). Uses only `values` up to the caller's as-of date — this
    function never receives or looks at anything beyond that, exactly like the formula it
    replaced.
    """
    v = pd.Series(values).dropna()
    n = len(v)
    if n < 6:
        return 1.0  # no usable history at all; the Monte Carlo layer's own priors take over

    diffs_1 = v.diff().dropna().to_numpy()
    if diffs_1.size == 0:
        return 1.0
    base_1 = _robust_scale(diffs_1)
    if not np.isfinite(base_1) or base_1 <= 0:
        base_1 = float(np.std(diffs_1)) or 1e-6

    max_lag = min(horizon, n - 2)
    if max_lag < 1:
        sigma = max(base_1, 1e-6)
    else:
        fit = _lag_scaling_exponent(v, max_lag)
        alpha, n_lags, base_from_fit = fit
        if alpha is not None:
            # Shrink the fitted exponent toward the random-walk prior (0.5) in proportion to how
            # many lags actually supported it -- a 3-4 lag fit is noisy and should barely move
            # the prior; a 12+ lag fit can be trusted close to fully.
            shrink = min(1.0, n_lags / 12.0)
            alpha = shrink * alpha + (1 - shrink) * 0.5
            base = 0.5 * base_1 + 0.5 * base_from_fit
            sigma = base * (horizon ** alpha)
        else:
            # Not enough overlapping lags to estimate any scaling behavior -- fall back to the
            # random-walk sqrt(h) assumption, but on the ROBUST 1-step scale so a single outlier
            # day cannot single-handedly blow up the estimate the way the old formula could.
            sigma = base_1 * np.sqrt(horizon)

    return apply_range_ceiling(sigma, v)


def apply_range_ceiling(sigma: float, values: pd.Series, range_ceiling_mult: float = 1.5) -> float:
    """Caps `sigma` at `range_ceiling_mult` times the full historical range `values` has shown.
    Generic and target-agnostic: it never binds for a series whose training window already
    reflects genuinely large historical moves (a real shock stays reflected, because the
    observed range itself is then large), and only bounds cases with no such evidence.
    """
    v = pd.Series(values).dropna()
    if len(v) < 2 or not np.isfinite(sigma):
        return max(float(sigma), 1e-6) if np.isfinite(sigma) else 1.0
    observed_range = float(v.max() - v.min())
    if observed_range > 0:
        sigma = min(float(sigma), range_ceiling_mult * observed_range)
    return max(float(sigma), 1e-6)
