"""Joint Monte Carlo sampling.

Produces `n_draws` scenarios over the full (asset x horizon) grid such that:

  1. **Cross-asset dependence** is preserved: the shared shock in draw d is drawn from the
     assets' empirical (shrinkage) correlation, not independently per asset. This is what the
     competition's 30%-weight joint variogram term checks for directly.
  2. **Cross-horizon coherence per asset** is preserved: each asset's horizons are built as
     partial realizations of the SAME underlying shock (shorter horizons share more of the
     longer horizon's draw, via `rho_h = h_min / h`), so draw d tells a single coherent story
     for that asset across all its horizons rather than an independent one per horizon.
  3. **Fat tails** come from two independent mechanisms, matching how real macro/rates/FX moves
     actually fatten: (a) a shared multivariate Student-t scaling factor (same chi-square
     realization applied to every asset/horizon within a draw, so tails co-move), and (b) a
     low-probability shared "shock" mixture component whose trigger probability and magnitude
     scale with the text-derived `tail_widen` — capturing "text flags a shock regime" (F4) as
     literally raising the odds of drawing from the shock component, not just widening every draw
     uniformly.
  4. **Skew** is applied per asset/horizon via a sinh-arcsinh transform of the (already
     correlated, already fat-tailed) standardized shock, then the transform's output is
     re-standardized numerically (its own empirical mean/std under this draw set) before scaling,
     so the requested `scale` remains the actual output dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MonteCarloConfig
from .fusion import FusedForecast


@dataclass(frozen=True, slots=True)
class Cell:
    asset: str
    horizon: int
    fused: FusedForecast


def shrinkage_correlation(diffs: pd.DataFrame, shrinkage: float) -> np.ndarray:
    """Ledoit-Wolf-style linear shrinkage of the empirical correlation toward the identity.
    Cheap, robust with few observations relative to the number of assets, and guaranteed PSD
    after the eigenvalue clip below (belt-and-braces against numerical indefiniteness).
    """
    corr = diffs.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    n = corr.shape[0]
    corr = (1 - shrinkage) * corr + shrinkage * np.eye(n)
    w, v = np.linalg.eigh(corr)
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _sinh_arcsinh(z: np.ndarray, skew: float) -> np.ndarray:
    """Jones & Pewsey (2009) sinh-arcsinh transform, tail parameter fixed at 1 (tail fatness is
    handled separately by the Student-t / shock-mixture machinery, so this transform contributes
    skew only, not kurtosis -- keeping the two effects independently controllable).
    """
    if abs(skew) < 1e-9:
        return z
    return np.sinh(np.arcsinh(z) + skew)


def draw_joint_forecast(
    cells: list[Cell],
    asset_order: list[str],
    horizons: list[int],
    corr: np.ndarray,          # (n_assets, n_assets) correlation of daily changes
    cfg: MonteCarloConfig,
    distribution_mode: str = "current",
    cross_horizon_mode: str = "legacy",
) -> np.ndarray:
    """Returns an array of shape (n_draws, n_assets, n_horizons) of forecast CHANGE draws
    (each `fused.mean` is itself a forecast of the change over the horizon, not a level -- the
    caller adds each asset's own anchor level on top of this array; see cli.py).
    """
    rng = np.random.default_rng(cfg.seed)
    n_draws = cfg.n_draws
    n_assets = len(asset_order)
    n_hz = len(horizons)
    h_min = min(horizons)

    chol = np.linalg.cholesky(corr)

    if distribution_mode not in {"current", "gaussian_oof", "student_t_oof"}:
        raise ValueError(
            f"unknown distribution_mode {distribution_mode!r}; expected current, gaussian_oof, or student_t_oof"
        )
    if cross_horizon_mode not in {"legacy", "preserve_asset_corr"}:
        raise ValueError(
            f"unknown cross_horizon_mode {cross_horizon_mode!r}; expected legacy or preserve_asset_corr"
        )

    # Current production mode retains the existing Student-t + shared-shock machinery.
    # The OOF modes deliberately isolate marginal distribution shape: they use the same
    # point forecasts/scales but remove the production shock/skew transforms so the ablation
    # answers whether the distributional construction itself is responsible for error.
    if distribution_mode == "gaussian_oof":
        chi_factor = np.ones(n_draws, dtype=float)
    else:
        df = cfg.student_t_df
        chi_factor = np.sqrt(df / rng.chisquare(df, size=n_draws))

    # Shared shock-mixture trigger, probability scaled by the max text-derived tail_widen across
    # cells (a stronger "text flags a shock" signal raises the odds of the fat regime for
    # everyone in that draw, not just the flagged asset -- macro shocks are rarely one-asset
    # events).
    max_tail_widen = max((c.fused.tail_widen for c in cells), default=1.0)
    if distribution_mode == "current":
        shock_p = float(np.clip(0.02 * (max_tail_widen - 1.0) * 3.0, 0.0, 0.20))
        shock_trigger = rng.random(n_draws) < shock_p
        shock_mult = np.where(shock_trigger, rng.uniform(2.0, 4.0, size=n_draws), 1.0)
    else:
        shock_mult = np.ones(n_draws, dtype=float)

    lookup = {(c.asset, c.horizon): c.fused for c in cells}
    out = np.empty((n_draws, n_assets, n_hz), dtype=float)

    # Generate one shared multivariate shock matrix per draw. Generating this inside the
    # asset loop silently destroys cross-asset dependence, even when `corr` is correct.
    z_assets = rng.standard_normal((n_draws, n_assets)) @ chol.T

    # For longer horizons, the legacy mode adds independent per-asset noise; that reduces
    # cross-asset correlation by rho_h. The candidate mode instead uses fresh multivariate
    # noise with the SAME cross-asset correlation at each horizon. Combining two independent
    # correlated vectors with sqrt(rho)/sqrt(1-rho) coefficients preserves the marginal
    # cross-asset correlation while keeping cross-horizon coherence through the shared factor.
    if cross_horizon_mode == "preserve_asset_corr":
        eps_assets_by_horizon = [
            rng.standard_normal((n_draws, n_assets)) @ chol.T for _ in horizons
        ]
    else:
        eps_assets_by_horizon = None

    for ai, asset in enumerate(asset_order):
        z_asset = z_assets[:, ai]
        for hi, h in enumerate(horizons):
            fused = lookup.get((asset, h))
            if fused is None:
                out[:, ai, hi] = np.nan
                continue
            rho_h = min(1.0, h_min / h)
            if eps_assets_by_horizon is None:
                eps = rng.standard_normal(n_draws)
            else:
                eps = eps_assets_by_horizon[hi][:, ai]
            z = np.sqrt(rho_h) * z_asset + np.sqrt(max(0.0, 1 - rho_h)) * eps
            z = z * chi_factor * shock_mult ** 0.5  # sqrt: shock adds vol, not mean, on average
            if distribution_mode == "current":
                z = _sinh_arcsinh(z, fused.skew)
            # Re-standardize the transform's realized dispersion so `scale` is the actual output
            # std, then apply the extra tail-only widening as a further pure scale bump.
            z = (z - z.mean()) / (z.std() + 1e-12)
            # `tail_widen` (per-cell, from fusion.py) applies as a direct scale multiplier on top
            # of the shared shock-mixture mechanism above -- the shared mixture makes tail events
            # co-move across assets/horizons; this term additionally widens THIS cell's own
            # dispersion when its own text signal (risk/regime flags) called for it.
            tail_multiplier = fused.tail_widen if distribution_mode == "current" else 1.0
            value = fused.mean + fused.scale * tail_multiplier * z
            out[:, ai, hi] = value
    return out
