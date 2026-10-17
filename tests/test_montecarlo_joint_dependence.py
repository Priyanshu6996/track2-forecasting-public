"""Regression tests for joint and cross-horizon Monte Carlo dependence."""
import numpy as np

from track2_agent.config import MonteCarloConfig
from track2_agent.fusion import FusedForecast
from track2_agent.montecarlo import Cell, draw_joint_forecast


def _cells(assets, horizons):
    fused = FusedForecast(mean=0.0, scale=1.0, tail_widen=1.0, skew=0.0,
                          text_contribution_sigma=0.0)
    return [Cell(asset=a, horizon=h, fused=fused) for a in assets for h in horizons]


def test_shared_asset_correlation_survives_sampling_for_single_horizon():
    target_corr = 0.82
    corr = np.array([[1.0, target_corr], [target_corr, 1.0]])
    draws = draw_joint_forecast(
        _cells(["A", "B"], [1]), ["A", "B"], [1], corr,
        MonteCarloConfig(n_draws=12_000, shrinkage=0.0, seed=17),
        distribution_mode="gaussian_oof",
    )
    observed = np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1]
    assert abs(observed - target_corr) < 0.04


def test_asset_correlation_is_preserved_at_each_horizon_without_losing_horizon_coherence():
    target_corr = 0.82
    corr = np.array([[1.0, target_corr], [target_corr, 1.0]])
    horizons = [21, 63]
    draws = draw_joint_forecast(
        _cells(["A", "B"], horizons), ["A", "B"], horizons, corr,
        MonteCarloConfig(n_draws=12_000, shrinkage=0.0, seed=23),
        distribution_mode="gaussian_oof",
        cross_horizon_mode="preserve_asset_corr",
    )
    corr_short = np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1]
    corr_long = np.corrcoef(draws[:, 0, 1], draws[:, 1, 1])[0, 1]
    corr_across_horizons = np.corrcoef(draws[:, 0, 0], draws[:, 0, 1])[0, 1]
    expected_cross_horizon = np.sqrt(horizons[0] / horizons[1])
    assert abs(corr_short - target_corr) < 0.04
    assert abs(corr_long - target_corr) < 0.04
    assert abs(corr_across_horizons - expected_cross_horizon) < 0.04
