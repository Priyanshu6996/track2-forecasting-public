"""Agenthon 2026 Track 2 — Reasoning-Augmented Time-Series Forecasting.

Numeric panels + LLM text reasoning -> calibrated joint Monte Carlo draws.

Package layout (import order = pipeline order):

    panels.py        leakage-safe panel loading / asset series extraction
    features.py       conventional time-series features (lags, returns, vol, regime)
    text_signals.py     LLM-assisted structured signal extraction from the text corpus
    numeric_model.py    numeric forecasting model (rw_drift/ewma/ar1/ridge/gbm, walk-forward selected)
    fusion.py          combine numeric forecast with text-derived signals
    montecarlo.py       joint correlated Monte Carlo sampling with fat tails
    calibration.py       post-hoc spread / tail calibration
    rationale.py        forecast_rationale.md generation
    metrics.py          local CRPS / variogram / pinball for internal validation only
    validation.py        walk-forward backtest harness over the public practice units
    cli.py             the `forecast` verb (SUBMISSION_CLI.md contract)

NOTE: an earlier revision also had `signatures.py` (rolling-window path-signature features) and
a `numerical_plus_sig` config preset. Walk-forward validation across the public practice units
showed signatures never net-improving the composite score (11.2993 numerical-only vs. 11.3755
with signatures, and `full` did not recover the degradation) and made it worse in aggregate, so
that module and every piece of infrastructure that existed solely to support it were removed —
see config.py's module docstring and docs/VALIDATION.md's "Update 3" for the evidence.
"""

__version__ = "1.0.0"
