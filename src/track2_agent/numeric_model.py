"""Numeric forecasting model: features -> (point forecast, residual scale) per (asset, horizon).

## Candidate set

Five candidates are evaluated by expanding-window walk-forward validation (see `ModelConfig`
for the full rationale): a driftless/drift random walk floor, an exponentially-weighted mean
(`ewma`, a fast-adapting drift distinct from the flat-window `rw_drift`), a one-feature
autoregressive-style regression (`ar1`, the target regressed on the asset's own most recent
realized change), regularized linear regression (`ridge`) and shallow gradient-boosted trees
(`gbm`) on the full engineered feature set (conventional lag/return/vol/regime/cross-asset
features from `features.py` — path-signature features were tried in an earlier revision and
removed after validation showed no net benefit; see `docs/VALIDATION.md`'s "Update 3"). This is
broader than a bare two-model-vs-baseline
comparison, per the competition-alignment review, while still staying well short of "try
everything": each candidate targets a genuinely different inductive bias (no drift, adaptive
drift, minimal autocorrelation, linear multi-feature, nonlinear multi-feature) rather than being
included for its own sake. Two further families discussed in the review — factor-based models
and pretrained time-series foundation models — are NOT included here: the former needs a
cross-sectional asset universe wider than most single practice units ship, and the latter are
not available in this sandboxed environment (no network access to fetch weights). Both are
documented as untested, not silently assumed to help, in docs/VALIDATION.md.

## Selection metric

The walk-forward comparison no longer ranks candidates by mean absolute error. A model-selection
decision here is, by construction, a single (asset, horizon) cell, so each candidate's pooled
out-of-fold (prediction, actual) pairs are scored with the SAME single-cell CRPS/tail combination
`metrics.composite_score` applies to a single cell (0.714 marginal-CRPS-proxy + 0.286
tail-pinball-proxy) — approximating CRPS as the average pinball loss over a quantile grid built
from the pooled out-of-fold residual distribution (a standard identity: CRPS is the integral of
pinball loss over all quantiles in [0,1]; a finite grid weighted toward both the center and the
competition's own tail levels approximates it directly instead of optimizing a metric, MAE, that
does not track it). MAE is still computed and reported per candidate (for the rationale / human
review), but it is never what decides the winner. This directly addresses the "MAE picks a
different model than CRPS/tail would" finding this review flagged as the highest-priority fix.

Everything here is fit ONLY on rows whose feature date is <= asof AND whose target (feature
date + horizon) is <= asof too — i.e., the training set never includes a row whose target would
require data beyond the as-of date. See `_build_training_rows`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .config import ModelConfig
from .features import asset_feature_frame, target_return
from .uncertainty import apply_range_ceiling, estimate_fallback_resid_std

# Quantile grid for the CRPS proxy: a broad central sweep plus the competition's own tail levels
# (0.01/0.05/0.95/0.99), so the same levels that carry 20% of the OFFICIAL score also get direct
# weight in which numeric candidate is selected internally.
_CRPS_GRID = tuple(round(q, 2) for q in np.linspace(0.05, 0.95, 19))
_TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)
_SINGLE_CELL_MARGINAL_WEIGHT = 0.714  # matches metrics.composite_score(single_cell=True)
_SINGLE_CELL_TAIL_WEIGHT = 0.286


@dataclass(frozen=True, slots=True)
class NumericForecast:
    point: float          # forecast CHANGE (not level) over the horizon, i.e. E[level_{t+h} - level_t]
    resid_std: float       # residual standard deviation of that change, from validation folds
    model_used: str
    n_train_rows: int
    cv_mae_by_candidate: dict[str, float]
    cv_score_by_candidate: dict[str, float]  # CRPS/tail-proxy composite; lower is better; this decides


def _build_training_rows(
    feats: pd.DataFrame, values: pd.Series, horizon: int, asof: str,
) -> tuple[pd.DataFrame, pd.Series]:
    # `target_return` is defined as values.shift(-horizon) - values, i.e. the change from the
    # feature row's own date out to horizon business days later. A target is usable for TRAINING
    # only when that future row already exists in `values` (so it is historical, not something
    # requiring data past the current as-of) -- `dropna` on "__y__" enforces exactly that, since
    # a future row past the end of `values` is NaN by construction of `.shift(-horizon)`.
    y = target_return(values, horizon)
    df = feats.copy()
    df["__y__"] = y
    df = df.loc[df.index <= pd.Timestamp(asof)]
    df = df.dropna(subset=["__y__"])
    df = df.dropna(axis=0, how="any")
    X = df.drop(columns=["__y__"])
    y_clean = df["__y__"]
    return X, y_clean


def _pinball(pred: np.ndarray, actual: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _crps_tail_proxy(oof_pred: np.ndarray, oof_actual: np.ndarray) -> tuple[float, float]:
    """Pooled-out-of-fold CRPS-proxy and tail-proxy for one candidate.

    Builds an empirical residual sample (actual - pred, pooled across every out-of-fold test
    point for this candidate), so a quantile forecast at point i is `pred_i + quantile(resid,
    q)`. This is a standard construction: it is exactly what the model's own point forecast plus
    its own out-of-fold error distribution implies about its predictive quantiles, without
    assuming Gaussianity. Averaging pinball loss over `_CRPS_GRID` approximates CRPS (the
    integral of pinball loss over the full quantile range); averaging it over the competition's
    own `_TAIL_LEVELS` gives a tail-focused companion metric with the same units.
    """
    resid = oof_actual - oof_pred
    if resid.size < 5:
        return float("inf"), float("inf")
    crps_losses = [_pinball(oof_pred + np.quantile(resid, q), oof_actual, q) for q in _CRPS_GRID]
    tail_losses = [_pinball(oof_pred + np.quantile(resid, q), oof_actual, q) for q in _TAIL_LEVELS]
    return float(np.mean(crps_losses)), float(np.mean(tail_losses))


def _walk_forward_oof(X: pd.DataFrame, y: pd.Series, fit_predict, n_folds: int = 3):
    """Returns (pooled_pred, pooled_actual, mae) over expanding-window walk-forward folds, or
    (None, None, inf) if there isn't enough history / a candidate fails a fold."""
    n = len(X)
    if n < 40:
        return None, None, float("inf")
    fold_size = max(10, n // (n_folds + 2))
    preds, actuals = [], []
    start = fold_size * 2
    while start + fold_size <= n:
        X_tr, y_tr = X.iloc[:start], y.iloc[:start]
        X_te, y_te = X.iloc[start:start + fold_size], y.iloc[start:start + fold_size]
        try:
            pred = fit_predict(X_tr, y_tr, X_te)
        except Exception:  # noqa: BLE001 - a candidate that fails a fold just scores inf
            return None, None, float("inf")
        preds.append(np.asarray(pred, dtype=float))
        actuals.append(y_te.to_numpy(dtype=float))
        start += fold_size
    if not preds:
        return None, None, float("inf")
    pooled_pred = np.concatenate(preds)
    pooled_actual = np.concatenate(actuals)
    mae = float(np.mean(np.abs(pooled_pred - pooled_actual)))
    return pooled_pred, pooled_actual, mae


def _fit_ridge(X_tr, y_tr, X_te, alpha: float):
    mu, sd = X_tr.mean(), X_tr.std().replace(0, 1.0)
    model = Ridge(alpha=alpha)
    model.fit((X_tr - mu) / sd, y_tr)
    return model.predict((X_te - mu) / sd)


def _fit_gbm(X_tr, y_tr, X_te, cfg: ModelConfig):
    model = GradientBoostingRegressor(
        max_depth=cfg.gbm_max_depth,
        n_estimators=cfg.gbm_n_estimators,
        learning_rate=cfg.gbm_learning_rate,
        random_state=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _fit_rw_drift(X_tr, y_tr, X_te):
    drift = float(y_tr.mean())
    return np.full(len(X_te), drift)


def _fit_ewma(X_tr, y_tr, X_te, span: float):
    """Exponentially-weighted mean of the training targets, recency-weighted rather than a flat
    average over the whole training window (`rw_drift`). Distinct failure mode from `rw_drift`:
    it adapts faster to a recent drift change, at the cost of more noise when there isn't one.
    """
    weight = float(y_tr.ewm(span=max(span, 2.0)).mean().iloc[-1]) if len(y_tr) else 0.0
    return np.full(len(X_te), weight)


def _fit_ar1(X_tr, y_tr, X_te):
    """Minimal one-feature autoregressive-style model: y (the horizon-ahead change) regressed
    on the asset's own most recent realized one-step change (`lag_diff_1`, always present in
    `asset_feature_frame`'s output). A deliberately small model distinct from `ridge`'s full
    feature set — a plain, interpretable autocorrelation check rather than a multi-feature fit.
    """
    col = "lag_diff_1"
    if col not in X_tr.columns:
        return np.full(len(X_te), float(y_tr.mean()))
    x_tr = X_tr[col].to_numpy(dtype=float).reshape(-1, 1)
    x_te = X_te[col].to_numpy(dtype=float).reshape(-1, 1)
    mu, sd = x_tr.mean(), x_tr.std() or 1.0
    model = Ridge(alpha=1.0)
    model.fit((x_tr - mu) / sd, y_tr)
    return model.predict((x_te - mu) / sd)


_FIT_FUNCS = {
    "rw_drift": lambda X_tr, y_tr, X_te, cfg: _fit_rw_drift(X_tr, y_tr, X_te),
    "ewma": lambda X_tr, y_tr, X_te, cfg: _fit_ewma(X_tr, y_tr, X_te, cfg.ewma_span),
    "ar1": lambda X_tr, y_tr, X_te, cfg: _fit_ar1(X_tr, y_tr, X_te),
    "ridge": lambda X_tr, y_tr, X_te, cfg: _fit_ridge(X_tr, y_tr, X_te, cfg.ridge_alpha),
    "gbm": lambda X_tr, y_tr, X_te, cfg: _fit_gbm(X_tr, y_tr, X_te, cfg),
}


def _refit_full(best: str, X: pd.DataFrame, y: pd.Series, cfg: ModelConfig) -> float:
    """Refit the winning candidate on ALL usable rows up to asof and return its point forecast
    for the final (most recent) row."""
    if best == "ridge":
        mu, sd = X.mean(), X.std().replace(0, 1.0)
        model = Ridge(alpha=cfg.ridge_alpha)
        model.fit((X - mu) / sd, y)
        return float(np.asarray(model.predict(((X.iloc[[-1]] - mu) / sd)), dtype=float).reshape(-1)[0])
    if best == "gbm":
        model = GradientBoostingRegressor(
            max_depth=cfg.gbm_max_depth, n_estimators=cfg.gbm_n_estimators,
            learning_rate=cfg.gbm_learning_rate, random_state=0,
        )
        model.fit(X, y)
        return float(np.asarray(model.predict(X.iloc[[-1]]), dtype=float).reshape(-1)[0])
    if best == "ewma":
        return float(y.ewm(span=max(cfg.ewma_span, 2.0)).mean().iloc[-1])
    if best == "ar1":
        col = "lag_diff_1"
        if col not in X.columns:
            return float(y.mean())
        x = X[col].to_numpy(dtype=float).reshape(-1, 1)
        mu, sd = x.mean(), x.std() or 1.0
        model = Ridge(alpha=1.0)
        model.fit((x - mu) / sd, y)
        return float(np.asarray(model.predict(((X[[col]].iloc[[-1]].to_numpy(dtype=float) - mu) / sd)), dtype=float).reshape(-1)[0])
    return float(y.mean())  # rw_drift


def fit_numeric_forecast(
    feats: pd.DataFrame,
    values: pd.Series,
    horizon: int,
    asof: str,
    cfg: ModelConfig,
) -> NumericForecast:
    X, y = _build_training_rows(feats, values, horizon, asof)

    if len(X) < cfg.min_train_rows:
        # Scale-aware, small-sample-aware fallback (see uncertainty.py's module docstring for
        # why this replaced a flat `values.diff().std() * sqrt(horizon)`, which produced
        # economically nonsensical distributions for short-history, long-horizon macro targets).
        resid_std = estimate_fallback_resid_std(values, horizon) if len(values) > 5 else 1.0
        return NumericForecast(
            point=0.0, resid_std=max(resid_std, 1e-6), model_used="rw_drift_insufficient_data",
            n_train_rows=len(X), cv_mae_by_candidate={}, cv_score_by_candidate={},
        )

    mae_scores: dict[str, float] = {}
    composite_scores: dict[str, float] = {}
    resid_pools: dict[str, np.ndarray] = {}
    for name in cfg.candidates:
        fit_fn = _FIT_FUNCS.get(name)
        if fit_fn is None:
            continue
        pooled_pred, pooled_actual, mae = _walk_forward_oof(
            X, y, lambda a, b, c, _fn=fit_fn: _fn(a, b, c, cfg),
        )
        mae_scores[name] = mae
        if pooled_pred is None:
            composite_scores[name] = float("inf")
            continue
        crps_proxy, tail_proxy = _crps_tail_proxy(pooled_pred, pooled_actual)
        composite_scores[name] = (
            _SINGLE_CELL_MARGINAL_WEIGHT * crps_proxy + _SINGLE_CELL_TAIL_WEIGHT * tail_proxy
        )
        resid_pools[name] = pooled_actual - pooled_pred

    finite = {k: v for k, v in composite_scores.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else "rw_drift"

    point = _refit_full(best, X, y, cfg)

    # Residual scale: prefer the pooled out-of-fold residual std for the WINNING candidate (an
    # honest, already-computed out-of-sample dispersion estimate) over any in-sample residual,
    # which would understate uncertainty (especially for the GBM candidate).
    if best in resid_pools and resid_pools[best].size >= 5:
        resid_std = float(np.std(resid_pools[best]))
    else:
        resid_std = float(y.std())
    if not np.isfinite(resid_std) or resid_std <= 0:
        resid_std = max(float(y.std()), 1e-6)
    # Generic sanity ceiling (see uncertainty.py): never binds for a genuinely well-evidenced
    # wide distribution (e.g. a training window that itself spans a real historical shock), only
    # for cases with no such evidence -- applied here too, not just in the small-sample fallback
    # above, as a light-touch safety net against any other pathological blow-up.
    resid_std = apply_range_ceiling(resid_std, values)

    return NumericForecast(
        point=point, resid_std=float(resid_std), model_used=best,
        n_train_rows=len(X),
        cv_mae_by_candidate={k: v for k, v in mae_scores.items() if np.isfinite(v)},
        cv_score_by_candidate={k: v for k, v in composite_scores.items() if np.isfinite(v)},
    )
