"""Conventional time-series features (lags, returns, vol, regime, cross-asset correlation).

Every feature here is computed with `pandas.Series.rolling`, which by construction only looks
backward from (and including) the row it lands on — so a feature value at date `t` never uses
`t' > t`. This is what makes `build_feature_frame` safe to call at every historical date when
assembling a walk-forward training set: the row at date t is built only from data <= t.

Feature families (mirrors the "NUMERICAL FORECASTING MODEL" section of the brief):
  lagged values, returns/changes, momentum, rolling volatility, rolling higher moments,
  cross-asset correlation, regime indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panels import diff_without_gaps

LAG_WINDOWS = (1, 5, 21, 63)
VOL_WINDOWS = (10, 21, 63)
MOM_WINDOWS = (21, 63, 126)


def asset_feature_frame(
    values: pd.Series,
    *,
    peer_diffs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix for one asset's level series, indexed the same as `values`.

    `peer_diffs`, if given, is a DataFrame of OTHER assets' first differences on a shared date
    index (already leakage-truncated by the caller) — used for the cross-asset correlation and
    co-movement features. Passing the asset's own diffs in `peer_diffs` is harmless; the
    self-correlation column is dropped internally by column-name match against `values.name`.
    """
    d = diff_without_gaps(values)
    feats: dict[str, pd.Series] = {}

    # --- lags / returns ---
    for w in LAG_WINDOWS:
        feats[f"lag_diff_{w}"] = values.diff(w)
        feats[f"lag_pct_{w}"] = values.pct_change(w)

    # --- momentum: cumulative change normalized by realized vol over the same window ---
    roll_std_21 = d.rolling(21, min_periods=10).std()
    for w in MOM_WINDOWS:
        cum = values.diff(w)
        feats[f"momentum_{w}"] = cum / (roll_std_21 * np.sqrt(w)).replace(0, np.nan)

    # --- rolling volatility / dispersion ---
    for w in VOL_WINDOWS:
        feats[f"vol_{w}"] = d.rolling(w, min_periods=max(5, w // 3)).std()
    feats["vol_ratio_short_long"] = feats["vol_10"] / feats["vol_63"].replace(0, np.nan)

    # --- rolling higher moments (regime / shock proneness) ---
    feats["skew_63"] = d.rolling(63, min_periods=21).skew()
    feats["kurt_63"] = d.rolling(63, min_periods=21).kurt()

    # --- regime indicator: current vol z-scored against its own trailing history ---
    long_mean = feats["vol_21"].rolling(252, min_periods=60).mean()
    long_std = feats["vol_21"].rolling(252, min_periods=60).std().replace(0, np.nan)
    feats["vol_regime_z"] = (feats["vol_21"] - long_mean) / long_std

    # --- level-relative features (mean reversion proxies) ---
    ma_63 = values.rolling(63, min_periods=21).mean()
    ma_252 = values.rolling(252, min_periods=60).mean()
    feats["dev_from_ma63"] = (values - ma_63) / (feats["vol_63"] * np.sqrt(63)).replace(0, np.nan)
    feats["ma_slope"] = (ma_63 - ma_252) / (feats["vol_63"] * np.sqrt(63)).replace(0, np.nan)

    # --- cross-asset co-movement ---
    if peer_diffs is not None and not peer_diffs.empty:
        peers = peer_diffs.drop(columns=[c for c in peer_diffs.columns if c == values.name],
                                 errors="ignore")
        if not peers.empty:
            roll_corr = peers.rolling(63, min_periods=21).corr(d)
            # pandas rolling().corr() can return a MultiIndex/union-index shape when peer
            # series have different histories. Re-align explicitly to the target series before
            # joining; assigning the union-length result to values.index caused the F3 BOJ/UST
            # channel length-mismatch failure.
            roll_corr = roll_corr.reindex(values.index)
            feats["mean_peer_corr_63"] = roll_corr.mean(axis=1)
            feats["max_abs_peer_corr_63"] = roll_corr.abs().max(axis=1)

    frame = pd.DataFrame(feats)
    frame = frame.reindex(values.index)
    return frame


def target_return(values: pd.Series, horizon_days: int) -> pd.Series:
    """Forward change over `horizon_days`, aligned to the ANCHOR date (leaks the future on
    purpose — this is the training target, used only at fit time on rows whose anchor date is
    strictly before the current as-of; see numeric_model.py for how the leakage boundary is
    enforced when building the walk-forward training set).
    """
    return values.shift(-horizon_days) - values
