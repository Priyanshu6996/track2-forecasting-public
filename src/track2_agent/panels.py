"""Leakage-safe panel loading.

The competition's own leakage rule (README, "Leakage rules" #1) is enforced upstream, at
publication time: no panel we are ever handed should contain a row with `date > asof`. We do
not rely on that alone — belt and braces, and it also protects the validation harness, which
*constructs* its own as-of cutoffs from full-history panels where the guarantee does not hold.

Every function that touches a date takes an explicit `asof` and filters to `date <= asof`
before returning anything. Nothing here reads a date column and forgets to filter it.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pandas as pd

_ASSET_COLS = ("asset", "asset_id")
_DATE_FMT = "%Y-%m-%d"


def _asset_col(df: pd.DataFrame) -> str | None:
    return next((c for c in _ASSET_COLS if c in df.columns), None)


def read_panels(panels_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Every parquet under `panels_dir`, keyed by filename stem.

    Tolerates the unit keeping panels one level up (the shipped exemplar does this) as well as
    the staged contract layout `<unit>/panels/*.parquet`.
    """
    panels_dir = pathlib.Path(panels_dir)
    found = sorted(panels_dir.glob("*.parquet"))
    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no .parquet found under {panels_dir} (or its parent)")
    return {p.stem: pd.read_parquet(p) for p in found}


def enforce_cutoff(panels: dict[str, pd.DataFrame], asof: str) -> dict[str, pd.DataFrame]:
    """Return copies of every panel with any `date > asof` row dropped.

    This is a defensive filter, not the primary leakage control (see module docstring) — but it
    means a bug upstream, or a validation cutoff constructed in-process, cannot leak a future row
    into a downstream feature by accident.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in panels.items():
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        out[name] = d[d["date"] <= asof].reset_index(drop=True)
    return out


@dataclass(frozen=True, slots=True)
class AssetSeries:
    asset: str
    panel: str
    dates: pd.DatetimeIndex
    values: pd.Series  # indexed by `dates`, float64, ascending, deduplicated


def all_asset_ids(panels: dict[str, pd.DataFrame]) -> list[str]:
    seen: set[str] = set()
    for df in panels.values():
        col = _asset_col(df)
        if col is not None:
            seen.update(str(v) for v in df[col].unique())
    return sorted(seen)


def asset_series(panels: dict[str, pd.DataFrame], asset: str, asof: str) -> AssetSeries:
    """History of one asset, `date <= asof`, from whichever panel carries it.

    Raises a clear `SystemExit` (not a KeyError) if the asset is absent, so a malformed
    `card.toml`/panel pairing fails loudly during a smoke run rather than crashing deep inside
    feature code with a stack trace that hides the actual cause.
    """
    for name, df in panels.items():
        col = _asset_col(df)
        if col is None:
            continue
        sub = df[df[col].astype(str) == asset]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["date"] = sub["date"].astype(str).str.slice(0, 10)
        sub = sub[sub["date"] <= asof].sort_values("date").drop_duplicates("date", keep="last")
        if sub.empty:
            continue
        idx = pd.to_datetime(sub["date"], format=_DATE_FMT, errors="coerce")
        vals = pd.Series(sub["value"].astype(float).to_numpy(), index=idx, name=asset).sort_index()
        return AssetSeries(asset=asset, panel=name, dates=vals.index, values=vals)
    raise SystemExit(
        f"asset {asset!r} not present in any panel at or before {asof}. "
        f"Panels carry: {', '.join(all_asset_ids(panels)) or '(no asset column found)'}"
    )


def diff_without_gaps(s: pd.Series) -> pd.Series:
    """First differences, dropping any difference that spans an unusually large gap.

    A transfer-style card can ship an asset as an early window plus a single later row, with
    years withheld in between. Differenced naively, that gap reads as one giant one-day move.
    The threshold adapts to the series' own typical spacing (10x the median step, floor 5 days),
    so daily and monthly panels are both handled correctly and a gapless panel is untouched.
    """
    d = s.diff()
    step = pd.Series(s.index).diff().dt.days
    step.index = s.index
    if step.notna().sum() == 0:
        return d
    thresh = max(float(step.median()) * 10.0, 5.0)
    return d.where(step <= thresh)


def log_return_without_gaps(s: pd.Series) -> pd.Series:
    """Log returns with the same gap guard as `diff_without_gaps`. Used for FX / factor levels
    that are always positive; falls back to a plain diff-ratio if any value is non-positive
    (e.g. a factor return series already expressed as decimal daily returns, which can be
    negative and near zero — log returns are undefined there, so callers choose per panel type).
    """
    import numpy as np

    if (s <= 0).any():
        raise ValueError("log_return_without_gaps requires a strictly positive series")
    lr = pd.Series(np.log(s.to_numpy()), index=s.index)
    return diff_without_gaps(lr)
