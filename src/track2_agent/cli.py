"""Track-2 submission CLI: the `forecast` verb.

    forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD \
             --out /output/forecast.parquet

Contract (SUBMISSION_CLI.md): reads --panels (parquet, date <= asof already enforced upstream
at publication time, defensively re-enforced here), --text (documents timestamp <= asof,
enforced by gate g2 and defensively re-checked here), writes forecast.parquet + forecast_meta.json
+ forecast_rationale.md next to --out. Exits 0 on success.

Pipeline: panels -> features -> numeric model -> text signals (LLM, house endpoint only) ->
fusion -> calibration -> joint Monte Carlo -> forecast.parquet.

NOTE: an earlier revision computed rolling-window path-signature features here and, later, a
per-cell walk-forward gate deciding whether to use them. Two rounds of validation evidence showed
them never net-improving the composite score (see `config.py`'s module docstring and
`docs/VALIDATION.md`'s "Update 3"), so that entire code path — the signature feature join, the
per-cell gate, and the `signature_gate` field this used to write to forecast_meta.json — was
removed rather than kept as inert, always-off infrastructure.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd

from .calibration import calibrate_scale, load_calibration_table
from .config import AgentConfig
from .features import asset_feature_frame
from .fusion import fuse
from .montecarlo import Cell, draw_joint_forecast, shrinkage_correlation
from .numeric_model import fit_numeric_forecast
from .panels import all_asset_ids, asset_series, diff_without_gaps, enforce_cutoff, read_panels
from .rationale import build_rationale
from .text_calibration import load_text_calibration
from .text_signals import extract_text_signals


def _card_family(card: dict) -> str:
    """Card family (F1..F4) from `metadata.category`, e.g. "T2-F3" -> "F3". Empty string if the
    card carries no recognizable family tag, in which case `AgentConfig.from_name_and_family`
    falls through to the un-overridden base config (see config.py's FAMILY_OVERRIDES docstring).
    """
    category = str(card.get("metadata", {}).get("category", ""))
    return category.upper().removeprefix("T2-")

DEFAULT_DRAWS = 500
MIN_DRAWS_FLOOR = 200
MAX_DRAWS_CEILING = 20_000
_RATIONALE_NAME = "forecast_rationale.md"
_META_NAME = "forecast_meta.json"


def _resolve_seed(cli_seed: int | None) -> int:
    """Return the explicit CLI seed, otherwise the harness seed, otherwise the pinned default.

    The Final harness sets ``QFBENCH_SEED`` for reproducibility checks.  Respect it when a
    caller has not explicitly supplied ``--seed``; a malformed environment value is ignored so
    a local run remains usable rather than failing before it can write a forecast.
    """
    if cli_seed is not None:
        return cli_seed
    raw = os.environ.get("QFBENCH_SEED")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return AgentConfig.from_name("full").mc.seed


def _find_card(panels_dir: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--card {explicit} does not exist")
        return explicit
    for cand in (panels_dir / "card.toml", panels_dir.parent / "card.toml"):
        if cand.exists():
            return cand
    raise SystemExit(
        f"card.toml not found in {panels_dir} or {panels_dir.parent}; pass --card explicitly"
    )


def _aligned_channel_matrix(
    panels: dict[str, pd.DataFrame], assets: list[str], asof: str,
) -> pd.DataFrame:
    """A common-date-indexed levels matrix over the target assets, used for cross-asset
    peer-correlation features (`features.py`) and for the joint Monte Carlo correlation
    structure (`montecarlo.shrinkage_correlation`). Assets that share a panel/frequency align
    cleanly; assets on different frequencies simply produce more NaNs, which downstream rolling
    operations already tolerate (they require a full, non-NaN window).
    """
    series = {a: asset_series(panels, a, asof).values for a in assets}
    return pd.DataFrame(series).sort_index()


def _to_feature_level(raw: pd.Series, target_type: str) -> pd.Series:
    """Card `target_type` decides what "the level series" even means (verified against the
    actual shipped panels, not assumed from the card alone -- e.g. `factors_daily` MOM really
    is a raw daily return series, mean ~0.0001, std ~0.011, not a price level):

    - "level": the panel value already IS a level (a yield, an FX quote, a macro index). Used
      as-is; the numeric model's target is the CHANGE in this level over the horizon, and the
      output `value` is anchor + that change.
    - "log_return": the panel value is a DAILY return. `cumsum` turns it into a synthetic
      cumulative-return index so every existing feature/model machinery (all written in terms of
      "change in a level") applies unmodified; the numeric model's target then becomes the
      CHANGE in this cumulative index over the horizon, which is exactly the cumulative log
      return over that horizon -- and critically the output `value` is that change ALONE, with
      no anchor added back (see cli.run / validation._forecast_one), because
      `value_unit = "cumulative_log_return"` means "return since as-of", not "index level".
    """
    if target_type == "log_return":
        return raw.cumsum()
    return raw


ADD_ANCHOR_BY_TARGET_TYPE = {"level": True, "log_return": False}



def run(args: argparse.Namespace) -> int:
    panels_dir = args.panels
    text_dir = args.text
    asof = args.asof

    card_path = _find_card(panels_dir, args.card)
    card = tomllib.loads(card_path.read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    unit_id = card["task"]["id"]

    card_floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    floor = max(card_floor, MIN_DRAWS_FLOOR)
    n_draws = max(args.n_draws or max(floor, DEFAULT_DRAWS), floor)
    if n_draws > MAX_DRAWS_CEILING:
        raise SystemExit(f"--n-draws {n_draws} exceeds the {MAX_DRAWS_CEILING} contract ceiling")

    target_type = str(tgt.get("target_type", "level"))
    add_anchor = ADD_ANCHOR_BY_TARGET_TYPE.get(target_type, True)

    family = _card_family(card)
    cfg = AgentConfig.from_name_and_family(args.config, family)

    raw_panels = read_panels(panels_dir)
    panels = enforce_cutoff(raw_panels, asof)  # defensive re-filter; see panels.py docstring

    available = set(all_asset_ids(panels))
    missing = [a for a in assets if a not in available]
    if missing:
        raise SystemExit(
            f"card targets {missing} not present in any panel at/before {asof}. "
            f"Panels carry: {', '.join(sorted(available))}"
        )

    channels_raw = _aligned_channel_matrix(panels, assets, asof)
    channels = channels_raw.apply(lambda s: _to_feature_level(s, target_type))
    diffs = channels.apply(diff_without_gaps).dropna(how="all")

    text_signals = extract_text_signals(text_dir, asof, assets, cfg.text) if cfg.use_text else \
        extract_text_signals(text_dir, asof, assets, type(cfg.text)(enabled=False))

    anchors: dict[str, float] = {}
    numeric_results: dict[tuple[str, int], object] = {}
    fused_results: dict[tuple[str, int], object] = {}
    calib_table = load_calibration_table(args.calibration)
    text_calib = load_text_calibration(args.calibration)  # None -> fusion.py's own conservative default

    for asset in assets:
        series = asset_series(panels, asset, asof)
        level_values = _to_feature_level(series.values, target_type)
        anchors[asset] = float(level_values.iloc[-1]) if add_anchor else 0.0

        feats = asset_feature_frame(level_values, peer_diffs=diffs)

        for h in horizons:
            nf = fit_numeric_forecast(feats, level_values, h, asof, cfg.model)
            numeric_results[(asset, h)] = nf
            fz = fuse(nf, text_signals, asset, cfg.use_text, cfg.fusion, text_calib)
            scale = calibrate_scale(asset, anchors[asset], fz.scale, calib_table)
            fz = type(fz)(mean=fz.mean, scale=scale, tail_widen=fz.tail_widen, skew=fz.skew,
                          text_contribution_sigma=fz.text_contribution_sigma)
            fused_results[(asset, h)] = fz

    corr = shrinkage_correlation(diffs.dropna(how="any"), cfg.mc.shrinkage) if len(assets) > 1 \
        else np.array([[1.0]])

    cells = [Cell(asset=a, horizon=h, fused=fused_results[(a, h)]) for a in assets for h in horizons]
    mc_cfg = type(cfg.mc)(n_draws=n_draws, student_t_df=cfg.mc.student_t_df,
                           shrinkage=cfg.mc.shrinkage, seed=_resolve_seed(args.seed))
    # Phase-3 winner: Gaussian OOF marginals + no correlation shrinkage.
    # Keep the same joint Monte Carlo machinery so cross-asset/horizon dependence is preserved,
    # but remove the Student-t/shock/skew construction that hurt the public ablation.
    distribution_mode = "gaussian_oof"
    cross_horizon_mode = "preserve_asset_corr" if family == "F3" else "legacy"
    change_draws = draw_joint_forecast(
        cells, assets, horizons, corr, mc_cfg, distribution_mode=distribution_mode,
        # F3 is scored on joint dependence.  A recorded 66-fold, second-seed-confirmed
        # pseudo-as-of ablation found this preserves the empirical asset correlation at longer
        # horizons without changing the marginal sampler.  Other families retain the validated
        # legacy behavior until their own time-ordered evidence supports a change.
        cross_horizon_mode=cross_horizon_mode,
    )

    level_draws = change_draws.copy()
    for ai, a in enumerate(assets):
        level_draws[:, ai, :] += anchors[a]

    out_path = args.out
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"draw": d, "asset": a, "horizon": h, "value": float(level_draws[d, ai, hi])}
        for d in range(n_draws)
        for ai, a in enumerate(assets)
        for hi, h in enumerate(horizons)
    ]
    pd.DataFrame(rows).to_parquet(out_path, index=False)

    (out_dir / _META_NAME).write_text(json.dumps({
        "unit_id": unit_id,
        "asof": asof,
        "representation": "samples",
        "asset_ids": assets,
        "horizons": horizons,
        "n_draws": n_draws,
        "target": target_type,
        "units": tgt.get("value_unit", ""),
        "rationale": {"file": _RATIONALE_NAME, "method": f"track2_agent/{cfg.name}"},
        "family": family or "unrecognized",
        "text_calibration_source": "learned" if text_calib is not None else "default_prior",
    }, indent=2) + "\n")

    rationale = build_rationale(
        unit_id, asof, assets, horizons, n_draws, cfg, anchors,
        numeric_results, fused_results, text_signals, distribution_mode, cross_horizon_mode,
    )
    (out_dir / _RATIONALE_NAME).write_text(rationale, encoding="utf-8")

    print(f"wrote {out_path.name}, {_META_NAME} and {_RATIONALE_NAME} to {out_dir}")
    print(f"  {len(assets)} asset(s) x {len(horizons)} horizon(s), {n_draws} draws, "
          f"config={cfg.name}, family={family or 'n/a'}, text_method={text_signals.method}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast", description="Agenthon 2026 Track 2 submission.")
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--card", type=pathlib.Path, default=None)
    p.add_argument("--n-draws", type=int, default=None)
    p.add_argument(
        "--seed", type=int, default=None,
        help="Monte Carlo seed; defaults to QFBENCH_SEED when supplied by the harness.",
    )
    p.add_argument(
        "--config", default="full",
        choices=["numerical_only", "full"],
        help="ablation switch: numerical_only | full (default)",
    )
    p.add_argument("--calibration", type=pathlib.Path, default=None,
                    help="path to a calibration.json produced by validation.py")
    args = p.parse_args(argv)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any unexpected failure to a clean exit
        print(f"forecast: fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
