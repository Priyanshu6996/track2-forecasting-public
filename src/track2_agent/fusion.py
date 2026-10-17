"""Combine the numeric forecast with LLM-derived text signals into fused distribution
parameters: a mean shift (in the same units as the numeric model's own residual scale), a
volatility multiplier, and pass-through flags for tail widening.

Design principle: text can only ever ADJUST the numeric forecast, scaled by the numeric model's
own residual uncertainty and by the LLM's stated confidence — it never replaces it and it can
never dominate it. `FusionConfig` (config.py) sets the hard rails on how far that adjustment can
go; those rails are a safety backstop and are family-aware (F2 cards, whose whole premise is a
text-cued regime shift, get a wider mean-shift rail; F4 cards get a stronger risk-flag tail
response — see `config.FAMILY_OVERRIDES`).

WITHIN those rails, the actual weight given to a text signal is no longer a hand-picked constant
("hawkish -> add X sigma"). `text_calibration.py` learns it from historical walk-forward data
(text-signal-at-cutoff -> realized-move-after-cutoff, fit only on information available before
each cutoff) and writes it to a calibration file; `fuse` uses those learned coefficients when a
calibration table is supplied, and falls back to conservative, clearly-labeled manual priors
otherwise (see `DEFAULT_CALIBRATION`). Either way, the ablation harness (`validation.py`) is what
actually tells you whether a given fusion setting is net-positive for CRPS/tail — the caps here
are a safety rail, not a substitute for that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FusionConfig
from .numeric_model import NumericForecast
from .text_signals import TextSignalBundle

# Conservative manual priors, used only when no learned calibration table is available (e.g. the
# very first run against a unit, before any walk-forward calibration has been produced). These
# are deliberately weak: a bias-weight of 0.5 means an asset-specific view and the broad policy
# stance are blended equally when nothing historical says otherwise, and a vol-sensitivity of
# 1.0 means `expected_vol_change` maps linearly onto the rail defined by `FusionConfig` at full
# confidence. `text_calibration.py`'s learned values REPLACE these two numbers with ones fit to
# each unit's own history; they do not change the hard rails from `FusionConfig`.
DEFAULT_CALIBRATION = {"bias_blend_weight": 0.5, "vol_sensitivity": 1.0}


@dataclass(frozen=True, slots=True)
class FusedForecast:
    mean: float           # point forecast for the CHANGE over the horizon, after text adjustment
    scale: float           # residual scale (std-like) after text adjustment
    tail_widen: float        # extra multiplicative widening applied only to the tail (calibration.py)
    skew: float            # target skew parameter in [-1, 1] for montecarlo.py's sinh-arcsinh shape
    text_contribution_sigma: float  # how many sigma of the final mean came from text (for the rationale)


def fuse(
    numeric: NumericForecast,
    text: TextSignalBundle,
    asset: str,
    use_text: bool,
    fusion_cfg: FusionConfig | None = None,
    learned_calibration: dict[str, float] | None = None,
) -> FusedForecast:
    fcfg = fusion_cfg or FusionConfig()
    calib = {**DEFAULT_CALIBRATION, **(learned_calibration or {})}

    if not use_text or text.method == "none":
        return FusedForecast(
            mean=numeric.point, scale=numeric.resid_std, tail_widen=1.0, skew=0.0,
            text_contribution_sigma=0.0,
        )

    # A lexical (non-LLM) proxy is a meaningfully less trustworthy signal than a real LLM
    # extraction -- see `FusionConfig.non_llm_damping`'s docstring for the concrete evidence.
    # This damps EVERY rail below by the same factor for any non-"llm" method, generically.
    damping = 1.0 if text.method == "llm" else fcfg.non_llm_damping

    bias = text.directional_bias.get(asset, 0.0)
    # Blend the asset-specific directional view with the broad policy stance. The blend weight
    # is `calib["bias_blend_weight"]` -- a historically-fit value when available (see
    # text_calibration.py), the 0.5 default otherwise (a text-blind numeric model has no
    # per-asset text signal to check `bias` against, so equal weight is the uninformative prior).
    w = min(1.0, max(0.0, calib["bias_blend_weight"]))
    combined_signal = w * bias + (1.0 - w) * text.policy_stance
    shift_sigma = (
        max(-1.0, min(1.0, combined_signal)) * text.confidence
        * fcfg.max_mean_shift_sigma * damping
    )
    mean = numeric.point + shift_sigma * numeric.resid_std

    vol_sensitivity = max(0.0, calib["vol_sensitivity"])
    vol_mult_raw = 1.0 + (
        text.confidence * text.expected_vol_change * vol_sensitivity * damping
        * (fcfg.max_vol_mult - 1.0)
    )
    vol_mult = max(fcfg.min_vol_mult, min(fcfg.max_vol_mult, vol_mult_raw))
    scale = numeric.resid_std * vol_mult

    tail_widen = 1.0
    if text.risk_event_flag:
        tail_widen *= 1.0 + fcfg.risk_tail_widen * text.confidence * damping
    if text.regime_change_flag:
        tail_widen *= 1.0 + fcfg.regime_tail_widen * text.confidence * damping

    # Skew: a hawkish surprise on a still-uncertain read skews the up-tail; symmetric otherwise.
    skew = max(-0.6, min(0.6, fcfg.skew_scale * text.macro_surprise * text.confidence * damping))

    return FusedForecast(
        mean=mean, scale=scale, tail_widen=tail_widen, skew=skew,
        text_contribution_sigma=shift_sigma,
    )
