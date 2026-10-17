"""forecast_rationale.md generation.

Per SUBMISSION_CLI.md's T2 note: numbered steps -- the anchor, each adjustment with its size and
what supports it, then the scale and shape -- naming the series/dates computed from and the
documents cited, ending with an adjustment ledger so the arithmetic can be followed. Required,
never scored (gate g1 checks only non-emptiness) -- written for a human reviewer, so every number
in it is one actually used in the forecast, not a post-hoc gloss.
"""

from __future__ import annotations

from .config import AgentConfig
from .fusion import FusedForecast
from .numeric_model import NumericForecast
from .text_signals import TextSignalBundle


def build_rationale(
    unit_id: str,
    asof: str,
    assets: list[str],
    horizons: list[int],
    n_draws: int,
    cfg: AgentConfig,
    anchors: dict[str, float],
    numeric: dict[tuple[str, int], NumericForecast],
    fused: dict[tuple[str, int], FusedForecast],
    text: TextSignalBundle,
    distribution_mode: str = "current",
    cross_horizon_mode: str = "legacy",
) -> str:
    lines: list[str] = []
    lines.append(f"# Forecast rationale — {unit_id}\n")
    lines.append(
        f"As of **{asof}**, joint distribution over {', '.join(assets)} at horizon(s) "
        f"{', '.join(str(h) for h in horizons)} business days. {n_draws} draws. "
        f"Configuration: `{cfg.name}` (text={'on' if cfg.use_text else 'off'}, family="
        f"`{cfg.family or 'unrecognized'}`).\n"
    )

    lines.append("## 1. Anchor\n")
    for a in assets:
        lines.append(f"- **{a}**: last observed value = `{anchors.get(a, float('nan')):.4f}`.")
    lines.append("")

    lines.append("## 2. Numeric model\n")
    for a in assets:
        for h in horizons:
            nf = numeric.get((a, h))
            if nf is None:
                continue
            score = ", ".join(f"{k}={v:.5f}" for k, v in sorted(nf.cv_score_by_candidate.items()))
            mae = ", ".join(f"{k}={v:.5f}" for k, v in sorted(nf.cv_mae_by_candidate.items()))
            lines.append(
                f"- **{a} @ {h}bd**: selected `{nf.model_used}` "
                f"(selection metric — pooled out-of-fold CRPS/tail-proxy composite, lower is "
                f"better: {score or 'n/a — insufficient history'}; MAE shown for reference only, "
                f"it did NOT decide the winner: {mae or 'n/a'}); "
                f"{nf.n_train_rows} training rows; point change forecast = "
                f"`{nf.point:+.4f}`, residual scale = `{nf.resid_std:.4f}`."
            )
    lines.append("")

    lines.append("## 3. Text-derived adjustment\n")
    if not cfg.use_text or text.method == "none":
        lines.append(
            "**None.** Text component disabled for this run, or no text corpus documents were "
            "available at or before the as-of date. The numeric model's own forecast is used "
            "unadjusted."
        )
    else:
        lines.append(
            f"Method: `{text.method}`" + (" (model endpoint unreachable; deterministic lexical "
            "fallback used — see text_signals.py)" if text.method == "lexical_fallback" else "")
            + f". {text.n_documents} document(s) considered, confidence = `{text.confidence:.2f}`."
        )
        lines.append(f"\n> {text.rationale}\n")
        lines.append(
            f"- policy stance = `{text.policy_stance:+.2f}` (-1 dovish .. +1 hawkish)\n"
            f"- macro surprise = `{text.macro_surprise:+.2f}` (-1 downside .. +1 upside)\n"
            f"- regime-change flag = `{text.regime_change_flag}`, risk-event flag = "
            f"`{text.risk_event_flag}`\n"
            f"- expected vol change = `{text.expected_vol_change:+.2f}`"
        )
        lines.append(
            f"\nPer-asset adjustment applied (bounded to ±{cfg.fusion.max_mean_shift_sigma:.2f}σ "
            "of the numeric model's own residual scale for this card's family; see `fusion.py` "
            "and `config.FAMILY_OVERRIDES`):"
        )
        for a in assets:
            h = horizons[0]
            fz = fused.get((a, h))
            if fz is None:
                continue
            lines.append(
                f"- **{a}**: mean shift = `{fz.text_contribution_sigma:+.3f}σ`, "
                f"vol multiplier applied to residual scale, tail-widen = `{fz.tail_widen:.2f}x`, "
                f"skew = `{fz.skew:+.2f}`."
            )
    lines.append("")

    lines.append("## 4. Scale and shape\n")
    if distribution_mode == "gaussian_oof":
        lines.append(
            "Joint draws use Gaussian innovations with the empirical cross-asset correlation "
            "estimated from historical daily changes. Student-t scaling, the shock mixture, "
            "tail widening, and skew transforms are deliberately disabled in this validated "
            "out-of-fold marginal mode."
        )
    elif distribution_mode == "student_t_oof":
        lines.append(
            f"Joint draws use Student-t innovations (df = {cfg.mc.student_t_df:g}) with the "
            "empirical cross-asset correlation estimated from historical daily changes. The "
            "shock mixture, tail widening, and skew transforms are disabled in this "
            "out-of-fold marginal mode."
        )
    else:
        lines.append(
            "Joint draws use a shared correlated shock across assets (shrinkage correlation of "
            "historical daily changes), a shared Student-t fat-tail scaling factor per draw (df = "
            f"{cfg.mc.student_t_df:g}) so extreme outcomes co-move across assets, a low-probability "
            "shared shock-mixture component whose trigger probability scales with the text-derived "
            "tail-widen signal, and a sinh-arcsinh skew transform per cell."
        )
    lines.append(
        "Cross-horizon residual mode: `" + cross_horizon_mode + "`. See `montecarlo.py`."
    )
    lines.append("")

    lines.append("## 5. Adjustment ledger\n")
    lines.append("| asset | horizon | anchor | numeric Δ | text shift (σ) | final scale | tail-widen |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in assets:
        for h in horizons:
            nf = numeric.get((a, h))
            fz = fused.get((a, h))
            if nf is None or fz is None:
                continue
            lines.append(
                f"| {a} | {h} | {anchors.get(a, float('nan')):.4f} | {nf.point:+.4f} | "
                f"{fz.text_contribution_sigma:+.3f} | {fz.scale:.4f} | {fz.tail_widen:.2f} |"
            )
    lines.append("")

    lines.append("## 6. What would change this forecast\n")
    lines.append(
        "A materially different signal in the text corpus (a policy surprise, a data release "
        "not yet reflected in the panel, or a flagged risk event), or new panel data that shifts "
        "the numeric model's own trained trend/volatility estimate. The ablation comparison in "
        "`docs/VALIDATION.md` reports how much this text signal typically moves the score, "
        "measured on the public practice units — not asserted here."
    )
    return "\n".join(lines) + "\n"
