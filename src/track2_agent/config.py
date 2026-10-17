"""Pipeline configuration.

One dataclass, two named presets. This is the entire ablation surface described in the
competition guide's "text uplift" diagnostic:

    numerical_only        time-series features only, no text
    full                  + LLM-derived structured text signals (the actual submission)

`AgentConfig.from_name` is the single place that maps a name to a configuration, so the CLI,
the validation harness and the ablation script all stay in sync.

NOTE ON PATH SIGNATURES (removed): an earlier revision of this pipeline computed rolling-window
path-signature features (a `numerical_plus_sig` preset sat between the two above) and, later, a
per-cell walk-forward gate that used them only when they measurably helped. Two rounds of
walk-forward validation evidence — most recently an aggregate composite of 11.2993
(numerical-only) vs. 11.3755 (with signatures, and `full` did not recover the degradation) across
overall and F1/F3 family breakdowns — showed signatures never net-improving and, in the
aggregate, making the composite score WORSE. Per this project's explicit "remove a component that
doesn't earn its complexity" decision rule, path signatures and every piece of infrastructure
that existed solely to support them (`signatures.py`, `SignatureConfig`, the per-cell signature
gate in `cli.py`, the `numerical_plus_sig` preset) were removed rather than kept as inert,
never-enabled infrastructure. See `docs/VALIDATION.md`'s "Update 3" section for the full evidence
and `docs/QUANT_NOTES.md`'s KEEP/REMOVE table for the final disposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class TextConfig:
    """LLM text-reasoning configuration. Model access follows SUBMISSION_CLI.md exactly:
    the ONLY reachable model is the organizer house endpoint at $MODEL_ENDPOINT / $MODEL_NAME
    in `restricted` network mode. No vendor API keys exist in the scoring environment.
    """

    enabled: bool = True
    max_docs: int = 40
    temperature: float = 0.0   # pinned per SUBMISSION_CLI.md rule 4
    seed: int = 20260101       # pinned per SUBMISSION_CLI.md rule 4
    max_output_tokens: int = 900
    request_timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Numeric forecasting model selection. `candidates` are tried and the best is chosen by
    walk-forward validation (numeric_model.py); this record only pins the search space + the
    regularization defaults used when there isn't enough history to cross-validate.

    Candidate set (broadened per the competition-alignment review; see numeric_model.py and
    docs/VALIDATION.md's model-comparison table for which of these actually earn their place on
    a given asset/horizon):

        rw_drift    driftless / drift random walk (the floor every other candidate must beat)
        ewma        exponentially-weighted mean of the historical target change (a fast-adapting
                    drift estimate, distinct from the flat-window `rw_drift` mean)
        ar1         one-feature regression of the horizon target on the asset's own most recent
                    realized change (a minimal autoregressive-style model)
        ridge       regularized linear regression on the full engineered feature set
        gbm         shallow gradient-boosted trees on the full engineered feature set

    Selection metric: `numeric_model.py` no longer picks the candidate with the lowest walk-
    forward MAE. It pools each candidate's out-of-fold (prediction, actual) pairs and scores it
    with the SAME marginal-CRPS / tail-pinball combination `metrics.composite_score` uses in the
    single-cell case (a model-selection step is, by construction, a single (asset, horizon)
    cell) — i.e. selection is now aligned with the competition's actual 50/30/20 objective
    rather than a point-accuracy proxy that does not correlate with it. MAE is still recorded
    per candidate for diagnostics/the rationale, but it no longer decides the winner.
    """

    candidates: tuple[str, ...] = ("rw_drift", "ewma", "ar1", "ridge", "gbm")
    ridge_alpha: float = 3.0
    gbm_max_depth: int = 3
    gbm_n_estimators: int = 120
    gbm_learning_rate: float = 0.05
    ewma_span: float = 21.0    # business days; ~1 trading month half-life-ish decay
    min_train_rows: int = 60   # below this, fall back to the driftless random-walk floor


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    n_draws: int = 500
    student_t_df: float = 6.0   # fat-tailed innovations; df=6 -> excess kurtosis ~1.5 (moderate)
    shrinkage: float = 0.0      # Phase-3 winner: use empirical dependence without shrinkage
    tail_widen_min: float = 1.0  # calibration.py may scale this up per asset/horizon
    seed: int = 20260101


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Text/numeric fusion caps (see fusion.py). These are the safety RAILS — the actual weight
    given to a text signal within those rails comes from `text_calibration.py`'s historically
    learned coefficients when a calibration file is available, and from the conservative
    defaults below otherwise. Per-family overrides (see `FAMILY_OVERRIDES`) widen or tighten
    these rails where the card family's own structure justifies it; they never remove the rails.
    """

    max_mean_shift_sigma: float = 0.75
    max_vol_mult: float = 2.5
    min_vol_mult: float = 0.6
    risk_tail_widen: float = 0.5
    regime_tail_widen: float = 0.3
    skew_scale: float = 0.4
    non_llm_damping: float = 0.35
    # Multiplicative damping applied to ALL of the above rails when `text.method` is anything
    # other than "llm" (i.e. the deterministic lexical fallback, or any other non-LLM proxy
    # added later). Justification: 96/96 practice-unit forecasts in the diagnostics that
    # motivated this were produced by the lexical fallback, never a real LLM call
    # ($MODEL_ENDPOINT unreachable in that environment), and a concrete measured case showed the
    # lexical fallback making a unit's composite score WORSE than the text-blind numeric-only
    # baseline (~6.400 -> ~6.769 on one F3 funding-flip unit). A hand-designed lexical heuristic
    # (keyword counting into a hawkish/dovish score, etc.) is a meaningfully less trustworthy
    # signal than a real LLM extraction, and this damping encodes exactly that epistemic
    # difference generically, via `text.method`, not via any per-asset or per-family special
    # case. It does not disable text entirely (the review explicitly allows "reduce its
    # influence" as an alternative to fully disabling it), and it composes with
    # `text_calibration.py`'s learned weights and with the family-aware overrides below:
    # whichever combination produces the narrowest net rail applies.


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str = "full"
    use_text: bool = True
    family: str = ""  # e.g. "F1".."F4", parsed from card metadata.category; "" = unknown/generic
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mc: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    @staticmethod
    def from_name(name: str) -> "AgentConfig":
        if name == "numerical_only":
            return AgentConfig(name=name, use_text=False, text=TextConfig(enabled=False))
        if name == "full":
            return AgentConfig(name=name, use_text=True)
        raise ValueError(f"unknown config {name!r}; expected one of 'numerical_only', 'full'")

    @staticmethod
    def from_name_and_family(name: str, family: str) -> "AgentConfig":
        """`from_name` plus the card family's structural overrides (see `FAMILY_OVERRIDES`).

        The family only ever adjusts fusion sensitivity / Monte Carlo shape parameters that the
        card's own text-corpus structure or asset-cardinality justify (see docstrings on
        `FAMILY_OVERRIDES`); it never changes which base config (`numerical_only` / `full`) is
        used, and every override still runs through the same walk-forward-validated numeric
        model and the same safety-rail bounds in `fusion.py`.
        An unrecognized or empty family falls through to the un-overridden base config.
        """
        base = AgentConfig.from_name(name)
        fam = (family or "").upper().removeprefix("T2-")
        overrides = FAMILY_OVERRIDES.get(fam)
        if overrides is None:
            return base
        return replace(base, family=fam, **overrides)


# Family-specific structural overrides, justified per the card families in the competition
# guide (see feedback item 6 / docs/ARCHITECTURE.md "Family-specific handling" section):
#
#   F1  standard time-series forecasting: no override; the base numeric+text pipeline
#       already targets this directly.
#   F2  text-cued regime-shift / transfer: text may need to move the mean MORE (a regime shift
#       is, by definition, something the numerical series has not yet fully reflected) and the
#       regime-change flag should widen tails more aggressively, since a missed regime shift is
#       exactly the failure mode this family is designed to probe.
#   F3  cross-asset / joint forecasting: trust the empirical cross-asset correlation more (less
#       shrinkage toward the diagonal) since these cards are scored 30% on the joint variogram,
#       which directly rewards a well-estimated joint dependence structure over a conservative
#       (shrunk-toward-independent) one, and these cards also tend to ship more assets, which
#       gives the empirical correlation more support.
#   F4  tail / shock-from-text forecasting: lower the baseline Student-t degrees of freedom
#       (fatter unconditional tails) and let a risk-event flag widen the tail further, since
#       this family is scored on tail calibration under text-flagged shock conditions.
FAMILY_OVERRIDES: dict[str, dict] = {
    "F1": {},
    "F2": {
        "fusion": FusionConfig(max_mean_shift_sigma=1.1, regime_tail_widen=0.45),
    },
    "F3": {},
    "F4": {
        "mc": MonteCarloConfig(student_t_df=4.0),
        "fusion": FusionConfig(risk_tail_widen=0.9),
    },
}
