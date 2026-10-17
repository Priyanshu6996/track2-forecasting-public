"""LLM-assisted structured text-signal extraction.

Per SUBMISSION_CLI.md, in `restricted` network mode the ONLY reachable model is the
organizer-hosted house endpoint at `$MODEL_ENDPOINT` (OpenAI-compatible), named by `$MODEL_NAME`.
No participant API key exists in the scoring environment, and vendor-side tools (web search,
retrieval, code execution) must be disabled on every call. This module therefore:

  - reads `$MODEL_ENDPOINT` / `$MODEL_NAME` from the environment and calls nothing else;
  - sends NO tools/functions in the request (vendor-side tools off, by construction);
  - pins temperature and seed (`TextConfig.temperature`, `TextConfig.seed`);
  - asks for STRUCTURED JSON only — never a numerical forecast (per the brief: "The LLM should
    NOT simply hallucinate a numerical forecast").

The LLM never sees anything except the text corpus documents (`timestamp <= asof`, enforced
upstream by gate g2 / the publication-time cutoff) and the target asset/horizon metadata. No web
search, no retrieval, no external context is added.

## Offline fallback

Local smoke runs use `--network=none` (README, "Network modes"): the house endpoint is
unreachable there by design, and that is expected and fine for a structural test. When the
endpoint cannot be reached, `extract_text_signals` falls back to a transparent, deterministic
lexical scorer (`_lexical_fallback`) so the CLI still produces a complete, valid forecast
offline. The fallback's provenance is always recorded in the returned `TextSignalBundle.method`
field and surfaces in the rationale — it is never silently presented as LLM reasoning.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import TextConfig

_SCHEMA_INSTRUCTIONS = """\
You are a fixed-income / FX / macro analyst extracting STRUCTURED signals from a dated text \
corpus for a quantitative forecasting model. You are NOT forecasting a number. Read the \
documents and return ONLY a JSON object (no prose, no markdown fences) with this exact shape:

{
  "policy_stance": <float in [-1, 1], -1=very dovish/easing, 0=neutral, 1=very hawkish/tightening>,
  "macro_surprise": <float in [-1, 1], -1=large downside surprise, 1=large upside surprise>,
  "regime_change_flag": <bool, true if the text signals a material regime change vs. priors>,
  "risk_event_flag": <bool, true if the text flags an imminent risk/shock event>,
  "expected_vol_change": <float in [-1, 1], -1=vol likely falls, 1=vol likely rises>,
  "directional_bias": {"<ASSET_ID>": <float in [-1, 1], per target asset, your directional view>, ...},
  "confidence": <float in [0, 1], your confidence in the above>,
  "rationale": "<one or two sentences citing which document(s) drove this>"
}

Only use information in the documents provided. Do not invent facts. If the corpus contains no \
relevant signal, return near-zero values and confidence <= 0.2 rather than guessing.
"""


@dataclass(frozen=True, slots=True)
class TextSignalBundle:
    policy_stance: float = 0.0
    macro_surprise: float = 0.0
    regime_change_flag: bool = False
    risk_event_flag: bool = False
    expected_vol_change: float = 0.0
    directional_bias: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = "no text signal extracted"
    method: str = "none"  # "llm" | "lexical_fallback" | "none"
    n_documents: int = 0

    def to_dict(self) -> dict:
        return {
            "policy_stance": self.policy_stance,
            "macro_surprise": self.macro_surprise,
            "regime_change_flag": self.regime_change_flag,
            "risk_event_flag": self.risk_event_flag,
            "expected_vol_change": self.expected_vol_change,
            "directional_bias": self.directional_bias,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "method": self.method,
            "n_documents": self.n_documents,
        }


def _load_documents(text_dir: pathlib.Path, asof: str, max_docs: int) -> list[dict]:
    """Every corpus document at or before `asof`. Trusts gate g2 / the publication-time cutoff
    for the guarantee, but still filters defensively: a doc with no readable timestamp, or a
    timestamp after `asof`, is excluded rather than included-by-default.
    """
    docs: list[dict] = []
    index_path = text_dir / "corpus_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {}
        for entry in index.get("documents", []):
            ts = str(entry.get("timestamp", ""))[:10]
            if not ts or ts > asof:
                continue
            fp = text_dir / entry.get("file", "")
            if not fp.exists():
                continue
            docs.append({
                "doc_id": entry.get("doc_id", fp.stem),
                "timestamp": ts,
                "doc_type": entry.get("doc_type", "unknown"),
                "text": fp.read_text(errors="ignore"),
            })
    else:
        # No index: fall back to reading every .txt directly and trust the upstream cutoff.
        for fp in sorted(text_dir.glob("*.txt")):
            docs.append({
                "doc_id": fp.stem, "timestamp": asof, "doc_type": "unknown",
                "text": fp.read_text(errors="ignore"),
            })
    docs.sort(key=lambda d: d["timestamp"])
    return docs[-max_docs:]


_HAWKISH = re.compile(
    r"\b(restrictive|tighten|hike|inflation risk|elevated inflation|above target|"
    r"further (progress|tightening)|higher for longer)\b", re.I,
)
_DOVISH = re.compile(
    r"\b(accommodat|cut rates|ease|reduce the target|greater confidence|slowdown|"
    r"downside risk|weakening|softening)\b", re.I,
)
_SURPRISE_UP = re.compile(r"\b(upside surprise|higher than expected|beat expectations|hot print)\b", re.I)
_SURPRISE_DOWN = re.compile(r"\b(downside surprise|lower than expected|missed expectations|cool(?:er|ing) print)\b", re.I)
_RISK = re.compile(r"\b(shock|crisis|contagion|liquidity stress|de-?peg|default|volatil(?:e|ity) spik)\b", re.I)
_REGIME = re.compile(r"\b(regime|pivot|shift in stance|change of course|unexpected(?:ly)?)\b", re.I)


def _lexical_fallback(docs: list[dict], assets: list[str]) -> TextSignalBundle:
    """Deterministic keyword scorer used only when the model endpoint is unreachable (offline
    smoke runs). Transparent by construction: every weight below is a plain count difference,
    clipped to [-1, 1], with no hidden state and no randomness.
    """
    if not docs:
        return TextSignalBundle(method="lexical_fallback", n_documents=0,
                                 rationale="no documents in corpus at or before as-of")
    text = "\n".join(d["text"] for d in docs)
    hawk = len(_HAWKISH.findall(text))
    dove = len(_DOVISH.findall(text))
    stance = 0.0 if (hawk + dove) == 0 else (hawk - dove) / (hawk + dove)
    up = len(_SURPRISE_UP.findall(text))
    down = len(_SURPRISE_DOWN.findall(text))
    surprise = 0.0 if (up + down) == 0 else (up - down) / (up + down)
    risk_hits = len(_RISK.findall(text))
    regime_hits = len(_REGIME.findall(text))
    # Capped well below 1.0: this is a crude keyword count, not reasoning, and fusion.py's
    # safety bounds (MAX_MEAN_SHIFT_IN_SIGMA, MAX_VOL_MULT) only actually protect the forecast
    # if confidence itself stays low for a method this weak. 7 documents saturating to
    # confidence=1.0 under the old `min(1.0, 0.15*n)` formula was measured (a real practice
    # unit) to blow a 0.10 residual scale out to 0.50 -- a 5x inflation from keyword counts
    # alone. Capped at 0.35 so the fallback can nudge, never dominate.
    conf = min(0.35, 0.05 * len(docs))
    bias = {a: 0.5 * stance for a in assets}  # naive uniform pass-through; a real LLM call
    #  would differentiate assets. Kept intentionally weak so the fallback cannot dominate fusion.
    return TextSignalBundle(
        policy_stance=round(float(stance), 3),
        macro_surprise=round(float(surprise), 3),
        regime_change_flag=regime_hits >= 2,
        risk_event_flag=risk_hits >= 1,
        expected_vol_change=round(min(1.0, 0.3 * risk_hits), 3),
        directional_bias={k: round(v, 3) for k, v in bias.items()},
        confidence=round(conf, 3),
        rationale=(
            f"Lexical fallback (no model endpoint reachable): {hawk} hawkish / {dove} dovish, "
            f"{up} upside / {down} downside surprise cues, {risk_hits} risk cues across "
            f"{len(docs)} document(s)."
        ),
        method="lexical_fallback",
        n_documents=len(docs),
    )


def _call_house_endpoint(prompt: str, cfg: TextConfig) -> str | None:
    endpoint = os.environ.get("MODEL_ENDPOINT")
    model_name = os.environ.get("MODEL_NAME")
    model_token = os.environ.get("MODEL_TOKEN")
    if not endpoint or not model_name or not model_token:
        return None
    body = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SCHEMA_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "max_tokens": cfg.max_output_tokens,
        # No `tools` / `functions` field: vendor-side tools are off by construction, per
        # SUBMISSION_CLI.md rule 1.
    }).encode("utf-8")
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError):
        return None


def _parse_llm_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _clip(x, lo, hi, default=0.0) -> float:
    try:
        return float(min(hi, max(lo, float(x))))
    except (TypeError, ValueError):
        return default


def extract_text_signals(
    text_dir: pathlib.Path,
    asof: str,
    assets: list[str],
    cfg: TextConfig,
) -> TextSignalBundle:
    if not cfg.enabled:
        return TextSignalBundle(method="none", rationale="text component disabled by config")

    docs = _load_documents(text_dir, asof, cfg.max_docs) if text_dir and text_dir.is_dir() else []
    if not docs:
        return TextSignalBundle(method="none", n_documents=0,
                                 rationale="no text corpus documents at or before as-of")

    corpus_block = "\n\n---\n\n".join(
        f"[{d['timestamp']} | {d['doc_type']} | {d['doc_id']}]\n{d['text']}" for d in docs
    )
    prompt = (
        f"As-of date: {asof}\nTarget assets: {', '.join(assets)}\n\n"
        f"Documents (all dated on or before the as-of date):\n\n{corpus_block}"
    )
    raw = _call_house_endpoint(prompt, cfg)
    if raw is None:
        return _lexical_fallback(docs, assets)

    parsed = _parse_llm_json(raw)
    if parsed is None:
        # Model reachable but returned unparsable output: fall back rather than propagate junk.
        fb = _lexical_fallback(docs, assets)
        return TextSignalBundle(**{**fb.to_dict(), "method": "lexical_fallback",
                                    "rationale": "model output was not valid JSON; " + fb.rationale})

    bias_raw = parsed.get("directional_bias", {}) or {}
    bias = {a: _clip(bias_raw.get(a, 0.0), -1.0, 1.0) for a in assets}
    return TextSignalBundle(
        policy_stance=_clip(parsed.get("policy_stance"), -1.0, 1.0),
        macro_surprise=_clip(parsed.get("macro_surprise"), -1.0, 1.0),
        regime_change_flag=bool(parsed.get("regime_change_flag", False)),
        risk_event_flag=bool(parsed.get("risk_event_flag", False)),
        expected_vol_change=_clip(parsed.get("expected_vol_change"), -1.0, 1.0),
        directional_bias=bias,
        confidence=_clip(parsed.get("confidence"), 0.0, 1.0),
        rationale=str(parsed.get("rationale", ""))[:800] or "(model returned no rationale text)",
        method="llm",
        n_documents=len(docs),
    )
