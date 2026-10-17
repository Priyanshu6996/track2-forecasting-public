"""## Executive summary (read this first)

These focused tests lock the submission's seed precedence: an explicit command-line seed wins,
the Final harness seed is otherwise honored, and an absent or malformed harness value falls back
to the pinned agent default.  This makes reproducibility behavior testable without a Docker run.
"""

from track2_agent.cli import _resolve_seed
from track2_agent.config import AgentConfig


def test_explicit_cli_seed_has_priority(monkeypatch):
    monkeypatch.setenv("QFBENCH_SEED", "200")
    assert _resolve_seed(17) == 17


def test_harness_seed_is_used_when_cli_seed_is_absent(monkeypatch):
    monkeypatch.setenv("QFBENCH_SEED", "200")
    assert _resolve_seed(None) == 200


def test_missing_or_invalid_harness_seed_uses_pinned_default(monkeypatch):
    monkeypatch.delenv("QFBENCH_SEED", raising=False)
    assert _resolve_seed(None) == AgentConfig.from_name("full").mc.seed
    monkeypatch.setenv("QFBENCH_SEED", "not-an-integer")
    assert _resolve_seed(None) == AgentConfig.from_name("full").mc.seed
