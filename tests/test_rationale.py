"""## Executive summary (read this first)

The required human-readable rationale must describe the sampler that actually made the submitted
draws.  These tests prevent a Gaussian out-of-fold forecast from being mislabeled as fat-tailed.
"""

from track2_agent.config import AgentConfig
from track2_agent.rationale import build_rationale
from track2_agent.text_signals import TextSignalBundle


def test_gaussian_oof_rationale_does_not_claim_student_t_sampling():
    rationale = build_rationale(
        "synthetic", "2020-01-01", ["SYN"], [1], 200, AgentConfig.from_name("full"),
        {"SYN": 1.0}, {}, {}, TextSignalBundle(), "gaussian_oof", "legacy",
    )
    scale_section = rationale.split("## 4. Scale and shape", 1)[1].split("## 5.", 1)[0]
    assert "Gaussian innovations" in scale_section
    assert "Student-t scaling" in scale_section
    assert "Cross-horizon residual mode: `legacy`" in scale_section
