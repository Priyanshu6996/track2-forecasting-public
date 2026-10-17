import numpy as np
import pandas as pd

from track2_agent.config import ModelConfig
from track2_agent.numeric_model import _refit_full


def test_refit_full_regression_predictions_are_numpy2_safe():
    rng = np.random.default_rng(20261009)
    x1 = rng.normal(size=90)
    x2 = rng.normal(size=90)
    X = pd.DataFrame({"lag_diff_1": x1, "feature": x2})
    y = pd.Series(0.25 * x1 - 0.1 * x2 + rng.normal(scale=0.2, size=90))
    cfg = ModelConfig(gbm_n_estimators=10)

    for candidate in ("ridge", "gbm", "ar1"):
        prediction = _refit_full(candidate, X, y, cfg)
        assert isinstance(prediction, float)
        assert np.isfinite(prediction)
