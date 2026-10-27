"""Models: a transparent scorecard and a gradient-boosted challenger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class FittedModel:
    name: str
    estimator: Any

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score in a points-like scale (higher = riskier)."""
        return self.predict_proba(X)

    def coefficients(self, feature_names) -> pd.DataFrame:
        if not hasattr(self.estimator, "coef_"):
            raise AttributeError("Only the logistic scorecard exposes coefficients.")
        return (
            pd.DataFrame(
                {
                    "feature": list(feature_names),
                    "woe_coefficient": self.estimator.coef_.ravel(),
                }
            )
            .assign(odds_multiplier=lambda d: np.exp(d["woe_coefficient"]))
            .sort_values("woe_coefficient", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )


def fit_scorecard(X: pd.DataFrame, y: pd.Series, C: float = 1.0, max_iter: int = 1000) -> FittedModel:
    """Logistic regression on WOE inputs = a classical scorecard."""
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X, y)
    return FittedModel("scorecard", clf)


def fit_gbm(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any] | None = None) -> FittedModel:
    """Gradient boosting challenger (in-tree implementation, no extra deps)."""
    p = {
        "max_iter": 400,
        "learning_rate": 0.04,
        "max_depth": 4,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": 42,
    }
    p.update(params or {})
    clf = HistGradientBoostingClassifier(**p)
    clf.fit(X, y)
    return FittedModel("gbm", clf)
