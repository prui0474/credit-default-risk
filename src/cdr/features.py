"""Weight-of-evidence (WOE) binning and information value (IV).

Why WOE instead of raw features
-------------------------------
A discrete scorecard has to map every input onto a points scale. WOE binning
does three jobs at once: it handles missing values, it tames outliers and
non-linearity, and it produces a monotone-per-bin signal that a logistic
regression can consume. IV then ranks candidate variables by their marginal
separating power, which is the usual first screen in a credit scorecard build.

The encoder is fitted on the training split only. Applying a pre-fitted
binning to validation and out-of-time data is what keeps the OOT numbers
honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace smoothing so that an empty cell never produces inf
MISSING_LABEL = "__MISSING__"
OTHER_LABEL = "__OTHER__"


def _apply_edges(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Bin a numeric series with pre-fitted edges; missing gets its own code."""
    cut = pd.cut(values, bins=edges, include_lowest=True)
    codes = cut.cat.codes.to_numpy().astype(int)
    nan_code = len(cut.cat.categories)
    codes[codes == -1] = nan_code
    return codes, nan_code


def _woe_table(bad: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_bad = max(bad.sum(), 1.0)
    total_good = max(good.sum(), 1.0)
    bad_dist = (bad + EPS) / (total_bad + EPS * len(bad))
    good_dist = (good + EPS) / (total_good + EPS * len(good))
    woe = np.log(bad_dist / good_dist)
    iv = (bad_dist - good_dist) * woe
    return woe, iv


@dataclass
class WOEEncoder:
    numeric_cols: List[str]
    categorical_cols: List[str]
    max_bins: int = 10
    min_share: float = 0.02
    edges_: Dict[str, np.ndarray] = field(default_factory=dict)
    maps_: Dict[str, Dict[int | str, float]] = field(default_factory=dict)
    iv_: Dict[str, float] = field(default_factory=dict)
    detail_: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, target: str = "target") -> "WOEEncoder":
        y = df[target].astype(int).to_numpy()
        for col in self.numeric_cols:
            self._fit_numeric(pd.to_numeric(df[col], errors="coerce"), y, col)
        for col in self.categorical_cols:
            self._fit_categorical(df[col].astype("object").fillna(MISSING_LABEL), y, col)
        return self

    def _fit_numeric(self, values: pd.Series, y: np.ndarray, col: str) -> None:
        valid = values.dropna()
        if valid.nunique() < 3:
            self._fit_categorical(values.astype("object").fillna(MISSING_LABEL), y, col)
            return

        edges = np.unique(np.quantile(valid.to_numpy(), np.linspace(0, 1, self.max_bins + 1)))
        if len(edges) < 3:
            edges = np.unique(np.array([valid.min(), valid.max()], dtype=float))
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        codes, nan_code = _apply_edges(values, edges)
        borders = [f"<= {e:,.4g}" for e in edges[1:-1]] + [f"> {edges[-2]:,.4g}"]
        labels = borders[: len(edges) - 1] + [MISSING_LABEL]

        agg = pd.DataFrame({"code": codes, "y": y}).groupby("code")["y"].agg(["size", "sum"])
        agg = agg.reindex(range(nan_code + 1), fill_value=0)
        bad = agg["sum"].to_numpy(dtype=float)
        good = (agg["size"] - agg["sum"]).to_numpy(dtype=float)
        woe, iv = _woe_table(bad, good)

        self.edges_[col] = edges
        self.maps_[col] = {int(c): float(w) for c, w in enumerate(woe)}
        self.iv_[col] = float(iv.sum())
        self.detail_[col] = pd.DataFrame(
            {
                "bin": labels,
                "n": agg["size"].to_numpy(),
                "bad": bad,
                "bad_rate": bad / np.maximum(agg["size"].to_numpy(), 1),
                "woe": woe,
                "iv": iv,
            }
        )

    def _fit_categorical(self, values: pd.Series, y: np.ndarray, col: str) -> None:
        share = values.value_counts(normalize=True)
        rare = set(share[share < self.min_share].index)
        values = values.where(~values.isin(rare), OTHER_LABEL)

        agg = pd.DataFrame({"value": values.to_numpy(), "y": y}).groupby("value")["y"].agg(
            ["size", "sum"]
        )
        bad = agg["sum"].to_numpy(dtype=float)
        good = (agg["size"] - agg["sum"]).to_numpy(dtype=float)
        woe, iv = _woe_table(bad, good)

        self.maps_[col] = {str(k): float(w) for k, w in zip(agg.index, woe)}
        self.iv_[col] = float(iv.sum())
        self.detail_[col] = pd.DataFrame(
            {
                "bin": agg.index.astype(str),
                "n": agg["size"].to_numpy(),
                "bad": bad,
                "bad_rate": bad / np.maximum(agg["size"].to_numpy(), 1),
                "woe": woe,
                "iv": iv,
            }
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            if col in self.edges_:
                codes, _ = _apply_edges(pd.to_numeric(df[col], errors="coerce"), self.edges_[col])
                out[col] = pd.Series(codes, index=df.index).map(self.maps_[col])
            else:  # was fitted as a categorical because it had too few levels
                values = df[col].astype("object").fillna(MISSING_LABEL).astype(str)
                values = values.where(values.isin(self.maps_.get(col, {})), OTHER_LABEL)
                out[col] = values.map(self.maps_.get(col, {})).astype(float)
        for col in self.categorical_cols:
            values = df[col].astype("object").fillna(MISSING_LABEL).astype(str)
            values = values.where(values.isin(self.maps_.get(col, {})), OTHER_LABEL)
            out[col] = values.map(self.maps_.get(col, {})).astype(float)
        return out.fillna(0.0)

    def iv_table(self) -> pd.DataFrame:
        tbl = (
            pd.DataFrame({"feature": list(self.iv_.keys()), "iv": list(self.iv_.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        tbl["strength"] = pd.cut(
            tbl["iv"],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "suspicious"],
        )
        return tbl
