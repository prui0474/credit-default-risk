"""Evaluation: discrimination, stability and calibration.

The metrics here are the ones a model-validation team would ask for:
AUC/Gini/KS for ranking power, PSI and per-vintage AUC for stability, and a
calibration table plus expected-loss math, because a PD that ranks well but is
mis-calibrated still mis-prices the loan.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def auc_gini_ks(y_true: Sequence[int], scores: Sequence[float]) -> dict:
    """AUC, Gini (=2*AUC-1) and the Kolmogorov-Smirnov statistic."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "gini": np.nan, "ks": np.nan, "n": int(len(y))}

    auc = float(roc_auc_score(y, s))
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    n_bad = max(ys.sum(), 1.0)
    n_good = max(len(ys) - ys.sum(), 1.0)
    cum_bad = np.cumsum(ys) / n_bad
    cum_good = np.cumsum(1.0 - ys) / n_good
    ks = float(np.max(np.abs(cum_bad - cum_good)))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks, "n": int(len(y))}


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population stability index of one score or feature distribution."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) < bins or len(a) == 0 or np.nanstd(e) == 0:
        return float("nan")

    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    e_share = np.histogram(e, bins=edges)[0] / len(e)
    a_share = np.histogram(a, bins=edges)[0] / len(a)
    e_share = np.clip(e_share, EPS, None)
    a_share = np.clip(a_share, EPS, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_verdict(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "monitor"
    return "shifted"


def decile_table(y_true: Sequence[int], scores: Sequence[float], n_bands: int = 10) -> pd.DataFrame:
    """Risk bands (highest first) with bad rate and lift."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "s": np.asarray(scores, dtype=float)})
    df["band"] = pd.qcut(df["s"].rank(method="first"), n_bands, labels=False)
    overall = max(df["y"].mean(), EPS)
    tbl = (
        df.groupby("band")
        .agg(n=("y", "size"), bad=("y", "sum"), min_score=("s", "min"), max_score=("s", "max"))
        .reset_index()
    )
    tbl["bad_rate"] = tbl["bad"] / tbl["n"]
    tbl["lift"] = tbl["bad_rate"] / overall
    tbl["cum_bad_capture"] = tbl.sort_values("band", ascending=False)["bad"].cumsum() / tbl["bad"].sum()
    tbl = tbl.sort_values("band", ascending=False).reset_index(drop=True)
    tbl["band"] = [f"D{i + 1}" for i in range(len(tbl))]
    return tbl


def calibration_table(y_true: Sequence[int], prob: Sequence[float], n_bands: int = 10) -> pd.DataFrame:
    """Predicted vs realised default rate by probability decile."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(prob, dtype=float)})
    df["band"] = pd.qcut(df["p"].rank(method="first"), n_bands, labels=False)
    tbl = (
        df.groupby("band")
        .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        .reset_index(drop=True)
    )
    tbl["gap"] = tbl["actual"] - tbl["predicted"]
    return tbl


def stability_by_vintage(df: pd.DataFrame, score_col: str, target: str = "target",
                         baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-vintage discrimination plus PSI against a baseline cohort."""
    rows = []
    base_scores = (
        baseline[score_col].to_numpy() if baseline is not None else df[score_col].to_numpy()
    )
    for vintage, group in df.groupby("vintage_year"):
        metrics = auc_gini_ks(group[target], group[score_col])
        rows.append(
            {
                "vintage_year": int(vintage),
                "n": metrics["n"],
                "bad_rate": float(group[target].mean()),
                "auc": metrics["auc"],
                "gini": metrics["gini"],
                "ks": metrics["ks"],
                "score_psi": psi(base_scores, group[score_col].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("vintage_year").reset_index(drop=True)


def approval_scenarios(
    y_true: Sequence[int],
    prob: Sequence[float],
    cutoffs: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> pd.DataFrame:
    """Decision layer: approve everyone below the cutoff and price the book."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    rows = []
    for cutoff in cutoffs:
        approved = p <= cutoff
        rejected = ~approved
        rows.append(
            {
                "cutoff_pd": cutoff,
                "approval_rate": float(approved.mean()),
                "approved_bad_rate": float(y[approved].mean()) if approved.any() else np.nan,
                "rejected_bad_rate": float(y[rejected].mean()) if rejected.any() else np.nan,
                "bad_capture_of_rejected": float(y[rejected].sum() / max(y.sum(), 1)),
            }
        )
    return pd.DataFrame(rows)
