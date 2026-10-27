"""End-to-end pipeline: load -> split -> WOE -> model -> evaluate -> report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import ensure_dir, resolve
from .data import load_data
from .evaluate import (
    approval_scenarios,
    auc_gini_ks,
    calibration_table,
    decile_table,
    psi,
    psi_verdict,
    stability_by_vintage,
)
from .features import WOEEncoder
from .models import fit_gbm, fit_scorecard
from .report import build_report
from .schema import TARGET
from .split import make_split

DEFAULT_LIMITATIONS = [
    "AUC/KS measure ranking, not profit. Origination-level economics need rate, "
    "loss-given-default and prepayment assumptions to be judged properly.",
    "The target is a binary 90+ DPD flag within a fixed window; it ignores "
    "time-to-default, which a survival model would exploit.",
    "No reject inference: the training book only contains approved loans, so the "
    "model is blind to the applicants that were previously declined.",
    "Protected attributes are not in the data. Fairness testing therefore needs a "
    "proxy analysis (e.g. geography) which is indicative, not conclusive.",
    "The note rate is risk-priced: it is partly a function of the credit score and "
    "LTV, so feeding it to the model can double-count information that is already "
    "there. Check whether the scorecard holds up without it.",
]


def run(cfg: Dict[str, Any], source: str | None = None, write_report: bool = True,
        verbose: bool = True) -> Dict[str, Any]:
    source = source or cfg["data"]["source"]
    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)

    df = load_data(cfg, source)
    if verbose:
        notes = []
        for key, label in (
            ("unobserved_dropped", "no_opening_window"),
            ("out_of_scope_dropped", "outside_split_vintages"),
        ):
            value = int(df.attrs.get(key, 0) or 0)
            if value:
                notes.append(f"{label}={value:,}")
        suffix = (" dropped(" + ", ".join(notes) + ")") if notes else ""
        print(f"[data] source={source} rows={len(df):,} "
              f"vintages={df['vintage_year'].min()}-{df['vintage_year'].max()} "
              f"default_rate={df[TARGET].mean():.3%}{suffix}")

    split = make_split(df, cfg)
    split_table = pd.DataFrame(
        [
            {"split": name, "rows": len(part),
             "vintages": f"{int(part['vintage_year'].min())}-{int(part['vintage_year'].max())}",
             "default_rate": part[TARGET].mean()}
            for name, part in split.as_dict().items()
        ]
    )

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    encoder = WOEEncoder(numeric, categorical, max_bins=int(cfg["features"]["woe_max_bins"]))
    encoder.fit(split.train, TARGET)

    matrices = {name: encoder.transform(part) for name, part in split.as_dict().items()}
    targets = {name: part[TARGET].astype(int) for name, part in split.as_dict().items()}

    scorecard = fit_scorecard(matrices["train"], targets["train"], **cfg["model"]["scorecard"])
    gbm = fit_gbm(matrices["train"], targets["train"], cfg["model"]["gbm"])
    models = {"scorecard": scorecard, "gbm": gbm}

    metrics_rows = []
    predictions: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in split.as_dict()}
    for name, model in models.items():
        for part_name, part in split.as_dict().items():
            prob = model.predict_proba(matrices[part_name])
            predictions[part_name][name] = prob
            m = auc_gini_ks(targets[part_name], prob)
            metrics_rows.append(
                {"model": name, "split": part_name, "n": m["n"],
                 "bad_rate": float(targets[part_name].mean()),
                 "auc": m["auc"], "gini": m["gini"], "ks": m["ks"]}
            )
    model_metrics = pd.DataFrame(metrics_rows)

    # --- Stability: per-vintage AUC for both models --------------------------
    vintage_parts = []
    for name, model in models.items():
        scored = df.copy()
        scored["_score"] = model.predict_proba(encoder.transform(df))
        v = stability_by_vintage(scored, "_score", TARGET, baseline=scored)
        v = v.rename(columns={"auc": f"{name}_auc", "gini": f"{name}_gini",
                              "ks": f"{name}_ks", "score_psi": f"{name}_psi"})
        vintage_parts.append(v.drop(columns=["bad_rate", "n"]).set_index("vintage_year"))
    vintage = pd.concat(vintage_parts, axis=1).reset_index()
    vintage.insert(1, "n", vintage["vintage_year"].map(df.groupby("vintage_year").size()))
    keep = ["vintage_year", "n", "scorecard_auc", "gbm_auc", "scorecard_gini",
            "gbm_gini", "scorecard_ks", "gbm_ks", "gbm_psi"]
    vintage = vintage[[c for c in keep if c in vintage.columns]]

    # --- Feature drift -------------------------------------------------------
    train_woe, oot_woe = matrices["train"], matrices["oot"]
    psi_rows = [
        {"feature": col, "psi": psi(train_woe[col].to_numpy(), oot_woe[col].to_numpy())}
        for col in train_woe.columns
    ]
    feature_psi = pd.DataFrame(psi_rows)
    feature_psi["verdict"] = feature_psi["psi"].map(psi_verdict)
    feature_psi = feature_psi.sort_values("psi", ascending=False).reset_index(drop=True)

    # --- Calibration / bands / decisions on the OOT cohort -------------------
    oot_prob = predictions["oot"]
    calibration = {
        name: calibration_table(targets["oot"], prob) for name, prob in oot_prob.items()
    }
    deciles = {name: decile_table(targets["oot"], prob) for name, prob in oot_prob.items()}
    approval = approval_scenarios(targets["oot"], oot_prob["gbm"])

    coefficients = None
    try:
        coefficients = scorecard.coefficients(matrices["train"].columns)
    except AttributeError:
        pass

    roc_curves = {
        f"{name} (OOT)": (targets["oot"].to_numpy(), oot_prob[name]) for name in models
    }

    processed_dir = ensure_dir(resolve(cfg, cfg["data"]["processed_dir"]))
    df.to_csv(processed_dir / f"{source}_unified.csv", index=False)

    limitations = list(DEFAULT_LIMITATIONS)
    if source == "demo":
        limitations.insert(
            0,
            "This run uses the synthetic smoke-test cohort (`data.source: demo`). "
            "Every number here is illustrative; re-run on Freddie Mac or SBA data "
            "before quoting anything.",
        )
    if source == "sba":
        limitations.insert(
            0,
            "SBA records have no bureau score, DTI or collateral value, so the "
            "achievable discrimination is structurally lower than for mortgages.",
        )

    results: Dict[str, Any] = {
        "meta": {
            "source": source,
            "n_rows": len(df),
            "bad_rate": float(df[TARGET].mean()),
            "target_definition": cfg["target"]["definition"],
            "performance_window_months": cfg["target"]["performance_window_months"],
            "split_table": split_table,
        },
        "iv_table": encoder.iv_table(),
        "woe_detail": encoder.detail_,
        "model_metrics": model_metrics,
        "vintage": vintage,
        "feature_psi": feature_psi,
        "calibration": calibration,
        "deciles": deciles,
        "approval": approval,
        "coefficients": coefficients,
        "roc_curves": roc_curves,
        "oot_target": targets["oot"].to_numpy(),
        "oot_prob": oot_prob,
        "limitations": limitations,
    }

    if write_report:
        report_path = resolve(cfg, cfg["report"]["output"])
        figures_dir = resolve(cfg, cfg["report"]["figures_dir"])
        build_report(results, report_path, figures_dir)
        results["report_path"] = str(report_path)
        if verbose:
            print(f"[report] written to {report_path}")

    if verbose:
        print("\n[metrics]")
        print(model_metrics.pivot(index="split", columns="model", values="auc").round(4).to_string())
    return results


def run_all_sources(cfg: Dict[str, Any]) -> None:
    """Convenience helper used by tests / notebooks."""
    for src in ("demo",):
        run(cfg, source=src)
