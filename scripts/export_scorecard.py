"""Export a fitted scorecard to JSON for the static demo page.

The demo runs in a browser with no Python behind it, so the fitted model has to
travel as data: bin edges, weight-of-evidence maps and the logistic coefficients.
Nothing here is loan-level. What leaves this script is a parameter file of a few
kilobytes, which is why the 2 GB of raw Freddie Mac files stay where they are.

The script ends with a self-check: it re-scores the out-of-time cohort from the
exported JSON alone and compares that against scikit-learn's predict_proba. If the
two disagree the JavaScript port would disagree too, and the demo would be lying.

Usage:
    python scripts/export_scorecard.py                 # reads data/processed
    python scripts/export_scorecard.py --source demo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdr.config import load_config, resolve  # noqa: E402
from cdr.evaluate import auc_gini_ks, calibration_table, decile_table  # noqa: E402
from cdr.features import MISSING_LABEL, OTHER_LABEL, WOEEncoder  # noqa: E402
from cdr.models import fit_scorecard  # noqa: E402
from cdr.schema import TARGET  # noqa: E402
from cdr.split import make_split  # noqa: E402

OUT_PATH = ROOT / "docs" / "demo" / "scorecard.json"


def _finite(value: float) -> float | None:
    """JSON has no infinity; edges that are open at either end become null."""
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="freddie", help="processed cohort to fit on")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "config.json"))
    processed = resolve(cfg, cfg["data"]["processed_dir"]) / f"{args.source}_unified.csv"
    if not processed.exists():
        raise SystemExit(f"missing {processed}; run scripts/run_pipeline.py first")

    df = pd.read_csv(processed)
    split = make_split(df, cfg)

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    encoder = WOEEncoder(numeric, categorical, max_bins=int(cfg["features"]["woe_max_bins"]))
    encoder.fit(split.train, TARGET)

    matrices = {name: encoder.transform(part) for name, part in split.as_dict().items()}
    targets = {name: part[TARGET].astype(int) for name, part in split.as_dict().items()}
    model = fit_scorecard(matrices["train"], targets["train"], **cfg["model"]["scorecard"])

    coef = dict(zip(matrices["train"].columns, model.estimator.coef_.ravel().tolist()))
    intercept = float(model.estimator.intercept_[0])

    features = []
    for col in numeric:
        if col not in encoder.edges_:
            continue
        edges = [_finite(e) for e in encoder.edges_[col]]
        codes = encoder.maps_[col]
        woe = [float(codes[i]) for i in range(len(codes))]
        values = pd.to_numeric(split.train[col], errors="coerce").dropna()
        features.append(
            {
                "name": col,
                "type": "numeric",
                "edges": edges,
                "woe": woe,
                "iv": float(encoder.iv_[col]),
                "beta": coef[col],
                "stats": {
                    "p01": float(values.quantile(0.01)),
                    "p50": float(values.median()),
                    "p99": float(values.quantile(0.99)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                },
            }
        )
    for col in categorical:
        mapping = {str(k): float(v) for k, v in encoder.maps_[col].items()}
        levels = [k for k in mapping if k not in (MISSING_LABEL, OTHER_LABEL)]
        features.append(
            {
                "name": col,
                "type": "categorical",
                "woe": mapping,
                "iv": float(encoder.iv_[col]),
                "beta": coef[col],
                "levels": sorted(levels),
            }
        )

    # --- Re-score from the exported structure only ---------------------------
    def score_rows(frame: pd.DataFrame) -> np.ndarray:
        """Mirror of what the page's JavaScript does, written independently."""
        logit = np.full(len(frame), intercept, dtype=float)
        for spec in features:
            col = spec["name"]
            if spec["type"] == "numeric":
                raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
                edges = [
                    (-np.inf if i == 0 else np.inf if e is None else e)
                    for i, e in enumerate(spec["edges"])
                ]
                codes = np.full(len(raw), len(edges) - 1, dtype=int)
                for i in range(len(edges) - 1):
                    if i == 0:
                        hit = raw <= edges[1]
                    else:
                        hit = (raw > edges[i]) & (raw <= edges[i + 1])
                    codes = np.where(hit, i, codes)
                woe = np.array(spec["woe"], dtype=float)[codes]
            else:
                text = frame[col].astype("object").fillna(MISSING_LABEL).astype(str)
                mapping = spec["woe"]
                fallback = mapping.get(OTHER_LABEL, 0.0)
                woe = text.map(lambda v: mapping.get(v, fallback)).to_numpy(dtype=float)
            logit += spec["beta"] * woe
        return 1.0 / (1.0 + np.exp(-logit))

    check_rows = []
    for name in ("train", "valid", "oot"):
        part = split.as_dict()[name]
        ref = model.predict_proba(matrices[name])
        alt = score_rows(part)
        m = auc_gini_ks(targets[name], alt)
        check_rows.append(
            {
                "split": name,
                "n": int(len(part)),
                "auc": round(float(m["auc"]), 4),
                "max_abs_diff_vs_sklearn": float(np.max(np.abs(ref - alt))),
            }
        )
    worst = max(r["max_abs_diff_vs_sklearn"] for r in check_rows)
    if worst > 1e-9:
        raise SystemExit(f"export is not faithful to the fitted model: diff {worst:.3e}")

    # --- Operating curve for the cut-off slider -----------------------------
    oot_pd = score_rows(split.oot)
    y = targets["oot"].to_numpy()
    curve = []
    for cutoff in np.round(np.arange(0.005, 0.205, 0.005), 3):
        approved = oot_pd <= cutoff
        if approved.sum() == 0:
            continue
        curve.append(
            {
                "cutoff": float(cutoff),
                "approval": round(float(approved.mean()), 4),
                "approved_bad": round(float(y[approved].mean()), 5),
                "rejected_bad": round(float(y[~approved].mean()), 5) if (~approved).any() else None,
                "bad_capture": round(float(y[~approved].sum() / max(y.sum(), 1)), 4),
            }
        )

    deciles = decile_table(y, oot_pd).to_dict(orient="records")
    calibration = calibration_table(y, oot_pd).to_dict(orient="records")
    under = sum(1 for row in calibration if row["actual"] > row["predicted"])

    payload = {
        "meta": {
            "trained_on": f"{args.source} vintages "
            f"{int(split.train['vintage_year'].min())}-{int(split.train['vintage_year'].max())}",
            "n_train": int(len(split.train)),
            "n_oot": int(len(split.oot)),
            "bad_rate_train": round(float(targets["train"].mean()), 4),
            "bad_rate_oot": round(float(y.mean()), 4),
            "target": cfg["target"]["definition"],
            "window_months": int(cfg["target"]["performance_window_months"]),
            "exported": date.today().isoformat(),
            "regularisation_C": float(cfg["model"]["scorecard"]["C"]),
        },
        "intercept": intercept,
        "features": features,
        "checks": check_rows,
        "cutoff_curve": curve,
        "pd_quantiles": [
            round(float(q), 5) for q in np.quantile(oot_pd, np.linspace(0.0, 1.0, 101))
        ],
        "deciles_oot": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in deciles
        ],
        "calibration_oot": [
            {k: round(float(v), 5) for k, v in row.items()} for row in calibration
        ],
        "calibration_bands_under": under,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[export] {OUT_PATH.relative_to(ROOT)}  {size_kb:.1f} KB")
    for row in check_rows:
        print(
            f"  {row['split']:<5} n={row['n']:>7,}  auc={row['auc']:.4f}  "
            f"max|json-sklearn|={row['max_abs_diff_vs_sklearn']:.2e}"
        )
    print(f"[export] {len(features)} features, intercept {intercept:+.4f}")
    for spec in sorted(features, key=lambda s: -abs(s["beta"])):
        print(f"    {spec['name']:<15} beta={spec['beta']:+.4f}  IV={spec['iv']:.3f}")


if __name__ == "__main__":
    main()
