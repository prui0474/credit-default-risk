"""Report assembly: figures plus a markdown model report.

Everything the pipeline prints is also written down, because "the notebook
cell said AUC 0.81" is not evidence you can hand to a reviewer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

GOOD, BAD = "#4C78A8", "#E45756"


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small dependency-free markdown table writer."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def fig_roc(curves: Dict[str, tuple], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.6, label=label)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#888")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - out-of-time cohort")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_score_dist(y: np.ndarray, p: np.ndarray, out_path: Path, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist(p[np.asarray(y) == 0], bins=40, alpha=0.65, label="Non-default", color=GOOD, density=True)
    ax.hist(p[np.asarray(y) == 1], bins=40, alpha=0.65, label="Default", color=BAD, density=True)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title(f"Score separation - {name} (OOT)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_vintage(vintage: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(vintage))
    width = 0.38
    ax.bar(x - width / 2, vintage["scorecard_auc"], width, label="Scorecard", color=GOOD)
    ax.bar(x + width / 2, vintage["gbm_auc"], width, label="GBM", color=BAD)
    ax.set_xticks(x)
    ax.set_xticklabels(vintage["vintage_year"].astype(int).astype(str))
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Origination vintage")
    ax.set_ylabel("AUC")
    ax.set_title("Discrimination by vintage (generalisation over time)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_calibration(calib: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(calib["predicted"], calib["actual"], "o-", ms=3.5, color=BAD, label="Observed")
    lim = [0, max(calib["actual"].max(), calib["predicted"].max()) * 1.1 + 1e-3]
    ax.plot(lim, lim, ls="--", lw=0.8, color="#888", label="Perfect calibration")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration - OOT")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_psi(psi_tbl: pd.DataFrame, out_path: Path, top: int = 12) -> Path:
    sub = psi_tbl.dropna(subset=["psi"]).nlargest(top, "psi").sort_values("psi")
    fig, ax = plt.subplots(figsize=(4.8, max(2.4, 0.28 * len(sub) + 1.2)))
    colors = ["#E45756" if v >= 0.25 else "#F2B134" if v >= 0.10 else "#4C78A8" for v in sub["psi"]]
    ax.barh(sub["feature"], sub["psi"], color=colors, height=0.62)
    ax.axvline(0.10, ls="--", lw=0.8, color="#888")
    ax.axvline(0.25, ls="--", lw=0.8, color="#888")
    ax.set_xlabel("PSI (train vs OOT)")
    ax.set_title("Feature population stability")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_deciles(deciles: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(deciles["band"], deciles["bad_rate"] * 100, color=BAD)
    ax.set_xlabel("Risk band (D1 = riskiest)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Risk ranking - OOT")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_report(results: Dict[str, Any], out_path: Path, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    meta = results["meta"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    fig_roc(results["roc_curves"], figures_dir / "01_roc_oot.png")
    fig_score_dist(results["oot_target"], results["oot_prob"]["gbm"], figures_dir / "02_score_separation.png", "GBM")
    fig_vintage(results["vintage"], figures_dir / "03_auc_by_vintage.png")
    fig_calibration(results["calibration"]["gbm"], figures_dir / "04_calibration.png")
    fig_psi(results["feature_psi"], figures_dir / "05_feature_psi.png")
    fig_deciles(results["deciles"]["gbm"], figures_dir / "06_risk_bands.png")

    parts: list[str] = []
    parts.append(f"# Credit Default Risk - Model Report\n")
    parts.append(f"*Data source:* `{meta['source']}` &nbsp;|&nbsp; *Rows:* {meta['n_rows']:,} "
                 f"&nbsp;|&nbsp; *Defaults:* {meta['bad_rate']:.2%} &nbsp;|&nbsp; *Generated:* {stamp}\n")

    parts.append("## 1. Target and sample\n")
    parts.append(f"- **Target definition:** `{meta['target_definition']}` - a loan is a default if it "
                 f"reaches 90+ days past due within {meta['performance_window_months']} months of origination.")
    parts.append("- **Model inputs:** origination-time attributes only. Payment, recovery and "
                 "delinquency fields are excluded to prevent target leakage.")
    parts.append("- **Split:** out-of-time by origination vintage, so the test cohort is booked in a "
                 "different rate environment than the training cohort.\n")
    parts.append(_md_table(meta["split_table"]))
    parts.append("")

    parts.append("## 2. Feature screening (WOE / IV)\n")
    parts.append("Information value on the training cohort: <0.02 useless, 0.02-0.1 weak, "
                 "0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious (often a leakage smell).\n")
    parts.append(_md_table(results["iv_table"]))
    parts.append("")

    parts.append("## 3. Discrimination\n")
    parts.append(_md_table(results["model_metrics"]))
    parts.append("")
    parts.append("Gini = 2*AUC-1. KS is the maximum gap between the cumulative default and "
                 "non-default score distributions.\n")

    parts.append("## 4. Stability over time\n")
    parts.append("A model that holds up in-sample but decays across vintages is not deployable. "
                 "PSI below 0.10 is stable, 0.10-0.25 needs monitoring, above 0.25 means the "
                 "population has moved.\n")
    parts.append(_md_table(results["vintage"]))
    parts.append("")
    parts.append("Feature-level PSI (train vs OOT):\n")
    parts.append(_md_table(results["feature_psi"].sort_values("psi", ascending=False).head(12)))
    parts.append("")

    parts.append("## 5. Calibration\n")
    parts.append("Ranking power alone is not enough: an uncalibrated PD mis-prices the loan and "
                 "mis-states expected loss.\n")
    parts.append(_md_table(results["calibration"]["gbm"]))
    parts.append("")

    parts.append("## 6. Decision layer\n")
    parts.append("Cutoff scenarios on the out-of-time cohort (approve every applicant below the PD "
                 "cutoff):\n")
    parts.append(_md_table(results["approval"]))
    parts.append("")

    parts.append("## 7. Risk ranking\n")
    parts.append(_md_table(results["deciles"]["gbm"]))
    parts.append("")

    if "coefficients" in results:
        parts.append("## 8. Scorecard coefficients\n")
        parts.append(_md_table(results["coefficients"]))
        parts.append("")

    parts.append("## 9. Limitations\n")
    for line in results["limitations"]:
        parts.append(f"- {line}")
    parts.append("")
    parts.append("## Figures\n")
    for name in ["01_roc_oot.png", "02_score_separation.png", "03_auc_by_vintage.png",
                 "04_calibration.png", "05_feature_psi.png", "06_risk_bands.png"]:
        parts.append(f"![{name}](figures/{name})")
    parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
