"""SBA 7(a) / 504 FOIA loan loader.

Source: https://data.sba.gov/dataset/7-a-504-foia (public domain, quarterly
refresh, loan-level records since FY1991).

Why this source is useful
-------------------------
* Free, no registration, plain CSV, refreshed quarterly (latest vintage
  reaches the current fiscal year).
* Real money movement with a hard outcome label (``loanstatus == 'CHGOFF'``
  plus ``chargeoffdate``).

Its limitation is equally important for the write-up: there is no bureau
score, no DTI and no collateral value, so the achievable discrimination is
lower than with mortgage data. That contrast is itself a finding worth
reporting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..schema import empty_frame_columns

# Only matured outcomes are usable as labels.
_KEEP_STATUS = {"PIF", "CHGOFF"}
# A loan needs time to go bad; cohorts younger than this stay censored.
_MIN_SEASONING_YEARS = 4


def _series(df: pd.DataFrame, name: str, default: object = np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]
    return df


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, low_memory=False)
    return pd.read_csv(path, dtype=str, low_memory=False)


def load_sba(
    files: Sequence[Path] | Iterable[Path],
    max_vintage_year: int = 2020,
    current_year: int | None = None,
) -> pd.DataFrame:
    """Load SBA FOIA CSVs into the unified schema.

    Parameters
    ----------
    files:
        One or more of the published CSVs (they are split by decade).
    max_vintage_year:
        Latest approval cohort to keep. The default leaves at least
        ``_MIN_SEASONING_YEARS`` of observation time, which avoids the classic
        mistake of comparing a fresh cohort with a matured one.
    """
    frames = [_norm_cols(_read_one(Path(f))) for f in files]
    raw = pd.concat(frames, ignore_index=True)

    status = _series(raw, "loanstatus", "").astype(str).str.upper().str.strip()
    approval_fy = pd.to_numeric(_series(raw, "approvalfy"), errors="coerce")
    approval_date = pd.to_datetime(_series(raw, "approvaldate"), errors="coerce", format="mixed")

    out = pd.DataFrame(index=raw.index)
    out["loan_id"] = [
        hashlib.md5(
            f"{n}|{d}|{a}".encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        for n, d, a in zip(
            _series(raw, "borrname", "").astype(str),
            _series(raw, "approvaldate", "").astype(str),
            _series(raw, "grossapproval", "").astype(str),
        )
    ]
    out["vintage_year"] = approval_fy.astype("Float64").astype("Int64")
    out["vintage_month"] = approval_date.dt.month.astype("Int64")

    # Attributes SBA does not publish -> explicitly left empty, not imputed.
    out["fico"] = np.nan
    out["dti"] = np.nan
    out["ltv"] = np.nan
    out["cltv"] = np.nan
    out["mi_pct"] = np.nan
    out["num_borrowers"] = np.nan

    out["orig_upb"] = pd.to_numeric(_series(raw, "grossapproval"), errors="coerce")
    out["orig_rate"] = pd.to_numeric(_series(raw, "initialinterestrate"), errors="coerce")
    out["term_months"] = pd.to_numeric(_series(raw, "terminmonths"), errors="coerce")
    out["loan_purpose"] = _series(raw, "subprogram", "Unknown").astype(str).str.strip()
    out["occupancy"] = "Unknown"
    out["property_type"] = _series(raw, "businessage", "Unknown").astype(str).str.strip()
    out["state"] = _series(raw, "projectstate", "Unknown").astype(str).str.strip()
    out["channel"] = _series(raw, "bankstate", "Unknown").astype(str).str.strip()

    charge_off_date = pd.to_datetime(_series(raw, "chargeoffdate"), errors="coerce", format="mixed")
    out["zero_balance_code"] = np.where(status == "CHGOFF", "03", np.where(status == "PIF", "01", ""))
    out["ever_90dpd"] = (status == "CHGOFF").astype(int)
    out["target"] = (status == "CHGOFF").astype(int)

    # Seasoning: months from approval to charge-off (or to observation end).
    obs_end = pd.Timestamp(year=int(current_year or (max_vintage_year + _MIN_SEASONING_YEARS)), month=12, day=31)
    months_to_charge_off = (charge_off_date - approval_date).dt.days / 30.44
    months_to_obs = (obs_end - approval_date).dt.days / 30.44
    out["months_observed"] = months_to_charge_off.fillna(months_to_obs).round(0)

    keep = status.isin(_KEEP_STATUS) & approval_fy.notna() & (approval_fy <= int(max_vintage_year))
    out = out[keep].reset_index(drop=True)

    out["vintage_year"] = out["vintage_year"].astype(int)
    out["vintage_month"] = out["vintage_month"].fillna(1).astype(int)
    out = out.dropna(subset=["orig_upb"])

    return out[empty_frame_columns()]
