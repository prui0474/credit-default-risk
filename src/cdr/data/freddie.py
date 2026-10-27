"""Freddie Mac Single-Family Loan-Level Dataset loader.

Source
------
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

* Free for academic / non-commercial use, but the bulk files sit behind a free
  registration (Clarity Data Intelligence).
* Pipe-delimited, **no header row**. The official ``file_layout.xlsx`` is only a
  guide, and it is currently out of date: the sample files actually served for
  the 2015-2021 vintages hold **31 fields** for the origination file and
  **35 fields** for the performance file, not the published 32/32. The
  origination file has dropped the published field 25 ("Servicer Name"), and the
  performance file carries three extra columns after field 32.
* Every field this loader needs therefore sits at the **same position** in both
  the published and the served layout, so columns are selected **by position,
  never by name**. Selecting by name is what silently breaks when a publisher
  adds or removes a column.
* Loan Sequence Number is field 20 of the origination file and field 1 of the
  performance file - that is the join key. It is pattern-checked on read, so a
  future layout change fails loudly instead of quietly producing wrong labels.
* The "sample" files (50,000 loans per vintage year) are the right starting
  point on a laptop: one origination file plus one performance file per year.

Default flag (this is a modelling choice, not a Freddie Mac field)
------------------------------------------------------------------
A loan is labelled as defaulted when, inside the performance window, it
either reaches 90+ days past due (delinquency status >= 3) or terminates with
a credit-event zero balance code (third-party sale, charge-off, REO
disposition, note sale).

Sentinel values ("Not Available") are mapped to NaN, never to zero: 9999 for
the credit score, 999 for CLTV/DTI/LTV/MI%, 99 for units and property type.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..schema import empty_frame_columns

# --- Where every field we care about sits (1-based) ---------------------------
#
# Select BY POSITION, never by name. The published ``file_layout.xlsx`` lists 32
# fields for each file, but the sample files actually served in 2026 have drifted:
#
#     origination : 31 fields  (published field 25, "Servicer Name", is gone)
#     performance : 35 fields  (three columns appended after published field 32)
#
# Positions 1-24 of the origination file and 1-32 of the performance file are
# identical in both layouts, which is what makes positional selection safe.
# Verified 2026-10-15 against sample_2015 .. sample_2021: the field count is
# constant across every row of every file, the join key sits at origination
# field 20 / performance field 1, and delinquency + zero balance sit at
# performance fields 4 and 9.
#
# Anchors taken from the first row of sample_2015:
#   1=734  2=201504  6=0  8=P  9=80  10=15  11=417000  12=65  13=2.5
#   14=R  17=CO  18=SF  20=F15Q10000025  21=P  22=180  23=2

ORIG_POS: dict[str, int] = {
    "fico": 1,
    "first_payment_date": 2,
    "mi_pct": 6,
    "occupancy": 8,
    "cltv": 9,
    "dti": 10,
    "orig_upb": 11,
    "ltv": 12,
    "orig_rate": 13,
    "channel": 14,
    "state": 17,
    "property_type": 18,
    "loan_id": 20,
    "loan_purpose": 21,
    "term_months": 22,
    "num_borrowers": 23,
}

PERF_POS: dict[str, int] = {
    "loan_id": 1,
    "delinq": 4,
    "loan_age": 5,
    "zero_balance": 9,
}

# Refuse to guess if a file is shorter than the layout this loader understands.
ORIG_MIN_FIELDS = 23
PERF_MIN_FIELDS = 9

# A loan whose first reported month is later than this entered the sample after
# origination, so its opening months are missing and its default label would be
# fabricated. 1 tolerates the usual month-0/month-1 reporting convention.
MAX_UNOBSERVED_START_MONTH = 1

# Loan Sequence Number format: PYYQnXXXXXXX, e.g. F15Q10000025.
LOAN_ID_PATTERN = re.compile(r"^[A-Za-z]\d{2}Q\d+$")

# Terminations that represent a credit event rather than a clean payoff.
LOSS_ZERO_BALANCE_CODES = {"02", "03", "09", "96"}

OCCUPANCY_MAP = {"P": "Primary Residence", "I": "Investment Property", "S": "Second Home"}
CHANNEL_MAP = {"R": "Retail", "B": "Broker", "C": "Correspondent", "T": "TPO Unspecified"}
PROPERTY_MAP = {
    "SF": "Single Family",
    "CO": "Condominium",
    "PU": "Planned Unit Development",
    "MH": "Manufactured Housing",
    "CP": "Co-operative",
}
PURPOSE_MAP = {"P": "Purchase", "C": "Cash-out Refinance", "N": "No Cash-out Refinance"}

# Sentinel ("Not Available") codes, keyed by the short names this loader uses.
_SENTINELS = {"cltv": 999, "dti": 999, "ltv": 999, "mi_pct": 999}


def _ordered_positions(mapping: dict[str, int]) -> tuple[list[int], list[str]]:
    """0-based ``usecols`` indices plus matching names, both in ascending position.

    pandas returns ``usecols`` columns in *file* order, so the names must be
    applied in that same ascending order or every column lands on a wrong field.
    """
    ordered = sorted(mapping.items(), key=lambda item: item[1])
    return [pos - 1 for _, pos in ordered], [name for name, _ in ordered]


def _field_count(path: Path) -> int:
    with path.open(encoding="latin-1") as handle:
        return len(handle.readline().rstrip("\n").split("|"))


def _check_layout(path: Path, min_fields: int, kind: str) -> None:
    """Guard rail: the publisher has changed this layout before, without notice."""
    count = _field_count(path)
    if count < min_fields:
        raise ValueError(
            f"{path.name}: {kind} file holds only {count} pipe-delimited fields "
            f"but this loader needs at least {min_fields}. Compare "
            "docs/refs/file_layout.xlsx against the raw file, then update "
            "ORIG_POS / PERF_POS in src/cdr/data/freddie.py."
        )


def _check_loan_ids(frame: pd.DataFrame, where: str, position: int, sample: int = 500) -> None:
    """Guard rail: fail loudly if the join key is not where we think it is."""
    values = frame["loan_id"].head(sample).astype(str)
    bad = values[~values.str.match(LOAN_ID_PATTERN)]
    if len(bad):
        raise ValueError(
            f"{where}: field {position} does not look like a Freddie Mac loan "
            "sequence number (expected PYYQnXXXXXXX, e.g. F15Q10000025). "
            f"Offending values: {bad.head(3).tolist()}. The file layout has "
            "probably changed again - verify before trusting any output."
        )


def _numeric(series: pd.Series, sentinel: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if sentinel is not None:
        values = values.mask(values == sentinel)
    return values


def _find_files(raw_dir: Path, prefixes: Sequence[str], years: Iterable[int] | None) -> dict[str, Path]:
    """Map each vintage year to the first file matching one of ``prefixes``."""
    wanted = {int(y) for y in years} if years else None
    found: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*.txt")):
        name = path.name
        if not any(name.startswith(p) for p in prefixes):
            continue
        digits = "".join(ch for ch in name.replace(".txt", "") if ch.isdigit())
        if len(digits) < 4:
            continue
        year = digits[-4:]
        if wanted is not None and int(year) not in wanted:
            continue
        found.setdefault(year, path)
    return found


def _read_orig(path: Path) -> pd.DataFrame:
    """Read the origination file, selecting fields by position (never by name)."""
    _check_layout(path, ORIG_MIN_FIELDS, "origination")
    positions, names = _ordered_positions(ORIG_POS)
    frame = pd.read_csv(
        path, sep="|", header=None, usecols=positions, dtype=str,
        encoding="latin-1", na_filter=False, low_memory=False,
    )
    frame.columns = names
    _check_loan_ids(frame, path.name, ORIG_POS["loan_id"])
    return frame


def _read_perf(path: Path, window_months: int, chunksize: int = 500_000) -> pd.DataFrame:
    """Loan-level outcome summary, aggregated in chunks to stay memory-safe.

    One row per loan: earliest observed loan age, worst delinquency inside the
    window, last observed month and final zero balance code.

    ``min_age`` matters. A slice of every vintage only enters the sample long
    after origination (seasoned loans transferred into the portfolio), so their
    opening months are simply absent from the file. Those loans must not be read
    as "no default seen", which is why the earliest observed age is carried
    through and checked by the caller.
    """
    _check_layout(path, PERF_MIN_FIELDS, "performance")
    positions, names = _ordered_positions(PERF_POS)
    reduced: list[pd.DataFrame] = []
    checked = False

    reader = pd.read_csv(
        path, sep="|", header=None, usecols=positions, dtype=str,
        encoding="latin-1", na_filter=False, chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = names
        if not checked:
            _check_loan_ids(chunk, path.name, PERF_POS["loan_id"])
            checked = True
        chunk["delinq"] = pd.to_numeric(chunk["delinq"], errors="coerce")
        chunk["age"] = pd.to_numeric(chunk["loan_age"], errors="coerce")

        part = pd.DataFrame(index=chunk.groupby("loan_id").size().index)
        part.index.name = "loan_id"
        part["min_age"] = chunk.groupby("loan_id")["age"].min()

        inside = chunk[chunk["age"].le(window_months)].sort_values("age")
        if inside.empty:
            part["max_delinq"] = np.nan
            part["months_observed"] = np.nan
            part["zb"] = ""
        else:
            grouped = inside.groupby("loan_id")
            part["max_delinq"] = grouped["delinq"].max()
            part["months_observed"] = grouped["age"].max()
            part["zb"] = grouped["zero_balance"].last()
        reduced.append(part.reset_index())

    if not reduced:
        return pd.DataFrame(
            columns=["loan_id", "min_age", "max_delinq", "months_observed", "zb"]
        )

    combined = pd.concat(reduced, ignore_index=True).sort_values(
        "months_observed", na_position="first"
    )
    grouped = combined.groupby("loan_id")
    final = pd.DataFrame(index=grouped.size().index)
    final.index.name = "loan_id"
    final["min_age"] = grouped["min_age"].min()
    final["max_delinq"] = grouped["max_delinq"].max()
    final["months_observed"] = grouped["months_observed"].max()
    final["zb"] = grouped["zb"].last()
    return final.reset_index()


def load_freddie(
    raw_dir: str | Path,
    years: Sequence[int] | None = None,
    orig_prefix: str = "sample_orig_",
    perf_prefix: str | None = None,
    performance_window_months: int = 36,
) -> pd.DataFrame:
    """Load one or more vintage years of Freddie Mac sample/standard files."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Download the sample files from Clarity "
            "Data Intelligence and unzip them here (see docs/guide-zh.md)."
        )

    perf_prefixes = [perf_prefix] if perf_prefix else ["sample_perf_", "sample_svcg_", "perf_"]
    orig_files = _find_files(raw_dir, [orig_prefix], years)
    perf_files = _find_files(raw_dir, perf_prefixes, years)

    if not orig_files:
        raise FileNotFoundError(
            f"No origination files matching {orig_prefix}*.txt under {raw_dir}."
        )

    frames = []
    for year, orig_path in sorted(orig_files.items()):
        orig = _read_orig(orig_path)
        outcome = pd.DataFrame(
            columns=["loan_id", "min_age", "max_delinq", "months_observed", "zb"]
        )
        if year in perf_files:
            outcome = _read_perf(perf_files[year], window_months=performance_window_months)

        merged = orig.merge(outcome, on="loan_id", how="left")

        out = pd.DataFrame(index=merged.index)
        out["loan_id"] = merged["loan_id"]
        first_payment = pd.to_datetime(merged["first_payment_date"], format="%Y%m", errors="coerce")
        out["vintage_year"] = first_payment.dt.year
        out["vintage_month"] = first_payment.dt.month

        fico = _numeric(merged["fico"])
        out["fico"] = fico.mask((fico < 300) | (fico > 850))
        out["dti"] = _numeric(merged["dti"], _SENTINELS["dti"])
        out["ltv"] = _numeric(merged["ltv"], _SENTINELS["ltv"])
        out["cltv"] = _numeric(merged["cltv"], _SENTINELS["cltv"])
        out["orig_upb"] = _numeric(merged["orig_upb"])
        out["orig_rate"] = _numeric(merged["orig_rate"])
        out["term_months"] = _numeric(merged["term_months"])
        out["num_borrowers"] = _numeric(merged["num_borrowers"])
        out["mi_pct"] = _numeric(merged["mi_pct"], _SENTINELS["mi_pct"])

        out["loan_purpose"] = merged["loan_purpose"].map(PURPOSE_MAP).fillna("Unknown")
        out["occupancy"] = merged["occupancy"].map(OCCUPANCY_MAP).fillna("Unknown")
        out["property_type"] = merged["property_type"].map(PROPERTY_MAP).fillna("Unknown")
        out["state"] = merged["state"].str.strip().replace("", "Unknown")
        out["channel"] = merged["channel"].map(CHANNEL_MAP).fillna("Unknown")

        zb = merged["zb"].fillna("")
        max_delinq = pd.to_numeric(merged["max_delinq"], errors="coerce")
        ever_90dpd = (max_delinq >= 3) | zb.isin(LOSS_ZERO_BALANCE_CODES)
        out["ever_90dpd"] = ever_90dpd.astype(int)
        out["target"] = out["ever_90dpd"]
        out["zero_balance_code"] = zb
        out["months_observed"] = pd.to_numeric(merged["months_observed"], errors="coerce")
        out["min_age"] = pd.to_numeric(merged["min_age"], errors="coerce")

        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["vintage_year"].notna()]
    df["vintage_year"] = df["vintage_year"].astype(int)
    df["vintage_month"] = df["vintage_month"].fillna(1).astype(int)

    # A sample_YYYY file is not the same thing as the YYYY vintage: roughly 16%
    # of each file holds loans whose first payment falls in YYYY+1, and a handful
    # fall further out. Those rows belong to no configured split, so drop them
    # here - visibly - instead of letting the splitter discard them in silence.
    if years:
        wanted = {int(y) for y in years}
        in_scope = df["vintage_year"].isin(wanted)
        out_of_scope = int((~in_scope).sum())
        df = df[in_scope]
    else:
        out_of_scope = 0

    # Observation-integrity filter. Loans whose first reported month is already
    # past month 1 are seasoned loans that joined the sample late; their opening
    # months are not in the file, so labelling them "no default" would invent a
    # clean record. Drop them and expose the count instead of hiding it.
    observed_from_start = df["min_age"].le(MAX_UNOBSERVED_START_MONTH).fillna(False)
    unobserved = int((~observed_from_start).sum())
    df = df[observed_from_start]

    out_frame = df[empty_frame_columns()].reset_index(drop=True)
    out_frame.attrs["unobserved_dropped"] = unobserved
    out_frame.attrs["out_of_scope_dropped"] = out_of_scope
    return out_frame
