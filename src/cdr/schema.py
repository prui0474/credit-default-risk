"""The unified column contract every data loader must satisfy.

Rationale: the pipeline is data-source agnostic. Freddie Mac mortgage data,
SBA small-business loans and the synthetic smoke-test data all get mapped
onto this schema so that features, models and reports stay unchanged when a
new source is plugged in.

Feature availability rule
-------------------------
Only *origination-time* attributes are used as model inputs. Anything that is
only known after the loan has been running (payments, recoveries, delinquency
status) is reserved for the target definition and is never fed to the model.
This is the standard anti-leakage rule in credit scoring.
"""

from __future__ import annotations

LOAN_ID = "loan_id"
VINTAGE_YEAR = "vintage_year"
VINTAGE_MONTH = "vintage_month"
TARGET = "target"

# Inputs available at application / origination time.
NUMERIC_FEATURES = [
    "fico",          # borrower credit score at origination
    "dti",           # debt-to-income ratio (%)
    "ltv",           # loan-to-value at origination (%)
    "cltv",          # combined loan-to-value (%)
    "orig_upb",      # original unpaid principal balance
    "orig_rate",     # note rate at origination (%)
    "term_months",   # contractual amortisation term
    "num_borrowers", # number of borrowers
    "mi_pct",        # mortgage insurance coverage (%)
]

CATEGORICAL_FEATURES = [
    "loan_purpose",
    "occupancy",
    "property_type",
    "state",
    "channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Non-model columns kept for auditability / slicing.
AUDIT_COLUMNS = [
    LOAN_ID,
    VINTAGE_YEAR,
    VINTAGE_MONTH,
    "months_observed",
    "ever_90dpd",
    "zero_balance_code",
    TARGET,
]

ALL_COLUMNS = AUDIT_COLUMNS + FEATURES


def empty_frame_columns() -> list[str]:
    """Column order used when materialising a processed dataset."""
    return [LOAN_ID, VINTAGE_YEAR, VINTAGE_MONTH] + FEATURES + [
        "months_observed",
        "ever_90dpd",
        "zero_balance_code",
        TARGET,
    ]
