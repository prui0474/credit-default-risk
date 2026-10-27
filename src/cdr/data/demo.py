"""Synthetic smoke-test data.

This exists so that the whole pipeline (features -> models -> stability report)
can be exercised end to end in a few seconds, on any machine, without a data
licence. It is **not** a substitute for real data: every headline number in the
report must be produced from Freddie Mac or SBA data.

The generator deliberately builds in a credit cycle (a 2020 stress vintage) so
that out-of-time degradation is visible and the stability machinery has
something to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import CATEGORICAL_FEATURES, LOAN_ID, NUMERIC_FEATURES, TARGET

STATES = ["CA", "TX", "FL", "NY", "WA", "IL", "NJ", "AZ", "GA", "NC", "OH", "PA"]
PURPOSES = ["Purchase", "Rate/Term Refinance", "Cash-out Refinance"]
OCCUPANCY = ["Primary Residence", "Second Home", "Investment Property"]
PROPERTY_TYPES = ["Single Family", "Condominium", "Townhouse", "Manufactured Housing"]
CHANNELS = ["Retail", "Broker", "Correspondent"]

# Macro overlay: rate environment and credit-cycle stress by vintage.
_RATE_SHIFT = {2015: 0.0, 2016: -0.10, 2017: 0.0, 2018: 0.45, 2019: 0.10, 2020: -0.75, 2021: -0.60}
_CYCLE_SHIFT = {2015: 0.0, 2016: -0.05, 2017: 0.0, 2018: 0.10, 2019: 0.15, 2020: 0.85, 2021: 0.35}


def make_demo(
    n_per_vintage: int = 900,
    years: range | list[int] = range(2015, 2022),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a schema-compliant synthetic loan book."""
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for year in years:
        n = int(n_per_vintage)
        fico = np.clip(rng.normal(722, 62, n), 300, 850).round(0)
        ltv = np.clip(rng.normal(76, 13, n), 30, 105)
        cltv = np.clip(ltv + rng.gamma(2.0, 1.6, n), 30, 130)
        dti = np.clip(rng.normal(35, 10, n), 5, 60)
        purpose = rng.choice(PURPOSES, n, p=[0.45, 0.35, 0.20])
        occupancy = rng.choice(OCCUPANCY, n, p=[0.86, 0.06, 0.08])
        property_type = rng.choice(PROPERTY_TYPES, n, p=[0.62, 0.22, 0.11, 0.05])
        state = rng.choice(STATES, n)
        channel = rng.choice(CHANNELS, n, p=[0.45, 0.30, 0.25])
        num_borrowers = rng.choice([1.0, 2.0], n, p=[0.60, 0.40])
        mi_pct = np.where(ltv > 80, rng.choice([6.0, 12.0, 25.0], n), 0.0)
        term_months = rng.choice([360.0, 240.0, 180.0], n, p=[0.82, 0.10, 0.08])

        base_rate = 3.40 + 0.90 * (720 - fico) / 100.0 + 0.035 * (ltv - 75.0)
        orig_rate = np.round(
            np.clip(base_rate + _RATE_SHIFT.get(year, 0.0) + rng.normal(0, 0.18, n), 2.4, 9.5), 3
        )
        orig_upb = np.round(np.clip(rng.lognormal(12.2, 0.55, n), 25_000, 1_500_000), -3)

        z = (
            -3.55
            - 0.020 * (fico - 722.0)
            + 0.042 * (ltv - 76.0)
            + 0.022 * (dti - 35.0)
            + 0.90 * (occupancy == "Investment Property")
            + 0.45 * (purpose == "Cash-out Refinance")
            + 0.25 * (channel == "Broker")
            + _CYCLE_SHIFT.get(year, 0.0)
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        target = rng.binomial(1, prob)

        # Time to event: defaults happen inside the performance window.
        months_observed = np.where(
            target == 1, rng.integers(6, 37, n), rng.integers(24, 121, n)
        )
        zero_balance_code = np.where(target == 1, "03", "01")

        frames.append(
            pd.DataFrame(
                {
                    LOAN_ID: [f"DEMO{year}{i:06d}" for i in range(n)],
                    "vintage_year": year,
                    "vintage_month": rng.integers(1, 13, n),
                    "fico": fico,
                    "dti": dti.round(1),
                    "ltv": ltv.round(1),
                    "cltv": cltv.round(1),
                    "orig_upb": orig_upb,
                    "orig_rate": orig_rate,
                    "term_months": term_months,
                    "num_borrowers": num_borrowers,
                    "mi_pct": mi_pct,
                    "loan_purpose": purpose,
                    "occupancy": occupancy,
                    "property_type": property_type,
                    "state": state,
                    "channel": channel,
                    "months_observed": months_observed,
                    "ever_90dpd": target,
                    "zero_balance_code": zero_balance_code,
                    TARGET: target,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    return df[[LOAN_ID, "vintage_year", "vintage_month"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
              + ["months_observed", "ever_90dpd", "zero_balance_code", TARGET]]
