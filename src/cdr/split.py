"""Sampling: vintage-based out-of-time (OOT) splits.

Random splits flatter a credit model. In production the model is fitted on
loans booked up to some date and then scored on loans booked later, in a
different rate and unemployment environment. Splitting by origination vintage
reproduces that setting, so the OOT numbers are the ones worth quoting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

TRAIN, VALID, OOT = "train", "valid", "oot"


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    oot: pd.DataFrame

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {TRAIN: self.train, VALID: self.valid, OOT: self.oot}


def vintage_split(df: pd.DataFrame, cfg: Dict[str, Any]) -> Split:
    """Split by origination year using the cohorts listed in the config."""
    split_cfg = cfg["split"]
    tr_years = set(split_cfg["train_vintages"])
    va_years = set(split_cfg["valid_vintages"])
    oot_years = set(split_cfg["oot_vintages"])

    train = df[df["vintage_year"].isin(tr_years)]
    valid = df[df["vintage_year"].isin(va_years)]
    oot = df[df["vintage_year"].isin(oot_years)]

    if min(len(train), len(valid), len(oot)) == 0:
        raise ValueError(
            "Vintage split produced an empty split. Check that the loaded data "
            "actually contains the vintages listed under `split` in config.yaml."
        )
    return Split(train=train, valid=valid, oot=oot)


def random_split(df: pd.DataFrame, seed: int = 42, fracs=(0.6, 0.2, 0.2)) -> Split:
    """Fallback random split, kept for comparison with the OOT split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * fracs[0])
    n_valid = int(len(df) * fracs[1])
    return Split(
        train=df.iloc[idx[:n_train]].copy(),
        valid=df.iloc[idx[n_train : n_train + n_valid]].copy(),
        oot=df.iloc[idx[n_train + n_valid :]].copy(),
    )


def make_split(df: pd.DataFrame, cfg: Dict[str, Any]) -> Split:
    strategy = cfg["split"].get("strategy", "vintage")
    if strategy == "random":
        return random_split(df, seed=int(cfg["project"]["seed"]))
    return vintage_split(df, cfg)
