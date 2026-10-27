"""Data loaders. Each loader returns a frame following :mod:`cdr.schema`."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..config import ensure_dir, resolve


def load_data(cfg: Dict[str, Any], source: str | None = None) -> pd.DataFrame:
    """Load the configured data source as a schema-compliant frame."""
    source = source or cfg["data"]["source"]
    data_cfg = cfg["data"]

    if source == "demo":
        from .demo import make_demo

        out_dir = ensure_dir(resolve(cfg, data_cfg["demo_dir"]))
        df = make_demo(
            n_per_vintage=int(data_cfg.get("demo_rows_per_vintage", 900)),
            seed=int(cfg["project"]["seed"]),
        )
        df.to_csv(out_dir / "demo_loans.csv", index=False)
        return df

    if source == "sba":
        from .sba import load_sba

        files = [resolve(cfg, f) for f in data_cfg["sba"].get("files", [])]
        files = [f for f in files if f is not None]
        if not files:
            raise FileNotFoundError(
                "No SBA files configured. Download the 7(a)/504 FOIA CSVs from "
                "https://data.sba.gov/dataset/7-a-504-foia and list them under "
                "data.sba.files in config.yaml."
            )
        return load_sba(files, max_vintage_year=int(data_cfg["sba"].get("max_vintage_year", 2020)))

    if source == "freddie":
        from .freddie import load_freddie

        raw_dir = resolve(cfg, data_cfg["raw_dir"]) / "freddie"
        return load_freddie(
            raw_dir,
            years=data_cfg["freddie"].get("vintage_years"),
            orig_prefix=data_cfg["freddie"].get("orig_prefix", "sample_orig_"),
            perf_prefix=data_cfg["freddie"].get("perf_prefix", "sample_svcg_"),
            performance_window_months=int(
                cfg.get("target", {}).get("performance_window_months", 36)
            ),
        )

    raise ValueError(f"Unknown data source: {source!r}")
