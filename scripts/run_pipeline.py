#!/usr/bin/env python
"""Run the whole pipeline from the command line.

Examples
--------
    python scripts/run_pipeline.py                       # synthetic smoke test
    python scripts/run_pipeline.py --source sba          # real SBA data
    python scripts/run_pipeline.py --source freddie      # real mortgage data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdr.config import load_config  # noqa: E402
from cdr.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credit default risk pipeline.")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--source", default=None, choices=["demo", "freddie", "sba"])
    parser.add_argument("--no-report", action="store_true", help="skip markdown/figures")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = run(cfg, source=args.source, write_report=not args.no_report)

    oot = results["model_metrics"].query("split == 'oot'")
    for _, row in oot.iterrows():
        print(f"  OOT {row['model']:<10} AUC={row['auc']:.4f}  Gini={row['gini']:.4f}  KS={row['ks']:.4f}")
    if "report_path" in results:
        print(f"  report: {results['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
