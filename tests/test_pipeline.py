"""Smoke tests.

Run either way::

    python tests/test_pipeline.py        # no pytest required
    pytest -q                            # if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cdr.config import load_config  # noqa: E402
from cdr.data.demo import make_demo  # noqa: E402
from cdr.evaluate import auc_gini_ks, decile_table, psi  # noqa: E402
from cdr.features import WOEEncoder  # noqa: E402
from cdr.pipeline import run  # noqa: E402
from cdr.schema import TARGET  # noqa: E402
from cdr.split import make_split  # noqa: E402


def test_demo_schema_is_complete():
    df = make_demo(n_per_vintage=200, years=range(2015, 2022), seed=1)
    assert len(df) == 200 * 7
    assert set(df["vintage_year"]) == set(range(2015, 2022))
    assert df[TARGET].isin([0, 1]).all()
    assert df["loan_id"].is_unique
    assert not df[["fico", "ltv", "dti"]].isna().any().any()


def test_vintage_split_has_no_overlap():
    df = make_demo(n_per_vintage=300, seed=7)
    cfg = load_config()
    split = make_split(df, cfg)
    assert set(split.train["vintage_year"]) & set(split.oot["vintage_year"]) == set()
    assert len(split.train) + len(split.valid) + len(split.oot) == len(df)


def test_woe_encoder_handles_unseen_categories_and_missing():
    train = make_demo(n_per_vintage=400, seed=3)
    enc = WOEEncoder(["fico", "ltv", "dti"], ["state"], max_bins=8).fit(train)

    unseen = train.head(50).copy()
    unseen.loc[:, "state"] = "ZZ"          # never seen during fit
    unseen.loc[:, "fico"] = np.nan         # missing at scoring time

    encoded = enc.transform(unseen)
    assert encoded.isna().sum().sum() == 0
    assert enc.iv_["fico"] > 0
    assert set(enc.iv_table()["strength"].dropna()).issubset(
        {"useless", "weak", "medium", "strong", "suspicious"}
    )


def test_metrics_behave_as_expected():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ordered = np.array([0.10, 0.15, 0.20, 0.25, 0.80, 0.85, 0.90, 0.95])
    reversed_scores = ordered[::-1]

    perfect = auc_gini_ks(y, ordered)
    assert abs(perfect["auc"] - 1.0) < 1e-9, perfect
    assert abs(perfect["gini"] - 1.0) < 1e-9, perfect
    assert abs(perfect["ks"] - 1.0) < 1e-9, perfect
    assert abs(auc_gini_ks(y, reversed_scores)["auc"] - 0.0) < 1e-9

    rng = np.random.default_rng(0)
    base = rng.beta(2, 5, 4000)
    moved = rng.beta(5, 2, 4000)
    assert psi(base, base) < 1e-6
    assert psi(base, moved) > 0.10
    assert psi(base, rng.beta(2, 5, 5)) != psi(base, moved)  # tiny samples -> nan, not a crash

    tbl = decile_table(rng.integers(0, 2, 500), rng.random(500), n_bands=5)
    assert len(tbl) == 5
    assert tbl["band"].tolist() == ["D1", "D2", "D3", "D4", "D5"]


def _freddie_row(n_fields: int, **positions) -> list[str]:
    """Build a pipe-delimited row with values at 1-based field positions."""
    row = [""] * n_fields
    for key, value in positions.items():
        row[int(key[1:]) - 1] = str(value)
    return row


def _write_freddie_fixture(tmp: Path, orig_fields: int, perf_fields: int) -> None:
    """Two-loan Freddie fixture, width configurable so both layouts get pinned.

    ``orig_fields`` / ``perf_fields`` let the same logical fixture be written in
    the published 32/32 layout *and* in the 31/35 layout actually served in 2026.
    """
    orig_rows = [
        # f1=FICO f2=first payment f8=occupancy f9=CLTV f10=DTI f11=UPB f12=LTV
        # f13=rate f14=channel f17=state f18=property f20=loan id f21=purpose f22=term f23=borrowers
        _freddie_row(orig_fields, f1=750, f2=201501, f8="P", f9=80, f10=32, f11=250000, f12=80,
                     f13=3.875, f14="R", f17="CA", f18="SF", f20="F15Q10000001",
                     f21="P", f22=360, f23=2),
        _freddie_row(orig_fields, f1=620, f2=201502, f8="I", f9=95, f10=45, f11=180000, f12=95,
                     f13=5.500, f14="B", f17="TX", f18="CO", f20="F15Q10000002",
                     f21="C", f22=360, f23=1),
    ]
    perf_rows = []
    for age in range(1, 13):  # loan 1: current throughout, then prepaid
        perf_rows.append(_freddie_row(perf_fields, f1="F15Q10000001", f2=201501, f4=0, f5=age,
                                      f9="01" if age == 12 else ""))
    for age, delinq, zb in [(1, 0, ""), (2, 0, ""), (3, 0, ""), (4, 0, ""), (5, 0, ""),
                            (6, 3, ""), (7, 6, ""), (8, 6, "03")]:  # loan 2: charge-off
        perf_rows.append(_freddie_row(perf_fields, f1="F15Q10000002", f2=201501, f4=delinq, f5=age, f9=zb))

    (tmp / "sample_orig_2015.txt").write_text(
        "\n".join("|".join(r) for r in orig_rows) + "\n", encoding="latin-1")
    (tmp / "sample_perf_2015.txt").write_text(
        "\n".join("|".join(r) for r in perf_rows) + "\n", encoding="latin-1")


def _assert_freddie_fixture(df) -> None:
    df = df.set_index("loan_id").sort_index()
    assert len(df) == 2
    assert df.loc["F15Q10000001", TARGET] == 0
    assert df.loc["F15Q10000002", TARGET] == 1
    assert df.loc["F15Q10000002", "months_observed"] == 8
    assert df.loc["F15Q10000001", "fico"] == 750
    assert df.loc["F15Q10000001", "cltv"] == 80
    assert df.loc["F15Q10000001", "occupancy"] == "Primary Residence"
    assert df.loc["F15Q10000002", "occupancy"] == "Investment Property"
    assert df.loc["F15Q10000002", "loan_purpose"] == "Cash-out Refinance"
    assert df.loc["F15Q10000002", "channel"] == "Broker"
    assert (df["vintage_year"] == 2015).all()


def test_freddie_loader_parses_published_32_field_layout():
    """The layout published in file_layout.xlsx (32 fields each) still parses."""
    import tempfile

    from cdr.data.freddie import load_freddie

    tmp = Path(tempfile.mkdtemp())
    _write_freddie_fixture(tmp, orig_fields=32, perf_fields=32)
    _assert_freddie_fixture(load_freddie(tmp, years=[2015]))


def test_freddie_loader_survives_the_2026_layout_drift():
    """Sample files served in 2026 carry 31 / 35 fields, not the published 32 / 32.

    Regression test. Reading by column *name* aligned the 32 published names onto
    the last 32 of the 35 performance fields, shifting loan id, delinquency status
    and zero balance code - i.e. silently corrupting the default label.
    """
    import tempfile

    from cdr.data.freddie import load_freddie

    tmp = Path(tempfile.mkdtemp())
    _write_freddie_fixture(tmp, orig_fields=31, perf_fields=35)
    _assert_freddie_fixture(load_freddie(tmp, years=[2015]))


def test_freddie_loader_drops_loans_whose_early_months_were_never_observed():
    """A late-arriving loan must never be labelled "no default"."""
    import tempfile

    from cdr.data.freddie import load_freddie

    tmp = Path(tempfile.mkdtemp())
    orig = [
        _freddie_row(31, f1=750, f2=201501, f8="P", f20="F15Q10000001", f22=360),
        _freddie_row(31, f1=700, f2=201501, f8="P", f20="F15Q10000002", f22=360),
    ]
    # Loan 2 first appears at month 44: its opening 36 months are absent entirely.
    perf = [_freddie_row(35, f1="F15Q10000001", f4=0, f5=age) for age in range(0, 36)]
    perf += [_freddie_row(35, f1="F15Q10000002", f4=0, f5=age) for age in range(44, 50)]

    (tmp / "sample_orig_2015.txt").write_text(
        "\n".join("|".join(r) for r in orig) + "\n", encoding="latin-1")
    (tmp / "sample_perf_2015.txt").write_text(
        "\n".join("|".join(r) for r in perf) + "\n", encoding="latin-1")

    df = load_freddie(tmp, years=[2015])

    assert df["loan_id"].tolist() == ["F15Q10000001"], df["loan_id"].tolist()
    assert df.attrs["unobserved_dropped"] == 1, df.attrs


def test_freddie_loader_drops_vintages_outside_the_requested_years():
    """sample_YYYY carries ~16% YYYY+1 first payments; those must not ride along.

    They belong to no configured split, and the splitter used to discard them
    silently.
    """
    import tempfile

    from cdr.data.freddie import load_freddie

    tmp = Path(tempfile.mkdtemp())
    orig = [
        _freddie_row(31, f1=750, f2=201506, f8="P", f20="F15Q10000001", f22=360),
        _freddie_row(31, f1=760, f2=201602, f8="P", f20="F16Q10000002", f22=360),
    ]
    perf = [_freddie_row(35, f1="F15Q10000001", f4=0, f5=age) for age in range(0, 4)]
    perf += [_freddie_row(35, f1="F16Q10000002", f4=0, f5=age) for age in range(0, 4)]

    (tmp / "sample_orig_2015.txt").write_text(
        "\n".join("|".join(r) for r in orig) + "\n", encoding="latin-1")
    (tmp / "sample_perf_2015.txt").write_text(
        "\n".join("|".join(r) for r in perf) + "\n", encoding="latin-1")

    df = load_freddie(tmp, years=[2015])

    assert df["loan_id"].tolist() == ["F15Q10000001"], df["loan_id"].tolist()
    assert df["vintage_year"].tolist() == [2015]
    assert df.attrs["out_of_scope_dropped"] == 1, df.attrs


def test_freddie_loader_rejects_a_file_that_is_too_narrow():
    """A shrunken file must raise, never be read on a guessed layout."""
    import tempfile

    from cdr.data.freddie import load_freddie

    tmp = Path(tempfile.mkdtemp())
    narrow = [""] * 20
    narrow[0], narrow[1], narrow[19] = "750", "201501", "F15Q10000001"
    (tmp / "sample_orig_2015.txt").write_text("|".join(narrow) + "\n", encoding="latin-1")
    (tmp / "sample_perf_2015.txt").write_text(
        "|".join(["F15Q10000001"] + [""] * 34) + "\n", encoding="latin-1")

    try:
        load_freddie(tmp, years=[2015])
    except ValueError as exc:
        assert "pipe-delimited fields" in str(exc), exc
    else:
        raise AssertionError("a 20-field origination file should have been rejected")


def test_pipeline_end_to_end_on_synthetic_data():
    cfg = load_config()
    cfg["data"]["source"] = "demo"
    cfg["data"]["demo_rows_per_vintage"] = 400
    cfg["model"]["gbm"]["max_iter"] = 60

    results = run(cfg, verbose=False, write_report=False)

    assert results["meta"]["n_rows"] > 0
    oot = results["model_metrics"].query("split == 'oot'")
    assert len(oot) == 2
    assert (oot["auc"] > 0.55).all(), oot
    assert results["vintage"]["vintage_year"].nunique() >= 5
    assert np.isfinite(results["feature_psi"]["psi"]).all()


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
