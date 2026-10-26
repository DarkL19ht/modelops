# Automated tests
import json

import numpy as np
import pytest

from src.modelops.drift import (
    calculate_drift,
    numeric_psi,
)
from src.modelops.generate_data import make_customers
from src.modelops.monitor import save_reports


def test_same_distribution_has_zero_drift():
    reference = make_customers(
        n=300,
        seed=7,
    )

    report = calculate_drift(
        reference,
        reference.copy(),
    )

    assert report["drifted_features"] == []

    assert all(
        item["psi"] == 0
        for item in report["features"]
    )


def test_known_shift_flags_activity_and_complaints():
    reference = make_customers(
        n=5000,
        seed=42,
    )

    incoming = make_customers(
        n=1200,
        seed=202,
        shifted=True,
    ).drop(columns="churn")

    report = calculate_drift(
        reference,
        incoming,
    )

    assert (
        "logins_30d"
        in report["drifted_features"]
    )

    assert (
        "support_tickets_90d"
        in report["drifted_features"]
    )

    assert "churn" not in [
        row["feature"]
        for row in report["features"]
    ]


def test_missing_feature_is_rejected():
    reference = make_customers(
        n=100,
        seed=1,
    )

    incoming = reference.drop(
        columns="monthly_spend"
    )

    with pytest.raises(
        ValueError,
        match="missing features",
    ):
        calculate_drift(
            reference,
            incoming,
        )


def test_nonfinite_feature_is_rejected():
    reference = make_customers(
        n=100,
        seed=1,
    )

    incoming = reference.copy()

    incoming.loc[
        0,
        "monthly_spend",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        calculate_drift(
            reference,
            incoming,
        )


def test_constant_reference_detects_new_values():
    reference = np.zeros(100)
    incoming = np.ones(100)

    assert numeric_psi(
        reference,
        incoming,
    ) > 0.20

    assert numeric_psi(
        reference,
        reference,
    ) == 0


def test_report_files_are_created(tmp_path):
    reference = make_customers(
        n=100,
        seed=1,
    )

    report = calculate_drift(
        reference,
        reference.copy(),
    )

    save_reports(
        report,
        tmp_path,
    )

    saved = json.loads(
        (tmp_path / "drift_report.json").read_text()
    )

    assert saved["reference_rows"] == 100

    assert "Data Drift Report" in (
        tmp_path / "drift_report.md"
    ).read_text()