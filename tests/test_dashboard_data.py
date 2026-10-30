# Test the data transformation functions that power the dashboard

import json

import pytest

from src.modelops.dashboard_data import (
    read_report,
    drift_table,
    performance_table,
    retraining_table,
)


def test_missing_report_returns_none(tmp_path):
    result = read_report(
        tmp_path / "missing.json"
    )

    assert result is None


def test_report_loads_json(tmp_path):
    path = tmp_path / "example.json"

    path.write_text(
        json.dumps({"status": "stable"}),
        encoding="utf-8",
    )

    assert read_report(path) == {
        "status": "stable"
    }


def test_drift_table():
    report = {
        "features": [
            {
                "feature": "logins_30d",
                "psi": 0.5,
                "status": "drift",
                "reference_mean": 15.0,
                "incoming_mean": 8.0,
                "ks_statistic": 0.4,
            }
        ]
    }

    table = drift_table(report)

    assert len(table) == 1
    assert table.iloc[0]["feature"] == "logins_30d"
    assert table.iloc[0]["status"] == "drift"


def test_performance_table():
    report = {
        "baseline": {
            "accuracy": 0.80,
            "f1": 0.70,
            "roc_auc": 0.85,
        },
        "metrics": {
            "accuracy": 0.72,
            "f1": 0.75,
            "roc_auc": 0.84,
        },
    }

    table = performance_table(report)

    assert len(table) == 3

    accuracy = table[
        table["metric"] == "accuracy"
    ].iloc[0]

    assert accuracy["baseline"] == 0.80
    assert accuracy["incoming"] == 0.72


def test_retraining_table():
    metrics = {
        "accuracy": 0.80,
        "f1": 0.70,
        "roc_auc": 0.85,
    }

    report = {
        "evaluation": {
            "reference_holdout": {
                "champion": metrics,
                "challenger": metrics,
            },
            "incoming_holdout": {
                "champion": metrics,
                "challenger": metrics,
            },
        }
    }

    table = retraining_table(report)

    assert len(table) == 6

    assert set(table["cohort"]) == {
        "reference_holdout",
        "incoming_holdout",
    }