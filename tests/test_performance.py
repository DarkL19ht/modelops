#Automated tests
import json

import pandas as pd
import pytest

from src.modelops.performance import (
    assess_performance,
    evaluate_predictions,
    save_reports,
)


def example():
    predictions = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "predicted_churn": [1, 0, 1, 0],
        "churn_probability": [0.9, 0.2, 0.8, 0.1],
    })

    labels = pd.DataFrame({
        "customer_id": ["C4", "C1", "C3", "C2"],
        "churn": [0, 1, 1, 0],
    })

    return predictions, labels


def test_join_by_customer_id_not_row_position():
    predictions, labels = example()

    metrics = evaluate_predictions(
        predictions,
        labels,
        min_labels=4,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0

    assert metrics["confusion_matrix"] == {
        "tn": 2,
        "fp": 0,
        "fn": 0,
        "tp": 2,
    }


def test_missing_or_extra_labels_are_rejected():
    predictions, labels = example()

    with pytest.raises(
        ValueError,
        match="ID mismatch",
    ):
        evaluate_predictions(
            predictions,
            labels.iloc[:3],
            min_labels=1,
        )


def test_duplicate_customer_ids_are_rejected():
    predictions, labels = example()

    predictions.loc[
        1, "customer_id"
    ] = "C1"

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        evaluate_predictions(
            predictions,
            labels,
            min_labels=1,
        )


def test_bad_probability_is_rejected():
    predictions, labels = example()

    predictions.loc[
        1, "churn_probability"
    ] = 1.5

    with pytest.raises(
        ValueError,
        match="churn_probability",
    ):
        evaluate_predictions(
            predictions,
            labels,
            min_labels=1,
        )


def test_small_sample_is_rejected():
    predictions, labels = example()

    with pytest.raises(
        ValueError,
        match="Need at least",
    ):
        evaluate_predictions(
            predictions,
            labels,
            min_labels=100,
        )


def test_degradation_alert_detects_accuracy_drop():
    baseline = {
        "accuracy": 0.90,
        "f1": 0.70,
        "roc_auc": 0.85,
    }

    current = {
        "accuracy": 0.80,
        "f1": 0.72,
        "roc_auc": 0.85,
    }

    result = assess_performance(
        current,
        baseline,
    )

    assert result["status"] == "degraded"
    assert len(result["alerts"]) == 1
    assert "Accuracy" in result["alerts"][0]


def test_one_class_cohort_marks_auc_unavailable():
    predictions, labels = example()

    labels["churn"] = 0

    metrics = evaluate_predictions(
        predictions,
        labels,
        min_labels=1,
    )

    assert metrics["roc_auc"] is None

    baseline = {
        "accuracy": 0.4,
        "f1": 0.0,
        "roc_auc": 0.85,
    }

    assessment = assess_performance(
        metrics,
        baseline,
    )

    assert (
        assessment["status"]
        == "insufficient_evidence"
    )


def test_report_history_is_idempotent(tmp_path):
    metrics = {
        "labeled_rows": 4,
        "accuracy": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "actual_churn_rate": 0.5,
        "predicted_churn_rate": 0.5,
    }

    baseline = {
        "accuracy": 0.8,
        "f1": 0.7,
        "roc_auc": 0.8,
    }

    report = {
        "run_id": "same-run",
        "baseline": baseline,
        "metrics": metrics,
        "assessment": assess_performance(
            metrics,
            baseline,
        ),
    }

    save_reports(report, tmp_path)
    save_reports(report, tmp_path)

    lines = (
        tmp_path / "performance_history.jsonl"
    ).read_text().splitlines()

    assert len(lines) == 1

    saved = json.loads(
        (tmp_path / "performance_report.json")
        .read_text()
    )

    assert saved["run_id"] == "same-run"

    assert "Performance Monitoring Report" in (
        tmp_path / "performance_report.md"
    ).read_text()