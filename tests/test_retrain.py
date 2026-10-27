from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.modelops.retrain import (
    assess_candidate,
    prepare_cohorts,
    promote_candidate,
    run_retraining,
)


def make_customers(n, prefix):
    return pd.DataFrame({
        "customer_id": [
            f"{prefix}{i:05d}"
            for i in range(n)
        ],
        "tenure_months": list(range(n)),
        "monthly_spend": [
            50 + (i % 12)
            for i in range(n)
        ],
        "support_tickets_90d": [
            i % 3
            for i in range(n)
        ],
        "logins_30d": [
            i % 22
            for i in range(n)
        ],
        "monthly_contract": [
            i % 2
            for i in range(n)
        ],
        "churn": [
            int(i % 4 == 0)
            for i in range(n)
        ],
    })


def test_splits_do_not_overlap():
    reference = make_customers(500, "R")
    incoming = make_customers(500, "P")

    (
        historical_train,
        historical_holdout,
        incoming_train,
        incoming_holdout,
    ) = prepare_cohorts(
        reference,
        incoming.drop(columns="churn"),
        incoming[["customer_id", "churn"]],
    )

    assert len(historical_train) == 400
    assert len(historical_holdout) == 100

    assert len(incoming_train) == 400
    assert len(incoming_holdout) == 100

    assert set(
        historical_train["customer_id"]
    ).isdisjoint(
        historical_holdout["customer_id"]
    )

    assert set(
        incoming_train["customer_id"]
    ).isdisjoint(
        incoming_holdout["customer_id"]
    )


def test_rejects_missing_labels():
    reference = make_customers(500, "R")
    incoming = make_customers(500, "P")

    with pytest.raises(
        ValueError,
        match="ID mismatch",
    ):
        prepare_cohorts(
            reference,
            incoming.drop(columns="churn"),
            incoming[
                ["customer_id", "churn"]
            ].iloc[:-1],
        )


def test_rejects_overlapping_ids():
    reference = make_customers(500, "R")
    incoming = make_customers(500, "R")

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        prepare_cohorts(
            reference,
            incoming.drop(columns="churn"),
            incoming[["customer_id", "churn"]],
        )


def test_rejects_insufficient_f1_improvement():
    decision = assess_candidate(
        {"accuracy": 0.80, "roc_auc": 0.80},
        {"accuracy": 0.80, "roc_auc": 0.80},
        {"f1": 0.70, "roc_auc": 0.80},
        {"f1": 0.705, "roc_auc": 0.80},
    )

    assert not decision["eligible"]

    assert (
        "incoming_f1_gain"
        in decision["reasons"]
    )


def test_acceptance_requires_guardrails():
    decision = assess_candidate(
        {"accuracy": 0.80, "roc_auc": 0.80},
        {"accuracy": 0.78, "roc_auc": 0.79},
        {"f1": 0.70, "roc_auc": 0.80},
        {"f1": 0.73, "roc_auc": 0.79},
    )

    assert decision["eligible"]
    assert all(decision["checks"].values())


def test_old_model_is_archived_before_promotion(tmp_path):
    champion = tmp_path / 'active.joblib'
    candidate = tmp_path / 'new.joblib'


    joblib.dump(
        {'version': 1},
        champion,
    )

    joblib.dump(
        {'version': 2},
        candidate,
    )

    backup = promote_candidate(
        candidate,
        champion,
        tmp_path / 'archive',
    )

    assert joblib.load(backup) == {
        'version': 1
    }

    assert joblib.load(champion) == {
        'version': 2
    }


def test_dry_run_never_changes_champion(tmp_path):
    root = Path(__file__).resolve().parents[1]

    original = (
        root / "artifacts/churn_model.joblib"
    )

    if not original.exists():
        pytest.skip(
            "Baseline model has not been generated"
        )

    champion = tmp_path / "champion.joblib"
    champion.write_bytes(
        original.read_bytes()
    )

    before = champion.read_bytes()

    report = run_retraining(
        reference_path=(
            root / "data/generated/reference.csv"
        ),
        incoming_path=(
            root / "data/generated/incoming.csv"
        ),
        labels_path=(
            root / "data/generated/delayed_labels.csv"
        ),
        champion_path=champion,
        candidate_path=(
            tmp_path / "candidate.joblib"
        ),
        archive_dir=(
            tmp_path / "archive"
        ),
        report_dir=(
            tmp_path / "reports"
        ),
        promote=False,
    )

    assert (
        report["promotion"]["status"]
        == "not_requested"
    )

    assert champion.read_bytes() == before

    assert (
        tmp_path
        / "reports/retraining_decision.md"
    ).exists()