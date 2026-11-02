"""Tests for uploaded-model workspaces; no uploaded pickle files are loaded."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.model_workspace import (
    Workspace,
    candidate_decision,
    compute_drift,
    read_csv,
    score,
    validate_model,
)

FEATURES = ["amount", "late", "score"]


def sample(n, seed, prefix, shifted=False):
    rng = np.random.default_rng(seed)
    amount = rng.normal(50 if shifted else 30, 10, n)
    late = rng.poisson(3 if shifted else 1, n)
    score_value = rng.normal(0, 1, n)
    logits = -0.5 + .05 * amount + .5 * late + .3 * score_value
    labels = rng.binomial(1, 1/(1+np.exp(-logits)))
    return pd.DataFrame({
        "id": [f"{prefix}{i:05d}" for i in range(n)],
        "amount": amount, "late": late, "score": score_value,
        "label": labels,
    })


@pytest.fixture
def setup():
    reference = sample(500, 1, "R")
    validation = sample(150, 2, "V")
    incoming = sample(250, 3, "I", shifted=True)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(reference[FEATURES], reference["label"])
    return model, reference, validation, incoming


def workspace(setup, labels=True, validation=True):
    model, ref, val, incoming = setup
    return Workspace(
        name="other-model", champion=model, reference=ref,
        incoming=incoming.drop(columns="label"),
        labels=incoming[["id", "label"]] if labels else None,
        validation=val if validation else None,
        id_col="id", target_col="label",
    )


def test_model_feature_contract(setup):
    model, *_ = setup
    assert validate_model(model) == tuple(FEATURES)


def test_identical_features_produce_zero_drift(setup):
    _, reference, _, _ = setup
    same = reference.copy()
    same["id"] = [f"N{i:05d}" for i in range(len(same))]
    result = compute_drift(reference, same, tuple(FEATURES))
    assert result["drifted_features"] == []
    assert all(row["psi"] == 0 for row in result["features"])


def test_shift_detected(setup):
    _, reference, _, incoming = setup
    result = compute_drift(reference, incoming, tuple(FEATURES))
    assert "amount" in result["drifted_features"]


def test_monitor_without_labels(setup):
    ws = workspace(setup, labels=False, validation=False)
    result = ws.monitor()
    assert result["drift"]["features"]
    assert result["performance"] is None


def test_monitor_with_labels(setup):
    ws = workspace(setup)
    result = ws.monitor()
    assert result["performance"]["incoming"]["rows"] == 250
    assert 0 <= result["performance"]["incoming"]["roc_auc"] <= 1
    assert result["performance"]["historical_holdout"]["rows"] == 150


def test_retrain_isolated_and_decision_generated(setup):
    ws = workspace(setup)
    old_hash = ws.model_sha256
    result = ws.train_challenger()
    assert result["decision"]["eligible"] in (True, False)
    assert ws.challenger is not None
    assert ws.model_sha256 == old_hash
    assert ws.champion is not ws.challenger
    assert result["evaluation"]["incoming_holdout"]["champion"]["rows"] == 50


def test_retrain_needs_independent_validation(setup):
    ws = workspace(setup, validation=False)
    with pytest.raises(ValueError, match="historical holdout"):
        ws.train_challenger()


def test_reject_overlapping_validation(setup):
    model, ref, validation, incoming = setup
    validation.loc[0, "id"] = ref.loc[0, "id"]
    with pytest.raises(ValueError, match="overlaps"):
        Workspace(
            name="bad", champion=model, reference=ref,
            incoming=incoming.drop(columns="label"),
            labels=incoming[["id", "label"]], validation=validation,
            id_col="id", target_col="label",
        )


def test_reject_missing_labels(setup):
    model, ref, validation, incoming = setup
    with pytest.raises(ValueError, match="identical ID sets"):
        Workspace(
            name="bad", champion=model, reference=ref,
            incoming=incoming.drop(columns="label"),
            labels=incoming[["id", "label"]].iloc[:-1],
            validation=validation, id_col="id", target_col="label",
        )


def test_failed_candidate_cannot_promote(setup):
    ws = workspace(setup)
    ws.train_challenger(min_f1_gain=0.9)
    assert not ws.decision["eligible"]
    with pytest.raises(ValueError, match="No eligible"):
        ws.promote()


def test_promotion_and_rollback_are_session_only(setup):
    ws = workspace(setup)
    ws.train_challenger()
    old_hash = ws.model_sha256
    ws.decision = {"eligible": True, "checks": {}, "reasons": []}  # exercise archive mechanics
    ws.promote()
    assert len(ws.previous) == 1
    assert ws.model_sha256 != old_hash
    ws.rollback()
    assert ws.model_sha256 == old_hash
    assert len(ws.previous) == 0


def test_candidate_decision_rejects_regression():
    result = candidate_decision(
        {"accuracy": .8, "roc_auc": .8},
        {"accuracy": .7, "roc_auc": .8},
        {"f1": .5, "roc_auc": .8},
        {"f1": .6, "roc_auc": .8},
    )
    assert not result["eligible"]
    assert "historical_accuracy_guardrail" in result["reasons"]


def test_csv_limit_and_validation():
    assert len(read_csv(b"id,a\n1,2\n", "a.csv")) == 1
    with pytest.raises(ValueError, match="between 1 byte"):
        read_csv(b"", "a.csv")
