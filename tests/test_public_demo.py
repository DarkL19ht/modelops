"""Regression checks for the public, read-only demonstration."""

import pytest

from src.modelops.public_demo import build_demo, predict_demo
from src.modelops.train import FEATURES


@pytest.fixture(scope="module")
def workspace():
    return build_demo()


def test_demo_uses_real_holdout_evaluations(workspace):
    assert workspace.reference_rows == 5000
    assert workspace.incoming_rows == 1200
    assert len(workspace.drift["features"]) == len(FEATURES)
    assert set(workspace.evaluations) == {"reference_holdout", "incoming_holdout"}
    for cohort in workspace.evaluations.values():
        for role in ("champion", "challenger"):
            assert 0 <= cohort[role]["accuracy"] <= 1
            assert 0 <= cohort[role]["f1"] <= 1
    assert workspace.decision["eligible"] == all(workspace.decision["checks"].values())


def test_demo_inference_on_both_versions(workspace):
    features = {
        "tenure_months": 12,
        "monthly_spend": 75.5,
        "support_tickets_90d": 2,
        "logins_30d": 10,
        "monthly_contract": 1,
    }
    for version in ("champion", "challenger"):
        result = predict_demo(workspace, features, version=version)
        assert result["predicted_churn"] in (0, 1)
        assert 0 <= result["churn_probability"] <= 1
        assert result["version"] == version


def test_demo_rejects_wrong_features(workspace):
    with pytest.raises(ValueError, match="schema"):
        predict_demo(workspace, {"monthly_spend": 50})


def test_demo_rejects_nonfinite_values(workspace):
    features = {
        "tenure_months": 12,
        "monthly_spend": float("nan"),
        "support_tickets_90d": 2,
        "logins_30d": 10,
        "monthly_contract": 1,
    }
    with pytest.raises(ValueError, match="finite"):
        predict_demo(workspace, features)


def test_demo_rejects_unknown_version(workspace):
    features = dict.fromkeys(FEATURES, 1)
    with pytest.raises(ValueError, match="version"):
        predict_demo(workspace, features, version="uploaded")
