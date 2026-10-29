
"""Tests for ModelOps HTTP model serving."""

import joblib
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.api import create_app
from src.modelops.train import FEATURES


SAMPLE = {
    "customer_id": "example-01",
    "tenure_months": 12,
    "monthly_spend": 75.5,
    "support_tickets_90d": 2,
    "logins_30d": 10,
    "monthly_contract": 1,
}


def write_model(path, random_seed=42, c=1.0):
    rng = np.random.default_rng(random_seed)

    features = pd.DataFrame({
        "tenure_months": rng.integers(1, 73, 500),
        "monthly_spend": rng.uniform(20, 150, 500),
        "support_tickets_90d": rng.integers(0, 6, 500),
        "logins_30d": rng.integers(0, 30, 500),
        "monthly_contract": rng.integers(0, 2, 500),
    })[FEATURES]

    labels = (
        (
            features["support_tickets_90d"] * 2
            + features["monthly_contract"]
            - features["logins_30d"] / 8
        ) > 1
    ).astype(int)

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=c,
                max_iter=1000,
            ),
        ),
    ])

    model.fit(features, labels)

    joblib.dump(model, path)


@pytest.fixture
def api(tmp_path):
    path = tmp_path / "champion.joblib"

    write_model(path)

    app = create_app(
        model_path=path,
        admin_token="local-test-token",
    )

    with TestClient(app) as client:
        yield client, path


def test_health_and_model_info(api):
    client, _ = api

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    info = client.get("/model").json()

    assert info["features"] == FEATURES
    assert len(info["model_sha256"]) == 64


def test_single_prediction(api):
    client, _ = api

    response = client.post(
        "/predict",
        json=SAMPLE,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["customer_id"] == "example-01"
    assert result["predicted_churn"] in (0, 1)

    assert (
        0 <= result["churn_probability"] <= 1
    )


def test_batch_prediction(api):
    client, _ = api

    second = {
        **SAMPLE,
        "customer_id": "example-02",
        "logins_30d": 23,
    }

    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                SAMPLE,
                second,
            ]
        },
    )

    assert response.status_code == 200

    assert len(
        response.json()["predictions"]
    ) == 2


def test_invalid_input_and_duplicates(api):
    client, _ = api

    assert client.post(
        "/predict",
        json={
            **SAMPLE,
            "monthly_spend": -5,
        },
    ).status_code == 422

    assert client.post(
        "/predict",
        json={
            **SAMPLE,
            "unexpected_feature": 2,
        },
    ).status_code == 422

    assert client.post(
        "/predict/batch",
        json={
            "customers": [SAMPLE, SAMPLE]
        },
    ).status_code == 422

    assert client.post(
        "/predict/batch",
        json={"customers": []},
    ).status_code == 422


def test_reload_requires_token(api):
    client, _ = api

    assert client.post(
        "/admin/reload"
    ).status_code == 403

    assert client.post(
        "/admin/reload",
        headers={
            "X-Admin-Token": "wrong"
        },
    ).status_code == 403


def test_reload_activates_new_champion(api):
    client, path = api

    old_sha = client.get(
        "/model"
    ).json()["model_sha256"]

    write_model(
        path,
        random_seed=17,
        c=0.5,
    )

    # The running API still has the old model.
    assert client.get(
        "/model"
    ).json()["model_sha256"] == old_sha

    response = client.post(
        "/admin/reload",
        headers={
            "X-Admin-Token": "local-test-token"
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["previous_sha256"] == old_sha
    assert result["model_sha256"] != old_sha


def test_failed_reload_keeps_previous_model(api):
    client, path = api

    old_sha = client.get(
        "/model"
    ).json()["model_sha256"]

    # Deliberately make the next artifact unreadable.
    path.write_bytes(b"invalid-joblib")

    response = client.post(
        "/admin/reload",
        headers={
            "X-Admin-Token": "local-test-token"
        },
    )

    assert response.status_code == 503

    # The previous in-memory model is preserved.
    assert client.get(
        "/model"
    ).json()["model_sha256"] == old_sha

    assert client.post(
        "/predict",
        json=SAMPLE,
    ).status_code == 200


def test_missing_champion_is_unavailable(tmp_path):
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        admin_token="",
    )

    with TestClient(app) as client:
        assert client.get(
            "/health"
        ).status_code == 503

        assert client.post(
            "/predict",
            json=SAMPLE,
        ).status_code == 503

        assert client.post(
            "/admin/reload"
        ).status_code == 503
