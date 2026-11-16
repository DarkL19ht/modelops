"""Portable numeric model validation and reproducible inference."""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.portable_models import export_portable_model, load_portable_model


@pytest.fixture
def trained_model():
    rng = np.random.default_rng(41)
    x = pd.DataFrame({
        "annual_income": rng.uniform(25000, 110000, 300),
        "debt_ratio": rng.uniform(0.05, 0.9, 300),
        "late_payments": rng.integers(0, 5, 300).astype(float),
    })
    y = (x.debt_ratio * 5 + x.late_payments * .8 - x.annual_income / 80000 > 2.5).astype(int)
    model = Pipeline([("scale", StandardScaler()), ("classify", LogisticRegression(max_iter=500))]).fit(x, y)
    return x, model


def test_export_import_predicts_identically(trained_model):
    x, model = trained_model
    restored = load_portable_model(export_portable_model(model))
    np.testing.assert_allclose(model.predict_proba(x), restored.predict_proba(x), atol=1e-12)
    assert list(restored.feature_names_in_) == list(x.columns)


def test_rebuilt_model_is_retrainable(trained_model):
    x, model = trained_model
    restored = load_portable_model(export_portable_model(model))
    from sklearn.base import clone
    challenger = clone(restored)
    challenger.fit(x, model.predict(x))
    assert challenger.predict_proba(x).shape == (len(x), 2)


@pytest.mark.parametrize("field,value", [
    ("coefficients", [0, float("nan"), 1]),
    ("coefficients", [0, 1]),
    ("intercept", "__import__('os')"),
    ("features", ["feature", "feature", "three"]),
    ("features", ["a", "b", "__class__"]),
    ("scaler", {"mean": [0,0,0], "scale": [1,0,1]}),
])
def test_rejects_bad_json_fields(field, value, trained_model):
    _, model = trained_model
    data = json.loads(export_portable_model(model))
    data[field] = value
    with pytest.raises(ValueError):
        load_portable_model(json.dumps(data).encode())


def test_rejects_large_payload():
    with pytest.raises(ValueError):
        load_portable_model(b"{" + b" " * 20000 + b"}")


def test_does_not_support_arbitrary_estimators(trained_model):
    x, _ = trained_model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(x, (x.debt_ratio > .5).astype(int))
    with pytest.raises(ValueError):
        export_portable_model(model)
