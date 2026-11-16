"""Strict, data-only export for bounded binary linear classifiers.

No file supplied by a user is deserialized as Python or executed by this module.
Public uploads support logistic regression and optional StandardScaler only.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FORMAT = "modelops-linear-v1"
MAX_BYTES = 16 * 1024
MAX_FEATURES = 15
_FEATURE_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$")


def _reject_constant(value: str):
    raise ValueError("Non-finite numeric values are not accepted")


def _number(value: Any, label: str, max_abs: float = 1e9) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    output = float(value)
    if not math.isfinite(output) or abs(output) > max_abs:
        raise ValueError(f"{label} is out of range")
    return output


def _vector(value: Any, label: str, count: int, max_abs: float = 1e9) -> np.ndarray:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} requires {count} values")
    return np.asarray([_number(v, label, max_abs) for v in value], dtype=np.float64)


def load_portable_model(data: bytes) -> Pipeline:
    """Reconstruct a limited, known sklearn pipeline from validated JSON."""
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_BYTES:
        raise ValueError("Portable model must be between 1 byte and 16 KB")
    try:
        spec = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid portable model JSON") from exc
    if not isinstance(spec, dict) or set(spec) != {
        "format", "features", "coefficients", "intercept", "scaler"
    }:
        raise ValueError("Portable model has unexpected or missing fields")
    if spec["format"] != FORMAT:
        raise ValueError("Unsupported portable model format")
    features = spec["features"]
    if (not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES
            or any(not isinstance(v, str) or not _FEATURE_NAME.fullmatch(v) for v in features)
            or len(features) != len(set(features))):
        raise ValueError("Feature names must be unique identifiers (up to 15)")
    n = len(features)
    coefs = _vector(spec["coefficients"], "coefficients", n, 1e4)
    intercept = _number(spec["intercept"], "intercept", 1e4)
    scaler_spec = spec["scaler"]
    if not isinstance(scaler_spec, dict) or set(scaler_spec) != {"mean", "scale"}:
        raise ValueError("A scaler with mean and scale is required")
    mean = _vector(scaler_spec["mean"], "scaler.mean", n)
    scale = _vector(scaler_spec["scale"], "scaler.scale", n)
    if (scale <= 1e-9).any():
        raise ValueError("Each scaler scale must be greater than 1e-9")

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = n
    scaler.feature_names_in_ = np.asarray(features, dtype=object)
    scaler.n_samples_seen_ = 2

    classifier = LogisticRegression(max_iter=500, random_state=42)
    classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
    classifier.coef_ = coefs.reshape(1, n)
    classifier.intercept_ = np.asarray([intercept])
    classifier.n_features_in_ = n
    classifier.n_iter_ = np.asarray([1], dtype=np.int32)
    model = Pipeline([("scaler", scaler), ("classifier", classifier)])
    return model


def export_portable_model(model) -> bytes:
    """Export only a trusted, already-fitted logistic classifier or simple pipeline.

    Never load untrusted joblib/pickle models before calling this function.
    """
    if isinstance(model, Pipeline):
        if len(model.steps) != 2 or not isinstance(model.steps[0][1], StandardScaler) or not isinstance(model.steps[1][1], LogisticRegression):
            raise ValueError("Only StandardScaler + LogisticRegression pipelines are supported")
        scaler, clf = model.steps[0][1], model.steps[1][1]
        mean = np.asarray(scaler.mean_, dtype=float)
        scale = np.asarray(scaler.scale_, dtype=float)
    elif isinstance(model, LogisticRegression):
        clf = model
        n = clf.coef_.shape[1]
        mean = np.zeros(n)
        scale = np.ones(n)
    else:
        raise ValueError("Only LogisticRegression or StandardScaler + LogisticRegression is supported")
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Fitted model must have feature_names_in_; fit using a pandas DataFrame")
    if list(getattr(clf, "classes_", [])) != [0, 1]:
        raise ValueError("Only binary classifiers with labels 0 and 1 are supported")
    if clf.coef_.shape != (1, len(names)) or len(clf.intercept_) != 1:
        raise ValueError("Unexpected fitted classifier dimensions")
    obj = {"format": FORMAT, "features": [str(x) for x in names],
           "coefficients": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
           "scaler": {"mean": mean.tolist(), "scale": scale.tolist()}}
    data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
    load_portable_model(data)  # round-trip checks bounds and schema
    return data
