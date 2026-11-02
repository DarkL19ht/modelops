"""Independent, session-scoped monitoring workspaces for binary sklearn models.

No arbitrary pickle/joblib loading is performed by this module.
Data and model versions stay in the caller's session, not in the deployed API.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000
MAX_FEATURES = 30


def load_uploaded_model(payload: bytes):
    """Load skops using only its built-in trusted types (never trust unknowns)."""
    if not payload or len(payload) > MAX_FILE_BYTES:
        raise ValueError("Model file must be between 1 byte and 5 MB")
    import skops.io as sio

    unknown = sio.get_untrusted_types(data=payload)
    if unknown:
        raise ValueError(
            "Model contains unsupported/untrusted types: "
            + ", ".join(unknown[:8])
            + ". Use a standard sklearn estimator/pipeline; do not auto-trust types."
        )
    model = sio.loads(payload, trusted=[])
    validate_model(model)
    return model


def serialize_model(model) -> bytes:
    import skops.io as sio
    return sio.dumps(model)


def read_csv(payload: bytes, filename: str) -> pd.DataFrame:
    if not payload or len(payload) > MAX_FILE_BYTES:
        raise ValueError(f"{filename} must be between 1 byte and 5 MB")
    try:
        data = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid CSV file {filename}: {exc}") from exc
    if not 1 <= len(data) <= MAX_ROWS:
        raise ValueError(f"{filename} must contain 1 to {MAX_ROWS:,} rows")
    if not data.columns.is_unique:
        raise ValueError(f"{filename} contains duplicate column names")
    return data


def validate_model(model) -> tuple[str, ...]:
    if not isinstance(model, BaseEstimator):
        raise ValueError("Only fitted scikit-learn classifiers are supported")
    names = getattr(model, "feature_names_in_", None)
    classes = getattr(model, "classes_", None)
    if names is None or not 1 <= len(names) <= MAX_FEATURES:
        raise ValueError("The fitted model must expose 1–30 feature_names_in_ columns")
    features = tuple(str(x) for x in names)
    if len(features) != len(set(features)):
        raise ValueError("Model feature names must be unique")
    if classes is None or list(classes) != [0, 1]:
        raise ValueError("Only fitted binary classifiers with classes [0, 1] are supported")
    for method in ("predict", "predict_proba", "fit", "get_params"):
        if not callable(getattr(model, method, None)):
            raise ValueError(f"Model must support {method}()")
    return features


def _ids(frame: pd.DataFrame, id_col: str, name: str):
    if id_col not in frame.columns:
        raise ValueError(f"{name} missing identifier column '{id_col}'")
    values = frame[id_col]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} has blank identifiers")
    if values.astype(str).duplicated().any():
        raise ValueError(f"{name} has duplicate identifiers")
    return set(values.astype(str))


def _features(frame: pd.DataFrame, features: tuple[str, ...], name: str):
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing model features: {missing}")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"{name} requires 1 to {MAX_ROWS:,} rows")
    try:
        matrix = frame[list(features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric model features") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains missing or non-finite model features")


def _labels(frame: pd.DataFrame, target_col: str, name: str):
    if target_col not in frame.columns:
        raise ValueError(f"{name} missing label column '{target_col}'")
    try:
        values = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} requires binary 0/1 labels") from exc
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} requires binary 0/1 labels")


def _bin_psi(ref, inc, categorical=False):
    ref = np.asarray(ref, dtype=float)
    inc = np.asarray(inc, dtype=float)
    if categorical:
        values = np.union1d(ref, inc)
        a = np.array([(ref == x).sum() for x in values])
        b = np.array([(inc == x).sum() for x in values])
    elif np.all(ref == ref[0]):
        a = np.array([len(ref), 0])
        b = np.array([(inc == ref[0]).sum(), (inc != ref[0]).sum()])
    else:
        cuts = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
        edges = np.concatenate(([-np.inf], cuts, [np.inf]))
        a = np.histogram(ref, bins=edges)[0]
        b = np.histogram(inc, bins=edges)[0]
    epsilon = 1e-6
    p = (a + epsilon) / (a.sum() + len(a) * epsilon)
    q = (b + epsilon) / (b.sum() + len(b) * epsilon)
    return max(0.0, float(np.sum((q - p) * np.log(q / p))))


def compute_drift(reference, incoming, features, categorical=(), threshold=0.2):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("PSI threshold must be positive")
    _features(reference, features, "Reference")
    _features(incoming, features, "Incoming")
    categorical = set(categorical)
    if not categorical.issubset(features):
        raise ValueError("Categorical columns must be model features")
    rows = []
    for feature in features:
        a, b = reference[feature].to_numpy(dtype=float), incoming[feature].to_numpy(dtype=float)
        psi = _bin_psi(a, b, categorical=(feature in categorical))
        ks = None if feature in categorical else float(ks_2samp(a, b).statistic)
        rows.append({
            "feature": feature,
            "psi": round(psi, 6),
            "status": "drift" if psi >= threshold else "stable",
            "reference_mean": round(float(a.mean()), 4),
            "incoming_mean": round(float(b.mean()), 4),
            "ks_statistic": None if ks is None else round(ks, 6),
        })
    return {
        "psi_threshold": threshold,
        "reference_rows": len(reference),
        "incoming_rows": len(incoming),
        "features": rows,
        "drifted_features": [row["feature"] for row in rows if row["status"] == "drift"],
    }


def score(model, frame, features, target_col):
    _features(frame, features, "Evaluation cohort")
    _labels(frame, target_col, "Evaluation cohort")
    actual = frame[target_col].to_numpy(dtype=int)
    pred = np.asarray(model.predict(frame[list(features)]))
    probability_matrix = np.asarray(model.predict_proba(frame[list(features)]), dtype=float)
    if pred.shape != actual.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("Classifier returned invalid class predictions")
    if probability_matrix.shape != (len(actual), 2):
        raise ValueError("Classifier must return two probability columns")
    probabilities = probability_matrix[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier returned invalid probabilities")
    return {
        "rows": len(frame),
        "accuracy": round(float(accuracy_score(actual, pred)), 4),
        "precision": round(float(precision_score(actual, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, pred, zero_division=0)), 4),
        "roc_auc": (
            round(float(roc_auc_score(actual, probabilities)), 4)
            if len(np.unique(actual)) == 2 else None
        ),
    }


def candidate_decision(champion_reference, challenger_reference, champion_incoming, challenger_incoming,
                       min_f1_gain=0.01, max_accuracy_drop=0.03, max_auc_drop=0.02):
    values = (min_f1_gain, max_accuracy_drop, max_auc_drop)
    if any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError("Acceptance thresholds must be non-negative and finite")
    if any(m["roc_auc"] is None for m in (
        champion_reference, challenger_reference, champion_incoming, challenger_incoming
    )):
        return {"eligible": False, "checks": {}, "reasons": ["ROC-AUC unavailable: holdout must contain both classes"]}
    checks = {
        "incoming_f1_gain": challenger_incoming["f1"] - champion_incoming["f1"] >= min_f1_gain,
        "historical_accuracy_guardrail": challenger_reference["accuracy"] >= champion_reference["accuracy"] - max_accuracy_drop,
        "historical_auc_guardrail": challenger_reference["roc_auc"] >= champion_reference["roc_auc"] - max_auc_drop,
        "incoming_auc_guardrail": challenger_incoming["roc_auc"] >= champion_incoming["roc_auc"] - max_auc_drop,
    }
    return {
        "eligible": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "reasons": [key for key, value in checks.items() if not value],
    }


@dataclass
class Workspace:
    name: str
    champion: object
    reference: pd.DataFrame
    incoming: pd.DataFrame
    labels: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    id_col: str = "record_id"
    target_col: str = "target"
    categorical: tuple[str, ...] = ()
    model_sha256: str = ""
    features: tuple[str, ...] = field(init=False)
    drift: dict | None = None
    performance: dict | None = None
    challenger: object | None = None
    decision: dict | None = None
    comparison: dict | None = None
    previous: list = field(default_factory=list)

    def __post_init__(self):
        self.features = validate_model(self.champion)
        if self.id_col == self.target_col or self.id_col in self.features or self.target_col in self.features:
            raise ValueError("ID and target columns must differ from each other and model features")
        self.validate()
        if not self.model_sha256:
            self.model_sha256 = hashlib.sha256(joblib.hash(self.champion).encode()).hexdigest()

    def validate(self):
        ref_ids = _ids(self.reference, self.id_col, "Reference")
        new_ids = _ids(self.incoming, self.id_col, "Incoming")
        _features(self.reference, self.features, "Reference")
        _features(self.incoming, self.features, "Incoming")
        if ref_ids.intersection(new_ids):
            raise ValueError("Reference and incoming IDs overlap")
        if self.target_col in self.reference.columns:
            _labels(self.reference, self.target_col, "Reference")
        if self.target_col in self.incoming.columns:
            raise ValueError("Incoming features must not include labels; upload them separately")
        if self.labels is not None:
            label_ids = _ids(self.labels, self.id_col, "Delayed labels")
            _labels(self.labels, self.target_col, "Delayed labels")
            if new_ids != label_ids:
                raise ValueError("Incoming records and delayed labels must have identical ID sets")
        if self.validation is not None:
            validation_ids = _ids(self.validation, self.id_col, "Historical holdout")
            _features(self.validation, self.features, "Historical holdout")
            _labels(self.validation, self.target_col, "Historical holdout")
            if validation_ids & (ref_ids | new_ids):
                raise ValueError("Historical holdout overlaps reference or incoming IDs")
        if not set(self.categorical).issubset(self.features):
            raise ValueError("Unknown categorical model features")

    def monitor(self, threshold: float = 0.20) -> dict:
        self.drift = compute_drift(self.reference, self.incoming, self.features, self.categorical, threshold)
        self.performance = None
        if self.labels is not None:
            labeled = self.incoming.merge(
                self.labels[[self.id_col, self.target_col]],
                on=self.id_col, validate="one_to_one", sort=False,
            )
            self.performance = {"incoming": score(self.champion, labeled, self.features, self.target_col)}
            if self.validation is not None:
                self.performance["historical_holdout"] = score(
                    self.champion, self.validation, self.features, self.target_col
                )
        return {"drift": self.drift, "performance": self.performance}

    def train_challenger(self, min_f1_gain=0.01) -> dict:
        if self.labels is None or self.validation is None:
            raise ValueError("Retraining requires both delayed labels and an independent historical holdout")
        if self.target_col not in self.reference.columns:
            raise ValueError("Retraining requires labels in the reference training dataset")
        if len(self.reference) < 100 or len(self.incoming) < 100 or len(self.validation) < 50:
            raise ValueError("Retraining requires >=100 reference, >=100 incoming and >=50 validation rows")
        joined = self.incoming.merge(
            self.labels[[self.id_col, self.target_col]],
            on=self.id_col, validate="one_to_one", sort=False,
        )
        if self.reference[self.target_col].nunique() < 2 or joined[self.target_col].nunique() < 2:
            raise ValueError("Training cohorts must contain both classes")
        incoming_train, incoming_holdout = train_test_split(
            joined, test_size=0.2, random_state=42, stratify=joined[self.target_col]
        )
        training = pd.concat([self.reference, incoming_train], ignore_index=True)
        proposed = clone(self.champion)
        proposed.fit(training[list(self.features)], training[self.target_col])
        challenger_features = validate_model(proposed)
        if challenger_features != self.features:
            raise ValueError("Retrained model changed its feature contract")
        evaluation = {
            "historical_holdout": {
                "champion": score(self.champion, self.validation, self.features, self.target_col),
                "challenger": score(proposed, self.validation, self.features, self.target_col),
            },
            "incoming_holdout": {
                "champion": score(self.champion, incoming_holdout, self.features, self.target_col),
                "challenger": score(proposed, incoming_holdout, self.features, self.target_col),
            },
        }
        decision = candidate_decision(
            evaluation["historical_holdout"]["champion"],
            evaluation["historical_holdout"]["challenger"],
            evaluation["incoming_holdout"]["champion"],
            evaluation["incoming_holdout"]["challenger"],
            min_f1_gain=min_f1_gain,
        )
        self.challenger = proposed
        self.decision = decision
        self.comparison = evaluation
        return {"decision": decision, "evaluation": evaluation}

    def promote(self):
        if self.challenger is None or not self.decision or not self.decision["eligible"]:
            raise ValueError("No eligible challenger exists for promotion")
        self.previous.append((self.champion, self.model_sha256))
        self.champion = self.challenger
        self.model_sha256 = hashlib.sha256(joblib.hash(self.champion).encode()).hexdigest()
        self.challenger = None
        self.decision = None
        self.comparison = None
        self.monitor()

    def rollback(self):
        if not self.previous:
            raise ValueError("No earlier session champion is available")
        self.champion, self.model_sha256 = self.previous.pop()
        self.challenger = None
        self.decision = None
        self.comparison = None
        self.monitor()

    def summary(self):
        return {
            "workspace": self.name,
            "model_sha256": self.model_sha256,
            "features": list(self.features),
            "drift": self.drift,
            "performance": self.performance,
            "decision": self.decision,
            "comparison": self.comparison,
            "session_previous_versions": len(self.previous),
            "note": "Session-only; changes are not deployed to the public FastAPI service.",
        }
