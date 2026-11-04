"""Deterministic, read-only portfolio demonstration using synthetic data.

No accounts, uploaded artifacts, external services, persistence, or model promotion.
Every score displayed by the dashboard is computed from the supplied ML pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.drift import calculate_drift
from src.modelops.generate_data import make_customers
from src.modelops.retrain import (
    assess_candidate,
    evaluate_model,
    prepare_cohorts,
    train_challenger,
)
from src.modelops.train import FEATURES


@dataclass(frozen=True)
class DemoWorkspace:
    """In-memory models and measured evaluation results for the public demo."""

    champion: Pipeline
    challenger: Pipeline
    drift: dict
    evaluations: dict
    decision: dict
    reference_rows: int
    incoming_rows: int


def build_demo() -> DemoWorkspace:
    """Generate trusted synthetic cohorts and reproduce the guarded comparison.

    This function has no filesystem writes, network calls or database access.
    The Streamlit page caches its return value to avoid repeated training.
    """

    reference = make_customers(n=5000, seed=42)
    incoming_labeled = make_customers(n=1200, seed=202, shifted=True)
    incoming_features = incoming_labeled.drop(columns="churn")
    delayed_labels = incoming_labeled[["customer_id", "churn"]]

    historical_train, historical_holdout, incoming_train, incoming_holdout = (
        prepare_cohorts(reference, incoming_features, delayed_labels)
    )

    champion = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )),
    ])
    champion.fit(historical_train[FEATURES], historical_train["churn"])
    challenger = train_challenger(historical_train, incoming_train)

    evaluations = {
        "reference_holdout": {
            "champion": evaluate_model(champion, historical_holdout),
            "challenger": evaluate_model(challenger, historical_holdout),
        },
        "incoming_holdout": {
            "champion": evaluate_model(champion, incoming_holdout),
            "challenger": evaluate_model(challenger, incoming_holdout),
        },
    }
    decision = assess_candidate(
        evaluations["reference_holdout"]["champion"],
        evaluations["reference_holdout"]["challenger"],
        evaluations["incoming_holdout"]["champion"],
        evaluations["incoming_holdout"]["challenger"],
    )
    return DemoWorkspace(
        champion=champion,
        challenger=challenger,
        drift=calculate_drift(reference, incoming_features),
        evaluations=evaluations,
        decision=decision,
        reference_rows=len(reference),
        incoming_rows=len(incoming_features),
    )


def predict_demo(
    workspace: DemoWorkspace,
    features: dict,
    *,
    version: str = "champion",
) -> dict:
    """Predict one synthetic customer with a named, prebuilt demo model."""

    if version not in {"champion", "challenger"}:
        raise ValueError("version must be champion or challenger")
    if set(features) != set(FEATURES):
        raise ValueError("Prediction fields must match the demo model schema")

    frame = pd.DataFrame([features], columns=FEATURES)
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All demo features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Demo features must be finite numbers")
    if values[0, FEATURES.index("monthly_contract")] not in (0, 1):
        raise ValueError("monthly_contract must be 0 or 1")

    model = workspace.champion if version == "champion" else workspace.challenger
    label = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Model returned an invalid probability")
    return {
        "version": version,
        "predicted_churn": label,
        "churn_probability": round(probability, 6),
    }
