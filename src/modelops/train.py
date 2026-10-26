# Train the baseline ML model

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "logins_30d",
    "monthly_contract",
]


def main() -> None:
    data_path = (
        ROOT / "data" / "generated" / "reference.csv"
    )

    frame = pd.read_csv(data_path)

    X = frame[FEATURES]
    y = frame["churn"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
    )

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    metrics = {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
            4,
        ),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }

    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        artifact_dir / "churn_model.joblib",
    )

    (
        artifact_dir / "baseline_metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("Saved model and metrics under artifacts/")


if __name__ == "__main__":
    main()
