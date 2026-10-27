# Automated tests
import joblib
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.generate_data import make_customers
from src.modelops.track_experiment import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    flatten_metrics,
    track_dry_run,
)
from src.modelops.train import FEATURES


def test_flatten_metrics_separates_cohorts():
    report = {
        "evaluation": {
            "reference_holdout": {
                "champion": {
                    "accuracy": 0.80,
                    "f1": 0.60,
                }
            },
            "incoming_holdout": {
                "champion": {
                    "accuracy": 0.70,
                    "f1": 0.50,
                }
            },
        }
    }

    result = flatten_metrics(report, "champion")

    assert result["reference_holdout_accuracy"] == 0.80
    assert result["incoming_holdout_accuracy"] == 0.70
    assert result["reference_holdout_f1"] == 0.60
    assert result["incoming_holdout_f1"] == 0.50


def test_mlflow_tracks_models_without_promotion(tmp_path):
    reference = make_customers(
        n=600,
        seed=42,
    )

    incoming = make_customers(
        n=600,
        seed=202,
        shifted=True,
    )

    reference_path = tmp_path / "reference.csv"
    incoming_path = tmp_path / "incoming.csv"
    labels_path = tmp_path / "delayed_labels.csv"

    reference.to_csv(
        reference_path,
        index=False,
    )

    incoming.drop(
        columns="churn"
    ).to_csv(
        incoming_path,
        index=False,
    )

    incoming[
        ["customer_id", "churn"]
    ].to_csv(
        labels_path,
        index=False,
    )

    model = Pipeline([
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

    model.fit(
        reference[FEATURES],
        reference["churn"],
    )

    champion_path = tmp_path / "champion.joblib"

    joblib.dump(
        model,
        champion_path,
    )

    original_bytes = champion_path.read_bytes()

    tracking_uri = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
    )

    result = track_dry_run(
        reference_path=reference_path,
        incoming_path=incoming_path,
        labels_path=labels_path,
        champion_path=champion_path,
        candidate_dir=tmp_path / "candidates",
        report_dir=tmp_path / "reports",
        tracking_uri=tracking_uri,
        artifact_root=tmp_path / "mlruns",
        register_candidate=True,
    )

    # MLflow logging must not promote the candidate.
    assert champion_path.read_bytes() == original_bytes

    assert result["champion_run_id"]
    assert result["challenger_run_id"]
    assert result["registered_version"] is not None

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) == 2

    roles = {
        run.data.tags["modelops.role"]
        for run in runs
    }

    assert roles == {
        "champion",
        "challenger",
    }

    version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=result["registered_version"],
    )

    assert (
        version.tags["deployment_status"]
        == "not_deployed"
    )

    assert version.tags["review_status"] in {
        "eligible",
        "rejected",
    }