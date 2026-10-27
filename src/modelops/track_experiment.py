# Experiment tracking module

"""Track safe ModelOps retraining comparisons in local MLflow."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.modelops.retrain import run_retraining
from src.modelops.train import FEATURES, ROOT


EXPERIMENT_NAME = "modelops-churn-retraining"
REGISTERED_MODEL_NAME = "modelops-churn-candidates"

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "actual_churn_rate",
    "predicted_churn_rate",
)

def flatten_metrics(report: dict, role:str) -> dict[str, float]:
    """Extract a model's evaluation metrics from both holdouts."""
    result = {}


    for cohort in ('reference_holdout', 'incoming_holdout'):
        metrics = report['evaluation'][cohort][role]


        for name in METRIC_NAMES:
            value = metrics.get(name)

            if value is not None:
                result[f"{cohort}_{name}"] = float(value)

    return result



def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track_dry_run(
    reference_path: Path = ROOT / "data/generated/reference.csv",
    incoming_path: Path = ROOT / "data/generated/incoming.csv",
    labels_path: Path = ROOT / "data/generated/delayed_labels.csv",
    champion_path: Path = ROOT / "artifacts/churn_model.joblib",
    candidate_dir: Path = ROOT / "artifacts/candidates",
    report_dir: Path = ROOT / "reports",
    tracking_uri: str | None = None,
    artifact_root: Path = ROOT / "mlruns",
    register_candidate: bool = False,
) -> dict:
    """Run and track a comparison without promoting a model."""

    reference_path = Path(reference_path)
    incoming_path = Path(incoming_path)
    labels_path = Path(labels_path)
    champion_path = Path(champion_path)
    candidate_dir = Path(candidate_dir)
    report_dir = Path(report_dir)
    artifact_root = Path(artifact_root)

    candidate_path = candidate_dir / (
        f"churn_candidate_{uuid.uuid4().hex[:12]}.joblib"
    )

    report = run_retraining(
        reference_path=reference_path,
        incoming_path=incoming_path,
        labels_path=labels_path,
        champion_path=champion_path,
        candidate_path=candidate_path,
        archive_dir=champion_path.parent / "archive",
        report_dir=report_dir,
        promote=False,
    )

    if report["promotion"]["status"] != "not_requested":
        raise RuntimeError(
            "Tracking requires a non-promoting retraining run"
        )

    tracking_uri = tracking_uri or (
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    previous_uri = mlflow.get_tracking_uri()

    result = {
        "decision": (
            "eligible"
            if report["decision"]["eligible"]
            else "rejected"
        ),
        "registered_version": None,
        "run_group": report["run_id"],
        "tracking_uri": tracking_uri,
    }

    try:
        mlflow.set_tracking_uri(tracking_uri)

        artifact_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        experiment = mlflow.get_experiment_by_name(
            EXPERIMENT_NAME
        )

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=(
                    artifact_root.resolve().as_uri()
                ),
            )
        else:
            experiment_id = experiment.experiment_id

        dataset_hashes = {
            "reference_sha256": sha256_file(reference_path),
            "incoming_sha256": sha256_file(incoming_path),
            "delayed_labels_sha256": sha256_file(labels_path),
        }

        input_example = pd.read_csv(
            reference_path
        )[FEATURES].head(3)

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        for role, model_path in (
            ("champion", champion_path),
            ("challenger", candidate_path),
        ):
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{role}-{report['run_id']}",
            ) as run:

                mlflow.set_tags({
                    "modelops.role": role,
                    "modelops.group_id": report["run_id"],
                    "modelops.decision": result["decision"],
                    "modelops.promoted": "false",
                    **dataset_hashes,
                })

                mlflow.log_params({
                    "classifier": "LogisticRegression",
                    "feature_count": len(FEATURES),
                    "historical_training_rows": (
                        report["cohorts"]["historical_training_rows"]
                    ),
                    "incoming_training_rows": (
                        report["cohorts"]["incoming_training_rows"]
                    ),
                    "reference_holdout_rows": (
                        report["cohorts"]["historical_holdout_rows"]
                    ),
                    "incoming_holdout_rows": (
                        report["cohorts"]["incoming_holdout_rows"]
                    ),
                    "min_incoming_f1_gain": (
                        report["decision"]["thresholds"][
                            "min_incoming_f1_gain"
                        ]
                    ),
                    "model_file_sha256": sha256_file(model_path),
                })

                mlflow.log_metrics(
                    flatten_metrics(report, role)
                )

                mlflow.log_artifact(
                    str(report_dir / "retraining_decision.json")
                )

                mlflow.log_artifact(
                    str(report_dir / "retraining_decision.md")
                )

                # Joblib models must come from trusted project files.
                model = joblib.load(model_path)

                signature = infer_signature(
                    input_example,
                    model.predict(input_example),
                )

                model_info = mlflow.sklearn.log_model(
                    model,
                    name=f"{role}_model",
                    signature=signature,
                    input_example=input_example,
                )

                if role == "challenger" and register_candidate:
                    version = mlflow.register_model(
                        model_uri=model_info.model_uri,
                        name=REGISTERED_MODEL_NAME,
                    )

                    client.set_model_version_tag(
                        REGISTERED_MODEL_NAME,
                        version.version,
                        "review_status",
                        result["decision"],
                    )

                    client.set_model_version_tag(
                        REGISTERED_MODEL_NAME,
                        version.version,
                        "deployment_status",
                        "not_deployed",
                    )

                    result["registered_version"] = str(
                        version.version
                    )

                result[f"{role}_run_id"] = run.info.run_id

    finally:
        mlflow.set_tracking_uri(previous_uri)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Log a non-promoting retraining comparison in MLflow"
        )
    )

    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help=(
            "Register the candidate as a version "
            "without deploying it"
        ),
    )

    args = parser.parse_args()

    result = track_dry_run(
        register_candidate=args.register_candidate
    )

    print(json.dumps(result, indent=2))
    print("Active champion was not changed.")


if __name__ == "__main__":
    main()