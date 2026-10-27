# Build the retraining pipeline
"""Guarded challenger training and manual model promotion."""

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.predict import create_predictions
from src.modelops.performance import evaluate_predictions
from src.modelops.train import FEATURES, ROOT


def _validate_ids(frame: pd.DataFrame, label: str) -> None:
    if "customer_id" not in frame.columns:
        raise ValueError(f"{label} missing customer_id")

    ids = frame["customer_id"]

    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains blank customer_id")

    if ids.duplicated().any():
        raise ValueError(f"{label} contains duplicate customer_id")


def _validate_labeled(frame: pd.DataFrame, label: str) -> None:
    required = set(FEATURES) | {"customer_id", "churn"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{label} missing columns: {sorted(missing)}"
        )

    _validate_ids(frame, label)

    try:
        values = frame[FEATURES].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=float)

        outcomes = pd.to_numeric(
            frame["churn"], errors="raise"
        ).to_numpy(dtype=float)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} contains nonnumeric values"
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} contains invalid feature values"
        )

    if (
        not np.isfinite(outcomes).all()
        or not np.isin(outcomes, [0, 1]).all()
    ):
        raise ValueError(
            f"{label} churn labels must be 0 or 1"
        )

    if len(frame) < 500 or frame["churn"].nunique() < 2:
        raise ValueError(
            f"{label} requires at least 500 rows "
            "and both outcome classes"
        )


def prepare_cohorts(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create training and evaluation partitions."""

    _validate_labeled(reference, "Reference")
    _validate_ids(incoming, "Incoming")
    _validate_ids(labels, "Delayed labels")

    if "churn" in incoming.columns:
        raise ValueError(
            "Incoming features must not contain churn labels"
        )

    if "churn" not in labels.columns:
        raise ValueError("Delayed labels missing churn")

    if set(incoming["customer_id"]) != set(labels["customer_id"]):
        raise ValueError(
            "Incoming and delayed labels have an ID mismatch"
        )

    if set(reference["customer_id"]) & set(incoming["customer_id"]):
        raise ValueError(
            "Reference and incoming customer IDs overlap"
        )

    joined = incoming.merge(
        labels[["customer_id", "churn"]],
        on="customer_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    _validate_labeled(joined, "Incoming")

    # Reproduce the original historical split from train.py.
    historical_train, historical_holdout = train_test_split(
        reference,
        test_size=0.20,
        stratify=reference["churn"],
        random_state=42,
    )

    # Hold incoming observations out of challenger fitting.
    incoming_train, incoming_holdout = train_test_split(
        joined,
        test_size=0.20,
        stratify=joined["churn"],
        random_state=42,
    )

    return (
        historical_train,
        historical_holdout,
        incoming_train,
        incoming_holdout,
    )


def train_challenger(
    historical_train: pd.DataFrame,
    incoming_train: pd.DataFrame,
):
    training = pd.concat(
        [historical_train, incoming_train],
        ignore_index=True,
    )

    challenger = Pipeline([
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

    challenger.fit(
        training[FEATURES],
        training["churn"],
    )

    return challenger


def evaluate_model(model, cohort: pd.DataFrame) -> dict:
    predictions = create_predictions(
        cohort.drop(columns="churn"),
        model,
    )

    labels = cohort[["customer_id", "churn"]]

    return evaluate_predictions(
        predictions,
        labels,
        min_labels=100,
    )


def assess_candidate(
    champion_reference: dict,
    challenger_reference: dict,
    champion_incoming: dict,
    challenger_incoming: dict,
    min_incoming_f1_gain: float = 0.01,
    max_reference_accuracy_drop: float = 0.03,
    max_reference_auc_drop: float = 0.02,
    max_incoming_auc_drop: float = 0.02,
) -> dict:

    limits = {
        "min_incoming_f1_gain": min_incoming_f1_gain,
        "max_reference_accuracy_drop": max_reference_accuracy_drop,
        "max_reference_auc_drop": max_reference_auc_drop,
        "max_incoming_auc_drop": max_incoming_auc_drop,
    }

    if any(
        not np.isfinite(value) or value < 0
        for value in limits.values()
    ):
        raise ValueError(
            "Acceptance thresholds must be finite "
            "and nonnegative"
        )

    evaluations = (
        ("Champion reference", champion_reference),
        ("Challenger reference", challenger_reference),
        ("Champion incoming", champion_incoming),
        ("Challenger incoming", challenger_incoming),
    )

    for name, metrics in evaluations:
        if metrics.get("roc_auc") is None:
            return {
                "eligible": False,
                "checks": {},
                "reasons": [f"{name} ROC-AUC unavailable"],
                "thresholds": limits,
            }

    checks = {
        "incoming_f1_gain": (
            challenger_incoming["f1"]
            - champion_incoming["f1"]
            >= min_incoming_f1_gain
        ),
        "reference_accuracy_guardrail": (
            challenger_reference["accuracy"]
            >= champion_reference["accuracy"]
            - max_reference_accuracy_drop
        ),
        "reference_auc_guardrail": (
            challenger_reference["roc_auc"]
            >= champion_reference["roc_auc"]
            - max_reference_auc_drop
        ),
        "incoming_auc_guardrail": (
            challenger_incoming["roc_auc"]
            >= champion_incoming["roc_auc"]
            - max_incoming_auc_drop
        ),
    }

    return {
        "eligible": all(checks.values()),
        "checks": {
            name: bool(passed)
            for name, passed in checks.items()
        },
        "reasons": [
            name
            for name, passed in checks.items()
            if not passed
        ],
        "thresholds": limits,
    }


def promote_candidate(
        candidate_path: Path,
        champion_path: Path,
        archive_dir: Path,
) -> Path:
    """""Back up the champion before replacing it."""

    candidate_path = Path(candidate_path)
    champion_path = Path(champion_path)
    archive_dir = Path(archive_dir)

    # Only load artifacts created by this trusted pipeline.
    joblib.load(candidate_path)

    if not champion_path.is_file():
        raise FileNotFoundError(
            f"Champion model missing: {champion_path}"
        )


    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    champion_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup = archive_dir / (
        f"champion_{timestamp}_{uuid.uuid4().hex[:8]}.joblib"
    )


    # Preserve the old model before changing anything.
    shutil.copy2(champion_path, backup)

    # Stage the new file in the same directory so the final replacement can be atomic on one filesystem.
    temporary = champion_path.with_name(
        f".{champion_path.name}.{uuid.uuid4().hex}.tmp"
    )


    try:
        shutil.copy2(candidate_path, temporary)
        os.replace(temporary, champion_path)
    finally:
        temporary.unlink(missing_ok=True)

    return backup



def save_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir / "retraining_decision.json"
    ).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ModelOps — Retraining Decision",
        "",
        f"- Run: `{report['run_id']}`",
        "- Decision: **"
        + (
            "eligible"
            if report["decision"]["eligible"]
            else "rejected"
        )
        + "**",
        f"- Promotion: **{report['promotion']['status']}**",
        f"- Historical training rows: "
        f"**{report['cohorts']['historical_training_rows']}**",
        f"- Incoming training rows: "
        f"**{report['cohorts']['incoming_training_rows']}**",
        "",
        "| Evaluation cohort | Metric | Champion | Challenger |",
        "|---|---|---:|---:|",
    ]

    for cohort in (
        "reference_holdout",
        "incoming_holdout",
    ):
        for metric in (
            "accuracy",
            "f1",
            "roc_auc",
        ):
            previous = report[
                "evaluation"
            ][cohort]["champion"][metric]

            proposed = report[
                "evaluation"
            ][cohort]["challenger"][metric]

            lines.append(
                f"| {cohort} | {metric} "
                f"| {previous:.4f} | {proposed:.4f} |"
            )

    lines.extend([
        "",
        "## Promotion checks",
        "",
    ])

    for name, passed in report["decision"]["checks"].items():
        result = "PASS" if passed else "FAIL"
        lines.append(f"- {name}: **{result}**")

    lines.extend([
        "",
        "## Model and backup paths",
        "",
        f"- Champion: `{report['champion_path']}`",
        f"- Candidate: `{report['candidate_path']}`",
        "- Backup: `"
        + (report["promotion"]["backup"] or "none")
        + "`",
        "",
        "## Interpretation",
        "",
        "The challenger is trained on historical training "
        "observations and an incoming training partition. "
        "Both models are evaluated on the same historical "
        "and incoming holdouts, which were not used for "
        "challenger fitting.",
        "",
        "The incoming holdout was previously visible "
        "during exploratory monitoring, so it is not "
        "an independent final external test set.",
        "",
        "Passing thresholds permits promotion only when "
        "explicitly requested. Synthetic test results "
        "do not establish real-world generalisation.",
        "",
    ])

    (
        output_dir / "retraining_decision.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    with (
        output_dir / "retraining_history.jsonl"
    ).open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report) + "\n"
        )


def run_retraining(
    reference_path: Path,
    incoming_path: Path,
    labels_path: Path,
    champion_path: Path,
    candidate_path: Path,
    archive_dir: Path,
    report_dir: Path,
    promote: bool = False,
    min_incoming_f1_gain: float = 0.01,
) -> dict:

    reference = pd.read_csv(
        reference_path,
        dtype={"customer_id": str},
    )

    incoming = pd.read_csv(
        incoming_path,
        dtype={"customer_id": str},
    )

    labels = pd.read_csv(
        labels_path,
        dtype={"customer_id": str},
    )

    (
        historical_train,
        historical_holdout,
        incoming_train,
        incoming_holdout,
    ) = prepare_cohorts(
        reference,
        incoming,
        labels,
    )

    if not champion_path.is_file():
        raise FileNotFoundError(
            f"Champion model missing: {champion_path}"
        )

    original_hash = hashlib.sha256(
        champion_path.read_bytes()
    ).hexdigest()

    # Joblib relies on pickle; load only your own model.
    champion = joblib.load(champion_path)

    challenger = train_challenger(
        historical_train,
        incoming_train,
    )

    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(challenger, candidate_path)

    results = {
        "reference_holdout": {
            "champion": evaluate_model(
                champion,
                historical_holdout,
            ),
            "challenger": evaluate_model(
                challenger,
                historical_holdout,
            ),
        },
        "incoming_holdout": {
            "champion": evaluate_model(
                champion,
                incoming_holdout,
            ),
            "challenger": evaluate_model(
                challenger,
                incoming_holdout,
            ),
        },
    }

    decision = assess_candidate(
        results["reference_holdout"]["champion"],
        results["reference_holdout"]["challenger"],
        results["incoming_holdout"]["champion"],
        results["incoming_holdout"]["challenger"],
        min_incoming_f1_gain=min_incoming_f1_gain,
    )

    promotion = {
        "status": "not_requested",
        "backup": None,
    }

    if promote and not decision["eligible"]:
        promotion["status"] = "rejected"

    elif promote:
        backup = promote_candidate(
            candidate_path,
            champion_path,
            archive_dir,
        )

        promotion = {
            "status": "promoted",
            "backup": str(backup),
        }

    report = {
        "run_id": uuid.uuid4().hex[:12],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "champion_sha256_before": original_hash,
        "champion_sha256_after": hashlib.sha256(
            champion_path.read_bytes()
        ).hexdigest(),
        "champion_path": str(champion_path),
        "candidate_path": str(candidate_path),
        "cohorts": {
            "historical_training_rows": len(historical_train),
            "historical_holdout_rows": len(historical_holdout),
            "incoming_training_rows": len(incoming_train),
            "incoming_holdout_rows": len(incoming_holdout),
        },
        "evaluation": results,
        "decision": decision,
        "promotion": promotion,
    }

    save_report(report, report_dir)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guarded churn model retraining"
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/generated/reference.csv",
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        default=ROOT / "data/generated/incoming.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/generated/delayed_labels.csv",
    )
    parser.add_argument(
        "--champion",
        type=Path,
        default=ROOT / "artifacts/churn_model.joblib",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "artifacts/candidates/churn_candidate.joblib",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "artifacts/archive",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports",
    )
    parser.add_argument(
        "--min-incoming-f1-gain",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote only if all checks pass",
    )

    args = parser.parse_args()

    report = run_retraining(
        args.reference,
        args.incoming,
        args.labels,
        args.champion,
        args.candidate,
        args.archive_dir,
        args.report_dir,
        args.promote,
        args.min_incoming_f1_gain,
    )

    print(
        "Candidate eligible:",
        report["decision"]["eligible"],
    )
    print(
        "Promotion:",
        report["promotion"]["status"],
    )
    print(
        "Failed checks:",
        report["decision"]["reasons"] or "none",
    )
    print(
        "Report:",
        (args.report_dir / "retraining_decision.md").resolve(),
    )


if __name__ == "__main__":
    main()