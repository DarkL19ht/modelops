#Evaluate model performance

"""Evaluate delayed churn labels against a historical baseline."""

import argparse
import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modelops.train import ROOT


def _check_ids(frame: pd.DataFrame, name: str) -> None:
    ids = frame["customer_id"]

    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            f"{name} contains missing/blank customer_id"
        )

    if ids.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate customer_id"
        )


def _binary(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(
            series,
            errors="raise",
        ).to_numpy(dtype=float)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain binary values"
        ) from exc

    if (
        not np.isfinite(values).all()
        or not np.isin(values, [0, 1]).all()
    ):
        raise ValueError(
            f"{name} must contain only 0 and 1"
        )

    return values.astype(int)


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    min_labels: int = 100,
) -> dict:

    if min_labels < 1:
        raise ValueError(
            "min_labels must be positive"
        )

    for frame, required, name in [
        (
            predictions,
            {
                "customer_id",
                "predicted_churn",
                "churn_probability",
            },
            "Predictions",
        ),
        (
            labels,
            {"customer_id", "churn"},
            "Labels",
        ),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

        _check_ids(frame, name)

    if len(predictions) < min_labels:
        raise ValueError(
            f"Need at least {min_labels} labeled predictions"
        )

    pred_ids = set(predictions["customer_id"])
    label_ids = set(labels["customer_id"])

    if pred_ids != label_ids:
        raise ValueError(
            f"ID mismatch: "
            f"{len(pred_ids - label_ids)} predictions without labels; "
            f"{len(label_ids - pred_ids)} labels without predictions"
        )

    paired = predictions.merge(
        labels[["customer_id", "churn"]],
        on="customer_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    actual = _binary(
        paired["churn"],
        "churn",
    )

    predicted = _binary(
        paired["predicted_churn"],
        "predicted_churn",
    )

    try:
        scores = pd.to_numeric(
            paired["churn_probability"],
            errors="raise",
        ).to_numpy(dtype=float)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "churn_probability must be numeric"
        ) from exc

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "churn_probability must be finite "
            "and between 0 and 1"
        )

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "labeled_rows": len(paired),

        "actual_churn_rate": round(
            float(actual.mean()), 4
        ),

        "predicted_churn_rate": round(
            float(predicted.mean()), 4
        ),

        "accuracy": round(
            float(accuracy_score(actual, predicted)),
            4,
        ),

        "precision": round(
            float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),

        "recall": round(
            float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),

        "f1": round(
            float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),

        "roc_auc": (
            round(
                float(
                    roc_auc_score(actual, scores)
                ),
                4,
            )
            if len(np.unique(actual)) == 2
            else None
        ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def assess_performance(
    current: dict,
    baseline: dict,
    accuracy_drop: float = 0.05,
    f1_drop: float = 0.05,
    auc_drop: float = 0.05,
) -> dict:

    if (
        accuracy_drop < 0
        or f1_drop < 0
        or auc_drop < 0
    ):
        raise ValueError(
            "Alert thresholds cannot be negative"
        )

    for key in ("accuracy", "f1", "roc_auc"):
        if key not in baseline:
            raise ValueError(
                f"Baseline missing {key}"
            )

    accuracy_delta = round(
        current["accuracy"] - float(baseline["accuracy"]),
        4,
    )

    f1_delta = round(
        current["f1"] - float(baseline["f1"]),
        4,
    )

    auc_delta = (
        round(
            current["roc_auc"] - float(baseline["roc_auc"]),
            4,
        )
        if current["roc_auc"] is not None
        else None
    )

    alerts = []

    if accuracy_delta <= -accuracy_drop:
        alerts.append(
            "Accuracy decreased by at least "
            "the configured threshold"
        )

    if f1_delta <= -f1_drop:
        alerts.append(
            "F1 decreased by at least "
            "the configured threshold"
        )

    if (
        auc_delta is not None
        and auc_delta <= -auc_drop
    ):
        alerts.append(
            "ROC-AUC decreased by at least "
            "the configured threshold"
        )

    status = (
        "degraded"
        if alerts
        else (
            "insufficient_evidence"
            if auc_delta is None
            else "stable"
        )
    )

    return {
        "status": status,
        "accuracy_delta": accuracy_delta,
        "f1_delta": f1_delta,
        "roc_auc_delta": auc_delta,
        "alerts": alerts,
        "accuracy_drop_threshold": accuracy_drop,
        "f1_drop_threshold": f1_drop,
        "auc_drop_threshold": auc_drop,
    }


def save_reports(report: dict, directory: Path) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (directory / "performance_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = report["metrics"]
    baseline = report["baseline"]
    assessment = report["assessment"]

    lines = [
        "# ModelOps — Performance Monitoring Report",
        "",
        f"- Monitored observations: **{metrics['labeled_rows']:,}**",
        f"- Assessment: **{assessment['status']}**",
        "",
        "| Metric | Holdout baseline | Incoming labeled cohort |",
        "|---|---:|---:|",
    ]

    for name in ("accuracy", "f1", "roc_auc"):
        value = metrics[name]
        current_text = (
            f"{value:.4f}"
            if value is not None
            else "N/A"
        )

        lines.append(
            f"| {name} | {baseline[name]:.4f} "
            f"| {current_text} |"
        )

    lines.extend([
        "",
        f"- Precision (incoming): {metrics['precision']:.4f}",
        f"- Recall (incoming): {metrics['recall']:.4f}",
        f"- Actual churn rate: {metrics['actual_churn_rate']:.4f}",
        f"- Predicted churn rate: {metrics['predicted_churn_rate']:.4f}",
        "",
        "## Alerts",
        "",
    ])

    lines.extend(
        [f"- {alert}" for alert in assessment["alerts"]]
        or ["No threshold breach detected."]
    )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- The baseline comes from a held-out historical split.",
        "- The incoming cohort has a shifted distribution.",
        "- Performance differences do not prove drift caused them.",
        "- All predictions must have matching delayed labels.",
        "- ROC-AUC is unavailable when only one outcome class exists.",
        "",
    ])

    (directory / "performance_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    history_path = (
        directory / "performance_history.jsonl"
    )

    existing_ids = set()

    if history_path.exists():
        for line in history_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                existing_ids.add(
                    json.loads(line).get("run_id")
                )

    if report["run_id"] not in existing_ids:
        with history_path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(report) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved predictions "
            "using delayed labels"
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "data/generated/predictions.csv",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/generated/delayed_labels.csv",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts/baseline_metrics.json",
    )

    parser.add_argument(
        "--min-labels",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--accuracy-drop",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--f1-drop",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--auc-drop",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
    )

    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions,
        dtype={"customer_id": str},
    )

    labels = pd.read_csv(
        args.labels,
        dtype={"customer_id": str},
    )

    baseline = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )

    metrics = evaluate_predictions(
        predictions,
        labels,
        min_labels=args.min_labels,
    )

    assessment = assess_performance(
        metrics,
        baseline,
        accuracy_drop=args.accuracy_drop,
        f1_drop=args.f1_drop,
        auc_drop=args.auc_drop,
    )

    contents = (
        args.predictions.read_bytes()
        + args.labels.read_bytes()
        + args.baseline.read_bytes()
        + (
            f"{args.min_labels}:"
            f"{args.accuracy_drop}:"
            f"{args.f1_drop}:"
            f"{args.auc_drop}"
        ).encode()
    )

    report = {
        "run_id": hashlib.sha256(
            contents
        ).hexdigest()[:16],

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),

        "baseline": {
            key: baseline[key]
            for key in (
                "accuracy",
                "f1",
                "roc_auc",
            )
        },

        "metrics": metrics,
        "assessment": assessment,
    }

    save_reports(
        report,
        args.output_dir,
    )

    print(
        f"Evaluated {metrics['labeled_rows']} customers"
    )

    print(
        f"Assessment: {assessment['status']}"
    )

    print(
        f"Alerts: {assessment['alerts'] or 'none'}"
    )

    print(
        f"Reports: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()