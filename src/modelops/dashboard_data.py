"""Read and format ModelOps monitoring reports."""

import json
from pathlib import Path

import pandas as pd


def read_report(path: Path) -> dict | None:
    """Read an optional JSON monitoring report."""

    path = Path(path)

    if not path.exists():
        return None

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return data


def drift_table(report: dict) -> pd.DataFrame:
    """Create a table of feature drift metrics."""

    rows = []

    for feature in report["features"]:
        rows.append({
            "feature": feature["feature"],
            "psi": feature["psi"],
            "status": feature["status"],
            "reference_mean": feature["reference_mean"],
            "incoming_mean": feature["incoming_mean"],
            "ks_statistic": feature["ks_statistic"],
        })

    return pd.DataFrame(rows)


def performance_table(report: dict) -> pd.DataFrame:
    """Compare holdout baseline and incoming metrics."""

    rows = []

    for metric in ("accuracy", "f1", "roc_auc"):
        rows.append({
            "metric": metric,
            "baseline": report["baseline"][metric],
            "incoming": report["metrics"][metric],
        })

    return pd.DataFrame(rows)


def retraining_table(report: dict) -> pd.DataFrame:
    """Compare champion and challenger evaluations."""

    rows = []

    for cohort in (
        "reference_holdout",
        "incoming_holdout",
    ):
        evaluation = report["evaluation"][cohort]

        for metric in ("accuracy", "f1", "roc_auc"):
            rows.append({
                "cohort": cohort,
                "metric": metric,
                "champion": evaluation["champion"][metric],
                "challenger": evaluation["challenger"][metric],
            })

    return pd.DataFrame(rows)