"""Feature-distribution monitoring for the ModelOps churn example."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.modelops.train import FEATURES


NUMERIC_FEATURES = [
    feature
    for feature in FEATURES
    if feature != "monthly_contract"
]

CATEGORICAL_FEATURES = ["monthly_contract"]


def _validate(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} dataset is empty")

    missing = sorted(
        set(FEATURES) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{name} dataset is missing features: {missing}"
        )

    try:
        matrix = (
            frame[FEATURES]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} dataset has non-numeric feature values"
        ) from exc

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} dataset contains missing or "
            "non-finite feature values"
        )

    if not set(
        frame["monthly_contract"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"{name} monthly_contract must contain only 0 or 1"
        )


def _psi_from_counts(
    reference_counts: np.ndarray,
    incoming_counts: np.ndarray,
) -> float:
    # Avoid division by zero for empty bins.
    smoothing = 1e-6
    n_bins = len(reference_counts)

    p = (
        (reference_counts + smoothing)
        / (
            reference_counts.sum()
            + smoothing * n_bins
        )
    )

    q = (
        (incoming_counts + smoothing)
        / (
            incoming_counts.sum()
            + smoothing * n_bins
        )
    )

    return max(
        0.0,
        float(
            np.sum((q - p) * np.log(q / p))
        ),
    )


def numeric_psi(
    reference: np.ndarray,
    incoming: np.ndarray,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")

    reference = np.asarray(
        reference,
        dtype=float,
    )

    incoming = np.asarray(
        incoming,
        dtype=float,
    )

    # Handle a constant reference distribution.
    if np.all(reference == reference[0]):
        ref_bins = np.where(
            reference == reference[0],
            0,
            1,
        )

        new_bins = np.where(
            incoming == reference[0],
            0,
            1,
        )

        return _psi_from_counts(
            np.bincount(ref_bins, minlength=2),
            np.bincount(new_bins, minlength=2),
        )

    # Build bin boundaries using reference data only.
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)[1:-1],
        )
    )

    full_edges = np.concatenate(
        ([-np.inf], edges, [np.inf])
    )

    reference_counts = np.histogram(
        reference,
        bins=full_edges,
    )[0]

    incoming_counts = np.histogram(
        incoming,
        bins=full_edges,
    )[0]

    return _psi_from_counts(
        reference_counts,
        incoming_counts,
    )


def categorical_psi(
    reference: np.ndarray,
    incoming: np.ndarray,
) -> float:
    categories = sorted(
        set(reference) | set(incoming)
    )

    ref_counts = np.array([
        (reference == value).sum()
        for value in categories
    ])

    new_counts = np.array([
        (incoming == value).sum()
        for value in categories
    ])

    return _psi_from_counts(
        ref_counts,
        new_counts,
    )


def calculate_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    threshold: float = 0.20,
) -> dict:
    """
    Compare the input features of two datasets.

    The incoming dataset does not need churn labels.
    """
    if threshold <= 0:
        raise ValueError(
            "PSI threshold must be positive"
        )

    _validate(reference, "Reference")
    _validate(incoming, "Incoming")

    results = []

    for feature in FEATURES:
        ref = reference[feature].to_numpy()
        new = incoming[feature].to_numpy()

        if feature in CATEGORICAL_FEATURES:
            psi = categorical_psi(ref, new)

            result = {
                "feature": feature,
                "type": "categorical",
                "psi": round(psi, 6),
                "reference_mean": round(
                    float(np.mean(ref)), 4
                ),
                "incoming_mean": round(
                    float(np.mean(new)), 4
                ),
                "ks_statistic": None,
                "ks_pvalue": None,
            }

        else:
            psi = numeric_psi(ref, new)

            ks = ks_2samp(
                ref,
                new,
                method="asymp",
            )

            result = {
                "feature": feature,
                "type": "numeric",
                "psi": round(psi, 6),
                "reference_mean": round(
                    float(np.mean(ref)), 4
                ),
                "incoming_mean": round(
                    float(np.mean(new)), 4
                ),
                "ks_statistic": round(
                    float(ks.statistic), 6
                ),
                "ks_pvalue": float(ks.pvalue),
            }

        result["status"] = (
            "drift"
            if psi >= threshold
            else "stable"
        )

        results.append(result)

    return {
        "reference_rows": len(reference),
        "incoming_rows": len(incoming),
        "psi_threshold": threshold,
        "drifted_features": [
            result["feature"]
            for result in results
            if result["status"] == "drift"
        ],
        "features": results,
    }