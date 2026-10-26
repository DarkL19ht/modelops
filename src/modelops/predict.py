# Generate predictions

"""Create predictions before delayed labels become available."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modelops.train import FEATURES, ROOT


def create_predictions(
    incoming: pd.DataFrame,
    model,
) -> pd.DataFrame:

    required = set(FEATURES) | {"customer_id"}
    missing = required - set(incoming.columns)

    if missing:
        raise ValueError(
            f"Missing incoming columns: {sorted(missing)}"
        )

    if incoming.empty:
        raise ValueError("Incoming dataset is empty")

    ids = incoming["customer_id"]

    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "Incoming customer_id contains missing/blank values"
        )

    if ids.duplicated().any():
        raise ValueError(
            "Incoming customer_id contains duplicates"
        )

    features = incoming[FEATURES]

    try:
        values = (
            features
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Incoming features must be numeric"
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            "Incoming features contain missing/non-finite values"
        )

    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Model produced invalid probabilities"
        )

    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError(
            "Model produced nonbinary predictions"
        )

    return pd.DataFrame({
        "customer_id": ids.to_numpy(),
        "predicted_churn": predictions.astype(int),
        "churn_probability": probabilities,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an incoming customer cohort"
    )

    parser.add_argument(
        "--incoming",
        type=Path,
        default=ROOT / "data/generated/incoming.csv",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts/churn_model.joblib",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/generated/predictions.csv",
    )

    args = parser.parse_args()

    # Load only your own trusted model artifacts.
    model = joblib.load(args.model)

    incoming = pd.read_csv(
        args.incoming,
        dtype={"customer_id": str},
    )

    predictions = create_predictions(
        incoming,
        model,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        args.output,
        index=False,
    )

    print(
        f"Saved {len(predictions)} predictions "
        f"to {args.output}"
    )

    print(
        "No ground-truth churn labels "
        "were used during prediction."
    )


if __name__ == "__main__":
    main()