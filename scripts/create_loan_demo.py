"""Generate a completely separate loan-default model and its own upload datasets."""
from pathlib import Path
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("demo_uploads/loan_default")
FEATURES = ["annual_income", "debt_ratio", "late_payments", "account_age_years"]


def generate(n, seed, prefix, shift=False):
    rng = np.random.default_rng(seed)
    income = np.clip(rng.normal(46000 if shift else 61000, 17000, n), 12000, 160000)
    ratio = np.clip(rng.beta(3.5 if shift else 2, 4, n), 0, 1)
    late = rng.poisson(2.2 if shift else 0.7, n)
    age = rng.integers(1, 25, n)
    z = -2.0 - 0.000017 * (income - 45000) + 2.4 * ratio + 0.55 * late - 0.025 * age
    labels = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({
        "record_id": [f"{prefix}{i:06d}" for i in range(n)],
        "annual_income": income.round(2),
        "debt_ratio": ratio.round(4),
        "late_payments": late,
        "account_age_years": age,
        "target": labels,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = generate(800, 101, "R")
    validation = generate(250, 102, "V")
    incoming = generate(500, 103, "I", shift=True)
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(reference[FEATURES], reference["target"])
    sio.dump(clf, OUT / "loan_default.skops")
    reference.to_csv(OUT / "reference.csv", index=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    incoming.drop(columns="target").to_csv(OUT / "incoming.csv", index=False)
    incoming[["record_id", "target"]].to_csv(OUT / "delayed_labels.csv", index=False)
    print("Created files in", OUT)
    print("Model features:", list(clf.feature_names_in_))
    print("Untrusted model types:", sio.get_untrusted_types(file=OUT / "loan_default.skops"))


if __name__ == "__main__":
    main()
