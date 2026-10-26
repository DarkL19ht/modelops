# Generate realistic simulated customer data

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def make_customers(
    n: int,
    seed: int,
    shifted: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n)

    spending = np.clip(
        rng.normal(
            85 if shifted else 70,
            22,
            n,
        ),
        15,
        200,
    )

    tickets = rng.poisson(
        2.0 if shifted else 0.8,
        size=n,
    )

    logins = rng.poisson(
        8 if shifted else 15,
        size=n,
    )

    monthly = rng.binomial(
        1,
        0.72 if shifted else 0.55,
        size=n,
    )

    # Simulated relationship between
    # customer behaviour and churn risk.
    logits = (
        -1.1
        + 1.7 * monthly
        + 0.8 * tickets
        - 0.11 * logins
        - 0.025 * tenure
        + 0.004 * (spending - 70)
    )

    churn_probability = 1 / (
        1 + np.exp(-logits)
    )

    churn = rng.binomial(
        1,
        churn_probability,
    )

    prefix = "P" if shifted else "R"

    return pd.DataFrame({
        "customer_id": [
            f"{prefix}{i:06d}"
            for i in range(n)
        ],
        "tenure_months": tenure,
        "monthly_spend": np.round(spending, 2),
        "support_tickets_90d": tickets,
        "logins_30d": logins,
        "monthly_contract": monthly,
        "churn": churn,
    })


def main() -> None:
    output_dir = ROOT / "data" / "generated"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = make_customers(
        n=5000,
        seed=42,
    )

    incoming = make_customers(
        n=1200,
        seed=202,
        shifted=True,
    )

    reference.to_csv(
        output_dir / "reference.csv",
        index=False,
    )

    # Production-style incoming data:
    # labels are not immediately available.
    incoming.drop(
        columns="churn"
    ).to_csv(
        output_dir / "incoming.csv",
        index=False,
    )

    # Keep labels separately to simulate
    # delayed ground-truth availability.
    incoming[
        ["customer_id", "churn"]
    ].to_csv(
        output_dir / "delayed_labels.csv",
        index=False,
    )

    print(
        f"Generated {len(reference)} "
        "reference customers"
    )

    print(
        f"Generated {len(incoming)} "
        "incoming customers"
    )

    print(
        "Saved datasets under data/generated/"
    )


if __name__ == "__main__":
    main()
