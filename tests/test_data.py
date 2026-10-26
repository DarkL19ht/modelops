# Add automated tests

from src.modelops.generate_data import make_customers
from src.modelops.train import FEATURES


def test_generated_data_has_required_columns():
    customers = make_customers(
        n=100,
        seed=1,
    )

    assert len(customers) == 100

    required = FEATURES + [
        "customer_id",
        "churn",
    ]

    assert set(required).issubset(
        customers.columns
    )

    assert not customers.isna().any().any()

    assert set(
        customers["churn"].unique()
    ).issubset({0, 1})


def test_shifted_data_has_lower_activity():
    reference = make_customers(
        n=2000,
        seed=1,
    )

    incoming = make_customers(
        n=2000,
        seed=2,
        shifted=True,
    )

    assert (
        incoming["logins_30d"].mean()
        < reference["logins_30d"].mean()
    )

    assert (
        incoming["support_tickets_90d"].mean()
        > reference["support_tickets_90d"].mean()
    )
