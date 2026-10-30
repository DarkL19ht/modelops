#Build the dashboard interface
"""Interactive ModelOps monitoring and prediction dashboard."""

import os

import pandas as pd
import requests
import streamlit as st

from src.modelops.dashboard_data import (
    read_report,
    drift_table,
    performance_table,
    retraining_table,
)
from src.modelops.train import ROOT


API_URL = os.getenv(
    "MODELOPS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REPORTS = ROOT / "reports"


st.set_page_config(
    page_title="ModelOps | Monitoring Workbench",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    div[data-testid="stMetric"] {
        background: rgba(110, 138, 150, 0.08);
        border: 1px solid rgba(128, 150, 160, 0.20);
        border-radius: 12px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 150, 160, 0.15);
    }

    .eyebrow {
        text-transform: uppercase;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: #3a9e86;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_api(path: str) -> dict | None:
    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def format_number(value, digits=3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def load_reports():
    return {
        "drift": read_report(
            REPORTS / "drift_report.json"
        ),
        "performance": read_report(
            REPORTS / "performance_report.json"
        ),
        "retraining": read_report(
            REPORTS / "retraining_decision.json"
        ),
    }


with st.sidebar:
    st.title("MODEL / OPS")
    st.caption("Machine Learning Operations")

    st.divider()

    st.markdown("**Inference service**")
    st.code(API_URL, language=None)

    health = fetch_api("/health")

    if health is not None:
        st.success("Inference engine online")
    else:
        st.warning("Inference engine offline")

    st.divider()

    st.caption(
        "Local monitoring workbench. "
        "Data and reports are generated "
        "by the ModelOps pipelines."
    )

    if st.button("Refresh reports"):
        st.rerun()


st.markdown(
    '<div class="eyebrow">ML SYSTEM OBSERVABILITY</div>',
    unsafe_allow_html=True,
)

st.title("ModelOps Monitoring Workbench")

st.write(
    "Monitor customer data drift, model performance, "
    "retraining decisions, and the active churn model."
)

try:
    reports = load_reports()
except (OSError, ValueError) as exc:
    st.error(f"Unable to read monitoring reports: {exc}")
    st.stop()

drift = reports["drift"]
performance = reports["performance"]
retraining = reports["retraining"]

overview, drift_tab, performance_tab, retrain_tab, predict_tab = (
    st.tabs([
        "Overview",
        "Data drift",
        "Performance",
        "Retraining",
        "Predict churn",
    ])
)


# -------------------------------------------
# OVERVIEW
# -------------------------------------------

with overview:
    st.subheader("System overview")

    feature_count = (
        len(drift["features"])
        if drift is not None
        else 0
    )

    drift_count = (
        len(drift["drifted_features"])
        if drift is not None
        else 0
    )

    monitored_rows = (
        performance["metrics"]["labeled_rows"]
        if performance is not None
        else 0
    )

    assessment = (
        performance["assessment"]["status"]
        if performance is not None
        else "not evaluated"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Features monitored",
        feature_count,
    )

    col2.metric(
        "Drift alerts",
        drift_count,
    )

    col3.metric(
        "Labeled predictions",
        monitored_rows,
    )

    col4.metric(
        "Performance status",
        assessment.replace("_", " ").title(),
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Active champion")

        model = fetch_api("/model")

        if model is not None:
            st.success("Model available")

            st.write(
                "**Type:**",
                model["model_type"],
            )

            st.write(
                "**Model fingerprint:**"
            )

            st.code(
                model["model_sha256"][:16] + "…",
                language=None,
            )

            st.write(
                "**Loaded at:**",
                model["loaded_at_utc"],
            )

            st.caption(
                "The fingerprint identifies the "
                "artifact currently loaded by the API."
            )
        else:
            st.info(
                "Start FastAPI to display live "
                "champion information."
            )

    with right:
        st.subheader("Latest retraining decision")

        if retraining:
            decision = retraining["decision"]
            promotion = retraining["promotion"]

            if decision["eligible"]:
                st.success("Challenger eligible")
            else:
                st.warning("Challenger rejected")

            st.write(
                "**Promotion status:**",
                promotion["status"],
            )

            st.write(
                "**Failed checks:**",
                ", ".join(
                    decision["reasons"]
                ) or "None",
            )
        else:
            st.info(
                "Run the retraining pipeline "
                "to generate a decision."
            )


# -------------------------------------------
# DATA DRIFT
# -------------------------------------------

with drift_tab:
    st.subheader("Feature distribution monitoring")

    st.caption(
        "Population Stability Index compares "
        "incoming feature distributions against "
        "the historical reference cohort."
    )

    if drift is None:
        st.info(
            "Run: python -m src.modelops.monitor"
        )

    else:
        data = drift_table(drift)
        threshold = drift["psi_threshold"]

        st.metric(
            "PSI alert threshold",
            format_number(threshold, 2),
        )

        st.bar_chart(
            data.set_index("feature")["psi"],
            horizontal=True,
        )

        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "PSI alerts describe feature-distribution "
            "changes, not confirmed model degradation."
        )


# -------------------------------------------
# PERFORMANCE
# -------------------------------------------

with performance_tab:
    st.subheader("Prediction quality")

    if performance is None:
        st.info(
            "Run: python -m src.modelops.performance"
        )

    else:
        metrics = performance["metrics"]
        assessment = performance["assessment"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Incoming accuracy",
            format_number(metrics["accuracy"]),
        )

        c2.metric(
            "Incoming F1",
            format_number(metrics["f1"]),
        )

        c3.metric(
            "Incoming ROC-AUC",
            format_number(metrics["roc_auc"]),
        )

        st.subheader("Baseline comparison")

        data = performance_table(performance)

        st.bar_chart(
            data.set_index("metric")[
                ["baseline", "incoming"]
            ]
        )

        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
        )

        st.subheader("Degradation alerts")

        if assessment["alerts"]:
            for alert in assessment["alerts"]:
                st.warning(alert)
        else:
            st.success(
                "No configured performance "
                "thresholds breached."
            )

        st.caption(
            "Accuracy, F1, and ROC-AUC measure "
            "different aspects of classification. "
            "The cohorts may have different churn prevalence."
        )


# -------------------------------------------
# RETRAINING
# -------------------------------------------

with retrain_tab:
    st.subheader("Champion versus challenger")

    if retraining is None:
        st.info(
            "Run: python -m src.modelops.retrain"
        )

    else:
        decision = retraining["decision"]

        if decision["eligible"]:
            st.success(
                "Candidate passed acceptance checks."
            )
        else:
            st.warning(
                "Candidate did not pass "
                "all acceptance checks."
            )

        data = retraining_table(retraining)

        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
        )

        st.subheader("Acceptance checks")

        for name, passed in decision["checks"].items():
            if passed:
                st.success(name.replace("_", " ").title())
            else:
                st.error(name.replace("_", " ").title())

        st.write(
            "**Promotion status:**",
            retraining["promotion"]["status"],
        )

        st.caption(
            "This dashboard is read-only. "
            "Model promotion remains controlled "
            "by the guarded retraining CLI."
        )


# -------------------------------------------
# CUSTOMER PREDICTION
# -------------------------------------------

with predict_tab:
    st.subheader("Customer churn prediction")

    st.caption(
        "Explore predictions from the active "
        "synthetic-data churn classifier."
    )

    with st.form("prediction_form"):
        customer_id = st.text_input(
            "Customer ID",
            value="demo-customer-001",
        )

        c1, c2 = st.columns(2)

        with c1:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=120,
                value=12,
                step=1,
            )

            spend = st.number_input(
                "Monthly spend",
                min_value=0.0,
                max_value=10000.0,
                value=75.50,
                step=1.0,
            )

            tickets = st.number_input(
                "Support tickets (90 days)",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
            )

        with c2:
            logins = st.number_input(
                "Logins (30 days)",
                min_value=0,
                max_value=10000,
                value=10,
                step=1,
            )

            contract = st.selectbox(
                "Contract type",
                options=[0, 1],
                format_func=lambda x: (
                    "Monthly contract"
                    if x == 1
                    else "Longer-term contract"
                ),
            )

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
        )

    if submitted:
        payload = {
            "customer_id": customer_id.strip() or None,
            "tenure_months": int(tenure),
            "monthly_spend": float(spend),
            "support_tickets_90d": int(tickets),
            "logins_30d": int(logins),
            "monthly_contract": int(contract),
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15,
            )

            response.raise_for_status()
            result = response.json()

        except (requests.RequestException, ValueError) as exc:
            st.error(
                f"Prediction request failed: {exc}"
            )

        else:
            probability = result["churn_probability"]

            if result["predicted_churn"] == 1:
                st.warning("Predicted churn: Yes")
            else:
                st.success("Predicted churn: No")

            st.metric(
                "Estimated churn probability",
                f"{probability:.1%}",
            )

            st.progress(
                min(
                    100,
                    max(0, int(probability * 100)),
                )
            )

            st.caption(
                "Model fingerprint: "
                + result["model_sha256"][:16]
                + "…"
            )

            st.info(
                "This demonstration model uses synthetic "
                "customer data and is not validated for "
                "real business decisions."
            )