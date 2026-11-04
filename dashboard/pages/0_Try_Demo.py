"""No-sign-in, read-only public ModelOps demonstration."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.modelops.public_demo import build_demo, predict_demo


st.set_page_config(
    page_title="Try Demo | ModelOps",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource(show_spinner="Preparing the demonstration models…")
def demo_workspace():
    return build_demo()


workspace = demo_workspace()

st.title("Explore ModelOps")
st.write(
    "Explore model predictions, feature drift, and champion-versus-challenger "
    "evaluation without signing in."
)
st.caption(
    "Interactive demonstration using reproducible synthetic customer data. "
    "Model training and evaluation run in memory; results are not production telemetry. "
    "This page neither accesses accounts nor modifies saved workspaces."
)

with st.sidebar:
    st.markdown("### ModelOps")
    st.success("Demo access · No account required")
    st.caption("This experience is read-only and uses preloaded models.")
    my_models = ROOT / "dashboard" / "pages" / "1_My_Models.py"
    alternative = ROOT / "dashboard" / "pages" / "4_My_Model_Workspaces.py"
    if my_models.exists():
        st.page_link("pages/1_My_Models.py", label="Open My Models", icon="📁")
    elif alternative.exists():
        st.page_link("pages/4_My_Model_Workspaces.py", label="Open My Models", icon="📁")

overview_tab, predict_tab, drift_tab, compare_tab = st.tabs([
    "Overview", "Try a prediction", "Data drift", "Model comparison"
])

with overview_tab:
    st.subheader("Model lifecycle overview")
    a, b, c = st.columns(3)
    a.metric("Reference observations", f"{workspace.reference_rows:,}")
    b.metric("Incoming observations", f"{workspace.incoming_rows:,}")
    c.metric("Drifted features", len(workspace.drift["drifted_features"]))
    st.write(
        "A baseline logistic-regression classifier is trained using historical "
        "observations. We then simulate a shift in customer behaviour and train "
        "a challenger. Both models are evaluated on corresponding holdout samples."
    )
    if workspace.decision["eligible"]:
        st.success("The challenger passes all configured acceptance checks.")
    else:
        st.info("The challenger does not pass every acceptance check; the champion remains the reference model.")

with predict_tab:
    st.subheader("Predict customer churn")
    st.caption("Real inference using the deterministic, preloaded scikit-learn models.")
    with st.form("demo_predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Tenure in months", min_value=0, max_value=120, value=12)
            spending = st.number_input("Monthly spend", min_value=0.0, max_value=10000.0, value=75.50)
            tickets = st.number_input("Support tickets (90 days)", min_value=0, max_value=100, value=2)
        with col2:
            logins = st.number_input("Logins (30 days)", min_value=0, max_value=10000, value=10)
            contract = st.selectbox(
                "Contract type", [0, 1],
                format_func=lambda value: "Monthly" if value == 1 else "Long-term",
            )
            model_choice = st.radio(
                "Compare model version", ["champion", "challenger"],
                format_func=lambda value: value.title(), horizontal=True,
            )
        submitted = st.form_submit_button("Generate prediction", type="primary")
    if submitted:
        result = predict_demo(workspace, {
            "tenure_months": int(tenure),
            "monthly_spend": float(spending),
            "support_tickets_90d": int(tickets),
            "logins_30d": int(logins),
            "monthly_contract": int(contract),
        }, version=model_choice)
        st.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
        st.write("**Predicted class:**", "Churn" if result["predicted_churn"] else "No churn")
        st.caption(f"Model: {result['version'].title()}")
        if model_choice == "challenger" and not workspace.decision["eligible"]:
            st.warning("This challenger is not approved for promotion. Its prediction is shown for comparison only.")
        st.info("Predictions are based on synthetic data and must not inform real customer decisions.")

with drift_tab:
    st.subheader("Feature distribution changes")
    st.caption("Population Stability Index (PSI) comparing historical and incoming samples.")
    data = pd.DataFrame(workspace.drift["features"])
    st.bar_chart(data.set_index("feature")["psi"], horizontal=True)
    st.dataframe(
        data[["feature", "psi", "status", "reference_mean", "incoming_mean", "ks_statistic"]],
        hide_index=True, use_container_width=True,
    )
    st.caption("Drift alone does not establish that model accuracy has degraded.")

with compare_tab:
    st.subheader("Champion versus challenger")
    rows = []
    for cohort in ("reference_holdout", "incoming_holdout"):
        for metric in ("accuracy", "f1", "roc_auc"):
            champion_score = workspace.evaluations[cohort]["champion"].get(metric)
            challenger_score = workspace.evaluations[cohort]["challenger"].get(metric)
            rows.append({
                "cohort": cohort.replace("_", " ").title(),
                "metric": metric.upper() if metric == "roc_auc" else metric.title(),
                "champion": champion_score,
                "challenger": challenger_score,
                "difference": challenger_score - champion_score
                if challenger_score is not None and champion_score is not None else None,
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Acceptance checks")
    for check, passed in workspace.decision["checks"].items():
        st.write(("✅" if passed else "❌") + " " + check.replace("_", " ").capitalize())
    if workspace.decision["eligible"]:
        st.success("The candidate is eligible for promotion; the demo does not modify its champion.")
    else:
        st.info("The candidate is not eligible for promotion under the configured safeguards.")

    report = {
        "demo": "synthetic customer churn model comparison",
        "evaluation": workspace.evaluations,
        "decision": workspace.decision,
        "drift": workspace.drift,
    }
    st.download_button(
        "Download evaluation report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="modelops_demo_evaluation.json",
        mime="application/json",
    )
    st.caption("This public page is read-only. Account holders manage their own models in My Models.")
