"""Optional, session-scoped lab for externally supplied sklearn models."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.modelops.model_workspace import (
    MAX_FILE_BYTES,
    Workspace,
    load_uploaded_model,
    read_csv,
    serialize_model,
)

st.set_page_config(page_title="ModelOps | Model Workspaces", page_icon="🧪", layout="wide")
st.title("Model Workspaces")
st.caption(
    "Upload a fitted binary scikit-learn classifier and its own datasets. "
    "Each workspace has isolated monitoring, evaluation, retraining and session-only promotion."
)
st.warning(
    "Experimental demo. Only standard sklearn .skops models are accepted. "
    "No .pkl/.joblib uploads, custom Python types or sensitive data. "
    "Files and trained candidates are held in this browser session; nothing is deployed to Render."
)

if "modelops_workspaces" not in st.session_state:
    st.session_state["modelops_workspaces"] = {}
workspaces = st.session_state["modelops_workspaces"]

with st.expander("1 · Register a model workspace", expanded=not bool(workspaces)):
    with st.form("register_model_workspace", clear_on_submit=False):
        name = st.text_input("Workspace name", value="Loan Default Demo", max_chars=60)
        model_file = st.file_uploader("Fitted sklearn model (.skops)", type=["skops"])
        reference_file = st.file_uploader("Reference training data (.csv)", type=["csv"], key="ref")
        incoming_file = st.file_uploader("Incoming features (.csv, no labels)", type=["csv"], key="inc")
        labels_file = st.file_uploader("Delayed labels (.csv, optional for monitoring)", type=["csv"], key="labs")
        validation_file = st.file_uploader(
            "Independent historical validation holdout (.csv, required for promotion)",
            type=["csv"], key="val"
        )
        c1, c2 = st.columns(2)
        id_col = c1.text_input("Record identifier column", value="record_id")
        target_col = c2.text_input("Binary target column (0/1)", value="target")
        categorical_text = st.text_input(
            "Categorical feature names, comma-separated (optional)", value=""
        )
        submitted = st.form_submit_button("Register workspace", type="primary")

    if submitted:
        try:
            if not name.strip() or name in workspaces:
                raise ValueError("Enter a unique non-empty workspace name")
            if len(workspaces) >= 3:
                raise ValueError("Maximum 3 active workspaces per session")
            if model_file is None or reference_file is None or incoming_file is None:
                raise ValueError("Model, reference and incoming files are required")
            for upload in (model_file, reference_file, incoming_file, labels_file, validation_file):
                if upload is not None and upload.size > MAX_FILE_BYTES:
                    raise ValueError(f"{upload.name} exceeds 5 MB")
            model_bytes = model_file.getvalue()
            model = load_uploaded_model(model_bytes)
            reference = read_csv(reference_file.getvalue(), "reference.csv")
            incoming = read_csv(incoming_file.getvalue(), "incoming.csv")
            labels = read_csv(labels_file.getvalue(), "labels.csv") if labels_file else None
            validation = read_csv(validation_file.getvalue(), "validation.csv") if validation_file else None
            import hashlib
            workspaces[name] = Workspace(
                name=name, champion=model, model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                reference=reference, incoming=incoming,
                labels=labels, validation=validation,
                id_col=id_col.strip(), target_col=target_col.strip(),
                categorical=tuple(x.strip() for x in categorical_text.split(",") if x.strip()),
            )
        except Exception as exc:
            st.error(f"Could not register workspace: {exc}")
        else:
            st.success(f"Workspace '{name}' registered. Select it below to run the pipeline.")

if not workspaces:
    st.info("Register your first model to begin. Generate demo files using scripts/create_loan_demo.py.")
    st.stop()

st.divider()
selected = st.selectbox("Active model workspace", list(workspaces), key="selected_model_workspace")
ws = workspaces[selected]
c1, c2, c3 = st.columns(3)
c1.metric("Model features", len(ws.features))
c2.metric("Incoming observations", len(ws.incoming))
c3.metric("Previous session versions", len(ws.previous))
st.caption(f"Model fingerprint: {ws.model_sha256[:20]}…")
st.write("**Feature contract:**", ", ".join(ws.features))

col1, col2, col3 = st.columns(3)
with col1:
    threshold = st.number_input("Drift threshold (PSI)", min_value=0.01, max_value=10.0, value=0.20, step=0.05)
    if st.button("Run drift + performance", type="primary", use_container_width=True):
        try:
            ws.monitor(threshold=threshold)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        else:
            st.success("Monitoring complete for this model workspace.")

with col2:
    min_f1_gain = st.number_input("Minimum F1 gain", min_value=0.0, max_value=0.5, value=0.01, step=0.01)
    if st.button("Train + evaluate challenger", use_container_width=True):
        try:
            ws.train_challenger(min_f1_gain=min_f1_gain)
        except Exception as exc:
            st.error(f"Retraining unavailable: {exc}")
        else:
            st.success("Challenger evaluated. Review acceptance checks below.")

with col3:
    st.write("**Model promotion**")
    eligible = bool(ws.decision and ws.decision["eligible"])
    if st.button("Promote in this session", disabled=not eligible, use_container_width=True):
        try:
            ws.promote()
        except Exception as exc:
            st.error(f"Promotion failed: {exc}")
        else:
            st.success("Session champion replaced; previous model archived in session memory.")
    if st.button("Restore previous session champion", disabled=not ws.previous, use_container_width=True):
        ws.rollback()
        st.success("Previous session model restored.")

st.divider()
monitor_tab, compare_tab, download_tab = st.tabs(["Monitoring", "Retraining comparison", "Export / reset"])

with monitor_tab:
    if ws.drift is None:
        st.info("Click 'Run drift + performance' to analyse this workspace.")
    else:
        st.subheader("Input feature drift")
        st.metric("Drift flags", len(ws.drift["drifted_features"]))
        rows = pd.DataFrame(ws.drift["features"])
        st.bar_chart(rows.set_index("feature")["psi"])
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("PSI is descriptive. Drift alone does not prove a model has degraded.")
        if ws.performance is None:
            st.info("Upload delayed labels to enable model performance evaluation.")
        else:
            st.subheader("Champion evaluation")
            perf = pd.DataFrame([
                {"cohort": cohort, **metrics} for cohort, metrics in ws.performance.items()
            ])
            st.dataframe(perf, hide_index=True, use_container_width=True)
            st.caption("A historical holdout must be independent of the model's training data.")

with compare_tab:
    if ws.comparison is None:
        st.info("Train a challenger to see side-by-side evaluations and guardrails.")
    else:
        st.subheader("Champion versus challenger")
        rows = []
        for cohort, groups in ws.comparison.items():
            for role, metrics in groups.items():
                rows.append({"cohort": cohort, "model": role, **metrics})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.write("**Eligibility:**", "eligible" if ws.decision["eligible"] else "rejected")
        for check, passed in ws.decision["checks"].items():
            (st.success if passed else st.warning)(f"{check}: {'PASS' if passed else 'FAIL'}")
        st.caption(
            "The incoming holdout is excluded from challenger fitting, but inspecting it repeatedly "
            "can still bias model selection. No cloud model is automatically updated."
        )

with download_tab:
    st.download_button(
        "Download workspace analysis (JSON)",
        data=json.dumps(ws.summary(), indent=2),
        file_name="modelops_workspace_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
    try:
        model_export = serialize_model(ws.champion)
    except Exception as exc:
        st.warning(f"Model export unavailable: {exc}")
    else:
        st.download_button(
            "Download session champion (.skops)",
            data=model_export,
            file_name="modelops_champion.skops",
            mime="application/octet-stream",
            use_container_width=True,
        )
    if ws.challenger is not None:
        try:
            challenger_export = serialize_model(ws.challenger)
        except Exception as exc:
            st.warning(f"Challenger export unavailable: {exc}")
        else:
            st.download_button(
                "Download challenger (.skops)",
                data=challenger_export,
                file_name="modelops_challenger.skops",
                mime="application/octet-stream",
                use_container_width=True,
            )
    if st.button("Remove this workspace"):
        del workspaces[selected]
        st.rerun()

st.caption("Limits: numeric features, binary sklearn classifiers, 5 MB/file, 10,000 rows/file, 3 session workspaces. No persistent storage or public model deployment.")
