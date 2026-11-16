"""Local operator console: persisted registry, versions and model inference.

Never expose this page publicly. Set MODELOPS_LOCAL_REGISTRY=1 on your OWN Mac.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ModelOps | Persistent Registry", page_icon="🗂️", layout="wide")
st.title("Persistent Model Registry")
st.caption("Local operator console · SQLite metadata · isolated .skops model versions")

if os.getenv("MODELOPS_LOCAL_REGISTRY") != "1":
    st.warning(
        "This administration page is disabled. For LOCAL development, set "
        "MODELOPS_LOCAL_REGISTRY=1 before starting Streamlit. "
        "Do not enable it on a public deployment."
    )
    st.stop()

from src.modelops.model_workspace import (
    MAX_FILE_BYTES, Workspace, load_uploaded_model, read_csv,
)
from src.modelops.workspace_registry import WorkspaceRegistry

registry = WorkspaceRegistry()
api_url = os.getenv("MODELOPS_WORKSPACE_API_URL", "http://127.0.0.1:8001").rstrip("/")

st.info(
    "For synthetic or otherwise non-sensitive data only. Local disk persistence "
    "is NOT persistent on free Render or Streamlit Cloud. No arbitrary pickle/joblib uploads."
)

with st.expander("Register a NEW independent model", expanded=not bool(registry.list_workspaces())):
    with st.form("persistent_workspace_registration"):
        name = st.text_input("Workspace name", value="Loan Default - Persistent", max_chars=60)
        model_file = st.file_uploader("Fitted binary sklearn classifier (.skops)", type=["skops"], key="persistent-model")
        ref_file = st.file_uploader("Reference training data CSV", type=["csv"], key="persistent-ref")
        incoming_file = st.file_uploader("Incoming features CSV", type=["csv"], key="persistent-incoming")
        labels_file = st.file_uploader("Delayed labels CSV (optional)", type=["csv"], key="persistent-labels")
        validation_file = st.file_uploader("Independent historical validation CSV (required for retraining)", type=["csv"], key="persistent-validation")
        a, b = st.columns(2)
        id_col = a.text_input("Identifier column", value="record_id")
        target_col = b.text_input("Binary target column", value="target")
        categorical_str = st.text_input("Categorical numeric features (comma separated)", value="")
        submitted = st.form_submit_button("Save workspace to local registry", type="primary")
    if submitted:
        try:
            if not all([model_file, ref_file, incoming_file]):
                raise ValueError("A model, reference data and incoming data are required")
            files = [f for f in (model_file, ref_file, incoming_file, labels_file, validation_file) if f]
            if any(f.size > MAX_FILE_BYTES for f in files):
                raise ValueError("Individual uploaded files must not exceed 5 MB")
            ws = Workspace(
                name=name.strip(),
                champion=load_uploaded_model(model_file.getvalue()),
                reference=read_csv(ref_file.getvalue(), "reference.csv"),
                incoming=read_csv(incoming_file.getvalue(), "incoming.csv"),
                labels=read_csv(labels_file.getvalue(), "labels.csv") if labels_file else None,
                validation=read_csv(validation_file.getvalue(), "validation.csv") if validation_file else None,
                id_col=id_col.strip(), target_col=target_col.strip(),
                categorical=tuple(x.strip() for x in categorical_str.split(",") if x.strip()),
            )
            wid = registry.register(ws)
        except Exception as exc:
            st.error(f"Registration failed: {exc}")
        else:
            st.success(f"Workspace saved: {wid}. Its data will survive local app restarts.")
            st.rerun()

all_workspaces = registry.list_workspaces()
if not all_workspaces:
    st.info("No registered models yet. Upload your loan-default demonstration model above.")
    st.stop()

choices = {f'{w["name"]} · {w["id"][:8]}': w["id"] for w in all_workspaces}
selected_name = st.selectbox("Select saved workspace", list(choices))
workspace_id = choices[selected_name]
info = registry.get_workspace_info(workspace_id)
versions = registry.versions(workspace_id)
events = registry.events(workspace_id, limit=15)

c1, c2, c3 = st.columns(3)
c1.metric("Current champion", f'v{info["champion_version"]}')
c2.metric("Saved model versions", len(versions))
c3.metric("Model features", len(info["features"]))
st.write("**Features:**", ", ".join(info["features"]))
st.code(workspace_id, language=None)

monitor_tab, train_tab, history_tab, predict_tab = st.tabs([
    "Drift + performance", "Candidate + promotion", "Version history", "Local predictions",
])

with monitor_tab:
    threshold = st.number_input("PSI threshold", value=0.20, min_value=0.01, max_value=10.0, step=0.05)
    if st.button("Run monitoring and SAVE results", type="primary"):
        try:
            results = registry.monitor(workspace_id, threshold)
        except Exception as exc:
            st.error(f"Monitoring failed: {exc}")
        else:
            st.success("Monitoring results saved in the registry history")
            st.json(results)
    last = next((x for x in events if x["type"] == "monitor"), None)
    if last:
        st.subheader("Latest persisted monitoring report")
        st.caption(last["at"])
        result = last["payload"]
        if result.get("drift"):
            st.dataframe(pd.DataFrame(result["drift"]["features"]), hide_index=True, use_container_width=True)
        if result.get("performance"):
            st.json(result["performance"])

with train_tab:
    st.warning("Retraining evaluates a cloned model. It does not replace the champion unless you explicitly promote an eligible candidate.")
    gain = st.number_input("Minimum incoming F1 improvement", value=0.01, min_value=0.0, max_value=0.5, step=0.01)
    if st.button("Train + evaluate + SAVE candidate"):
        try:
            outcome = registry.train_candidate(workspace_id, min_f1_gain=gain)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
        else:
            st.success(f'Candidate version v{outcome["candidate_version"]} saved; champion unchanged.')
            st.json(outcome)
            st.rerun()
    eligible = [v for v in versions if v["status"] == "eligible" and v["decision"] and v["decision"]["eligible"]]
    if eligible:
        version = st.selectbox("Eligible candidate", [v["version"] for v in eligible], format_func=lambda x: f"v{x}")
        confirm = st.checkbox("I have reviewed the guardrails and approve promotion")
        if st.button("Promote selected candidate", disabled=not confirm, type="primary"):
            try:
                outcome = registry.promote(workspace_id, version)
            except Exception as exc:
                st.error(f"Promotion failed: {exc}")
            else:
                st.success(f'Promoted candidate v{outcome["champion"]}')
                st.rerun()
    else:
        st.info("No eligible saved candidates are available for promotion.")
    if info["rollback_version"] is not None:
        confirm_rollback = st.checkbox("Confirm one-step rollback", key="rollback-check")
        if st.button("Rollback to previous champion", disabled=not confirm_rollback):
            try:
                result = registry.rollback(workspace_id)
            except Exception as exc:
                st.error(f"Rollback failed: {exc}")
            else:
                st.success(f'Restored champion v{result["champion"]}')
                st.rerun()

with history_tab:
    st.subheader("Saved versions")
    for version in versions:
        with st.expander(f'v{version["version"]} · {version["status"]} · {version["sha256"][:12]}…'):
            if version["decision"] is not None:
                st.json(version["decision"])
            if version["evaluation"] is not None:
                st.json(version["evaluation"])
    st.subheader("Audit events")
    for event in events:
        with st.expander(f'{event["at"]} · {event["type"]}'):
            st.json(event["payload"])

with predict_tab:
    st.caption("Start the LOCAL workspace API on port 8001 first. Model version is selected from this registry, not from the original churn API.")
    with st.form("workspace-inference"):
        values = {}
        for feature in info["features"]:
            values[feature] = st.number_input(feature, value=0.0, step=1.0, key=f'infer-{workspace_id}-{feature}')
        predict_clicked = st.form_submit_button("Predict with current workspace champion")
    if predict_clicked:
        try:
            response = requests.post(
                f"{api_url}/workspaces/{workspace_id}/predict",
                json={"features": values}, timeout=15,
            )
            response.raise_for_status()
            st.json(response.json())
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Inference API unavailable: {exc}")

st.caption("Local-only registry. To deploy persisted uploaded models publicly, use a separately authenticated API with durable external object storage and database.")
