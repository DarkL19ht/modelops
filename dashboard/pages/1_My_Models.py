"""Self-service upload, monitor, retrain, download and predict for each signed-in user."""
import io
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st
from dashboard.api_feedback import describe_api_error

logger = logging.getLogger(__name__)

st.set_page_config(page_title="ModelOps | My Models", page_icon="🧠", layout="wide")
st.title("My Models")
st.caption("Upload a trained model, evaluate its performance, train challengers, and run inference.")

API = os.getenv("MODELOPS_SELF_SERVICE_API_URL", "").rstrip("/")
if not (API.startswith("https://") or API.startswith("http://127.0.0.1:") or API.startswith("http://localhost:")):
    st.error("Configure MODELOPS_SELF_SERVICE_API_URL with your HTTPS API address.")
    st.stop()

SESSION_TOKEN = "modelops_ss_token"
SESSION_USER = "modelops_ss_user"


def call(method: str, path: str, *, data=None, files=None, params=None, json=None, timeout=90, binary=False):
    headers = {}
    token = st.session_state.get(SESSION_TOKEN)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.request(
            method, f"{API}{path}", headers=headers,
            data=data, files=files, params=params, json=json, timeout=timeout,
        )
        response.raise_for_status()
        return response.content if binary else response.json()
    except requests.HTTPError:
        status = response.status_code
        logger.warning("API request failed: %s %s (%d)", method, path, status)
        if status == 401 and path != "/auth/login":
            st.session_state.pop(SESSION_TOKEN, None)
            st.session_state.pop(SESSION_USER, None)
        message = describe_api_error(status, path)
        if path == "/auth/register" and status == 409:
            st.info(message)
        else:
            st.error(message)
    except requests.Timeout:
        st.error("ModelOps is taking longer than expected. Please try again shortly.")
    except requests.RequestException:
        logger.warning("API connectivity failure: %s %s", method, path)
        st.error("We couldn't connect to ModelOps. Please check your connection and try again.")
    except ValueError:
        st.error("We received an unexpected response. Please try again shortly.")
    return None

if SESSION_TOKEN not in st.session_state:
    st.info("Create your account or sign in to access your private model workspaces.")
    signin, signup = st.tabs(["Sign in", "Create account"])
    with signin:
        with st.form("login"):
            email = st.text_input("Email", key="ss_signin_email")
            password = st.text_input("Password", type="password", key="ss_signin_password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            result = call("POST", "/auth/login", json={"email": email, "password": password})
            if result:
                st.session_state[SESSION_TOKEN] = result["access_token"]
                st.session_state[SESSION_USER] = result["user"]
                st.rerun()
    with signup:
        with st.form("signup"):
            email_new = st.text_input("Email", key="ss_signup_email")
            password_new = st.text_input("Password (12+ characters)", type="password", key="ss_signup_password")
            created = st.form_submit_button("Create account")
        if created:
            result = call("POST", "/auth/register", json={"email": email_new, "password": password_new})
            if result:
                st.info("If this email is eligible for registration, an eight-digit verification code will be sent. "
                        "Already have an account? Use the Sign in tab instead.")
    with st.expander("Verify email address", expanded=True):
        with st.form("email_verification"):
            verify_address = st.text_input("Registration email", key="ss_verify_address")
            verify_code = st.text_input("Eight-digit verification code", max_chars=8,
                                        key="ss_verify_code")
            verify_submitted = st.form_submit_button("Verify email")
        if verify_submitted:
            result = call("POST", "/auth/verify-email", json={
                "email": verify_address, "code": verify_code.strip(),
            })
            if result and result.get("verified"):
                st.success("Email verified. You can now sign in.")

    with st.expander("Resend verification code"):
        with st.form("resend_email_verification"):
            resend_address = st.text_input("Email", key="ss_resend_address")
            resend_submitted = st.form_submit_button("Resend code")
        if resend_submitted:
            result = call("POST", "/auth/resend-verification", json={"email": resend_address})
            if result:
                st.info("If this address has a pending registration, a code will be sent.")
    st.stop()

with st.sidebar:
    st.write("**ModelOps**")
    user = st.session_state.get(SESSION_USER, {})
    st.caption(user.get("email", "Signed in"))
    if st.button("Sign out"):
        call("POST", "/auth/logout")
        st.session_state.pop(SESSION_TOKEN, None)
        st.session_state.pop(SESSION_USER, None)
        st.rerun()
    if st.button("Refresh my workspaces"):
        st.rerun()

st.caption("Public uploads accept a restricted portable-model format. "
           "Do not upload confidential or regulated datasets.")

upload_tab, manage_tab = st.tabs(["➕ Add my model", "📊 My workspaces"])

with upload_tab:
    st.subheader("Upload a model and its datasets")
    st.caption("Verified accounts can upload supported portable models without manual approval. "
               "Standard .skops uploads remain part of the controlled beta.")
    with st.expander("Upload a model · Public access", expanded=True):
        st.caption("Currently supported: numeric binary logistic regression models in portable JSON format.")
        st.info("Have a .skops model? That is a different format. Renaming the file will not convert it. "
                "Use the export instructions for a model you trust, or try the sample model below.")
        st.markdown("[Model format, export instructions and sample files]"
                    "(https://github.com/DarkL19ht/modelops/blob/main/docs/portable_model_access.md)")
        st.caption("Only supported fitted logistic regression models can be exported. "
                   "Never load unknown model files just to convert them.")
        sample_zip = ROOT / "dashboard" / "assets" / "modelops_employer_starter_files.zip"
        if sample_zip.is_file():
            st.download_button("Download sample model and datasets",
                               data=sample_zip.read_bytes(),
                               file_name="modelops_employer_starter_files.zip",
                               mime="application/zip")
        with st.form("portable_model_upload"):
            portable_name = st.text_input("Workspace name", value="Loan Default", key="portable_name")
            portable_model = st.file_uploader("Fitted portable model (.modelops.json)", type=["json"], key="portable_model")
            portable_ref = st.file_uploader("Reference CSV", type=["csv"], key="portable_ref")
            portable_inc = st.file_uploader("Incoming CSV", type=["csv"], key="portable_inc")
            portable_labels = st.file_uploader("Delayed labels CSV (optional)", type=["csv"], key="portable_labels")
            portable_validation = st.file_uploader("Historical holdout CSV (optional)", type=["csv"], key="portable_validation")
            portable_id = st.text_input("ID column", value="record_id", key="portable_id")
            portable_target = st.text_input("Target column", value="target", key="portable_target")
            portable_submit = st.form_submit_button("Create my workspace", type="primary")
        if portable_submit:
            if not all((portable_model, portable_ref, portable_inc)):
                st.error("The model, reference data and incoming data are required.")
            elif not portable_model.name.lower().endswith(".modelops.json"):
                st.error("This upload requires a portable .modelops.json file. "
                             "An ordinary .skops file cannot be renamed to this format.")
            else:
                files = {
                    "model": (portable_model.name, portable_model.getvalue(), "application/json"),
                    "reference": (portable_ref.name, portable_ref.getvalue(), "text/csv"),
                    "incoming": (portable_inc.name, portable_inc.getvalue(), "text/csv"),
                }
                if portable_labels:
                    files["labels"] = (portable_labels.name, portable_labels.getvalue(), "text/csv")
                if portable_validation:
                    files["validation"] = (portable_validation.name, portable_validation.getvalue(), "text/csv")
                result = call("POST", "/my/portable-workspaces", files=files,
                    data={"name": portable_name, "id_col": portable_id, "target_col": portable_target}, timeout=120)
                if result:
                    st.success("Model ready. Open My workspaces to predict, monitor and train challengers.")
    st.divider()
    with st.expander("Advanced · .skops upload (restricted beta)", expanded=False):
        st.caption("Legacy upload: .skops files are supported only for approved beta accounts.")
        st.write(
            "Your uploaded model becomes this workspace's champion immediately. "
            "You can make predictions as soon as registration succeeds. "
            "Delayed labels and a separate holdout are needed only for training a challenger."
        )
        with st.form("ss_upload", clear_on_submit=False):
            name = st.text_input("Model/workspace name", value="Loan Default", max_chars=60)
            model_file = st.file_uploader("My fitted model (.skops)", type=["skops"])
            ref_file = st.file_uploader("Reference data (.csv)", type=["csv"], key="ss_ref")
            new_file = st.file_uploader("New incoming features (.csv; no target)", type=["csv"], key="ss_inc")
            label_file = st.file_uploader("Delayed incoming labels (.csv; optional for monitoring)", type=["csv"], key="ss_labels")
            val_file = st.file_uploader("Independent historical holdout (.csv; required for retraining)", type=["csv"], key="ss_val")
            c1, c2 = st.columns(2)
            id_col = c1.text_input("ID column", value="record_id")
            target_col = c2.text_input("Binary target column", value="target")
            categorical = st.text_input("Categorical feature names (comma-separated, optional)")
            uploaded = st.form_submit_button("Create my workspace", type="primary")
        if uploaded:
            if model_file is None or ref_file is None or new_file is None:
                st.error("Your model, reference data and incoming data are required.")
            elif any(f is not None and f.size > 5 * 1024 * 1024 for f in
                     (model_file, ref_file, new_file, label_file, val_file)):
                st.error("Each file must be 5 MB or smaller.")
            else:
                upload_files = {
                    "model": (model_file.name, model_file.getvalue(), "application/octet-stream"),
                    "reference": (ref_file.name, ref_file.getvalue(), "text/csv"),
                    "incoming": (new_file.name, new_file.getvalue(), "text/csv"),
                }
                if label_file:
                    upload_files["labels"] = (label_file.name, label_file.getvalue(), "text/csv")
                if val_file:
                    upload_files["validation"] = (val_file.name, val_file.getvalue(), "text/csv")
                result = call("POST", "/my/workspaces", files=upload_files,
                              data={"name": name, "id_col": id_col,
                                    "target_col": target_col, "categorical": categorical}, timeout=120)
                if result:
                    st.success("Your model is uploaded. Open 'My workspaces' to run predictions and monitoring.")
                    st.caption(f'Workspace ID: {result["id"]}')

with manage_tab:
    listing = call("GET", "/my/workspaces")
    if listing is None:
        st.stop()
    choices = listing.get("workspaces", [])
    if not choices:
        st.info("You don't have a model workspace yet. Upload a model using the first tab.")
        st.stop()
    options = {f'{w["name"]} · champion v{w["champion_version"]}': w for w in choices}
    selection = st.selectbox("Choose one of my models", list(options))
    info = options[selection]
    wid = info["id"]
    base = f"/my/workspaces/{wid}"
    st.caption("Required model features: " + ", ".join(info["features"]))
    st.metric("Current champion version", info["champion_version"])
    predict_tab, monitor_tab, train_tab, version_tab = st.tabs([
        "🔮 Predict", "📈 Monitor", "🧪 Train & compare", "📦 Download & history"
    ])

    with predict_tab:
        st.subheader("Model inference")
        st.write("Predictions are available immediately after upload, even before retraining.")
        versions_data = call("GET", base + "/versions")
        versions = versions_data.get("versions", []) if versions_data else []
        allowed = [v for v in versions if v["status"] in ("champion", "eligible")]
        if not allowed:
            allowed = [{"version": info["champion_version"], "status": "champion"}]
        prediction_choice = st.selectbox(
            "Model version for prediction",
            options=allowed,
            format_func=lambda x: f'v{x["version"]} · {x["status"]}',
            key=f"ss_pred_ver_{wid}",
        )
        selected_version = prediction_choice["version"]
        with st.form(f"ss_predict_{wid}"):
            values = {}
            for feature in info["features"]:
                values[feature] = st.number_input(feature, value=0.0, step=0.1,
                                                  key=f"ss_field_{wid}_{feature}")
            do_predict = st.form_submit_button("Predict now", type="primary")
        if do_predict:
            result = call("POST", base + "/predict", json={"features": values},
                          params={"version": selected_version})
            if result:
                c1, c2 = st.columns(2)
                c1.metric("Predicted class", result["predicted_class"])
                c2.metric("Probability of class 1", f'{result["positive_class_probability"]:.1%}')
                st.caption(f'Model v{result["model_version"]}; fingerprint: {result["model_sha256"][:18]}…')

        st.divider()
        st.subheader("Batch predictions")
        st.caption("Upload a CSV with exactly the feature columns shown above (up to 100 rows).")
        batch_file = st.file_uploader("Prediction CSV", type=["csv"], key=f"ss_batch_{wid}")
        if st.button("Predict CSV rows", key=f"ss_run_batch_{wid}"):
            if batch_file is None:
                st.warning("Upload a CSV first.")
            else:
                try:
                    frame = pd.read_csv(io.BytesIO(batch_file.getvalue()))
                    if len(frame) < 1 or len(frame) > 100 or set(frame.columns) != set(info["features"]):
                        raise ValueError("CSV must have 1–100 rows and exactly the required feature columns")
                    rows = [{"features": record} for record in frame.to_dict(orient="records")]
                    result = call("POST", base + "/predict/batch", json={"rows": rows},
                                  params={"version": selected_version}, timeout=120)
                    if result:
                        output = pd.concat([frame.reset_index(drop=True),
                            pd.DataFrame(result["predictions"])], axis=1)
                        st.dataframe(output, use_container_width=True)
                        st.download_button("Download prediction results (.csv)",
                                           output.to_csv(index=False).encode("utf-8"),
                                           file_name="modelops_predictions.csv", mime="text/csv")
                except (ValueError, TypeError, pd.errors.ParserError) as exc:
                    st.error(str(exc))

    with monitor_tab:
        st.subheader("Run my model's drift and performance evaluation")
        threshold = st.number_input("PSI drift threshold", 0.01, 10.0, 0.20, step=0.05)
        if st.button("Run monitoring", type="primary", key=f"ss_mon_{wid}"):
            report = call("POST", base + "/monitor", json={"psi_threshold": threshold}, timeout=120)
            if report:
                st.session_state[f"ss_monitor_{wid}"] = report
        report = st.session_state.get(f"ss_monitor_{wid}")
        if report:
            st.metric("Drifted features", len(report["drift"]["drifted_features"]))
            drift_frame = pd.DataFrame(report["drift"]["features"])
            st.bar_chart(drift_frame.set_index("feature")["psi"])
            st.dataframe(drift_frame, use_container_width=True)
            if report.get("performance"):
                st.subheader("Classifier performance")
                st.json(report["performance"], expanded=False)
            st.download_button("Download monitoring report (.json)",
                data=__import__("json").dumps(report, indent=2),
                file_name=f"monitoring_{wid[:8]}.json", mime="application/json")

    with train_tab:
        st.subheader("Train a challenger for this workspace")
        st.caption("The challenger must improve incoming F1 and pass historical/incoming guardrails. "
                   "Your champion stays unchanged until you explicitly promote a better model.")
        gain = st.number_input("Minimum incoming F1 improvement", 0.0, 0.5, 0.01, step=0.01)
        if os.getenv("MODELOPS_TRAINING_JOBS_ENABLED", "0") == "1":
            if st.button("Queue challenger training", type="primary", key=f"ss_queue_{wid}"):
                started = call("POST", base + "/training-jobs", json={"min_f1_gain": gain})
                if started:
                    st.session_state[f"ss_training_job_{wid}"] = started["job"]["id"]
                    st.session_state.pop(f"ss_last_candidate_{wid}", None)
            history = call("GET", base + "/training-jobs")
            jobs = history.get("jobs", []) if history else []
            job_id = st.session_state.get(f"ss_training_job_{wid}")
            if not job_id and jobs:
                job_id = jobs[0]["id"]
            if job_id:
                response = call("GET", f"/my/training-jobs/{job_id}")
                if response:
                    job = response["job"]
                    st.write("**Training status:**", job["status"].title())
                    if job["status"] in ("queued", "running"):
                        st.info("Training is processing in the background. This page can be closed safely.")
                        if st.button("Refresh training status", key=f"ss_refresh_job_{wid}"):
                            st.rerun()
                    elif job["status"] == "succeeded":
                        st.session_state[f"ss_last_candidate_{wid}"] = job["result"]
                    elif job["status"] in ("failed", "interrupted"):
                        st.warning(job.get("error") or "Training did not complete. You can request a new job.")
        else:
            if st.button("Train and evaluate challenger", type="primary", key=f"ss_train_{wid}"):
                result = call("POST", base + "/train", json={"min_f1_gain": gain}, timeout=180)
                if result:
                    st.session_state[f"ss_last_candidate_{wid}"] = result
        candidate = st.session_state.get(f"ss_last_candidate_{wid}")
        if candidate:
            if candidate["decision"]["eligible"]:
                st.success(f'Challenger v{candidate["candidate_version"]} is BETTER by the configured criteria.')
                st.info("Download the improved model below, or promote it to use it for future predictions.")
                file_bytes = call("GET", base + f'/models/{candidate["candidate_version"]}/download', binary=True)
                if file_bytes:
                    st.download_button("Download improved challenger (.skops)", file_bytes,
                        file_name=f'modelops_{wid[:8]}_v{candidate["candidate_version"]}.skops',
                        mime="application/octet-stream", type="primary")
                if st.button("Promote this challenger", key=f"ss_promote_now_{wid}"):
                    promoted = call("POST", base + "/promote", json={"version": candidate["candidate_version"]})
                    if promoted:
                        st.success("Challenger promoted. Your next predictions will use the new champion.")
                        st.session_state.pop(f"ss_last_candidate_{wid}", None)
                        st.rerun()
            else:
                st.warning("Challenger did not improve enough. Your original champion remains active.")
                st.write("Failed checks:", ", ".join(candidate["decision"]["reasons"]) or "N/A")
            rows = []
            for cohort, models in candidate["evaluation"].items():
                for role, scores in models.items():
                    rows.append({"cohort": cohort, "model": role, **scores})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with version_tab:
        st.subheader("My saved model versions")
        current = call("GET", base + "/versions")
        if current:
            saved = current.get("versions", [])
            st.dataframe(pd.DataFrame([{
                "version": v["version"], "status": v["status"],
                "base_version": v.get("base_version"), "created_at": v.get("created_at")
            } for v in saved]), use_container_width=True)
            eligible = [v for v in saved if v["status"] == "eligible"]
            if eligible:
                chosen = st.selectbox("Eligible challenger to download or promote", eligible,
                                      format_func=lambda x: f'v{x["version"]}', key=f"ss_eligible_{wid}")
                if st.checkbox("Prepare selected challenger download", key=f"ss_dl_cand_{wid}"):
                    payload = call("GET", base + f'/models/{chosen["version"]}/download', binary=True)
                    if payload:
                        st.download_button("Download eligible challenger", payload,
                            file_name=f'candidate_v{chosen["version"]}.skops')
                if st.button("Promote selected eligible challenger", key=f"ss_selected_promote_{wid}"):
                    result = call("POST", base + "/promote", json={"version": chosen["version"]})
                    if result:
                        st.success("New champion promoted")
                        st.rerun()
            champion_v = info["champion_version"]
            if st.checkbox("Prepare champion download", key=f"ss_dl_champ_{wid}"):
                champion_file = call("GET", base + f"/models/{champion_v}/download", binary=True)
                if champion_file:
                    st.download_button("Download current champion", champion_file,
                        file_name=f"champion_v{champion_v}.skops")
        if info.get("rollback_version") is not None:
            if st.button("Roll back to previous champion", key=f"ss_rollback_{wid}"):
                result = call("POST", base + "/rollback")
                if result:
                    st.success("Previous champion restored")
                    st.rerun()
        with st.expander("Workspace audit history"):
            history = call("GET", base + "/events")
            if history:
                st.json(history.get("events", []), expanded=False)
