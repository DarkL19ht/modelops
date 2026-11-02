# ModelOps — Model Workspaces

Model Workspaces lets users investigate **multiple independent fitted binary scikit-learn classifiers** through a Streamlit page separate from the original customer churn dashboard.

Each workspace has its own uploaded model, reference and incoming cohorts, feature drift report, optional delayed-label evaluation, candidate retraining, acceptance checks, and in-session promotion/rollback. A model trained in one workspace never replaces a model in another workspace or the Render inference service.

## Supported models and scope

- Fitted scikit-learn classifiers / Pipelines with `feature_names_in_`, `classes_ == [0, 1]`, `predict`, `predict_proba`, `.fit`, and sklearn `clone()` compatibility.
- Up to 30 **numeric** model input features; integer-coded categorical columns can be specified for categorical PSI.
- Only `.skops` uploads; unknown or custom types are **rejected**, never automatically trusted. No `.pkl` or `.joblib` uploads.
- Each file at most 5 MB; at most 10,000 records per CSV; at most three workspaces per session.
- No regression, multiclass, deep-learning models, raw-string categorical features or arbitrary Python preprocessor code yet.
- Temporary **session-only** model promotion, rollback and export. No MLflow registry write, FastAPI activation, persistent model store or scheduled retraining yet.
- These bounded inputs do not substitute for authentication, quotas, workload isolation, background jobs, or malware security review. Use test data, not private customer datasets.

## Required files

Upload four or five files per new workspace:

1. `my_model.skops`: fitted model, features extracted from `model.feature_names_in_`.
2. `reference.csv`: historical **training** records, with unique record IDs, feature columns and a 0/1 target for retraining. For monitoring only, target can be omitted.
3. `incoming.csv`: new records with IDs and feature columns, **without** the target.
4. `delayed_labels.csv` (optional, needed to evaluate/retrain): record IDs and 0/1 target. Must match incoming IDs exactly.
5. `validation.csv` (needed to retrain and assess promotion): independent historical labeled holdout, with IDs, model features, and the target. IDs must be disjoint from reference and incoming. It must not have been used to train the uploaded model.

Specify the ID and label column names in the form (e.g. `record_id` and `target`). The model features are inferred, not hardcoded to churn.

**Data leakage limitation:** The app can check ID overlaps but cannot prove an uploaded model was not fitted on the holdout. The uploader is responsible for providing genuinely independent historical validation data. Repeated experimentation against the same incoming holdout can cause selection bias; a separate external final test is still needed for real adoption.

## Local setup

```bash
cd ~/Documents/modelops
source .venv/bin/activate
# Add skops>=0.11,<1 to requirements.txt and dashboard/requirements.txt
# Add numpy, scipy, scikit-learn, joblib to dashboard/requirements.txt
python -m pip install -r requirements.txt
python scripts/create_loan_demo.py
python -m pytest tests/test_model_workspace.py -v
python -m streamlit run dashboard/app.py --server.port 8501
```

Streamlit finds the new page automatically via `dashboard/pages/1_Model_Workspaces.py`.

For a locally created, **trusted** `.joblib` model, convert it to `.skops` using:

```bash
python scripts/export_trusted_model.py artifacts/churn_model.joblib demo_uploads/churn.skops
```

Never run that converter on an untrusted `.joblib`; loading pickle/joblib files can execute arbitrary code.

Use the four or five files from `demo_uploads/loan_default` to test a second unrelated model and different input features. The folder `demo_uploads/` can be added to `.gitignore`; files are generated reproducibly.

## Workflow

1. Register a model workspace. Upload the fitted model and matching datasets.
2. Run feature-drift and optional delayed-label performance monitoring.
3. Train a cloned challenger using reference training records plus 80% of labeled incoming records. Evaluate both models on the independent historical validation cohort and the remaining 20% of incoming records.
4. Apply F1 improvement and accuracy/ROC-AUC guardrails. Rejected challengers cannot be promoted.
5. Explicitly promote an eligible challenger **in that session only**, or restore the previous in-session champion.
6. Download the model as a `.skops` artifact and the analysis as JSON for further review.

## Deployment

Add the following packages to `dashboard/requirements.txt`, in addition to its existing Streamlit, requests and pandas:

```text
numpy>=1.26,<3
scipy>=1.13,<2
scikit-learn>=1.5,<2
joblib>=1.3,<2
skops>=0.11,<1
```

Add `skops>=0.11,<1` to the root `requirements.txt` for the local demo/test generator. `requirements-api.txt` remains unchanged because the existing public Render backend does **not** accept uploaded models.

**Cloud safety:** The Streamlit host executes user-supplied model inference and fits in the same process. This demo uses input limits and skops allowlisting, but it is not a hardened multi-tenant model execution service. For broadly accessible public upload capability, use authenticated workspaces, request throttling, per-job CPU/memory/time limits, sandboxed workers and persistent artifact storage.
