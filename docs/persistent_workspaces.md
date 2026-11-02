# Persistent Model Workspaces — Local operator guide

This upgrade builds on the existing `src/modelops/model_workspace.py` and `dashboard/pages/1_Model_Workspaces.py`. It **does not overwrite** any of the previous project files.

## Scope and security

- Intended for a **single, trusted operator on their own Mac**, using synthetic/non-sensitive datasets and models they trust.
- `.skops` models only, no arbitrary `.pkl`/`.joblib` uploads, no custom Python classes, and no user-supplied trusted type overrides. skops is safer than pickle, but not a sandbox for running untrusted estimators; the trained `predict()` or `fit()` methods still execute Python code.
- Local registry data lives in `var/modelops_registry/`, which must be excluded from Git; it contains datasets, model artifacts and a SQLite database.
- The operator Streamlit page is disabled unless `MODELOPS_LOCAL_REGISTRY=1`. **Never set that flag on Streamlit Community Cloud.** The page has no authentication.
- The model-specific API is **read-only**, is unauthenticated and is intended for `127.0.0.1` only. Never bind it to `0.0.0.0` or place it on public Render as-is.
- This is not MLflow integration. Existing MLflow experiments are unaffected. Each workspace has a lightweight independent SQLite-backed version history and audit log.
- On free Render and Streamlit Cloud, the filesystem is ephemeral. SQLite on those services is **not durable**. For real public persistence, we need external durable database/object storage, authentication, quotas, and background worker isolation in a later stage.

## Getting started

From repository root, in your existing Python 3.12 virtual environment:

```bash
python -m pip install -r requirements.txt
python scripts/create_loan_demo.py
python -m pytest tests/test_workspace_registry.py -v
python -m pytest -q
```

Ensure requirements include `skops>=0.11,<1`, `fastapi>=0.115,<1`, `httpx>=0.27,<1`, `streamlit>=1.40,<2`, `pandas>=2.2,<3` and `requests>=2.32,<3` (all installed in earlier stages).

### Terminal 1 — local registry inference API

```bash
cd ~/Documents/modelops
source .venv/bin/activate
python -m uvicorn src.modelops.workspace_api:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/docs

### Terminal 2 — local operator dashboard

```bash
cd ~/Documents/modelops
source .venv/bin/activate
MODELOPS_LOCAL_REGISTRY=1 python -m streamlit run dashboard/app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Use the sidebar page **Persistent Registry**, not the earlier **Model Workspaces** page, to register a model that survives local app restarts. Upload the loan-default files from `demo_uploads/loan_default/`:

- `loan_default.skops`
- `reference.csv`
- `incoming.csv`
- `delayed_labels.csv`
- `validation.csv`

Identifier: `record_id`. Target: `target`.

Perform monitoring and save the report. Run retraining; it creates a **new candidate model version** even if rejected. Only eligible candidates can be explicitly promoted. A one-step rollback is available after a successful promotion. The model-specific API reads the champion from SQLite, so it automatically follows the updated champion pointer on the next request.

### Example API call

Replace WORKSPACE_ID with the 32-character identifier shown on the Registry page:

```bash
curl -s http://127.0.0.1:8001/workspaces/WORKSPACE_ID

curl -s -X POST http://127.0.0.1:8001/workspaces/WORKSPACE_ID/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"annual_income":52000,"debt_ratio":0.42,"late_payments":2,"account_age_years":6}}'
```

The API includes `model_version`, `model_sha256`, `predicted_class`, and `positive_class_probability` in every prediction response. These predictions are for demonstration only.

## What survives a restart?

- Uploaded model and each saved candidate model file
- Reference and incoming data, optional labels and validation cohort
- Champion version pointer, one-step rollback pointer
- Monitoring report and candidate decision audit events

No Streamlit session state is needed to recover this data. To verify, stop and restart both local processes, and revisit **Persistent Registry**.

## Notes

- New models may have different sets of **numeric features**, but still must be fitted binary scikit-learn classifiers with class labels `[0, 1]` and `predict_proba()`.
- Model training is performed synchronously; it is not safe or appropriate for a public multi-user service.
- `.skops` uses version-sensitive model formats. Keep your environment versions consistent.
- Local SQLite is suitable for this demo and single operator, not a horizontally scaled model registry.
- `train_candidate` reuses the Stage 9 model evaluation and guardrails; it does not provide a guarantee of real-world validity of a model trained on synthetic data.
- Git ignore `var/` and `demo_uploads/` to avoid publishing saved model files and datasets.
