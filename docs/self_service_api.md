# ModelOps — Self-service workspaces

## Product contract

An ordinary account holder can:

1. Sign up and sign in.
2. Upload a fitted binary sklearn model in `.skops` format plus reference and incoming CSV datasets.
3. Immediately use that workspace to predict (single row or batch), without publication or administrator approval.
4. Run PSI/KS drift monitoring and, when delayed labels exist, evaluate its champion.
5. Train a challenger with new labels and an independent historical validation holdout.
6. See the *champion vs challenger* metrics and acceptance checks.
7. Download an **eligible** challenger `.skops` model, or explicitly promote it to the active version.
8. Predict with the newly promoted version, or compare predictions using an eligible challenger before promotion.
9. Keep every workspace separate from other users' workspaces, and view their saved audit history after an application restart.

Rejected challengers cannot be promoted or downloaded as improved versions; they remain in version history for audit.

## Uploaded data contracts

- Only standard fitted sklearn binary classifiers with numeric named features and class labels `[0,1]`, `predict`, `predict_proba` and `fit`, in `.skops` format, no unknown types.
- `reference.csv`: features, unique ID, training target (needed for challenger training).
- `incoming.csv`: features and unique IDs, no target.
- `labels.csv`: unique IDs and target `0/1` for incoming data, optional until performance monitoring/retraining.
- `validation.csv`: independent historical labeled holdout, required for retraining and promotion checks. Must not overlap reference or incoming IDs; must be genuinely out-of-sample to avoid misleading scores.
- Numeric features must be finite and match the uploaded fitted model's `feature_names_in_`.
- 5 MB per input file; 10,000 rows per dataset; 30 model features; 3 workspaces per user; 6 versions per workspace; batches up to 100.

### Challenger acceptance

The existing `model_workspace.py` logic trains a cloned model on reference data plus an 80% split of newly labeled incoming data and evaluates the untouched 20% incoming holdout and independent historical holdout. It requires at least 1 percentage point improvement in incoming holdout F1 by default, no more than a 3-point historical accuracy drop, and no more than 2-point historical or incoming ROC-AUC drop. Other metric improvements **do not guarantee** eligibility if a guardrail fails.

Repeated experiments on the same holdout can overfit model selection. For stronger evidence, reserve a final blind test cohort.

## Run locally

From the repository root in the existing `.venv`:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-self-service-api.txt
export MODELOPS_DATABASE_URL='YOUR_NEON_POSTGRES_CONNECTION_STRING'
python -m uvicorn src.modelops.self_service_api:app --host 127.0.0.1 --port 8003
```

Run the Streamlit UI in a second terminal:

```bash
export MODELOPS_SELF_SERVICE_API_URL='http://127.0.0.1:8003'
python -m streamlit run dashboard/app.py --server.port 8501
```

Open Streamlit's **My Models** page, create an account, and upload a model and its datasets. No admin token is involved. The API automatically creates its PostgreSQL tables at startup, using the `ss_` prefix to avoid prior cloud registry tables.

For an example model:

```bash
python scripts/create_loan_demo.py
```

Upload `demo_uploads/loan_default/loan_default.skops`, `reference.csv`, `incoming.csv`, `delayed_labels.csv` as labels, and `validation.csv`. Set ID column `record_id` and target column `target`.

## Deploy to Render and Streamlit Cloud

Create a dedicated Render Python web service for the self-service API. Set **Root Directory: empty**, **Python: 3.12**, and:

- Build command: `python -m pip install -r requirements-self-service-api.txt`
- Start command: `python -m uvicorn src.modelops.self_service_api:app --host 0.0.0.0 --port $PORT`
- Environment secret: `MODELOPS_DATABASE_URL=<Neon PostgreSQL URL>`

Do not use the old `cloud_api:app` service, and do not set or require `MODELOPS_CLOUD_ADMIN_TOKEN` for the new user interface.

In Streamlit Community Cloud → App settings → Secrets:

```toml
MODELOPS_SELF_SERVICE_API_URL = 'https://YOUR-SELF-SERVICE-API.onrender.com'
```

Ensure `dashboard/requirements.txt` continues to include Streamlit, pandas and Requests (the UI does not need psycopg or skops). After pushing the new page, restart Streamlit if needed.

## Routes

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Create bearer session |
| POST | `/auth/logout` | Revoke session |
| GET | `/my/workspaces` | List only caller's workspaces |
| POST | `/my/workspaces` | Upload fitted model + datasets |
| GET | `/my/workspaces/{id}` | Metadata and feature contract |
| POST | `/my/workspaces/{id}/predict` | Single prediction, optional `?version=` |
| POST | `/my/workspaces/{id}/predict/batch` | Batch predictions |
| POST | `/my/workspaces/{id}/monitor` | Drift and performance checks |
| POST | `/my/workspaces/{id}/train` | Train and evaluate challenger |
| GET | `/my/workspaces/{id}/versions` | Saved model history and decision |
| GET | `/my/workspaces/{id}/models/{version}/download` | Download eligible/champion `.skops` |
| POST | `/my/workspaces/{id}/promote` | Explicit eligible-challenger promotion |
| POST | `/my/workspaces/{id}/rollback` | Restore prior champion |
| GET | `/my/workspaces/{id}/events` | Event history |

Every `/my/` endpoint requires the caller's bearer session and checks workspace ownership. IDs are not authorization tokens.

## Security / operational limits

**This is a functional prototype for trusted testers, not yet a production-safe public upload service.**

- The built-in email/password flow has no email verification, password reset, MFA, centralized rate limiting, or anti-abuse signup controls. Add managed authentication, edge rate limiting and signup throttling before allowing unrestricted visitors.
- `.skops` blocks unknown Python types; nevertheless, executing uploaded sklearn estimators consumes server CPU/memory and is **not a security sandbox**. Use isolated, time-limited workers with fixed resource caps before untrusted public uploads; separate inference/training from the API workers.
- Model sizes, training times and database write volume still require budget/quota enforcement beyond the included demonstration caps; unbounded account creation remains an abuse risk.
- PostgreSQL stores uploaded bytes. Don't accept sensitive customer data until retention, encryption, deletion, audit and privacy controls have been reviewed.
- For long training, replace the synchronous `/train` endpoint with queued background workers and status polling.
- No automatic claim of superior model quality is made. If checks fail, users keep their original champion and can try again with new data.

## For users who have a `.joblib` model

Only convert a **model you trained yourself or otherwise independently trust**, on your own computer:

```bash
python scripts/export_trusted_model.py path/to/trusted_model.joblib path/to/my_model.skops
```

The `joblib.load()` conversion step can run arbitrary code from a malicious file, so **never** use this conversion command with an untrusted download. The hosted ModelOps app itself must never accept `.joblib` or `.pkl` uploads.
