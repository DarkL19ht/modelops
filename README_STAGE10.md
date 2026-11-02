# ModelOps — Persistent Multi-Model Registry (Stage 10)

Additive upgrade for an existing ModelOps repository that has completed the `Model Workspaces` upgrade. No original project source files are overwritten.

## Included

- `src/modelops/workspace_registry.py` — SQLite metadata and audit history; per-workspace `.skops` model versions; dataset snapshots; saved drift/performance reports; guarded candidate persistence, promotion and one-step rollback.
- `src/modelops/workspace_api.py` — read-only, local FastAPI inference service with model-specific feature schemas and versions.
- `dashboard/pages/2_Persistent_Registry.py` — local, explicitly enabled operator page for upload, monitoring, retraining, version history, promotion, rollback and prediction.
- `tests/test_workspace_registry.py` — 11 integration tests.
- `docs/persistent_workspaces.md` — full setup and constraints.
- `INSTALL_STAGE10.txt` — command checklist.

## Crucial boundaries

The operator UI is disabled unless `MODELOPS_LOCAL_REGISTRY=1`. This flag is only for a local trusted operator. Never set it on your public Streamlit app. The workspace API is unauthenticated; bind **only** to `127.0.0.1`. It is separate from the original public churn FastAPI service, which remains unchanged.

This registry persists on a **local, durable filesystem**, not on Render Free / Streamlit Community Cloud's ephemeral hosting. For cloud-wide persistence and authenticated public model uploads, a subsequent phase must use an external durable database and model object storage, plus authentication and workload isolation. MLflow is not connected to this local registry yet.

Only fitted, numeric-feature, binary scikit-learn models stored as `.skops` with no unknown trusted types are supported. `skops` is safer than pickle but is not a sandbox for untrusted code. Only work with synthetic or non-sensitive data you own.

Read `INSTALL_STAGE10.txt` first.
