## User-Uploaded Model Workspaces

ModelOps includes an experimental **Model Workspaces** Streamlit page for analysing multiple separately uploaded binary scikit-learn models. Users can supply model-specific reference and incoming datasets, optional delayed labels, and an independent historical holdout. The workspace calculates input distribution drift, evaluates performance, trains a cloned challenger, runs guarded acceptance checks, and supports explicit in-session promotion and rollback.

Models are uploaded in the `.skops` format; unknown serialized types and pickle/joblib uploads are not accepted. Each workspace remains isolated in session memory. The public Render API, persistent MLflow registry and default churn dashboard are not modified by session promotions. This feature supports standard numeric-input binary classifiers and synthetic or nonsensitive datasets; it is not a hardened multi-tenant model hosting service.

See [Model Workspaces](docs/model_workspaces.md) for upload formats, constraints and usage.
