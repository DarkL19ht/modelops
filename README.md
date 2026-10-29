
# ModelOps — ML Monitoring and Automated Retraining

An end-to-end machine learning engineering project
focused on monitoring model performance, identifying
data drift, and building reproducible retraining pipelines.

## Project Overview

ModelOps simulates a customer churn prediction system
where historical training data and incoming customer
activity exhibit different distributions.

The project will implement:

- Reproducible data generation and model training
- Data quality validation
- Statistical drift detection
- Model performance monitoring
- Automated retraining and evaluation
- Experiment tracking and model versioning
- FastAPI model serving
- A monitoring dashboard
- Automated tests and CI

## Current Implementation

- Synthetic customer data generator
- Reference and shifted incoming datasets
- Baseline logistic regression classifier
- Train/test evaluation
- Saved model and evaluation metrics
- Initial automated tests


## Data Drift Monitoring

ModelOps compares historical model-development data with
incoming observations to identify changes in input-feature
distributions.

The monitoring engine implements:

- Population Stability Index (PSI)
- Reference-based quantile binning
- KS distribution comparison for numeric features
- Categorical distribution comparison
- Configurable drift thresholds
- Feature validation and error handling
- JSON and Markdown monitoring reports

### Run the Monitor

```bash
python -m src.modelops.monitor

## Getting Started

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.modelops.generate_data
python -m src.modelops.train
python -m pytest -q
```

## Dataset

The dataset is entirely synthetic and is intended
for demonstrating machine learning monitoring workflows.

It must not be interpreted as evidence of real-world
customer churn behaviour.


### Evaluate Performance

```bash
python -m src.modelops.performance
```

### Monitoring Outputs

- 'reports/performance_report.md'
- 'repprts/performance_report.json'
- 'reports/performance_history.jsonl'


The demonstration uses fully labeled synthetic cohorts.
The initial baseline is derived from a held-out historical dataset.

Changes in metrics may reflect changes in outcome prevalence, the incoming population, or model behaviour.
A drift alert alone does not justify automatic model replacement.


## Guarded Model Retraining

ModelOps supports retraining a customer churn classifier
using newly available labeled customer observations.

The system trains a challenger model and compares it
with the active champion model before permitting promotion.

### Retraining Workflow

1. Load historical training data and newly labeled data.
2. Recreate the historical training and holdout partitions.
3. Separate incoming data into training and evaluation sets.
4. Train a challenger using historical and incoming training data.
5. Evaluate the champion and challenger on identical holdouts.
6. Apply configurable performance acceptance checks.
7. Archive the previous champion before explicit promotion.
8. Generate evaluation reports and a retraining audit history.

### Run a Dry-Run Evaluation

```bash
python -m src.modelops.retrain
```

A dry run never replaces the active model.

### Promote an Eligible Candidate

```bash
python -m src.modelops.retrain --promote
```

Promotion is permitted only when all acceptance checks pass.

### Retraining Reports

- `reports/retraining_decision.md`
- `reports/retraining_decision.json`
- `reports/retraining_history.jsonl`

### Model Artifacts

- `artifacts/churn_model.joblib` — active champion
- `artifacts/candidates/` — trained challenger
- `artifacts/archive/` — previous champion versions


## Experiment Tracking and Model Registry

ModelOps integrates MLflow to track and compare
champion and challenger machine-learning models.

The experiment-tracking workflow records:

- Historical and incoming holdout metrics
- Training and evaluation cohort sizes
- Model training configuration
- Source dataset fingerprints
- Champion and challenger model artifacts
- Retraining decisions and evaluation reports
- Registered challenger model versions

### Run a Tracked Experiment

```bash
python -m src.modelops.track_experiment
```

### Register a Candidate Version

```bash
python -m src.modelops.track_experiment --register-candidate
```

Candidate registration does not promote or deploy a model.

### Open the Tracking Dashboard

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root "$PWD/mlruns" \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000` and select
the `modelops-churn-retraining` experiment.

### Model Management

The local MLflow backend uses SQLite for experiment
metadata and local storage for model artifacts.

Candidate model versions are registered under
`modelops-churn-candidates`.

Registration is separate from model promotion.
The current inference champion remains managed by
the guarded local promotion workflow.


## Model Serving API

ModelOps provides a FastAPI inference service that
serves the active, locally promoted churn classifier.

The service supports:

- Single-customer churn predictions
- Batch predictions for up to 100 customers
- Pydantic input validation
- Model availability and health checks
- Model artifact SHA-256 fingerprints
- Token-protected model reloading
- Safe in-memory model replacement
- Automated API tests

### Start the API Locally

Activate the Python virtual environment.

```bash
source .venv/bin/activate
```

Load a locally configured admin token if model
reloading is required.

```bash
source .env.local
```

Run the development server:

```bash
python -m uvicorn src.modelops.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

The interactive API documentation is available at:

http://127.0.0.1:8000/docs

### API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | /health | Model availability |
| GET | /model | Active model metadata |
| POST | /predict | Single-customer prediction |
| POST | /predict/batch | Batch prediction |
| POST | /admin/reload | Reload the active model |

### Model Deployment Behaviour

The inference service loads the model from
`artifacts/churn_model.joblib`.

Retraining a challenger or registering a model
version in MLflow does not automatically change
the model served by the API.

A model must first be explicitly promoted through
the guarded retraining workflow.

The running inference service can then reload
the updated champion using the protected reload
endpoint.

### Limitations

The API currently runs locally and uses synthetic
training data. Its predictions should not be used
to make real customer decisions.

The admin reload endpoint uses a shared token,
and additional authentication, auditing, rate
limiting, monitoring, and deployment safeguards
would be required for public production use.

The service is designed for a single local worker.
Multiple worker processes would require coordinated
model-version management.



