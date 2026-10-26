
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
