#!/usr/bin/env bash
set -euo pipefail

echo "Installing ModelOps API dependencies..."
python -m pip install -r requirements-api.txt

echo "Generating reproducible synthetic customer data..."
python -m src.modelops.generate_data

echo "Training the baseline model..."
python -m src.modelops.train

echo "Checking generated model..."
python - <<'PY'
from pathlib import Path
import joblib

path = Path("artifacts/churn_model.joblib")

assert path.is_file(), "Model artifact was not created"

model = joblib.load(path)

assert hasattr(model, "predict")
assert hasattr(model, "predict_proba")

print("ModelOps baseline build completed successfully.")
PY