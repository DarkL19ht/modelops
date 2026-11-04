# Public Demonstration

ModelOps provides an account-free, read-only demonstration of its model lifecycle.

## Implementation

`dashboard/pages/0_Try_Demo.py` presents predictions, feature-distribution drift,
and champion-versus-challenger evaluation. `src/modelops/public_demo.py`
reuses the project's synthetic data generator and evaluated model lifecycle.

The demonstration:

- Trains two trusted, preconfigured scikit-learn pipelines in memory.
- Uses separate historical and incoming holdout cohorts for evaluation.
- Applies the same drift and candidate-acceptance functions as the core project.
- Generates predictions for visitors without uploading or executing user models.
- Displays acceptance decisions and provides a downloadable evaluation report.
- Does not connect to the database, modify workspaces, register accounts or promote models.

All observations are synthetic. The results demonstrate system behaviour, not
real-world generalisation or continuously updated production monitoring.

## Deployment

The demo depends on Streamlit, pandas, NumPy, SciPy, scikit-learn and joblib.
List those dependencies in `dashboard/requirements.txt` for Streamlit Community Cloud.

The demo runs independently of the FastAPI and PostgreSQL services. The
registered-user interface continues to provide model uploads, per-user
workspaces, predictions and challenger model management. Account registration
may remain invitation-restricted while the public demo is available to all.
