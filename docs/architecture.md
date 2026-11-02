# Architecture

ModelOps separates its synthetic churn reference workflow from the local multi-model workspace registry. The reference workflow trains a baseline classifier, generates monitoring reports and records experiments in MLflow. The workspace registry stores each uploaded compatible model, its feature schema, associated datasets, monitoring outcomes and champion/candidate versions independently.

The local workspace metadata is backed by SQLite and artifact files. The dedicated workspace FastAPI service selects a workspace's current champion for inference. Model promotion changes a workspace version pointer; it does not overwrite an unrelated workspace or automatically change the separately deployed churn reference service.

The Streamlit reference dashboard and local workspace administration interface are separate surfaces. Local workspaces are not authenticated multi-tenant cloud accounts.
