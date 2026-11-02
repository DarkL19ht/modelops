# Local Operations

## Requirements

Python 3.12, dependencies from `requirements.txt`, and compatible generated or uploaded model artifacts.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Start the local workspace inference service:

```bash
python -m uvicorn src.modelops.workspace_api:app --host 127.0.0.1 --port 8001
```

Start the local Streamlit dashboard with the registry management page enabled:

```bash
MODELOPS_LOCAL_REGISTRY=1 python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

The original reference-model inference service can be started independently:

```bash
python -m uvicorn src.modelops.api:app --host 127.0.0.1 --port 8000
```

Store model and dataset outputs outside version control. The local registry is intended for a trusted operator on one machine and does not provide durable, authenticated multi-user hosting.
