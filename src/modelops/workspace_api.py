"""Read-only, local-only inference API for registered model workspaces.

Run ONLY on loopback: uvicorn src.modelops.workspace_api:app --host 127.0.0.1 --port 8001
This is not authenticated and must not be exposed on the public internet.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.modelops.workspace_registry import WorkspaceRegistry

logger = logging.getLogger(__name__)


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1, max_length=30)


class BatchPredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PredictionPayload] = Field(min_length=1, max_length=100)


def _rows(registry: WorkspaceRegistry, workspace_id: str, entries: list[PredictionPayload]):
    try:
        info = registry.get_workspace_info(workspace_id)
        model, version, sha = registry.champion(workspace_id)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Workspace/model not available") from None
    except (OSError, ImportError):
        logger.exception("Champion could not be loaded")
        raise HTTPException(status_code=503, detail="Champion unavailable") from None

    expected = list(info["features"])
    records = []
    for payload in entries:
        provided = payload.features
        if set(provided) != set(expected):
            raise HTTPException(
                status_code=422,
                detail={"required_features": expected, "provided_features": sorted(provided)},
            )
        ordered = [float(provided[key]) for key in expected]
        if not np.isfinite(ordered).all():
            raise HTTPException(status_code=422, detail="Features must be finite")
        records.append(ordered)

    frame = pd.DataFrame(records, columns=expected)
    try:
        y = np.asarray(model.predict(frame))
        p = np.asarray(model.predict_proba(frame), dtype=float)
        if (
            y.shape != (len(frame),)
            or p.shape != (len(frame), 2)
            or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()
        ):
            raise ValueError("Invalid classifier output")
    except Exception:
        logger.exception("Registered model failed inference")
        raise HTTPException(status_code=503, detail="Model inference unavailable") from None

    return {
        "workspace_id": workspace_id,
        "model_version": version,
        "model_sha256": sha,
        "predictions": [
            {"predicted_class": int(y[i]), "positive_class_probability": round(float(p[i, 1]), 6)}
            for i in range(len(frame))
        ],
    }


def create_app(registry_root=None) -> FastAPI:
    registry = WorkspaceRegistry(registry_root)
    app = FastAPI(title="ModelOps Local Workspace Inference", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "registered_workspaces": len(registry.list_workspaces())}

    @app.get("/workspaces")
    def workspaces():
        return {"workspaces": registry.list_workspaces()}

    @app.get("/workspaces/{workspace_id}")
    def info(workspace_id: str):
        try:
            result = registry.get_workspace_info(workspace_id)
            result["versions"] = registry.versions(workspace_id)
            return result
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="Workspace not found") from None

    @app.post("/workspaces/{workspace_id}/predict")
    def predict(workspace_id: str, payload: PredictionPayload):
        result = _rows(registry, workspace_id, [payload])
        return {k: v for k, v in result.items() if k != "predictions"} | result["predictions"][0]

    @app.post("/workspaces/{workspace_id}/predict/batch")
    def predict_batch(workspace_id: str, payload: BatchPredictionPayload):
        return _rows(registry, workspace_id, payload.rows)

    return app


app = create_app()
