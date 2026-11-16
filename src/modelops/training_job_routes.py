"""Owner-scoped REST endpoints for durable challenger training jobs."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    min_f1_gain: float = Field(default=0.01, ge=0.0, le=0.5)


def build_training_router(jobs, bearer, operation, limit_request, registry):
    router = APIRouter()

    def beta_guard(user: dict, wid: str):
        if os.getenv("MODELOPS_PUBLIC_PORTABLE_MODE", "0") == "1":
            method = getattr(registry, "is_portable_workspace", None)
            if method is not None and operation(method, user["id"], wid):
                return
        allowed = os.getenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", "").strip()
        if (allowed or os.getenv("MODELOPS_PUBLIC_PORTABLE_MODE", "0") == "1") and user.get("email", "").strip().lower() not in {
            address.strip().lower() for address in allowed.split(",") if address.strip()
        }:
            raise HTTPException(403, "Workspace processing is restricted during beta")

    @router.post("/my/workspaces/{wid}/training-jobs", status_code=202)
    def submit(wid: str, payload: TrainingRequest, user=Depends(bearer)):
        beta_guard(user, wid)
        limit_request("train", user["id"], 5, 86400)
        # Validate workspace ownership BEFORE creating a job.
        operation(registry.owned_info, user["id"], wid)
        return {"job": operation(jobs.enqueue, user["id"], wid, payload.min_f1_gain)}

    @router.get("/my/training-jobs/{job_id}")
    def status(job_id: str, user=Depends(bearer)):
        return {"job": operation(jobs.get_owned, user["id"], job_id)}

    @router.get("/my/workspaces/{wid}/training-jobs")
    def history(wid: str, user=Depends(bearer)):
        return {"jobs": operation(jobs.list_owned, user["id"], wid)}

    return router
