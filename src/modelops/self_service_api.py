"""User-owned model upload, monitoring, challenger selection and prediction API.

No global admin/publication gate: authenticated users operate on their OWN workspaces.
Unrestricted public traffic requires edge rate limits,
account verification, resource quotas, and isolated model execution workers.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from src.modelops.model_workspace import MAX_FILE_BYTES, Workspace, load_uploaded_model, read_csv
from src.modelops.account_limits import RateLimitExceeded

logger = logging.getLogger(__name__)


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)


class VerifyEmailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(pattern=r"^[0-9]{8}$")


class ResendEmailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=254)


class PredictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1, max_length=30)


class PredictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PredictItem] = Field(min_length=1, max_length=100)


class MonitorOptions(BaseModel):
    psi_threshold: float = Field(default=0.20, gt=0, le=10)


class CandidateOptions(BaseModel):
    min_f1_gain: float = Field(default=0.01, ge=0, le=0.5)


class PromotionChoice(BaseModel):
    version: int = Field(ge=2)


async def read_upload(upload: UploadFile | None, suffix: str):
    if upload is None:
        return None
    if not upload.filename or not upload.filename.lower().endswith(suffix):
        raise HTTPException(422, f"Only {suffix} files are accepted")
    raw = await upload.read(MAX_FILE_BYTES + 1)
    if not (1 <= len(raw) <= MAX_FILE_BYTES):
        raise HTTPException(413, "Each uploaded file must be between 1 byte and 5 MB")
    return raw


def create_app(registry=None, auth=None) -> FastAPI:
    if registry is None:
        from src.modelops.self_service_registry import SelfServiceRegistry
        registry = SelfServiceRegistry()
    if auth is None:
        from src.modelops.email_verification import VerifiedAuthStore as AuthStore
        auth = AuthStore()

    @asynccontextmanager
    async def lifespan(app):
        if not hasattr(auth, "_test_stub"):
            auth.initialize()
        if not hasattr(registry, "_test_stub"):
            registry.initialize()
        yield

    app = FastAPI(title="ModelOps Self-Service API", version="0.1.0", lifespan=lifespan)

    def operation(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, "Workspace not found") from None
        except PermissionError:
            raise HTTPException(401, "Authentication required") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception as exc:
            if isinstance(exc, RateLimitExceeded):
                raise HTTPException(429, "Request limit reached. Please try again later") from None
            if getattr(exc, "sqlstate", None) == "23505":
                raise HTTPException(409, "Account or workspace name already exists") from None
            logger.exception("ModelOps self-service operation failed")
            raise HTTPException(503, "Operation temporarily unavailable") from None

    def limit_request(action: str, identity: str, limit: int, seconds: int):
        if os.getenv("MODELOPS_RATE_LIMITS_ENABLED", "1") != "0":
            if callable(getattr(auth, "enforce_limit", None)):
                operation(auth.enforce_limit, action, identity, limit=limit, seconds=seconds)

    def client_identity(request: Request) -> str:
        # This is the trusted ASGI client address, not an arbitrary X-Forwarded-For header.
        # Configure trusted proxy forwarding at the edge before relying on IP limits.
        return request.client.host if request.client else "unknown"

    def bearer(authorization: str | None = Header(default=None)):
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(401, "Sign in to access your workspaces")
        token = authorization[len(prefix):].strip()
        if not token:
            raise HTTPException(401, "Invalid access token")
        user = operation(auth.identify, token)
        return user

    @app.get("/health")
    def health():
        return {"status": "ok"} if operation(registry.ping) else {"status": "unavailable"}

    @app.post("/auth/register", status_code=201)
    def register(credentials: Credentials, request: Request):
        limit_request("registration-ip", client_identity(request), 4, 3600)
        limit_request("registration-email", credentials.email.strip().lower(), 4, 3600)
        verified_mode = os.getenv("MODELOPS_EMAIL_VERIFICATION_REQUIRED", "0") == "1"
        if not verified_mode and not getattr(auth, "_test_stub", False):
            # Never accidentally reopen public registration without verification.
            raise HTTPException(503, "Verified registration is not configured")
        return operation(auth.register, credentials.email, credentials.password)

    @app.post("/auth/verify-email")
    def verify_email(item: VerifyEmailInput, request: Request):
        limit_request("verify-ip", client_identity(request), 30, 3600)
        limit_request("verify-email", item.email.strip().lower(), 10, 3600)
        return operation(auth.verify_email, item.email, item.code)

    @app.post("/auth/resend-verification")
    def resend_verification(item: ResendEmailInput, request: Request):
        limit_request("resend-ip", client_identity(request), 20, 3600)
        limit_request("resend-email", item.email.strip().lower(), 6, 3600)
        return operation(auth.resend_verification, item.email)

    @app.post("/auth/login")
    def login(credentials: Credentials, request: Request):
        limit_request("login-ip", client_identity(request), 30, 3600)
        limit_request("login-email", credentials.email.strip().lower(), 10, 3600)
        return operation(auth.login, credentials.email, credentials.password)

    @app.get("/auth/me")
    def me(user=Depends(bearer)):
        return user

    @app.post("/auth/logout")
    def logout(authorization: str | None = Header(default=None), user=Depends(bearer)):
        operation(auth.logout, authorization[7:].strip())
        return {"status": "signed_out"}

    @app.get("/my/workspaces")
    def my_workspaces(user=Depends(bearer)):
        return {"workspaces": operation(registry.mine, user["id"])}

    @app.post("/my/workspaces", status_code=201)
    async def create_workspace(
        name: Annotated[str, Form(min_length=1, max_length=60)],
        model: Annotated[UploadFile, File()],
        reference: Annotated[UploadFile, File()],
        incoming: Annotated[UploadFile, File()],
        user=Depends(bearer),
        id_col: str = Form("record_id"),
        target_col: str = Form("target"),
        categorical: str = Form(""),
        labels: UploadFile | None = File(default=None),
        validation: UploadFile | None = File(default=None),
    ):
        limit_request("workspace-upload", user["id"], 3, 86400)
        model_bytes = await read_upload(model, ".skops")
        ref_bytes = await read_upload(reference, ".csv")
        inc_bytes = await read_upload(incoming, ".csv")
        label_bytes = await read_upload(labels, ".csv")
        validation_bytes = await read_upload(validation, ".csv")
        try:
            workspace = Workspace(
                name=name.strip(), champion=load_uploaded_model(model_bytes),
                reference=read_csv(ref_bytes, "reference.csv"),
                incoming=read_csv(inc_bytes, "incoming.csv"),
                labels=read_csv(label_bytes, "labels.csv") if label_bytes else None,
                validation=read_csv(validation_bytes, "validation.csv") if validation_bytes else None,
                id_col=id_col.strip(), target_col=target_col.strip(),
                categorical=tuple(x.strip() for x in categorical.split(",") if x.strip()),
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception:
            logger.exception("Uploaded workspace validation failed")
            raise HTTPException(422, "Unsupported model or invalid dataset") from None
        wid = operation(registry.register, workspace, user["id"])
        return {"id": wid, "champion_version": 1, "message": "Your model is ready for predictions"}

    @app.get("/my/workspaces/{wid}")
    def info(wid: str, user=Depends(bearer)):
        return operation(registry.owned_info, user["id"], wid)

    @app.get("/my/workspaces/{wid}/versions")
    def versions(wid: str, user=Depends(bearer)):
        return {"versions": operation(registry.owned_versions, user["id"], wid)}

    @app.get("/my/workspaces/{wid}/events")
    def events(wid: str, user=Depends(bearer)):
        return {"events": operation(registry.owned_events, user["id"], wid)}

    @app.post("/my/workspaces/{wid}/monitor")
    def monitor(wid: str, opts: MonitorOptions, user=Depends(bearer)):
        limit_request("monitor", user["id"], 30, 86400)
        return operation(registry.owned_monitor, user["id"], wid, threshold=opts.psi_threshold)

    @app.post("/my/workspaces/{wid}/train")
    def train(wid: str, opts: CandidateOptions, user=Depends(bearer)):
        limit_request("training", user["id"], 3, 86400)
        return operation(registry.owned_train, user["id"], wid, min_f1_gain=opts.min_f1_gain)

    @app.post("/my/workspaces/{wid}/promote")
    def promote(wid: str, choice: PromotionChoice, user=Depends(bearer)):
        return operation(registry.owned_promote, user["id"], wid, choice.version)

    @app.post("/my/workspaces/{wid}/rollback")
    def rollback(wid: str, user=Depends(bearer)):
        return operation(registry.owned_rollback, user["id"], wid)

    @app.post("/my/workspaces/{wid}/predict")
    def predict(wid: str, item: PredictItem, version: int | None = Query(default=None, ge=1), user=Depends(bearer)):
        limit_request("inference", user["id"], 300, 3600)
        result = operation(registry.owned_predict, user["id"], wid, [item.features], version=version)
        return {key: value for key, value in result.items() if key != "predictions"} | result["predictions"][0]

    @app.post("/my/workspaces/{wid}/predict/batch")
    def batch(wid: str, batch: PredictBatch, version: int | None = Query(default=None, ge=1), user=Depends(bearer)):
        limit_request("inference", user["id"], 300, 3600)
        return operation(registry.owned_predict, user["id"], wid, [r.features for r in batch.rows], version=version)

    @app.get("/my/workspaces/{wid}/models/{version}/download")
    def download(wid: str, version: int, user=Depends(bearer)):
        data = operation(registry.owned_artifact, user["id"], wid, version)
        return Response(
            content=data, media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="modelops-v{version}.skops"',
                     "Cache-Control": "no-store"},
        )

    return app


# Streamlit's public page calls this app over HTTPS.
# This module remains importable for tests without requiring a database at import time.
if os.getenv("MODELOPS_DATABASE_URL"):
    app = create_app()
else:
    app = FastAPI(title="ModelOps Self-Service (database not configured)")

    @app.get("/health")
    def unconfigured():
        raise HTTPException(503, "Configure MODELOPS_DATABASE_URL")
