"""Owner isolation and worker lifecycle for durable training job APIs."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.modelops.training_job_routes import build_training_router
from src.modelops.training_jobs import TrainingJobStore
from src.modelops.training_job_worker import process_one


class RegistryStub:
    def owned_info(self, owner, wid):
        if owner != "user-a" or wid != "w-a":
            raise KeyError("Not found")
        return {"id": wid}


class JobsStub:
    def enqueue(self, owner, wid, gain):
        assert owner == "user-a" and wid == "w-a"
        return {"id": "job-1", "status": "queued"}

    def get_owned(self, owner, jid):
        if owner != "user-a" or jid != "job-1":
            raise KeyError("Not found")
        return {"id": jid, "status": "succeeded", "result": {"candidate_version": 2}}

    def list_owned(self, owner, wid):
        if owner != "user-a" or wid != "w-a":
            raise KeyError("Not found")
        return [{"id": "job-1", "status": "succeeded"}]


def make_client(registry=None):
    app = FastAPI()

    def auth(authorization: str = ""):
        from fastapi import Header
        # Router dependency receives the caller's Authorization header.
        return {"id": "user-a", "email": "a@example.org"}

    def operation(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, "Not found") from None

    app.include_router(build_training_router(
        JobsStub(), auth, operation, lambda *a: None, registry if registry is not None else RegistryStub()
    ))
    return TestClient(app)


def test_enqueue_and_read_training_job(monkeypatch):
    monkeypatch.setenv("MODELOPS_PUBLIC_PORTABLE_MODE", "0")
    monkeypatch.delenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", raising=False)
    with make_client() as client:
        response = client.post("/my/workspaces/w-a/training-jobs", json={"min_f1_gain": 0.02})
        assert response.status_code == 202, response.text
        assert response.json()["job"]["id"] == "job-1"
        assert client.get("/my/training-jobs/job-1").json()["job"]["status"] == "succeeded"
        assert len(client.get("/my/workspaces/w-a/training-jobs").json()["jobs"]) == 1


def test_foreign_workspace_is_rejected(monkeypatch):
    monkeypatch.setenv("MODELOPS_PUBLIC_PORTABLE_MODE", "0")
    monkeypatch.delenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", raising=False)
    with make_client() as client:
        response = client.post("/my/workspaces/other/training-jobs", json={})
        assert response.status_code == 404
        assert client.get("/my/training-jobs/other-job").status_code == 404


def test_beta_allowlist_controls_training(monkeypatch):
    monkeypatch.setenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", "someone-else@example.org")
    with make_client() as client:
        assert client.post("/my/workspaces/w-a/training-jobs", json={}).status_code == 403


def test_invalid_gain_rejected(monkeypatch):
    monkeypatch.delenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", raising=False)
    with make_client() as client:
        assert client.post("/my/workspaces/w-a/training-jobs", json={"min_f1_gain": -0.2}).status_code == 422


def test_worker_records_success(monkeypatch):
    class Jobs:
        def __init__(self):
            self.result = None
        def claim(self, worker):
            return {"id": "j1", "owner_id": "u1", "workspace_id": "w1", "min_f1_gain": .01}
        def complete(self, jid, worker, result):
            self.result = result
        def fail(self, *args):
            pytest.fail("Unexpected failure")
    from src.modelops import training_job_worker
    monkeypatch.setattr(training_job_worker, "run_isolated", lambda *a, **kw: {"candidate_version": 2})
    jobs = Jobs()
    assert process_one(jobs, "worker-1")
    assert jobs.result["candidate_version"] == 2


def test_worker_records_failure(monkeypatch):
    class Jobs:
        def __init__(self):
            self.message = None
        def claim(self, worker):
            return {"id": "j1", "owner_id": "u1", "workspace_id": "w1", "min_f1_gain": .01}
        def complete(self, *args):
            pytest.fail("Unexpected success")
        def fail(self, jid, worker, message):
            self.message = message
    from src.modelops import training_job_worker
    def broken(*args, **kwargs):
        raise RuntimeError("internal DB credential should not be shown")
    monkeypatch.setattr(training_job_worker, "run_isolated", broken)
    jobs = Jobs()
    assert process_one(jobs, "worker-1")
    assert "credential" not in jobs.message


def test_worker_idle():
    class Jobs:
        def claim(self, worker):
            return None
    assert process_one(Jobs(), "worker-1") is False


def test_job_public_view():
    now = datetime.now(timezone.utc)
    result = TrainingJobStore._view({
        "id": "x", "workspace_id": "w", "status": "succeeded",
        "min_f1_gain": .01, "created_at": now, "started_at": now,
        "finished_at": now, "result_json": '{"candidate_version": 2}',
        "error_message": None,
    })
    assert result["result"] == {"candidate_version": 2}
    assert result["created_at"].endswith("+00:00")

def test_portable_user_can_queue_without_allowlist(monkeypatch):
    monkeypatch.setenv(
        "MODELOPS_PUBLIC_PORTABLE_MODE", "1"
    )
    monkeypatch.delenv(
        "MODELOPS_TRUSTED_OPERATOR_EMAILS",
        raising=False,
    )

    class PortableRegistry(RegistryStub):
        def is_portable_workspace(self, owner, wid):
            return owner == "user-a" and wid == "w-a"

    with make_client(registry=PortableRegistry()) as client:
        response = client.post(
            "/my/workspaces/w-a/training-jobs",
            json={"min_f1_gain": 0.02},
        )

        assert response.status_code == 202, response.text
        assert response.json()["job"]["status"] == "queued"