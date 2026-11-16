"""Contract: verified users can register safe JSON workspaces without an email allowlist."""
import pytest
from fastapi.testclient import TestClient
from src.modelops import self_service_api as api


class Auth:
    _test_stub = True

    def identify(self, token):
        if token != "valid":
            raise PermissionError("Invalid token")
        return {"id": "owner-a", "email": "unlisted@example.org"}


class Registry:
    _test_stub = True
    def ping(self): return True
    def is_portable_workspace(self, owner_id, wid):
        return owner_id == "owner-a" and wid == "c" * 32
    def owned_predict(self, owner_id, wid, rows, *, version=None):
        if not self.is_portable_workspace(owner_id, wid):
            raise KeyError("Not found")
        return {"model_version": 1, "predictions": [{"predicted_class": 1, "positive_class_probability": .84}]}
    def owned_train(self, owner_id, wid, *, min_f1_gain):
        if not self.is_portable_workspace(owner_id, wid):
            raise KeyError("Not found")
        return {"candidate_version": 2, "candidate_status": "eligible"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODELOPS_PUBLIC_PORTABLE_MODE", "1")
    monkeypatch.setenv("MODELOPS_EXECUTION_MODE", "subprocess")
    monkeypatch.setenv("MODELOPS_EMAIL_VERIFICATION_REQUIRED", "1")
    monkeypatch.setenv("MODELOPS_RATE_LIMITS_ENABLED", "1")
    monkeypatch.delenv("MODELOPS_TRUSTED_OPERATOR_EMAILS", raising=False)
    calls = []
    def fake_run(action, **kwargs):
        calls.append(action)
        if action == "register_portable":
            return {"id": "c"*32, "champion_version": 1}
        if action == "predict":
            return {"model_version": 1, "predictions": [{"predicted_class": 1, "positive_class_probability": 0.84}]}
        if action == "train":
            return {"candidate_version": 2, "candidate_status": "eligible"}
        if action == "monitor":
            return {"drift": {"features": []}}
        raise AssertionError("Unexpected worker action")
    monkeypatch.setattr(api, "run_isolated", fake_run)
    with TestClient(api.create_app(registry=Registry(), auth=Auth())) as test_client:
        yield test_client, calls


def test_verified_unlisted_user_uploads_public_portable(client):
    test_client, calls = client
    from src.modelops.portable_models import load_portable_model
    import json
    model_bytes = json.dumps({"format":"modelops-linear-v1", "features":["age"], "coefficients":[1.1], "intercept": -.3,
                              "scaler":{"mean":[40.],"scale":[10.]}}).encode()
    result = test_client.post("/my/portable-workspaces", headers={"Authorization": "Bearer valid"},
        data={"name": "My classifier"},
        files={"model": ("my.modelops.json", model_bytes),
               "reference": ("reference.csv",b"record_id,age,target\n1,20,0\n"),
               "incoming": ("incoming.csv",b"record_id,age\n2,40\n")})
    assert result.status_code == 201, result.text
    assert "register_portable" in calls


def test_public_model_predictions_and_training_do_not_require_allowlist(client):
    test_client, calls = client
    h = {"Authorization":"Bearer valid"}
    wid = "c"*32
    predict = test_client.post(f"/my/workspaces/{wid}/predict", headers=h, json={"features":{"age":30.}})
    assert predict.status_code == 200, predict.text
    train = test_client.post(f"/my/workspaces/{wid}/train", headers=h, json={"min_f1_gain":0.01})
    assert train.status_code == 200, train.text
    assert predict.json()["model_version"] == 1
    assert train.json()["candidate_status"] == "eligible"


def test_legacy_skops_upload_denied_without_beta_access(client):
    test_client, calls = client
    result = test_client.post("/my/workspaces", headers={"Authorization":"Bearer valid"},
        data={"name":"Legacy"}, files={
          "model":("m.skops", b"not-a-model"),
          "reference":("reference.csv",b"record_id,age,target\n1,20,0\n"),
          "incoming":("incoming.csv",b"record_id,age\n2,40\n")})
    assert result.status_code == 403, result.text
    assert "register" not in calls


def test_public_upload_rejects_wrong_format(client):
    test_client, calls = client
    result = test_client.post("/my/portable-workspaces", headers={"Authorization":"Bearer valid"},
        data={"name":"Bad"}, files={
          "model":("not-a-model.json", b"{}"),
          "reference":("reference.csv",b"record_id,age,target\n1,20,0\n"),
          "incoming":("incoming.csv",b"record_id,age\n2,40\n")})
    assert result.status_code == 422
    assert not calls
