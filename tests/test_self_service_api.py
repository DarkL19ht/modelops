"""API contract tests using a small in-memory fake instead of Neon."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.modelops import self_service_api as api


class FakeAuth:
    _test_stub = True

    def __init__(self):
        self.users = {"a": {"id": "a", "email": "a@example.org"}, "b": {"id": "b", "email": "b@example.org"}}
        self.tokens = {"token-a": "a", "token-b": "b"}

    def identify(self, token):
        if token not in self.tokens:
            raise PermissionError("Unauthorized")
        return self.users[self.tokens[token]]

    def login(self, email, password):
        if email not in {u["email"] for u in self.users.values()}:
            raise ValueError("Invalid credentials")
        u = next(u for u in self.users.values() if u["email"] == email)
        return {"access_token": f'token-{u["id"]}', "user": u}

    def logout(self, token):
        self.tokens.pop(token, None)

    def register(self, email, password):
        return {"id": "new", "email": email}


class FakeRegistry:
    _test_stub = True

    def __init__(self):
        self.workspaces = {}
        self.promotions = {}
        self.downloads = {}

    def ping(self):
        return True

    def register(self, ws, owner):
        wid = "a" * 32 if owner == "a" else "b" * 32
        self.workspaces[wid] = (owner, ws)
        self.promotions[wid] = 1
        return wid

    def _owned(self, user, wid):
        item = self.workspaces.get(wid)
        if not item or item[0] != user:
            raise KeyError("Workspace not found")
        return item[1]

    def mine(self, user):
        return [{"id": wid, "name": ws.name, "champion_version": self.promotions[wid],
                 "features": list(ws.features)}
                for wid, (uid, ws) in self.workspaces.items() if uid == user]

    def owned_info(self, user, wid):
        ws = self._owned(user, wid)
        return {"id": wid, "name": ws.name, "features": list(ws.features),
                "champion_version": self.promotions[wid]}

    def owned_versions(self, user, wid):
        self._owned(user, wid)
        return [{"version": 1, "status": "champion"},
                {"version": 2, "status": "eligible"}]

    def owned_events(self, user, wid):
        self._owned(user, wid)
        return [{"kind": "registered"}]

    def owned_monitor(self, user, wid, *, threshold):
        ws = self._owned(user, wid)
        return {"drift": ws.monitor(threshold)["drift"], "champion_version": 1}

    def owned_train(self, user, wid, *, min_f1_gain):
        ws = self._owned(user, wid)
        return {"candidate_version": 2, "candidate_status": "eligible",
                "decision": {"eligible": True, "checks": {"incoming_f1_gain": True}, "reasons": []},
                "evaluation": {"historical_holdout": {"champion": {"f1": 0.5}, "challenger": {"f1": 0.6}}}}

    def owned_promote(self, user, wid, version):
        self._owned(user, wid)
        if version != 2:
            raise ValueError("Only eligible challengers may be promoted")
        self.promotions[wid] = 2
        return {"champion": 2, "previous": 1}

    def owned_rollback(self, user, wid):
        self._owned(user, wid)
        self.promotions[wid] = 1
        return {"champion": 1}

    def owned_predict(self, user, wid, records, *, version=None):
        ws = self._owned(user, wid)
        if any(set(record) != set(ws.features) for record in records):
            raise ValueError("Incorrect feature contract")
        predictions = ws.champion.predict_proba(pd.DataFrame(records)[list(ws.features)])[:, 1]
        return {"model_version": version or self.promotions[wid], "model_sha256": "f"*64,
                "predictions": [{"predicted_class": int(p >= .5),
                                 "positive_class_probability": float(p)} for p in predictions]}

    def owned_artifact(self, user, wid, version):
        self._owned(user, wid)
        if version not in (1, 2):
            raise ValueError("Not eligible")
        return b"simulated-skops-file"


@pytest.fixture
def setup(monkeypatch):
    rng = np.random.default_rng(43)
    features = pd.DataFrame({"age": rng.uniform(18, 80, 180), "debt": rng.uniform(0, 1, 180)})
    target = (features.age / 80 + features.debt > 1.1).astype(int)
    model = LogisticRegression().fit(features, target)
    monkeypatch.setattr(api, "load_uploaded_model", lambda blob: model)
    registry = FakeRegistry()
    with TestClient(api.create_app(registry=registry, auth=FakeAuth())) as client:
        yield client, registry


def upload(client, token, name="Loan model"):
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"record_id": list(range(1, 121)),
        "age": rng.uniform(18, 80, 120), "debt": rng.uniform(0, 1, 120),
        "target": [0, 1] * 60})
    incoming = pd.DataFrame({"record_id": list(range(200, 320)),
        "age": rng.uniform(18, 80, 120), "debt": rng.uniform(0, 1, 120)})
    labels = pd.DataFrame({"record_id": range(200, 320), "target": [0, 1] * 60})
    validation = pd.DataFrame({"record_id": range(400, 460),
        "age": rng.uniform(18, 80, 60), "debt": rng.uniform(0, 1, 60),
        "target": [0, 1] * 30})
    files = {"model": ("model.skops", b"dummy-skops"),
             "reference": ("reference.csv", reference.to_csv(index=False)),
             "incoming": ("incoming.csv", incoming.to_csv(index=False)),
             "labels": ("labels.csv", labels.to_csv(index=False)),
             "validation": ("validation.csv", validation.to_csv(index=False))}
    return client.post("/my/workspaces", headers={"Authorization": f"Bearer {token}"},
                       data={"name": name}, files=files)


def test_authentication_required(setup):
    client, _ = setup
    assert client.get("/my/workspaces").status_code == 401
    assert client.get("/my/workspaces", headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_user_can_upload_and_predict_own_model(setup):
    client, _ = setup
    res = upload(client, "token-a")
    assert res.status_code == 201, res.text
    wid = res.json()["id"]
    prediction = client.post(f"/my/workspaces/{wid}/predict", json={"features": {"age": 40, "debt": 0.5}},
                             headers={"Authorization": "Bearer token-a"})
    assert prediction.status_code == 200, prediction.text
    assert 0 <= prediction.json()["positive_class_probability"] <= 1
    assert prediction.json()["model_version"] == 1


def test_other_user_cannot_view_predict_or_download(setup):
    client, _ = setup
    wid = upload(client, "token-a").json()["id"]
    headers = {"Authorization": "Bearer token-b"}
    for url in [f"/my/workspaces/{wid}", f"/my/workspaces/{wid}/versions",
                f"/my/workspaces/{wid}/events", f"/my/workspaces/{wid}/models/2/download"]:
        assert client.get(url, headers=headers).status_code == 404
    assert client.post(f"/my/workspaces/{wid}/predict", json={"features": {"age": 25, "debt": 0.1}},
                       headers=headers).status_code == 404
    assert client.post(f"/my/workspaces/{wid}/monitor", json={}, headers=headers).status_code == 404
    assert client.post(f"/my/workspaces/{wid}/train", json={}, headers=headers).status_code == 404
    assert client.post(f"/my/workspaces/{wid}/promote", json={"version": 2}, headers=headers).status_code == 404


def test_self_service_monitor_train_and_download(setup):
    client, _ = setup
    wid = upload(client, "token-a").json()["id"]
    h = {"Authorization": "Bearer token-a"}
    monitoring = client.post(f"/my/workspaces/{wid}/monitor", json={"psi_threshold": 0.2}, headers=h)
    assert monitoring.status_code == 200, monitoring.text
    assert len(monitoring.json()["drift"]["features"]) == 2
    train = client.post(f"/my/workspaces/{wid}/train", json={"min_f1_gain": 0.01}, headers=h)
    assert train.status_code == 200
    assert train.json()["decision"]["eligible"]
    file_response = client.get(f"/my/workspaces/{wid}/models/2/download", headers=h)
    assert file_response.status_code == 200
    assert file_response.content == b"simulated-skops-file"
    promote = client.post(f"/my/workspaces/{wid}/promote", json={"version": 2}, headers=h)
    assert promote.json()["champion"] == 2
    p = client.post(f"/my/workspaces/{wid}/predict", json={"features": {"age": 50, "debt": 0.4}}, headers=h)
    assert p.json()["model_version"] == 2


def test_invalid_feature_contract_is_rejected(setup):
    client, _ = setup
    wid = upload(client, "token-a").json()["id"]
    response = client.post(f"/my/workspaces/{wid}/predict", json={"features": {"foo": 1}},
                           headers={"Authorization": "Bearer token-a"})
    assert response.status_code == 422


def test_batch_predictions_owned(setup):
    client, _ = setup
    wid = upload(client, "token-a").json()["id"]
    r = client.post(f"/my/workspaces/{wid}/predict/batch", json={"rows": [
        {"features": {"age": 40, "debt": 0.4}}, {"features": {"age": 60, "debt": 0.7}}
    ]}, headers={"Authorization": "Bearer token-a"})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2
