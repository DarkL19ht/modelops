"""Local durable registry integration tests (requires skops installed)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops.model_workspace import Workspace
from src.modelops.workspace_api import create_app
from src.modelops.workspace_registry import WorkspaceRegistry

FEATURES = ["income", "debt_ratio", "late_payments"]


def records(n, prefix, seed, shifted=False):
    rng = np.random.default_rng(seed)
    income = rng.normal(40000 if shifted else 55000, 9000, n).clip(10000)
    ratio = rng.uniform(0.1, 0.9, n)
    late = rng.integers(0, 6, n)
    probability = 1 / (1 + np.exp(-(-1.5 + ratio * 2.1 + late * .3 - income / 70000)))
    labels = rng.binomial(1, probability)
    return pd.DataFrame({
        "id": [f"{prefix}{i:05d}" for i in range(n)],
        "income": income, "debt_ratio": ratio, "late_payments": late,
        "target": labels,
    })


def make_workspace(name="Loan Default"):
    ref = records(400, "R", 11)
    inc = records(220, "I", 12, shifted=True)
    val = records(150, "V", 13)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(ref[FEATURES], ref["target"])
    return Workspace(
        name=name, champion=model,
        reference=ref, incoming=inc.drop(columns="target"),
        labels=inc[["id", "target"]], validation=val,
        id_col="id", target_col="target",
    )


def test_workspace_is_persistent_across_registry_restarts(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    assert registry.get_workspace_info(wid)["champion_version"] == 1
    later_process = WorkspaceRegistry(tmp_path)
    ws = later_process.load(wid)
    assert ws.name == "Loan Default"
    assert ws.features == tuple(FEATURES)
    assert len(ws.incoming) == 220


def test_models_are_independent(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    first = registry.register(make_workspace("Credit A"))
    second = registry.register(make_workspace("Credit B"))
    assert first != second
    assert len(registry.list_workspaces()) == 2
    assert registry.champion(first)[1] == registry.champion(second)[1] == 1
    assert registry.get_workspace_info(second)["name"] == "Credit B"


def test_duplicate_workspace_name_rejected(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    registry.register(make_workspace("Duplicate"))
    with pytest.raises(sqlite3.IntegrityError):
        registry.register(make_workspace("Duplicate"))
    assert len(registry.list_workspaces()) == 1


def test_monitor_report_persists(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    report = registry.monitor(wid)
    assert len(report["drift"]["features"]) == 3
    later = WorkspaceRegistry(tmp_path)
    record = later.events(wid)[0]
    assert record["type"] == "monitor"
    assert record["payload"]["drift"] == report["drift"]


def test_challenger_creates_version_without_replacing_champion(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    before = registry.champion(wid)[2]
    result = registry.train_candidate(wid, min_f1_gain=0.9)
    assert result["champion_unchanged"]
    assert result["candidate_version"] == 2
    assert registry.champion(wid)[2] == before
    assert registry.versions(wid)[0]["status"] == "rejected"
    with pytest.raises(ValueError, match="Only an eligible"):
        registry.promote(wid, 2)


def test_eligible_promotion_and_explicit_rollback(tmp_path, monkeypatch):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    old_sha = registry.champion(wid)[2]

    def simulate_passed_guardrails(self, min_f1_gain=0.01):
        # Tests the registry pointer mechanics only, not acceptance thresholds.
        model = clone(self.champion)
        model.fit(self.reference[list(self.features)], self.reference[self.target_col])
        model.set_params(classifier__C=0.5)
        model.fit(self.reference[list(self.features)], self.reference[self.target_col])
        self.challenger = model
        self.decision = {"eligible": True, "checks": {"approved": True}, "reasons": []}
        self.comparison = {"historical_holdout": {}, "incoming_holdout": {}}
        return {"decision": self.decision, "evaluation": self.comparison}

    monkeypatch.setattr(Workspace, "train_challenger", simulate_passed_guardrails)
    outcome = registry.train_candidate(wid)
    assert outcome["candidate_version"] == 2
    assert registry.champion(wid)[2] == old_sha

    registry.promote(wid, 2)
    with TestClient(create_app(tmp_path)) as client:
        live = client.post(f"/workspaces/{wid}/predict", json={
            "features": {"income": 56000, "debt_ratio": .3, "late_payments": 2}
        })
        assert live.status_code == 200
        assert live.json()["model_version"] == 2
    assert registry.get_workspace_info(wid)["champion_version"] == 2
    assert registry.champion(wid)[2] != old_sha

    restarted = WorkspaceRegistry(tmp_path)
    assert restarted.champion(wid)[1] == 2
    restarted.rollback(wid)
    assert restarted.champion(wid)[1] == 1
    assert restarted.champion(wid)[2] == old_sha
    with pytest.raises(ValueError, match="No rollback"):
        restarted.rollback(wid)


def test_tampered_model_is_detected(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    registry._version_path(wid, 1).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        registry.champion(wid)


def test_invalid_workspace_identifier_is_rejected(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    with pytest.raises(ValueError, match="Invalid workspace identifier"):
        registry.get_workspace_info("../outside")


def test_workspace_api_dynamic_features_and_versions(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["registered_workspaces"] == 1
        assert client.get("/workspaces").status_code == 200
        info = client.get(f"/workspaces/{wid}")
        assert info.status_code == 200
        assert info.json()["features"] == FEATURES
        data = {"features": {"income": 54000, "debt_ratio": .4, "late_payments": 1}}
        result = client.post(f"/workspaces/{wid}/predict", json=data)
        assert result.status_code == 200, result.text
        assert result.json()["model_version"] == 1
        assert 0 <= result.json()["positive_class_probability"] <= 1
        assert len(result.json()["model_sha256"]) == 64
        batch = client.post(f"/workspaces/{wid}/predict/batch", json={"rows": [data, data]})
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2


def test_workspace_api_rejects_invalid_inputs(tmp_path):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())
    with TestClient(create_app(tmp_path)) as client:
        assert client.post(f"/workspaces/{wid}/predict", json={"features": {"income": 1}}).status_code == 422
        assert client.post(f"/workspaces/{wid}/predict", json={"features": {"income": 1, "debt_ratio": .2, "late_payments": 0, "extra": 1}}).status_code == 422
        assert client.post(f"/workspaces/{wid}/predict", json={"features": {"income": "NaN", "debt_ratio": .2, "late_payments": 0}}).status_code == 422
        assert client.get("/workspaces/" + "0" * 32).status_code == 404


def test_stale_eligible_candidate_cannot_replace_new_champion(tmp_path, monkeypatch):
    registry = WorkspaceRegistry(tmp_path)
    wid = registry.register(make_workspace())

    def accepted(self, min_f1_gain=0.01):
        proposed = clone(self.champion)
        proposed.set_params(classifier__C=0.8)
        proposed.fit(self.reference[list(self.features)], self.reference[self.target_col])
        self.challenger = proposed
        self.decision = {"eligible": True, "checks": {"approved": True}, "reasons": []}
        self.comparison = {"historical_holdout": {}, "incoming_holdout": {}}
        return {"decision": self.decision, "evaluation": self.comparison}

    monkeypatch.setattr(Workspace, "train_challenger", accepted)
    first = registry.train_candidate(wid)["candidate_version"]
    second = registry.train_candidate(wid)["candidate_version"]
    registry.promote(wid, second)
    with pytest.raises(ValueError, match="Stale candidate"):
        registry.promote(wid, first)
    assert registry.champion(wid)[1] == second
