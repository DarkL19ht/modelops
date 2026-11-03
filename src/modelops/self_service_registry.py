"""Isolated persistence for authenticated user model workspaces.

All operations are exposed through owner-scoped methods in SelfServiceRegistry.
Models must be standard scikit-learn .skops with no untrusted types.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row

from src.modelops.model_workspace import (
    MAX_FILE_BYTES, MAX_ROWS, Workspace, load_uploaded_model, serialize_model,
)

ALLOWED_FILES = ("reference.csv", "incoming.csv", "labels.csv", "validation.csv")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _check_id(workspace_id: str):
    try:
        value = uuid.UUID(hex=workspace_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid workspace identifier") from exc
    if value.hex != workspace_id:
        raise ValueError("Invalid workspace identifier")


def _model_blob(model) -> tuple[bytes, str]:
    payload = serialize_model(model)
    if not 1 <= len(payload) <= MAX_FILE_BYTES:
        raise ValueError("Serialized model must not exceed 5 MB")
    model_again = load_uploaded_model(payload)
    if tuple(str(x) for x in model_again.feature_names_in_) != tuple(str(x) for x in model.feature_names_in_):
        raise ValueError("Model features changed after serialization")
    return payload, _sha(payload)


def _frame_blob(frame):
    if frame is None:
        return None
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError("Dataset exceeds row limit")
    payload = frame.to_csv(index=False).encode("utf-8")
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("Dataset exceeds 5 MB")
    return payload


class SelfServiceRegistry:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("MODELOPS_DATABASE_URL", "")
        if not self.database_url:
            raise RuntimeError("MODELOPS_DATABASE_URL is not configured")

    def _connect(self):
        return psycopg.connect(
            self.database_url, connect_timeout=15, row_factory=dict_row,
        )

    def initialize(self):
        """Initialize tenant-scoped model and dataset tables after AuthStore.initialize()."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_id TEXT NOT NULL REFERENCES ss_users(id),
                    id_col TEXT NOT NULL,
                    target_col TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    categorical_json TEXT NOT NULL,
                    champion_version INTEGER NOT NULL,
                    rollback_version INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(owner_id,name)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_versions (
                    workspace_id TEXT NOT NULL REFERENCES ss_workspaces(id),
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('champion','eligible','rejected','archived')
                    ),
                    model BYTEA NOT NULL,
                    sha256 TEXT NOT NULL,
                    decision_json TEXT,
                    evaluation_json TEXT,
                    base_version INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (workspace_id,version)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_datasets (
                    workspace_id TEXT NOT NULL REFERENCES ss_workspaces(id),
                    filename TEXT NOT NULL,
                    content BYTEA NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, filename)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_events (
                    id BIGSERIAL PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES ss_workspaces(id),
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_ss_events_workspace ON ss_events(workspace_id,id DESC)")

    def ping(self) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1

    @staticmethod
    def _metadata(row):
        return {
            "id": row["id"], "name": row["name"],
            "features": json.loads(row["features_json"]),
            "categorical": json.loads(row["categorical_json"]),
            "id_col": row["id_col"], "target_col": row["target_col"],
            "champion_version": row["champion_version"],
            "rollback_version": row["rollback_version"],
            "created_at": row["created_at"].isoformat(),
        }

    @staticmethod
    def _event(conn, wid, kind, payload):
        conn.execute(
            "INSERT INTO ss_events(workspace_id,kind,payload_json) VALUES(%s,%s,%s)",
            (wid, kind, _json(payload)),
        )

    def register(self, workspace: Workspace, owner_id: str) -> str:
        if not 1 <= len(workspace.name.strip()) <= 60:
            raise ValueError("Workspace name must be 1–60 characters")
        workspace.validate()
        model_data, sha = _model_blob(workspace.champion)
        datasets = {
            name: _frame_blob(frame)
            for name, frame in zip(ALLOWED_FILES, (
                workspace.reference, workspace.incoming, workspace.labels,
                workspace.validation,
            ), strict=True)
        }
        wid = uuid.uuid4().hex
        with self._connect() as conn:
            # Avoid two concurrent uploads bypassing the three-workspace quota.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (owner_id,))
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM ss_workspaces WHERE owner_id=%s",
                (owner_id,),
            ).fetchone()["n"]
            if count >= 3:
                raise ValueError("Maximum three workspaces per account")
            conn.execute("""
                INSERT INTO ss_workspaces (
                    id,name,owner_id,id_col,target_col,features_json,categorical_json,champion_version
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,1)
            """, (wid, workspace.name.strip(), owner_id, workspace.id_col, workspace.target_col,
                  _json(workspace.features), _json(workspace.categorical)))
            conn.execute("""
                INSERT INTO ss_versions(workspace_id,version,status,model,sha256)
                VALUES(%s,1,'champion',%s,%s)
            """, (wid, model_data, sha))
            for filename, payload in datasets.items():
                if payload is not None:
                    conn.execute("""
                        INSERT INTO ss_datasets(workspace_id,filename,content,sha256)
                        VALUES(%s,%s,%s,%s)
                    """, (wid, filename, payload, _sha(payload)))
            self._event(conn, wid, "registered", {"version": 1, "model_sha256": sha})
        return wid

    def list_workspaces(self, *, owner_id: str) -> list[dict]:
        sql = "SELECT * FROM ss_workspaces WHERE owner_id=%s ORDER BY created_at DESC, name"
        with self._connect() as conn:
            return [self._metadata(row) for row in conn.execute(sql, (owner_id,)).fetchall()]

    def info(self, wid: str) -> dict:
        _check_id(wid)
        sql = "SELECT * FROM ss_workspaces WHERE id=%s"
        with self._connect() as conn:
            row = conn.execute(sql, (wid,)).fetchone()
        if row is None:
            raise KeyError("Workspace not found")
        return self._metadata(row)

    def _model_from_conn(self, conn, wid: str, version: int):
        row = conn.execute("""
            SELECT model,sha256 FROM ss_versions
            WHERE workspace_id=%s AND version=%s
        """, (wid, version)).fetchone()
        if row is None:
            raise KeyError("Model version not found")
        payload = bytes(row["model"])
        if _sha(payload) != row["sha256"]:
            raise ValueError("Stored model SHA-256 mismatch")
        return load_uploaded_model(payload), row["sha256"]

    def champion(self, wid: str):
        _check_id(wid)
        with self._connect() as conn:
            sql = "SELECT champion_version FROM ss_workspaces WHERE id=%s"
            current = conn.execute(sql, (wid,)).fetchone()
            if current is None:
                raise KeyError("Workspace not found")
            version = current["champion_version"]
            model, sha = self._model_from_conn(conn, wid, version)
        return model, version, sha

    def _load(self, wid: str) -> tuple[Workspace, int]:
        _check_id(wid)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ss_workspaces WHERE id=%s", (wid,)).fetchone()
            if row is None:
                raise KeyError("Workspace not found")
            model, sha = self._model_from_conn(conn, wid, row["champion_version"])
            data = {}
            for item in conn.execute("SELECT filename,content,sha256 FROM ss_datasets WHERE workspace_id=%s", (wid,)):
                payload = bytes(item["content"])
                if _sha(payload) != item["sha256"]:
                    raise ValueError(f"Stored dataset fingerprint mismatch: {item['filename']}")
                data[item["filename"]] = pd.read_csv(io.BytesIO(payload))
        workspace = Workspace(
            name=row["name"], champion=model,
            reference=data["reference.csv"], incoming=data["incoming.csv"],
            labels=data.get("labels.csv"), validation=data.get("validation.csv"),
            id_col=row["id_col"], target_col=row["target_col"],
            categorical=tuple(json.loads(row["categorical_json"])), model_sha256=sha,
        )
        if workspace.features != tuple(json.loads(row["features_json"])):
            raise ValueError("Stored feature contract mismatch")
        return workspace, row["champion_version"]

    def monitor(self, wid: str, *, threshold: float = 0.2) -> dict:
        ws, champion_version = self._load(wid)
        results = ws.monitor(threshold)
        with self._connect() as conn:
            self._event(conn, wid, "monitor", {
                "champion_version": champion_version, "results": results,
            })
        return {"champion_version": champion_version, **results}

    def train_candidate(self, wid: str, *, min_f1_gain: float = 0.01) -> dict:
        ws, base_version = self._load(wid)
        report = ws.train_challenger(min_f1_gain=min_f1_gain)
        payload, sha = _model_blob(ws.challenger)
        status = "eligible" if report["decision"]["eligible"] else "rejected"
        with self._connect() as conn:
            # Serialise competing promotions/training commits for this workspace.
            row = conn.execute(
                "SELECT champion_version FROM ss_workspaces WHERE id=%s FOR UPDATE", (wid,)
            ).fetchone()
            if row is None:
                raise KeyError("Workspace not found")
            previous_count = conn.execute(
                "SELECT COUNT(*) AS n FROM ss_versions WHERE workspace_id=%s", (wid,)
            ).fetchone()["n"]
            if previous_count >= 6:
                raise ValueError("Maximum six versions per workspace")
            if row["champion_version"] != base_version:
                raise ValueError("Champion changed while challenger trained; rerun evaluation")
            next_version = conn.execute("""
                SELECT COALESCE(MAX(version),0)+1 AS version
                FROM ss_versions WHERE workspace_id=%s
            """, (wid,)).fetchone()["version"]
            conn.execute("""
                INSERT INTO ss_versions(
                    workspace_id,version,status,model,sha256,decision_json,evaluation_json,base_version
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
            """, (wid, next_version, status, payload, sha,
                  _json(report["decision"]), _json(report["evaluation"]), base_version))
            self._event(conn, wid, "candidate_evaluated", {
                "candidate_version": next_version, "base_version": base_version,
                "status": status, "decision": report["decision"],
            })
        return {"candidate_version": next_version, "candidate_status": status,
                "champion_unchanged": True, **report}

    def promote(self, wid: str, version: int) -> dict:
        _check_id(wid)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT champion_version FROM ss_workspaces WHERE id=%s FOR UPDATE", (wid,)
            ).fetchone()
            if row is None:
                raise KeyError("Workspace not found")
            old = row["champion_version"]
            candidate = conn.execute("""
                SELECT status,decision_json,base_version FROM ss_versions
                WHERE workspace_id=%s AND version=%s
            """, (wid, version)).fetchone()
            if candidate is None or candidate["status"] != "eligible":
                raise ValueError("Only an eligible candidate may be promoted")
            if candidate["base_version"] != old:
                raise ValueError("Stale candidate; reevaluate against active champion")
            if not json.loads(candidate["decision_json"])["eligible"]:
                raise ValueError("Candidate failed promotion checks")
            self._model_from_conn(conn, wid, version)
            conn.execute(
                "UPDATE ss_versions SET status='archived' WHERE workspace_id=%s AND version=%s",
                (wid, old),
            )
            conn.execute(
                "UPDATE ss_versions SET status='champion' WHERE workspace_id=%s AND version=%s",
                (wid, version),
            )
            conn.execute("""
                UPDATE ss_workspaces SET champion_version=%s,rollback_version=%s WHERE id=%s
            """, (version, old, wid))
            self._event(conn, wid, "promoted", {"previous": old, "champion": version})
        return {"previous": old, "champion": version}

    def rollback(self, wid: str) -> dict:
        _check_id(wid)
        with self._connect() as conn:
            row = conn.execute("""
                SELECT champion_version,rollback_version FROM ss_workspaces
                WHERE id=%s FOR UPDATE
            """, (wid,)).fetchone()
            if row is None:
                raise KeyError("Workspace not found")
            previous = row["rollback_version"]
            if previous is None:
                raise ValueError("No rollback version is available")
            self._model_from_conn(conn, wid, previous)
            conn.execute(
                "UPDATE ss_versions SET status='archived' WHERE workspace_id=%s AND version=%s",
                (wid, row["champion_version"]),
            )
            conn.execute(
                "UPDATE ss_versions SET status='champion' WHERE workspace_id=%s AND version=%s",
                (wid, previous),
            )
            conn.execute("""
                UPDATE ss_workspaces SET champion_version=%s,rollback_version=NULL WHERE id=%s
            """, (previous, wid))
            self._event(conn, wid, "rolled_back", {"from": row["champion_version"], "champion": previous})
        return {"champion": previous, "previous": row["champion_version"]}

    def versions(self, wid: str) -> list[dict]:
        self.info(wid)
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT version,status,sha256,decision_json,evaluation_json,base_version,created_at
                FROM ss_versions WHERE workspace_id=%s ORDER BY version DESC
            """, (wid,)).fetchall()
        return [{
            "version": row["version"], "status": row["status"],
            "sha256": row["sha256"], "base_version": row["base_version"],
            "decision": json.loads(row["decision_json"]) if row["decision_json"] else None,
            "evaluation": json.loads(row["evaluation_json"]) if row["evaluation_json"] else None,
            "created_at": row["created_at"].isoformat(),
        } for row in rows]

    def events(self, wid: str, limit: int = 30) -> list[dict]:
        self.info(wid)
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT kind,payload_json,created_at FROM ss_events
                WHERE workspace_id=%s ORDER BY id DESC LIMIT %s
            """, (wid, min(100, max(1, int(limit))))).fetchall()
        return [{"kind": row["kind"], "at": row["created_at"].isoformat(),
                 "payload": json.loads(row["payload_json"])} for row in rows]

    def predict(self, wid: str, entries: list[dict[str, float]], *, version: int | None = None):
        info = self.info(wid)
        if version is None or version == info["champion_version"]:
            model, version, sha = self.champion(wid)
        else:
            with self._connect() as conn:
                v = conn.execute("SELECT status FROM ss_versions WHERE workspace_id=%s AND version=%s", (wid, version)).fetchone()
                if v is None or v["status"] not in ("eligible", "champion"):
                    raise ValueError("Only champion or eligible challenger may be used for inference")
                model, sha = self._model_from_conn(conn, wid, version)
        expected = info["features"]
        if not 1 <= len(entries) <= 100:
            raise ValueError("Batch must contain 1–100 records")
        matrix = []
        for item in entries:
            if set(item) != set(expected):
                raise ValueError(f"Exactly these features are required: {expected}")
            matrix.append([float(item[name]) for name in expected])
        array = np.asarray(matrix, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("All feature values must be finite")
        frame = pd.DataFrame(array, columns=expected)
        labels = np.asarray(model.predict(frame))
        probability = np.asarray(model.predict_proba(frame), dtype=float)
        if (labels.shape != (len(frame),)
            or probability.shape != (len(frame), 2)
            or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
            raise ValueError("Registered classifier returned invalid output")
        return {"workspace_id": wid, "model_version": version, "model_sha256": sha,
                "predictions": [{"predicted_class": int(labels[i]),
                                 "positive_class_probability": round(float(probability[i, 1]), 6)}
                                for i in range(len(frame))]}

    # All routes in self_service_api.py MUST use these owner-checked methods.
    # Underlying methods are internal helpers and must never be exposed directly.

    def require_owner(self, user_id: str, wid: str) -> None:
        _check_id(wid)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ss_workspaces WHERE id=%s AND owner_id=%s",
                (wid, user_id),
            ).fetchone()
        if row is None:
            raise KeyError("Workspace not found")

    def mine(self, user_id: str):
        return self.list_workspaces(owner_id=user_id)

    def owned_info(self, user_id: str, wid: str):
        self.require_owner(user_id, wid)
        return self.info(wid)

    def owned_versions(self, user_id: str, wid: str):
        self.require_owner(user_id, wid)
        return self.versions(wid)

    def owned_events(self, user_id: str, wid: str):
        self.require_owner(user_id, wid)
        return self.events(wid)

    def owned_monitor(self, user_id: str, wid: str, *, threshold: float = 0.20):
        self.require_owner(user_id, wid)
        return self.monitor(wid, threshold=threshold)

    def owned_train(self, user_id: str, wid: str, *, min_f1_gain: float = 0.01):
        self.require_owner(user_id, wid)
        return self.train_candidate(wid, min_f1_gain=min_f1_gain)

    def owned_promote(self, user_id: str, wid: str, version: int):
        self.require_owner(user_id, wid)
        return self.promote(wid, version)

    def owned_rollback(self, user_id: str, wid: str):
        self.require_owner(user_id, wid)
        return self.rollback(wid)

    def owned_predict(self, user_id: str, wid: str, records, *, version=None):
        self.require_owner(user_id, wid)
        return self.predict(wid, records, version=version)

    def owned_artifact(self, user_id: str, wid: str, version: int):
        self.require_owner(user_id, wid)
        with self._connect() as conn:
            row = conn.execute("""
                SELECT model, sha256, status FROM ss_versions
                WHERE workspace_id=%s AND version=%s
            """, (wid, version)).fetchone()
        if row is None:
            raise KeyError("Model version not found")
        if row["status"] not in ("eligible", "champion"):
            raise ValueError("Only an eligible challenger or champion may be downloaded")
        data = bytes(row["model"])
        if _sha(data) != row["sha256"]:
            raise ValueError("Model artifact integrity check failed")
        return data
