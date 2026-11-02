"""Local-only durable registry for independent binary sklearn workspaces.

All data is synthetic/non-sensitive. Filesystem and SQLite must be on a durable
*shared* volume for multi-process/cloud persistence; this module is local-first.
Uploads are .skops only, with zero user-supplied trusted types; never pickle loads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelops.model_workspace import (
    MAX_FILE_BYTES, MAX_ROWS, Workspace, load_uploaded_model, serialize_model,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "var" / "modelops_registry"
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_LOCK = threading.RLock()  # one-process lock; SQLite serializes writes across processes


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_id(workspace_id: str):
    if not ID_PATTERN.fullmatch(workspace_id):
        raise ValueError("Invalid workspace identifier")
    return workspace_id


def _model_bytes(model):
    binary = serialize_model(model)
    if len(binary) > MAX_FILE_BYTES:
        raise ValueError("Serialized model exceeds 5 MB")
    # Validate the exact bytes that will later be loaded from disk.
    verified = load_uploaded_model(binary)
    return binary, tuple(str(x) for x in verified.feature_names_in_)


def _csv_bytes(frame: pd.DataFrame | None):
    if frame is None:
        return None
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError("Dataset is outside accepted row limits")
    data = frame.to_csv(index=False).encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Dataset exceeds 5 MB")
    return data


class WorkspaceRegistry:
    """Persistent local metadata, sklearn models and per-workspace datasets."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("MODELOPS_REGISTRY_ROOT") or DEFAULT_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "registry.sqlite3"
        self._setup()

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.db, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def _setup(self):
        with self._connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    id_col TEXT NOT NULL,
                    target_col TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    categorical_json TEXT NOT NULL,
                    champion_version INTEGER NOT NULL,
                    rollback_version INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('champion','eligible','rejected','archived')
                    ),
                    sha256 TEXT NOT NULL,
                    decision_json TEXT,
                    evaluation_json TEXT,
                    base_version INTEGER,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, version)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def _directory(self, workspace_id: str) -> Path:
        return self.root / "workspaces" / _safe_id(workspace_id)

    def _version_path(self, workspace_id: str, version: int) -> Path:
        return self._directory(workspace_id) / "models" / f"v{int(version)}.skops"

    def _check(self, workspace_id: str):
        _safe_id(workspace_id)
        with self._connection() as con:
            row = con.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
        if row is None:
            raise KeyError("Workspace not found")
        return dict(row)

    def _log(self, con, workspace_id, type_, payload):
        con.execute(
            "INSERT INTO events(workspace_id,type,payload_json,created_at) VALUES(?,?,?,?)",
            (workspace_id, type_, _encode(payload), _now()),
        )

    def register(self, ws: Workspace) -> str:
        """Register validated sklearn workspace and copy model/data into local store."""
        if not ws.name.strip() or len(ws.name) > 60:
            raise ValueError("Workspace name must be 1–60 characters")
        ws.validate()
        model_blob, names = _model_bytes(ws.champion)
        if names != ws.features:
            raise ValueError("Model feature contract changed on serialization")
        data = {
            "reference.csv": _csv_bytes(ws.reference),
            "incoming.csv": _csv_bytes(ws.incoming),
            "labels.csv": _csv_bytes(ws.labels),
            "validation.csv": _csv_bytes(ws.validation),
        }
        workspace_id = uuid.uuid4().hex
        directory = self._directory(workspace_id)
        with _LOCK:
            directory.mkdir(parents=True)
            (directory / "models").mkdir()
            try:
                (directory / "models" / "v1.skops").write_bytes(model_blob)
                for filename, payload in data.items():
                    if payload is not None:
                        (directory / filename).write_bytes(payload)
                with self._connection() as con:
                    con.execute("BEGIN IMMEDIATE")
                    con.execute("""
                        INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        workspace_id, ws.name.strip(), ws.id_col, ws.target_col,
                        _encode(list(ws.features)), _encode(list(ws.categorical)),
                        1, None, _now(),
                    ))
                    con.execute("""
                        INSERT INTO versions VALUES(?,?,?,?,?,?,?,?)
                    """, (workspace_id, 1, "champion", _sha(model_blob), None, None, None, _now()))
                    self._log(con, workspace_id, "registered", {"version": 1})
            except BaseException:
                import shutil
                shutil.rmtree(directory, ignore_errors=True)
                raise
        return workspace_id

    def list_workspaces(self) -> list[dict]:
        with self._connection() as con:
            rows = con.execute("SELECT * FROM workspaces ORDER BY created_at, name").fetchall()
        return [self._public(row) for row in rows]

    def _public(self, row):
        return {
            "id": row["id"], "name": row["name"],
            "features": json.loads(row["features_json"]),
            "id_col": row["id_col"], "target_col": row["target_col"],
            "categorical": json.loads(row["categorical_json"]),
            "champion_version": row["champion_version"],
            "rollback_version": row["rollback_version"],
            "created_at": row["created_at"],
        }

    def get_workspace_info(self, workspace_id: str) -> dict:
        return self._public(self._check(workspace_id))

    def versions(self, workspace_id: str) -> list[dict]:
        self._check(workspace_id)
        with self._connection() as con:
            rows = con.execute("""
                SELECT * FROM versions WHERE workspace_id=? ORDER BY version DESC
            """, (workspace_id,)).fetchall()
        result = []
        for row in rows:
            result.append({
                "version": row["version"], "status": row["status"],
                "base_version": row["base_version"],
                "sha256": row["sha256"], "created_at": row["created_at"],
                "decision": json.loads(row["decision_json"]) if row["decision_json"] else None,
                "evaluation": json.loads(row["evaluation_json"]) if row["evaluation_json"] else None,
            })
        return result

    def events(self, workspace_id: str, limit=30) -> list[dict]:
        self._check(workspace_id)
        limit = max(1, min(int(limit), 100))
        with self._connection() as con:
            rows = con.execute("""
                SELECT * FROM events WHERE workspace_id=? ORDER BY id DESC LIMIT ?
            """, (workspace_id, limit)).fetchall()
        return [
            {"type": row["type"], "at": row["created_at"],
             "payload": json.loads(row["payload_json"])}
            for row in rows
        ]

    def _read_version(self, workspace_id: str, version: int):
        with self._connection() as con:
            row = con.execute("""
                SELECT sha256 FROM versions WHERE workspace_id=? AND version=?
            """, (workspace_id, int(version))).fetchone()
        if row is None:
            raise KeyError("Model version not found")
        path = self._version_path(workspace_id, version)
        payload = path.read_bytes()
        if _sha(payload) != row["sha256"]:
            raise ValueError("Model fingerprint mismatch: file may have been modified")
        return load_uploaded_model(payload), row["sha256"]

    def champion(self, workspace_id: str):
        row = self._check(workspace_id)
        model, fingerprint = self._read_version(workspace_id, row["champion_version"])
        return model, row["champion_version"], fingerprint

    def load(self, workspace_id: str) -> Workspace:
        row = self._check(workspace_id)
        directory = self._directory(workspace_id)
        model, _, sha = self.champion(workspace_id)
        def read(filename):
            path = directory / filename
            return pd.read_csv(path) if path.is_file() else None
        ws = Workspace(
            name=row["name"], champion=model,
            reference=read("reference.csv"), incoming=read("incoming.csv"),
            labels=read("labels.csv"), validation=read("validation.csv"),
            id_col=row["id_col"], target_col=row["target_col"],
            categorical=tuple(json.loads(row["categorical_json"])),
            model_sha256=sha,
        )
        if list(ws.features) != json.loads(row["features_json"]):
            raise ValueError("Persisted feature contract does not match model")
        return ws

    def monitor(self, workspace_id: str, threshold: float = 0.2):
        with _LOCK:
            ws = self.load(workspace_id)
            result = ws.monitor(threshold)
            with self._connection() as con:
                con.execute("BEGIN IMMEDIATE")
                self._log(con, workspace_id, "monitor", result)
            return result

    def train_candidate(self, workspace_id: str, min_f1_gain=0.01):
        """Save candidate even when rejected; leave champion unchanged."""
        with _LOCK:
            ws = self.load(workspace_id)
            base_version = self._check(workspace_id)["champion_version"]
            report = ws.train_challenger(min_f1_gain=min_f1_gain)
            payload, names = _model_bytes(ws.challenger)
            if names != ws.features:
                raise ValueError("Challenger feature contract changed")
            with self._connection() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("""
                    SELECT COALESCE(MAX(version),0)+1 AS next_version
                    FROM versions WHERE workspace_id=?
                """, (workspace_id,)).fetchone()
                version = int(row["next_version"])
                path = self._version_path(workspace_id, version)
                # Atomically write the model before committing its version row.
                temp = path.with_suffix(".tmp")
                try:
                    temp.write_bytes(payload)
                    os.replace(temp, path)
                    status = "eligible" if report["decision"]["eligible"] else "rejected"
                    con.execute("""
                        INSERT INTO versions VALUES(?,?,?,?,?,?,?,?)
                    """, (
                        workspace_id, version, status, _sha(payload),
                        _encode(report["decision"]), _encode(report["evaluation"]), base_version, _now(),
                    ))
                    self._log(con, workspace_id, "candidate_evaluated", {
                        "version": version, "status": status, "decision": report["decision"]
                    })
                finally:
                    temp.unlink(missing_ok=True)
        return {"candidate_version": version, "champion_unchanged": True, **report}

    def promote(self, workspace_id: str, candidate_version: int):
        """Approve only a previously evaluated eligible candidate."""
        self._check(workspace_id)
        with _LOCK:
            with self._connection() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("SELECT champion_version FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
                old_version = row["champion_version"]
                candidate = con.execute("""
                    SELECT status, decision_json, base_version FROM versions
                    WHERE workspace_id=? AND version=?
                """, (workspace_id, candidate_version)).fetchone()
                if candidate is None or candidate["status"] != "eligible":
                    raise ValueError("Only an eligible evaluated candidate can be promoted")
                if not json.loads(candidate["decision_json"])["eligible"]:
                    raise ValueError("Candidate does not meet promotion checks")
                if candidate["base_version"] != old_version:
                    raise ValueError("Stale candidate: evaluated against another champion version")
                model, _ = self._read_version(workspace_id, candidate_version)
                # Ensures file integrity and loadability before changing active pointer.
                if not callable(getattr(model, "predict_proba", None)):
                    raise ValueError("Invalid champion model")
                con.execute("""
                    UPDATE versions SET status='archived'
                    WHERE workspace_id=? AND version=?
                """, (workspace_id, old_version))
                con.execute("""
                    UPDATE versions SET status='champion'
                    WHERE workspace_id=? AND version=?
                """, (workspace_id, candidate_version))
                con.execute("""
                    UPDATE workspaces SET champion_version=?, rollback_version=? WHERE id=?
                """, (candidate_version, old_version, workspace_id))
                self._log(con, workspace_id, "promoted", {
                    "previous": old_version, "champion": candidate_version,
                })
        return {"previous": old_version, "champion": candidate_version}

    def rollback(self, workspace_id: str):
        self._check(workspace_id)
        with _LOCK:
            with self._connection() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("""
                    SELECT champion_version,rollback_version FROM workspaces WHERE id=?
                """, (workspace_id,)).fetchone()
                previous = row["rollback_version"]
                if previous is None:
                    raise ValueError("No rollback version available")
                self._read_version(workspace_id, previous)
                con.execute("""
                    UPDATE versions SET status='archived'
                    WHERE workspace_id=? AND version=?
                """, (workspace_id, row["champion_version"]))
                con.execute("""
                    UPDATE versions SET status='champion'
                    WHERE workspace_id=? AND version=?
                """, (workspace_id, previous))
                con.execute("""
                    UPDATE workspaces SET champion_version=?,rollback_version=NULL
                    WHERE id=?
                """, (previous, workspace_id))
                self._log(con, workspace_id, "rolled_back", {
                    "from": row["champion_version"], "champion": previous,
                })
        return {"champion": previous, "previous": row["champion_version"]}
