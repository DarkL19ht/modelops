"""Durable, owner-scoped challenger training jobs stored in PostgreSQL.

Jobs are at-most-once dispatched: a lost worker marks its job as interrupted
rather than automatically retrying, avoiding duplicate model-version commits.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime


class TrainingJobStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("MODELOPS_DATABASE_URL", "")
        if not self.database_url:
            raise RuntimeError("MODELOPS_DATABASE_URL is not configured")

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.database_url, connect_timeout=15, row_factory=dict_row)

    def initialize(self):
        """Idempotent schema creation; assumes ss_workspaces exists."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_training_jobs (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES ss_workspaces(id),
                    owner_id TEXT NOT NULL REFERENCES ss_users(id),
                    status TEXT NOT NULL CHECK (status IN
                        ('queued','running','succeeded','failed','interrupted')),
                    min_f1_gain DOUBLE PRECISION NOT NULL,
                    worker_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    lease_expires_at TIMESTAMPTZ,
                    result_json TEXT,
                    error_message TEXT
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_training_active_workspace
                ON ss_training_jobs(workspace_id)
                WHERE status IN ('queued','running')
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS ix_training_jobs_owner
                ON ss_training_jobs(owner_id,created_at DESC)
            """)

    @staticmethod
    def _view(row: dict) -> dict:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "status": row["status"],
            "min_f1_gain": row["min_f1_gain"],
            "created_at": row["created_at"].isoformat(),
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error_message"],
        }

    @staticmethod
    def _stale(conn):
        # An interrupted job is NOT automatically retried because training
        # might have saved a candidate just before a worker was terminated.
        conn.execute("""
            UPDATE ss_training_jobs
            SET status='interrupted', finished_at=NOW(),
                error_message='Worker stopped before confirming completion',
                worker_id=NULL, lease_expires_at=NULL
            WHERE status='running' AND lease_expires_at < NOW()
        """)

    def enqueue(self, owner_id: str, workspace_id: str, min_f1_gain: float) -> dict:
        if not (0 <= min_f1_gain <= 0.5):
            raise ValueError("min_f1_gain must be between 0 and 0.5")
        if not isinstance(workspace_id, str) or len(workspace_id) != 32:
            raise ValueError("Invalid workspace identifier")
        with self._connect() as conn:
            # Lock the workspace row so concurrent requests cannot both queue.
            workspace = conn.execute("""
                SELECT id FROM ss_workspaces WHERE id=%s AND owner_id=%s FOR UPDATE
            """, (workspace_id, owner_id)).fetchone()
            if workspace is None:
                raise KeyError("Workspace not found")
            self._stale(conn)
            existing = conn.execute("""
                SELECT * FROM ss_training_jobs
                WHERE workspace_id=%s AND status IN ('queued','running')
                ORDER BY created_at ASC LIMIT 1
            """, (workspace_id,)).fetchone()
            if existing:
                return self._view(existing)
            active_count = conn.execute("""
                SELECT COUNT(*) AS n FROM ss_training_jobs
                WHERE owner_id=%s AND status IN ('queued','running')
            """, (owner_id,)).fetchone()["n"]
            if active_count >= 2:
                raise ValueError("Maximum two active training jobs per account")
            job_id = uuid.uuid4().hex
            row = conn.execute("""
                INSERT INTO ss_training_jobs(id,workspace_id,owner_id,status,min_f1_gain)
                VALUES(%s,%s,%s,'queued',%s) RETURNING *
            """, (job_id, workspace_id, owner_id, min_f1_gain)).fetchone()
            return self._view(row)

    def get_owned(self, owner_id: str, job_id: str) -> dict:
        with self._connect() as conn:
            self._stale(conn)
            row = conn.execute("""
                SELECT * FROM ss_training_jobs WHERE id=%s AND owner_id=%s
            """, (job_id, owner_id)).fetchone()
        if row is None:
            raise KeyError("Training job not found")
        return self._view(row)

    def list_owned(self, owner_id: str, workspace_id: str) -> list[dict]:
        with self._connect() as conn:
            # Explicit owner check even if a workspace has no jobs.
            if not conn.execute("""
                SELECT 1 FROM ss_workspaces WHERE id=%s AND owner_id=%s
            """, (workspace_id, owner_id)).fetchone():
                raise KeyError("Workspace not found")
            self._stale(conn)
            rows = conn.execute("""
                SELECT * FROM ss_training_jobs
                WHERE workspace_id=%s AND owner_id=%s
                ORDER BY created_at DESC LIMIT 20
            """, (workspace_id, owner_id)).fetchall()
        return [self._view(r) for r in rows]

    def claim(self, worker_id: str) -> dict | None:
        """Claim a job atomically with SKIP LOCKED; safe across worker replicas."""
        with self._connect() as conn:
            self._stale(conn)
            row = conn.execute("""
                WITH next_job AS (
                    SELECT id FROM ss_training_jobs
                    WHERE status='queued'
                    ORDER BY created_at ASC, id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ss_training_jobs AS job
                SET status='running', worker_id=%s,
                    started_at=NOW(), lease_expires_at=NOW() + INTERVAL '5 minutes'
                FROM next_job WHERE job.id=next_job.id
                RETURNING job.*
            """, (worker_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "owner_id": row["owner_id"],
            "workspace_id": row["workspace_id"],
            "min_f1_gain": row["min_f1_gain"],
        }

    def complete(self, job_id: str, worker_id: str, result: dict):
        data = json.dumps(result, allow_nan=False)
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE ss_training_jobs
                SET status='succeeded', finished_at=NOW(),
                    result_json=%s, error_message=NULL, lease_expires_at=NULL
                WHERE id=%s AND worker_id=%s AND status='running'
                RETURNING id
            """, (data, job_id, worker_id)).fetchone()
            if row is None:
                raise RuntimeError("Job completion rejected: claim is no longer active")

    def fail(self, job_id: str, worker_id: str, public_message: str):
        # Do not expose worker exception details (may contain credentials).
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE ss_training_jobs
                SET status='failed', finished_at=NOW(),
                    error_message=%s, lease_expires_at=NULL
                WHERE id=%s AND worker_id=%s AND status='running'
                RETURNING id
            """, (public_message[:200], job_id, worker_id)).fetchone()
            if row is None:
                raise RuntimeError("Job failure update rejected")
