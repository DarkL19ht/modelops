"""PostgreSQL-backed account and session management for model workspaces.

Accounts are pseudonymous emails (not verified). Credentials are never logged or
stored in cleartext; bearer tokens are stored as SHA-256 digests. This is an MVP,
not a replacement for managed authentication and edge rate limits.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

EMAIL = re.compile(r"^[^\s@]{1,64}@[^\s@]{1,190}$")
SESSION_DAYS = 7


def _utcnow():
    return datetime.now(timezone.utc)


def _email(raw: str) -> str:
    normalized = raw.strip().lower()
    if not EMAIL.fullmatch(normalized) or len(normalized) > 254:
        raise ValueError("Enter a valid email address")
    return normalized


def _password(password: str):
    if not 12 <= len(password) <= 128:
        raise ValueError("Password must contain 12–128 characters")


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("MODELOPS_DATABASE_URL", "")
        if not self.database_url:
            raise RuntimeError("MODELOPS_DATABASE_URL is not configured")

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=15, row_factory=dict_row)

    def initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_salt BYTEA NOT NULL,
                    password_hash BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES ss_users(id),
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_ss_sessions_user ON ss_sessions(user_id)")

    def register(self, email: str, password: str) -> dict:
        address = _email(email)
        _password(password)
        salt = os.urandom(16)
        digest = _hash_password(password, salt)
        user_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ss_users(id,email,password_salt,password_hash) VALUES(%s,%s,%s,%s)",
                (user_id, address, salt, digest),
            )
        return {"id": user_id, "email": address}

    def login(self, email: str, password: str) -> dict:
        address = _email(email)
        # Defensive upper bound for potentially expensive KDF input.
        if len(password) > 128:
            raise ValueError("Invalid login credentials")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ss_users WHERE email=%s", (address,)).fetchone()
            if row is None:
                raise ValueError("Invalid login credentials")
            digest = _hash_password(password, bytes(row["password_salt"]))
            if not hmac.compare_digest(digest, bytes(row["password_hash"])):
                raise ValueError("Invalid login credentials")
            token = secrets.token_urlsafe(32)
            expires = _utcnow() + timedelta(days=SESSION_DAYS)
            conn.execute("DELETE FROM ss_sessions WHERE user_id=%s AND expires_at < NOW()", (row["id"],))
            conn.execute(
                "INSERT INTO ss_sessions(token_hash,user_id,expires_at) VALUES(%s,%s,%s)",
                (_token_hash(token), row["id"], expires),
            )
        return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat(),
                "user": {"id": row["id"], "email": row["email"]}}

    def identify(self, token: str) -> dict:
        if not token or len(token) > 256:
            raise PermissionError("Authentication required")
        with self._connect() as conn:
            row = conn.execute("""
                SELECT u.id, u.email FROM ss_sessions s
                JOIN ss_users u ON u.id=s.user_id
                WHERE s.token_hash=%s AND s.expires_at>NOW()
            """, (_token_hash(token),)).fetchone()
        if row is None:
            raise PermissionError("Session expired or invalid")
        return {"id": row["id"], "email": row["email"]}

    def logout(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ss_sessions WHERE token_hash=%s", (_token_hash(token),))
