"""Database-backed per-identity limits for self-service API operations.

Limits are a resource-management control, not a replacement for edge WAF,
container resource isolation, or abuse prevention at the hosting provider.
"""
from __future__ import annotations

import hashlib
import os
import time


class RateLimitExceeded(Exception):
    """The operation quota for the current window is exhausted."""


class RequestLimits:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def initialize(self) -> None:
        import psycopg
        with psycopg.connect(self.database_url, connect_timeout=15) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_request_limits (
                    identity_hash TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    window_start BIGINT NOT NULL,
                    hits INTEGER NOT NULL,
                    PRIMARY KEY(identity_hash, operation, window_start)
                )
            """)

    def hit(self, operation: str, identity: str, *, limit: int, seconds: int) -> None:
        import psycopg
        if not operation or not identity or not 1 <= limit <= 100000 or seconds < 60:
            raise ValueError("Invalid rate limit specification")
        # Do not persist raw email addresses, IPs, or bearer tokens in this table.
        key = hashlib.sha256(f"{operation}\x00{identity}".encode()).hexdigest()
        window_start = int(time.time()) // seconds * seconds
        with psycopg.connect(self.database_url, connect_timeout=15) as conn:
            row = conn.execute("""
                INSERT INTO ss_request_limits(identity_hash,operation,window_start,hits)
                VALUES(%s,%s,%s,1)
                ON CONFLICT(identity_hash,operation,window_start)
                DO UPDATE SET hits=ss_request_limits.hits+1
                WHERE ss_request_limits.hits < %s
                RETURNING hits
            """, (key, operation, window_start, limit)).fetchone()
            if row is None:
                raise RateLimitExceeded("Request limit reached. Please try again later.")
