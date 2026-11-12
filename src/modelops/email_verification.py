"""Verified-email accounts for ModelOps self-service workspaces.

Builds on the existing PostgreSQL AuthStore and preserves its session format.
Existing users are grandfathered only on the *first* schema migration.
Do not enable public registrations without properly configured email delivery.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import requests

from src.modelops.account_limits import RequestLimits
from src.modelops.self_service_auth import AuthStore, _email, _token_hash, _password

_CODE_LENGTH = 8
_LIFETIME_MINUTES = 10
_GENERIC_MESSAGE = "If this address can be verified, a verification code has been sent."


def verification_enabled() -> bool:
    return os.getenv("MODELOPS_EMAIL_VERIFICATION_REQUIRED", "0") == "1"


def _secret() -> str:
    secret = os.getenv("MODELOPS_EMAIL_TOKEN_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("MODELOPS_EMAIL_TOKEN_SECRET must be at least 32 characters")
    return secret


def _code_digest(user_id: str, code: str) -> str:
    return hmac.new(_secret().encode(), f"{user_id}:{code}".encode(), hashlib.sha256).hexdigest()



def _send_email(email: str, code: str) -> None:
    """Send an account verification code using Mailjet."""

    api_key = os.getenv("MODELOPS_MAILJET_API_KEY", "")
    secret_key = os.getenv("MODELOPS_MAILJET_SECRET_KEY", "")
    sender = os.getenv("MODELOPS_EMAIL_FROM", "")

    if not all((api_key, secret_key, sender)):
        raise RuntimeError(
            "Mailjet email delivery is not configured"
        )

    response = requests.post(
        "https://api.mailjet.com/v3.1/send",
        auth=(api_key, secret_key),
        json={
            "Messages": [
                {
                    "From": {
                        "Email": sender,
                        "Name": "ModelOps",
                    },
                    "To": [
                        {
                            "Email": email,
                        }
                    ],
                    "Subject": "Verify your ModelOps account",
                    "TextPart": (
                        f"Your ModelOps verification code is {code}.\n\n"
                        "This code expires in 10 minutes.\n\n"
                        "If you did not create a ModelOps account, "
                        "you can ignore this email."
                    ),
                }
            ]
        },
        timeout=15,
    )

    response.raise_for_status()

    result = response.json()

    messages = result.get("Messages", [])

    if (
        not messages
        or messages[0].get("Status") != "success"
    ):
        raise RuntimeError(
            "Mailjet did not accept the verification email"
        )



class VerifiedAuthStore(AuthStore):
    """Drop-in AuthStore with optional verified-email registration."""

    def __init__(self, database_url: str | None = None):
        super().__init__(database_url)
        self.limits = RequestLimits(self.database_url)

    def initialize(self) -> None:
        super().initialize()
        with self._connect() as conn:
            existing_column = conn.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema=current_schema()
                      AND table_name='ss_users' AND column_name='email_verified_at'
                ) AS present
            """).fetchone()["present"]
            conn.execute("ALTER TABLE ss_users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ")
            # Only grandfather accounts that existed before this column was introduced.
            if not existing_column:
                conn.execute("UPDATE ss_users SET email_verified_at=NOW()")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ss_email_codes (
                    user_id TEXT PRIMARY KEY REFERENCES ss_users(id),
                    code_hash TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_sent_at TIMESTAMPTZ NOT NULL,
                    sent_count INTEGER NOT NULL DEFAULT 1,
                    first_sent_at TIMESTAMPTZ NOT NULL
                )
            """)
        self.limits.initialize()
        if verification_enabled():
            _secret()
            if not all(
                os.getenv(name)
                for name in (
                    "MODELOPS_MAILJET_API_KEY",
                    "MODELOPS_MAILJET_SECRET_KEY",
                    "MODELOPS_EMAIL_FROM",
                )
            ):
                raise RuntimeError(
                    "Configure Mailjet before enabling email-verified registration"
                )

    def enforce_limit(self, action: str, identity: str, *, limit: int, seconds: int):
        self.limits.hit(action, identity, limit=limit, seconds=seconds)

    def _send_code(self, address: str) -> None:
        # Rates also enforced inside a locked row to prevent email floods.
        with self._connect() as conn:
            row = conn.execute("SELECT id,email_verified_at FROM ss_users WHERE email=%s", (address,)).fetchone()
            if row is None or row["email_verified_at"] is not None:
                return
            uid = row["id"]
            previous = conn.execute("SELECT * FROM ss_email_codes WHERE user_id=%s FOR UPDATE", (uid,)).fetchone()
            now = datetime.now(timezone.utc)
            if previous:
                elapsed = (now - previous["last_sent_at"]).total_seconds()
                if elapsed < 60:
                    return
                hour_passed = (now - previous["first_sent_at"]).total_seconds() >= 3600
                if not hour_passed and previous["sent_count"] >= 5:
                    return
                count = 1 if hour_passed else previous["sent_count"] + 1
                first = now if hour_passed else previous["first_sent_at"]
            else:
                count, first = 1, now
            code = f"{secrets.randbelow(10 ** _CODE_LENGTH):0{_CODE_LENGTH}d}"
            conn.execute("""
                INSERT INTO ss_email_codes
                  (user_id,code_hash,expires_at,attempts,last_sent_at,sent_count,first_sent_at)
                VALUES(%s,%s,%s,0,%s,%s,%s)
                ON CONFLICT(user_id) DO UPDATE SET
                  code_hash=EXCLUDED.code_hash,
                  expires_at=EXCLUDED.expires_at,
                  attempts=0,
                  last_sent_at=EXCLUDED.last_sent_at,
                  sent_count=EXCLUDED.sent_count,
                  first_sent_at=EXCLUDED.first_sent_at
            """, (uid, _code_digest(uid, code), now + timedelta(minutes=_LIFETIME_MINUTES), now, count, first))
        # Code is committed before delivery; a failed send can be retried after
        # the resend cooldown. Do not expose the code in responses or logs.
        _send_email(address, code)

    def register(self, email: str, password: str) -> dict:
        address = _email(email)
        _password(password)
        if not verification_enabled():
            user = super().register(address, password)
            with self._connect() as conn:
                conn.execute("UPDATE ss_users SET email_verified_at=NOW() WHERE id=%s", (user["id"],))
            return user
        # Use a generic response even for existing emails, reducing enumeration.
        try:
            super().register(address, password)
        except Exception as exc:
            if getattr(exc, "sqlstate", None) != "23505":
                raise
            return {"message": _GENERIC_MESSAGE}
        self._send_code(address)
        return {"message": _GENERIC_MESSAGE}

    def resend_verification(self, email: str) -> dict:
        address = _email(email)
        if verification_enabled():
            self._send_code(address)
        return {"message": _GENERIC_MESSAGE}

    def verify_email(self, email: str, code: str) -> dict:
        address = _email(email)
        if not re.fullmatch(r"[0-9]{8}", code):
            raise ValueError("Invalid or expired verification code")
        with self._connect() as conn:
            row = conn.execute("SELECT id,email_verified_at FROM ss_users WHERE email=%s", (address,)).fetchone()
            if row is None:
                raise ValueError("Invalid or expired verification code")
            if row["email_verified_at"] is not None:
                raise ValueError("Invalid or expired verification code")
            challenge = conn.execute("SELECT * FROM ss_email_codes WHERE user_id=%s FOR UPDATE", (row["id"],)).fetchone()
            if (
                challenge is None
                or challenge["attempts"] >= 5
                or challenge["expires_at"] < datetime.now(timezone.utc)
            ):
                raise ValueError("Invalid or expired verification code")
            conn.execute("UPDATE ss_email_codes SET attempts=attempts+1 WHERE user_id=%s", (row["id"],))
            matched = hmac.compare_digest(challenge["code_hash"], _code_digest(row["id"], code))
            if matched:
                conn.execute("UPDATE ss_users SET email_verified_at=NOW() WHERE id=%s", (row["id"],))
                conn.execute("DELETE FROM ss_email_codes WHERE user_id=%s", (row["id"],))
        # Commit failed attempts before raising, so guessing cannot bypass the cap.
        if not matched:
            raise ValueError("Invalid or expired verification code")
        return {"verified": True}

    def login(self, email: str, password: str) -> dict:
        session = super().login(email, password)
        if verification_enabled():
            with self._connect() as conn:
                row = conn.execute("SELECT email_verified_at FROM ss_users WHERE id=%s", (session["user"]["id"],)).fetchone()
                blocked = row is None or row["email_verified_at"] is None
                if blocked:
                    conn.execute("DELETE FROM ss_sessions WHERE token_hash=%s", (_token_hash(session["access_token"]),))
            # Commit session revocation before returning the sign-in error.
            if blocked:
                raise ValueError("Verify your email before signing in")
        return session

    def identify(self, token: str) -> dict:
        user = super().identify(token)
        if verification_enabled():
            with self._connect() as conn:
                row = conn.execute("SELECT email_verified_at FROM ss_users WHERE id=%s", (user["id"],)).fetchone()
            if row is None or row["email_verified_at"] is None:
                raise PermissionError("Email verification required")
        return user
