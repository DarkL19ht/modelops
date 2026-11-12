"""Contract checks for verification and limits without a live email/database service."""
from fastapi.testclient import TestClient

from src.modelops.account_limits import RateLimitExceeded
from src.modelops.self_service_api import create_app


class RegistrationAuth:
    _test_stub = True

    def __init__(self):
        self.codes = {}
        self.verified = set()
        self.limits = {}

    def register(self, email, password):
        self.codes[email] = "12345678"
        return {"message": "Check your email"}

    def verify_email(self, email, code):
        if self.codes.get(email) != code:
            raise ValueError("Invalid or expired verification code")
        self.verified.add(email)
        return {"verified": True}

    def resend_verification(self, email):
        return {"message": "If an unverified account exists, an email was sent"}

    def login(self, email, password):
        if email not in self.verified:
            raise ValueError("Verify your email before signing in")
        return {"access_token": "fake-token", "user": {"email": email, "id": "u1"}}

    def enforce_limit(self, action, identity, *, limit, seconds):
        key = (action, identity)
        self.limits[key] = self.limits.get(key, 0) + 1
        if self.limits[key] > limit:
            raise RateLimitExceeded("Exceeded")


class RegistrationRegistry:
    _test_stub = True

    def ping(self):
        return True


def test_registration_verification_login_without_invitation(monkeypatch):
    monkeypatch.setenv("MODELOPS_EMAIL_VERIFICATION_REQUIRED", "1")
    monkeypatch.setenv("MODELOPS_REQUIRE_INVITE", "1")
    with TestClient(create_app(auth=RegistrationAuth(), registry=RegistrationRegistry())) as client:
        credentials = {"email": "person@example.org", "password": "very-strong-password"}
        create = client.post("/auth/register", json=credentials)
        assert create.status_code == 201
        assert client.post("/auth/login", json=credentials).status_code == 422
        assert client.post("/auth/verify-email", json={"email": credentials["email"], "code": "00000000"}).status_code == 422
        assert client.post("/auth/verify-email", json={"email": credentials["email"], "code": "12345678"}).json() == {"verified": True}
        assert client.post("/auth/login", json=credentials).status_code == 200


def test_verification_bad_code_format_is_rejected():
    with TestClient(create_app(auth=RegistrationAuth(), registry=RegistrationRegistry())) as client:
        result = client.post("/auth/verify-email", json={"email": "a@example.org", "code": "12"})
        assert result.status_code == 422


def test_registration_rate_limit(monkeypatch):
    monkeypatch.setenv("MODELOPS_RATE_LIMITS_ENABLED", "1")
    with TestClient(create_app(auth=RegistrationAuth(), registry=RegistrationRegistry())) as client:
        for i in range(4):
            r = client.post("/auth/register", json={"email": f"person{i}@example.org", "password": "very-strong-password"})
            assert r.status_code == 201
        response = client.post("/auth/register", json={"email": "person5@example.org", "password": "very-strong-password"})
        assert response.status_code == 429


def test_resend_has_generic_response():
    with TestClient(create_app(auth=RegistrationAuth(), registry=RegistrationRegistry())) as client:
        result = client.post("/auth/resend-verification", json={"email": "a@example.org"})
        assert result.status_code == 200
        assert "message" in result.json()
