"""Public error feedback must stay informative without revealing internal errors."""
from dashboard.api_feedback import describe_api_error


def test_duplicate_registration_does_not_enumerate_addresses():
    message = describe_api_error(409, '/auth/register', 'Account or workspace name already exists')
    assert 'sign in' in message.lower()
    assert 'already exists' not in message.lower()
    assert '409' not in message


def test_registration_validation_has_human_guidance():
    assert 'password' in describe_api_error(422, '/auth/register').lower()


def test_login_error_is_not_a_raw_api_error():
    message = describe_api_error(422, '/auth/login', 'Invalid login credentials')
    assert 'sign you in' in message
    assert '422' not in message


def test_legacy_upload_is_explained():
    message = describe_api_error(403, '/my/workspaces')
    assert 'portable' in message
    assert '403' not in message


def test_internal_details_not_displayed():
    message = describe_api_error(503, '/my/workspaces', 'database-password-example')
    assert 'database-password-example' not in message
    assert '503' not in message


def test_rate_limits_are_actionable():
    assert 'wait' in describe_api_error(429, '/auth/register').lower()


def test_verification_failure_is_specific():
    assert 'code' in describe_api_error(422, '/auth/verify-email').lower()
