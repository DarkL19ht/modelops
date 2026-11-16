"""Safe, human-readable feedback for the public ModelOps application.

Do not expose raw HTTP errors, internal exception strings or account-existence
information in the browser. Preserve detailed diagnostics on the API server.
"""
from __future__ import annotations


def describe_api_error(status: int, path: str, detail: object = None) -> str:
    """Map API failures to actionable messages without leaking internals."""
    if path == "/auth/register":
        if status == 409:
            return ("If you already have an account, sign in instead. "
                    "Otherwise, check your email for a verification code.")
        if status in (400, 422):
            return "Enter a valid email and a password containing 12–128 characters."
    if path == "/auth/login" and status in (400, 401, 422):
        return ("We couldn't sign you in. Check your email and password, "
                "and make sure your email has been verified.")
    if path == "/auth/verify-email" and status in (400, 404, 422):
        return "That verification code is invalid or expired. Request a new code and try again."
    if status == 429:
        return "You've made too many requests. Please wait a little before trying again."
    if status == 401:
        return "Your session has expired. Please sign in again."
    if status == 403:
        if path == "/my/workspaces":
            return ("This upload format is currently limited to approved beta accounts. "
                    "For public access, choose the portable JSON upload instead.")
        return "You don't have access to this operation. Please check your workspace permissions."
    if status == 413:
        return "One of the uploaded files is too large. Please check the file-size limits."
    if status == 409:
        return "An item with these details already exists. Choose a different name and try again."
    if status == 422:
        # Backend error details are not necessarily safe or user-readable.
        # The upload page offers format guidance separately.
        return "Some information is invalid or incompatible. Check the model format and CSV columns."
    if status >= 500:
        return "ModelOps is temporarily unavailable. Please try again shortly."
    return "We couldn't complete that request. Please review the details and try again."
