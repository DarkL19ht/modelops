# Account Registration and Verification

## Purpose

ModelOps uses email verification for self-service registration while maintaining
private user-owned model workspaces. The public sample demonstration does not
require an account.

## Workflow

1. Users register using an email address and a password.
2. The service sends a one-time eight-digit verification code through Resend.
3. Codes expire after ten minutes and allow at most five verification attempts.
4. Users verify ownership of their email address before signing in.
5. Existing accounts from the original database schema are preserved by the
   one-time migration. New accounts are initially unverified.
6. An authenticated user can manage only their own model workspaces.

## Operational controls

- PostgreSQL stores rate-limit counters, challenge hashes, and verification state.
- Registration and login are limited by identity and client address.
- Workspace uploads, retraining, monitoring, and inference have server-side
  request budgets; changing Streamlit code does not bypass these limits.
- A signed-in user can upload at most three workspaces under the existing
  registry rule and files are bounded by the existing 5 MB limit.
- Each email address has a resend cooldown and hourly send allowance.
- Secret keys must be stored on Render, not in the Streamlit application.

## Security boundaries and known limitations

These controls do not constitute an execution sandbox. Loading a `.skops`
artifact with no explicitly trusted unknown types reduces risk, but executing
user-supplied estimators can still be resource-intensive or dangerous. For an
unrestricted public deployment, isolate model loading, inference, and training
in separate unprivileged workers with CPU, memory, time and network limits.

The synchronous FastAPI retraining endpoint does not provide durable background
job semantics. Limit exposure to a controlled beta until workers, quotas,
additional abuse controls, and observable jobs are introduced.

The `ss_` PostgreSQL tables remain in Neon. The migration grants verified status
to accounts created before the verification field existed, once only. This
retains existing accounts rather than invalidating them unexpectedly.

Email delivery requires an operational Resend account and sender verified for
sending to users outside your own account. Sandbox sender addresses may only
support testing with permitted recipients.
