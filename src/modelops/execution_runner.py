"""Bounded child-process execution for user-owned model operations.

This is crash/time containment, NOT a security sandbox. For untrusted public
uploads use separate, restricted compute without API/production DB credentials.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ALLOWED = {"register", "register_portable", "monitor", "train", "predict", "promote", "rollback"}
_TIMEOUTS = {"register": 90, "register_portable": 90, "monitor": 60, "train": 150,
             "predict": 30, "promote": 45, "rollback": 45}
_SLOTS = threading.BoundedSemaphore(2)


class ExecutionUnavailable(RuntimeError):
    """Execution worker timed out, failed, or is at capacity."""


def isolated_enabled() -> bool:
    """Isolated subprocesses are the default; explicit inline is local-only."""
    return os.getenv("MODELOPS_EXECUTION_MODE", "subprocess").lower() != "inline"


def _limited_environment() -> dict[str, str]:
    """Do not send Mailjet/session tokens to the model-executing process."""
    kept = ("MODELOPS_DATABASE_URL", "PATH", "HOME", "LANG", "LC_ALL",
            "SSL_CERT_FILE", "SSL_CERT_DIR", "SYSTEMROOT")
    env = {k: os.environ[k] for k in kept if k in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    return env


def _resource_limits(timeout: int):
    """POSIX best-effort CPU/address-space limits; no claim of sandboxing."""
    if os.name != "posix":
        return None

    def apply_limits():
        import resource
        cpu_limit = min(max(timeout + 5, 10), 300)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 2))
        # Address-space limiting is opt-in: BLAS/Python virtual memory varies.
        memory = int(os.getenv("MODELOPS_WORKER_MEMORY_MB", "0"))
        if memory > 0:
            limit = memory * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return apply_limits


def run_isolated(action: str, **payload):
    if action not in _ALLOWED:
        raise ValueError("Unsupported execution operation")
    if not isinstance(payload.get("owner_id"), str) or not payload["owner_id"]:
        raise ValueError("Authenticated owner identifier is required")
    if not _SLOTS.acquire(blocking=False):
        raise ExecutionUnavailable("Model workers are currently busy")
    try:
        timeout = _TIMEOUTS[action]
        with tempfile.TemporaryDirectory(prefix="modelops-exec-") as directory:
            workdir = Path(directory)
            content = {}
            for key, value in payload.items():
                if isinstance(value, bytes):
                    # Field names are controlled by API code, not filenames.
                    if key not in {"model_bytes", "ref_bytes", "inc_bytes",
                                   "label_bytes", "validation_bytes"}:
                        raise ValueError("Unexpected binary field")
                    filename = key + ".bin"
                    (workdir / filename).write_bytes(value)
                    content[key] = {"file": filename}
                else:
                    content[key] = value
            request = workdir / "request.json"
            response = workdir / "response.json"
            request.write_text(json.dumps({"action": action, "data": content}), encoding="utf-8")
            command = [sys.executable, "-m", "src.modelops.execution_worker",
                       str(request), str(response)]
            try:
                process = subprocess.Popen(
                    command, cwd=workdir, env=_limited_environment(),
                    start_new_session=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, preexec_fn=_resource_limits(timeout),
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                    raise ExecutionUnavailable("Model operation exceeded its time limit") from exc
            except OSError as exc:
                raise ExecutionUnavailable("Unable to start model worker") from exc
            if process.returncode != 0 or not response.exists():
                raise ExecutionUnavailable("Model worker terminated before completing")
            try:
                outcome = json.loads(response.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExecutionUnavailable("Model worker returned an invalid response") from exc
            if outcome.get("ok"):
                return outcome["result"]
            kind = outcome.get("error", "Unavailable")
            message = outcome.get("message", "Model operation failed")
            if kind == "ValidationError":
                raise ValueError(message)
            if kind == "NotFound":
                raise KeyError("Workspace not found")
            if kind == "PermissionDenied":
                raise PermissionError("Access denied")
            raise ExecutionUnavailable("Model operation unavailable")
    finally:
        _SLOTS.release()
