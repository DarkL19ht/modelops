"""Short-lived child process for trusted-beta model operations.

The parent authenticates the user; owner-scoped registry methods validate the
owner again before accessing a workspace. All model deserialization and model
execution happen here, outside the public API process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_blob(value, workdir: Path):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid uploaded file reference")
    filename = value.get("file", "")
    if not isinstance(filename, str) or filename not in {
        "model_bytes.bin", "ref_bytes.bin", "inc_bytes.bin",
        "label_bytes.bin", "validation_bytes.bin",
    }:
        raise ValueError("Unexpected uploaded file reference")
    return (workdir / filename).read_bytes()


def perform(action: str, data: dict, workdir: Path):
    from src.modelops.self_service_registry import SelfServiceRegistry
    registry = SelfServiceRegistry()
    owner_id = data["owner_id"]
    if action in {"register", "register_portable"}:
        from src.modelops.model_workspace import Workspace, load_uploaded_model, read_csv
        if action == "register_portable":
            from src.modelops.portable_models import load_portable_model
            model = load_portable_model(_read_blob(data["model_bytes"], workdir))
            source_format = "portable-linear-v1"
        else:
            model = load_uploaded_model(_read_blob(data["model_bytes"], workdir))
            source_format = "legacy-skops"
        ws = Workspace(
            name=data["name"], champion=model,
            reference=read_csv(_read_blob(data["ref_bytes"], workdir), "reference.csv"),
            incoming=read_csv(_read_blob(data["inc_bytes"], workdir), "incoming.csv"),
            labels=(read_csv(_read_blob(data["label_bytes"], workdir), "labels.csv")
                    if data.get("label_bytes") else None),
            validation=(read_csv(_read_blob(data["validation_bytes"], workdir), "validation.csv")
                        if data.get("validation_bytes") else None),
            id_col=data["id_col"], target_col=data["target_col"],
            categorical=tuple(data["categorical"]),
        )
        if source_format == "portable-linear-v1":
            if len(ws.reference) > 2000 or len(ws.incoming) > 2000:
                raise ValueError("Public workspaces are limited to 2,000 rows per dataset")
            for cohort in (ws.labels, ws.validation):
                if cohort is not None and len(cohort) > 2000:
                    raise ValueError("Public workspaces are limited to 2,000 rows per dataset")
            import numpy as np
            for cohort in (ws.reference, ws.incoming, ws.validation):
                if cohort is not None:
                    matrix = cohort[list(ws.features)].to_numpy(dtype=float)
                    if not np.isfinite(matrix).all() or (np.abs(matrix) > 1e9).any():
                        raise ValueError("Public feature values must be within finite bounds")
        wid = registry.register(ws, owner_id, source_format=source_format)
        return {"id": wid, "champion_version": 1, "message": "Your model is ready for predictions"}
    wid = data["workspace_id"]
    if action == "monitor":
        return registry.owned_monitor(owner_id, wid, threshold=data["threshold"])
    if action == "train":
        return registry.owned_train(owner_id, wid, min_f1_gain=data["min_f1_gain"])
    if action == "predict":
        return registry.owned_predict(owner_id, wid, data["records"], version=data.get("version"))
    if action == "promote":
        return registry.owned_promote(owner_id, wid, data["version"])
    if action == "rollback":
        return registry.owned_rollback(owner_id, wid)
    raise ValueError("Unsupported operation")


def main(argv):
    if len(argv) != 3:
        raise SystemExit(2)
    request_path = Path(argv[1])
    response_path = Path(argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = perform(request["action"], request["data"], request_path.parent)
        outcome = {"ok": True, "result": result}
    except (ValueError, KeyError, PermissionError) as exc:
        typ = {ValueError: "ValidationError", KeyError: "NotFound",
               PermissionError: "PermissionDenied"}[type(exc)] if type(exc) in (
                   ValueError, KeyError, PermissionError) else "ValidationError"
        # Never send raw DB or storage exception text to the browser.
        message = str(exc)[:250] if typ == "ValidationError" else "Operation unavailable"
        outcome = {"ok": False, "error": typ, "message": message}
    except Exception:
        outcome = {"ok": False, "error": "Unavailable", "message": "Operation unavailable"}
    response_path.write_text(json.dumps(outcome, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
