# Portable model workspaces

## Supported public model format

Verified accounts may register `.modelops.json` artifacts containing a fitted **binary logistic regression** model, optionally using a `StandardScaler` preprocessing step. The portable format is numeric JSON, not Python pickle or executable object code. Supported features are numerical; the model must have 1–15 named features and classify labels `0` and `1`.

The JSON contains the model's feature list, logistic coefficients, intercept, and standardisation parameters. Values, names, lengths and keys are validated before the backend reconstructs a fixed scikit-learn `Pipeline`. All processing remains subject to per-account request limits, bounded data files, and a finite subprocess timeout.

Users may run predictions immediately after upload, monitor drift, evaluate delayed-label outcomes, train and evaluate a challenger, download qualifying challenger `.skops` artifacts, and explicitly promote eligible versions. A challenger is **not** guaranteed to outperform the champion. Non-eligible candidates cannot be promoted.

## Exporting your own fitted model

Run this locally from trusted Python code with a fitted `LogisticRegression` or `Pipeline([StandardScaler(), LogisticRegression()])`. It must have been trained with a pandas DataFrame so the feature names are preserved.

```python
from pathlib import Path
from src.modelops.portable_models import export_portable_model

# model is the already-fitted, trusted estimator in your Python session.
Path("my_classifier.modelops.json").write_bytes(export_portable_model(model))
```

Do not load unknown `.pkl` or `.joblib` files to export them. Loading those formats can execute arbitrary code.

## Sample model and datasets

The `examples/portable_loan/` directory contains a reproducible synthetic loan-risk classifier and compatible datasets. Upload the model and the reference/incoming CSV files. For challenger training, also supply the delayed-label and validation CSV files. Use `record_id` for the identifier and `target` for the binary label.

## Existing model uploads

Previously registered `.skops` workspaces are tagged `legacy-skops` during the additive database migration and remain restricted to trusted beta accounts. The public-safe format does not automatically apply to existing uploaded artifacts. The legacy path must not become accessible to arbitrary users merely because the beta email allowlist is empty.

## Operational safeguards and limitations

- The public-safe flag is explicitly enabled with `MODELOPS_PUBLIC_PORTABLE_MODE=1` on the API and requires subprocess execution.
- Per-file size limit for public datasets: 512 KB; maximum 2,000 rows per dataset; maximum 15 model features; maximum 16 KB for model JSON.
- The model artifact is converted from declarative JSON to a fixed scikit-learn estimator. Untrusted `.skops`, `.pkl`, `.joblib`, Python callables, arbitrary pipelines and custom classes are not accepted through the public endpoint.
- Private legacy uploads remain behind the beta gate. They should be migrated only from models that their owners trust.
- This is **not** a universal model sandbox. Public infrastructure still needs edge-layer abuse protection, capacity monitoring and preferably independently restricted compute for wider model families.
- Protect sensitive data; the demonstration is not intended for confidential financial, health or personal datasets.
- Registered accounts remain subject to email verification, workspace quotas, and training rate limits. The source project should be deployed with HTTPS and private database credentials stored only in the API service.
