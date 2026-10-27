# ModelOps — Retraining Decision

- Run: `3290f6392f20`
- Decision: **eligible**
- Promotion: **promoted**
- Historical training rows: **4000**
- Incoming training rows: **960**

| Evaluation cohort | Metric | Champion | Challenger |
|---|---|---:|---:|
| reference_holdout | accuracy | 0.7010 | 0.7610 |
| reference_holdout | f1 | 0.4670 | 0.4816 |
| reference_holdout | roc_auc | 0.7968 | 0.7967 |
| incoming_holdout | accuracy | 0.5625 | 0.6333 |
| incoming_holdout | f1 | 0.6847 | 0.7143 |
| incoming_holdout | roc_auc | 0.7851 | 0.7859 |

## Promotion checks

- incoming_f1_gain: **PASS**
- reference_accuracy_guardrail: **PASS**
- reference_auc_guardrail: **PASS**
- incoming_auc_guardrail: **PASS**

## Model and backup paths

- Champion: `/Users/user/Documents/modelops/artifacts/churn_model.joblib`
- Candidate: `/Users/user/Documents/modelops/artifacts/candidates/churn_candidate.joblib`
- Backup: `/Users/user/Documents/modelops/artifacts/archive/champion_20261027T011325Z_90f58f7b.joblib`

## Interpretation

The challenger is trained on historical training observations and an incoming training partition. Both models are evaluated on the same historical and incoming holdouts, which were not used for challenger fitting.

The incoming holdout was previously visible during exploratory monitoring, so it is not an independent final external test set.

Passing thresholds permits promotion only when explicitly requested. Synthetic test results do not establish real-world generalisation.
