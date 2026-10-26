# ModelOps — Performance Monitoring Report

- Monitored observations: **1,200**
- Assessment: **degraded**

| Metric | Holdout baseline | Incoming labeled cohort |
|---|---:|---:|
| accuracy | 0.7010 | 0.5917 |
| f1 | 0.4670 | 0.6994 |
| roc_auc | 0.7968 | 0.7949 |

- Precision (incoming): 0.5470
- Recall (incoming): 0.9694
- Actual churn rate: 0.4900
- Predicted churn rate: 0.8683

## Alerts

- Accuracy decreased by at least the configured threshold

## Limitations

- The baseline comes from a held-out historical split.
- The incoming cohort has a shifted distribution.
- Performance differences do not prove drift caused them.
- All predictions must have matching delayed labels.
- ROC-AUC is unavailable when only one outcome class exists.
