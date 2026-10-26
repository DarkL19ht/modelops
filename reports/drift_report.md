# ModelOps — Data Drift Report

- Reference observations: **5,000**
- Incoming observations: **1,200**
- PSI alert threshold: **0.20**
- Features flagged: **3 of 5**

| Feature | Type | PSI | Status | Reference mean | Incoming mean | KS statistic |
|---|---|---:|---|---:|---:|---:|
| tenure_months | numeric | 0.006 | stable | 36.41 | 36.89 | 0.024 |
| monthly_spend | numeric | 0.414 | drift | 70.37 | 84.73 | 0.273 |
| support_tickets_90d | numeric | 0.951 | drift | 0.81 | 2.05 | 0.422 |
| logins_30d | numeric | 5.540 | drift | 15.07 | 8.02 | 0.714 |
| monthly_contract | categorical | 0.134 | stable | 0.55 | 0.73 | — |

## Interpretation

Drift flags compare input-feature distributions; they do not establish that model performance has deteriorated.

PSI at or above the configured threshold triggers a drift flag. PSI depends on reference-based binning and sample size. The KS statistic is descriptive for discrete count features; its p-value assumptions are most appropriate for continuous observations.

The incoming dataset is synthetically shifted by design. Results are a demonstration, not a real-world performance claim.
