# Cloud cost review: Q1 2026

Prepared by finance with the platform team. Amounts in thousands of euros.

## Summary

Cloud hosting cost **€512k** in the first quarter, against a budget of **€475k**: **€37k (7.8%)
over**. The overrun is within the 10% allowed by the *Engineering budget 2026*, because other
categories were under budget by the same amount in Q1:

| Category | Budget Q1 | Actual Q1 | Difference |
|---|---|---|---|
| Cloud hosting | 475 | 512 | +37 |
| Developer tooling | 60 | 41 | −19 |
| Contractors | 150 | 132 | −18 |
| Training and conferences | 15 | 15 | 0 |
| Total | 700 | 700 | 0 |

## Why cloud hosting was over

- After the dispatch backlog incident in **February**, the dispatch service ran with **12
  consumers**, its maximum, for three weeks instead of the usual 4. This cost about **€21k**.
- Storage for ledgerdb backups grew faster than planned: about **€11k**.
- The rest is normal growth in traffic.

## Actions

1. The platform team adds autoscaling to dispatch so consumers scale back down by themselves
   (owner: Ines Okafor, due **30 June 2026**).
2. Backups older than **35 days** move to cheaper archive storage (owner: platform team, due
   **31 May 2026**).
