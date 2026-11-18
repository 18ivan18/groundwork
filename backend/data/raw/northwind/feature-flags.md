# Feature flags

Feature flags let us ship code turned off and turn it on without a deploy. Northwind Labs uses the
internal flag service **Semaphore**.

## Rules

- Every flag has an **owner** (a person, not a team) and a **removal date**.
- The removal date is at most **90 days** after the flag is created. Semaphore lists overdue flags
  every Monday in #engineering.
- Flags that change billing or pricing need approval from the finance team before they are turned
  on for any customer.

## Rolling out

1. Turn the flag on for Northwind Labs' internal test accounts.
2. Then for **5%** of customers for at least one day.
3. Then for everyone.

Watch the service's error rate on Lantern after each step.

## In an incident

Turning a flag off is often faster than a rollback. In Semaphore, open the flag and choose **Kill**:
it turns the flag off for every customer within **one minute** and records who did it.
