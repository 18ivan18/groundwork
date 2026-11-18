# Month-end close runbook

The finance team closes the books for each month by the **5th working day** of the next month.
This runbook lists the steps in order. The finance controller signs off the close.

## Day 1

1. Export last month's invoices from billing-worker's report in the finance portal.
2. Reconcile the export with the bank statements. Differences above **€100** are investigated
   before the close continues; smaller ones are booked to the rounding account.
3. Book card payments that settled after the month ended to the right month.

## Days 2 and 3

4. Book accruals for services received but not yet invoiced: cloud hosting, contractors and the
   push notification provider. Use the usage reports in the finance portal.
5. Book prepaid annual licences as prepaid expenses and release one twelfth each month.
6. Check payroll against the payroll provider's report (see *Payroll run checklist*).

## Days 4 and 5

7. Review the profit and loss against budget with the department heads. Explain any line more than
   **10%** away from budget.
8. The finance controller signs off and locks the month in the accounting system.

## After the close

A locked month can only be reopened by the CFO. Corrections found later are booked in the current
month with a note referring to the original entry.
