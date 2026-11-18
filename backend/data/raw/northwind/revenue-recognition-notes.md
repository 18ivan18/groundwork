# Revenue recognition notes

How the finance team recognises Tern revenue. These notes summarise the accounting policy approved by
the CFO; the auditors have a copy.

## Subscriptions

- Customers pay for Tern per courier per month, billed **monthly** or **annually in advance**.
- Revenue is recognised **evenly over the subscription period**, whatever the billing schedule.
  An annual invoice is booked as deferred revenue and one twelfth is released each month.

## Setup fees

One-off setup fees for integrating a customer's systems are recognised over **12 months** from the
go-live date, not when they are invoiced, because the setup has no value to the customer without
the subscription.

## Usage above the plan

Jobs dispatched above the customer's plan are billed in arrears and recognised in the month the
jobs were dispatched. billing-worker's monthly usage report is the source.

## Credits

Service credits granted after an incident reduce revenue in the month they are granted. The credit
policy in customer contracts gives **5%** of the monthly fee for each full hour of a SEV1 outage, up
to **50%**.
