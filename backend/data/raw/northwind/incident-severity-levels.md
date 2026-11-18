# Incident severity levels

Every incident at Northwind Labs gets a severity when it is declared. The severity decides how fast
we respond, who joins and how we communicate. When in doubt, pick the higher severity.

## SEV1: critical

- Tern is down or unusable for many customers, couriers cannot receive jobs, or customer data may be
  lost or exposed.
- Acknowledge within **5 minutes**.
- An incident commander is named at once; the on-call engineering manager takes the role if nobody
  else does.
- Update the Lantern status page within **15 minutes** of declaring, then at least every **30
  minutes** until resolved.
- A postmortem is mandatory and is due within **5 working days**.

## SEV2: major

- A feature is degraded or slow for many customers, but a workaround exists, or one large customer
  is fully affected.
- Acknowledge within **15 minutes**.
- Status page update if customers are likely to notice.
- A postmortem is mandatory and is due within **10 working days**.

## SEV3: minor

- A small group of users is affected, or an internal tool is broken.
- Acknowledge within **60 minutes** during working hours; outside working hours it can wait for the
  next working day.
- A postmortem is optional.

## Changing severity

The incident commander can raise or lower the severity at any time. Write the change and the reason
in the incident channel, so the timeline in the postmortem is accurate.
