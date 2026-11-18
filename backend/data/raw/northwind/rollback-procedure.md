# Rollback procedure

Rolling back is the fastest way to stop damage caused by a release. Roll back first and investigate
afterwards; nobody will ask why you rolled back a release that turned out to be fine.

## When

- Shipyard stopped a canary because its error rate went above 2%.
- An incident started within an hour of a deploy of the affected service.
- Within **15 minutes** of a full rollout, the service's error rate is above 2%.

## How

Find the previous release in Shipyard's release history, then:

```shell
shipyard rollback <service> --to <release>
```

For example, `shipyard rollback orders-api --to 2026.03.17-2`. A rollback skips the canary and takes
about **3 minutes** for orders-api and **5 minutes** for dispatch.

## Database migrations

A rollback does not undo database migrations. This is why *Deploying to production* requires
migrations that the previous release can run against. If a migration itself is the problem, page
the platform team: they decide whether to write a reverse migration.

## Afterwards

Post the rollback in #deploys and in the incident channel if there is one. Block the release in
Shipyard with `shipyard release block <service> <release>` so nobody deploys it again by accident.
