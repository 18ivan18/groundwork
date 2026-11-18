# Deploying to production

Every change to Tern reaches production through **Shipyard**, our deploy pipeline. Nobody deploys
from a laptop.

## Deploy windows

- Production deploys run **Monday to Thursday, 09:00 to 16:00 CET**.
- **No deploys on Fridays**, on weekends, or on the working day before an office holiday (see
  *Office holidays 2026*).
- Outside these windows, only fixes for an open SEV1 or SEV2 incident may be deployed, with the
  incident commander's approval written in the incident channel.

## Before you deploy

- The change has at least **one approving review** from someone other than the author.
- Continuous integration is green on the exact commit being deployed.
- Database migrations are backwards compatible: the previous release must still run against the
  migrated schema, so a rollback stays possible.

## How a deploy runs

1. Shipyard releases the new version to a **canary** that takes **10% of traffic** for
   **20 minutes**.
2. If the canary's error rate stays below **2%** and its p95 latency within 20% of the rest of the
   fleet, Shipyard rolls out to all replicas.
3. Otherwise Shipyard stops and pages the author. Follow *Rollback procedure*.

Announce each production deploy in #deploys; Shipyard does it for you if you start the deploy from
the pull request.
