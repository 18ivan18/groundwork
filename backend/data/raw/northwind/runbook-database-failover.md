# Runbook: failing over ledgerdb

ledgerdb is the PostgreSQL database behind orders-api and billing-worker. It runs as a **primary in
zone A** with a streaming **replica in zone B**.

## When to fail over

Only when the primary is down or so degraded that Tern is unusable, and restarting it has not
helped. A failover needs **two people**: the on-call engineer and the incident commander, who
approves it in the incident channel.

## What to expect

- Writes stop for **2 to 4 minutes** while the replica is promoted and services reconnect.
- The replica lags the primary by less than **30 seconds**, so at most the last 30 seconds of
  writes can be lost. Note the time of the failover in the timeline so billing can check those
  orders afterwards.

## Steps

1. Announce the failover in the incident channel and on the Lantern status page.
2. Stop writes from billing-worker: `shipyard scale billing-worker --replicas 0`.
3. Promote the replica:

   ```shell
   tern-admin db promote ledgerdb --replica zone-b
   ```

4. Check that orders-api reconnected: the error rate on its dashboard falls back to normal.
5. Start billing-worker again with its usual **2 replicas**.

## Afterwards

The old primary must not come back as a primary. The platform team rebuilds it as the new replica,
usually within a working day. Until then ledgerdb has no replica, so another failover is impossible:
say so in the incident summary.
