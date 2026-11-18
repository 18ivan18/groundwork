# Runbook: orders-api high latency

**Alert:** `OrdersApiLatencyHigh` fires when the p95 latency of orders-api is above **800 ms for
10 minutes**. orders-api takes new delivery orders from customers' systems, so slow responses mean
customers' integrations time out and retry.

## Check

1. Open the *orders-api* dashboard on Lantern. Look at request rate, p95 latency and error rate
   together: latency with a traffic spike is a capacity problem; latency at normal traffic usually
   comes from the database.
2. Check the ledgerdb panel. orders-api has a connection pool of **200** connections. If the pool
   is above 90% used, requests are queuing for a connection.
3. Check Shipyard for a deploy of orders-api in the last hour.

## Act

- **Recent deploy:** roll it back (see *Rollback procedure*). Do this first; debug afterwards.
- **Traffic spike:** scale out. orders-api normally runs **6 replicas**; scale to at most **10**:

  ```shell
  shipyard scale orders-api --replicas 10
  ```

  More than 10 replicas exhausts the ledgerdb connection limit, so do not go higher.
- **Database CPU above 85% for 10 minutes:** page the database owner (platform team) and consider
  *Failing over ledgerdb* with the incident commander.

## Afterwards

Scale back to 6 replicas once p95 latency has been below 500 ms for an hour, and note the incident
number in the scaling change.
