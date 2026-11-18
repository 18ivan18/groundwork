# Runbook: dispatch queue backlog

The dispatch service reads jobs from the `dispatch-jobs` queue and sends them to couriers' phones.
A backlog means couriers get their next job late.

**Alert:** `DispatchQueueBacklog` fires when `dispatch-jobs` holds more than **5,000 messages for
5 minutes**.

## Check

1. On the *dispatch* dashboard in Lantern, compare the publish rate with the consume rate. A backlog
   with a normal publish rate means the consumers are slow or stuck.
2. Look at the dead-letter queue, `dispatch-jobs-dead`. A message moves there after **5 failed
   attempts**. A fast-growing dead-letter queue points to a bad message or a broken downstream
   service rather than to load.
3. Check the push notification provider's status on Lantern's *dependencies* page.

## Act

- **Consumers are slow but healthy:** add consumers. The default is **4**, the maximum **12**:

  ```shell
  shipyard scale dispatch --replicas 8
  ```

- **One message keeps failing:** move it to the dead-letter queue by hand so the rest can flow:

  ```shell
  tern-admin queue park dispatch-jobs <message-id>
  ```

- **Consumers crash on start:** roll back the last dispatch deploy.

Do not purge `dispatch-jobs`. Every message is a courier's job; purged jobs are lost and customers
must re-send them.

## Afterwards

Replay parked messages with `tern-admin queue replay dispatch-jobs-dead` once the cause is fixed,
and scale back to 4 consumers.
