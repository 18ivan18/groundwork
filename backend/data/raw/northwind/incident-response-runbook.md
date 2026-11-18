# Incident response runbook

Use this runbook as soon as you think something is wrong in production. It applies to every
severity; *Incident severity levels* says how fast each step must happen.

## 1. Declare

Run `/incident declare` in the #incidents channel with a one-line summary. The bot creates a
dedicated channel named `inc-<number>` and a timeline document.

## 2. Assign roles

- **Incident commander (IC):** coordinates, decides, keeps the timeline. The IC does not debug.
- **Operations lead:** the engineer working on the fix, usually the primary on-call.
- **Communications lead:** writes status page updates and answers customer support. For SEV1 this
  must be a different person from the IC.

## 3. Stabilise before you diagnose

Prefer actions that stop the damage quickly: roll back the last deploy (see *Rollback procedure*),
turn off a feature flag, scale up, or fail over the database. Finding the root cause can wait until
customers are no longer affected.

## 4. Communicate

Post an update in the incident channel at least every 30 minutes, even if nothing changed. Say what
we know, what we are trying and when the next update will come.

## 5. Resolve and hand over

The IC declares the incident resolved once the impact has stopped and monitoring has been normal for
**30 minutes**. Then the IC names a postmortem owner and closes the channel with `/incident resolve`.
