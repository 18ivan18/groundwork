# On-call handbook

Northwind Labs runs Tern, our route planning and dispatch platform for regional couriers. Tern has
to stay up while couriers are on the road, so every production service has an on-call engineer.
This handbook explains how the rotation works and what we expect from you while you hold the pager.

## The rotation

- Each team runs a weekly rotation with a **primary** and a **secondary** engineer.
- Handover is every **Monday at 10:00 CET**. The outgoing primary writes a short handover note in
  the #oncall-handover channel: open incidents, noisy alerts and anything still being watched.
- Pages go to the primary first. If the primary does not acknowledge within the time set for the
  severity (see *Incident severity levels*), Beacon pages the secondary, then the engineering manager.
- You join the rotation after your first **eight weeks** at Northwind Labs, once you have shadowed
  one full week.

## What we expect

- Keep your laptop and a working internet connection within reach. You do not have to stay at home.
- Acknowledge pages within the severity's acknowledgement time, even if you cannot fix the problem
  yet. Acknowledging tells everyone a human is looking.
- If you are unsure of the severity, treat it as the higher one. Downgrading later is cheap.
- Hand over to the secondary if you are ill or have an emergency. Tell them in a call, not only
  in chat.

## Compensation

- The on-call allowance is **€350 per week** as primary and **€150 per week** as secondary.
- If you work more than **two hours** on an incident between 22:00 and 07:00, you get the next
  working day off in lieu. Log it in the time-off tool with the incident number.

## Tools

- **Beacon** sends pages and holds the rotation schedule.
- **Lantern** is our status page and dashboard site.
- **Shipyard** is the deploy pipeline; see *Deploying to production*.
