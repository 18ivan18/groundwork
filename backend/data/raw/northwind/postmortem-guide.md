# Writing a postmortem

A postmortem explains what happened in an incident and what we will change so it does not happen
again. Postmortems at Northwind Labs are **blameless**: we write about systems and decisions, not
about who made a mistake.

## When one is required

SEV1 and SEV2 incidents always need a postmortem; SEV3 incidents only if the incident commander
asks for one. Deadlines are in *Incident severity levels*.

## Sections

1. **Summary:** two or three sentences a customer support agent could read.
2. **Impact:** who was affected, for how long, and how many jobs, orders or customers.
3. **Timeline:** in CET, from the first sign of trouble to resolution. Copy it from the incident
   channel.
4. **Root cause and contributing factors:** ask "why" until you reach something we can change.
5. **What went well and what did not.**
6. **Action items:** each with an owner and a due date, filed in the tracker with the label
   `postmortem`.

## Review

The postmortem is reviewed in the weekly operations review on **Wednesday at 14:00 CET**. Action
items from SEV1 postmortems must be finished within **30 days**; the platform team reports open
items every month.
