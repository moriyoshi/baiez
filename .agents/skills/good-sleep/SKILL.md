---
name: good-sleep
description: Consolidate accumulated baiez engineering journal findings into durable topic notes and keep the project backlog accurate.
---

# Consolidate engineering notes

Read `.agents/docs/JOURNAL.md`, `.agents/docs/TODO.md`, and
`.agents/docs/LTM/INDEX.md`. Use this workflow when several journal entries
cover the same design or performance question and a topic note would make the
finding easier to reuse.

Move unresolved follow-ups to `TODO.md` with their evidence and source
heading. Create or update focused notes under `.agents/docs/LTM/`, merging
related evidence by topic rather than copying entries by date. Add each note
to `LTM/INDEX.md`. Preserve measurements, workload definitions, rejected
hypotheses, and exact compatibility rules. Leave the chronological journal
intact and append a consolidation record there.

Promote a stable current rule to `OVERVIEW.md`, `ARCHITECTURE.md`, or
`QUALITY_GATE.md` when the topic note supports it. Do not turn an unverified
idea into a project rule.
