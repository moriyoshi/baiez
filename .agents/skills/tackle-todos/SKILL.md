---
name: tackle-todos
description: Work through a selected baiez engineering backlog item, verify it at the relevant test layer, and update the agent notes with the outcome.
---

# Tackle a baiez to-do

Read `.agents/docs/TODO.md` and choose the item named by the user, or the
highest-impact item that is actionable with the current checkout and tools.
Read its source evidence, the related architecture notes, and relevant code.
Carry the item to a concrete result; do not mark it done because a partial
step succeeded.

Use `.agents/docs/TESTING.md` to pick the check that can catch a regression
and run `./scripts/gate.sh` after code changes. Update the item with the
result or the precise remaining dependency. Append a finding to
`JOURNAL.md` when the work changes a design decision or reveals a reusable
failure mode.
