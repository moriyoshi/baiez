---
name: quality-gate
description: Check baiez code or release changes against the repository's Rust, Python, archive, and behavioral quality gate; fix failures and report the evidence.
---

# Quality gate for baiez

Read `.agents/docs/QUALITY_GATE.md` and the relevant section of
`.agents/docs/ARCHITECTURE.md`. Identify the change scope from the task and,
when available, the Git diff. This checkout may have no `.git` directory.

Run `./scripts/gate.sh`. Treat a missing dependency or unavailable tool as a
failed check, not as a pass. Fix code failures in scope and rerun the affected
steps, then the full gate. Do not change score references, tolerances, or
archive requirements merely to silence a failure.

For changed behavior, inspect its test layer in `.agents/docs/TESTING.md`.
The gate runs a small saved LightGBM fixture. For changes that affect wider
cohorts or performance, also run the full fixture attestation when available;
state any unavailable coverage. For performance changes, confirm the
benchmark reaches the changed code and records a comparable baseline.

Report the gate verdict, focused checks, any unresolved failures, and files
changed. Append a durable finding to `.agents/docs/JOURNAL.md` only when the
work establishes a new decision or failure mode.
