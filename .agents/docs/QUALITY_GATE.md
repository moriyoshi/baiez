# Quality gate

`./scripts/gate.sh` is the routine local and CI check. It changes to the
repository root, reports each result, and exits nonzero if a check fails or
the expected sequence is incomplete. Run it after code changes. Its checks
are:

1. `cargo fmt --package baiez -- --check` (the package selector keeps the
   sibling yesno checkout out of this gate).
2. `cargo clippy --offline --all-targets --all-features -- -D warnings`.
3. `cargo test --offline --all-features`.
4. Ruff lint and format checks for `python/` and `scripts/`. Monty scenario
   syntax is checked by the separate Rust E2E crate.
5. Local Markdown links resolve in project and agent documents.
6. Build the Rust CLI and PyO3 extension, then run
   `scripts/check_python_workflow.py` and the Flight integration test with
   NumPy to test Python prediction, namespace handling, and Flight load/bind.
7. Verify a hash-pinned fixture and compare CLI and Python scores with native
   LightGBM scores for contiguous, scattered, and shuffled cohorts.
8. `cargo package --offline --allow-dirty --list`, with required release files
   checked and generated artifacts excluded.

The Cargo checks require the sibling `../yesno/yesno-core` checkout. `uvx`
provides Ruff if no standalone `ruff` executable is installed. The Python
runtime step uses the active Python when it has NumPy, or a cached NumPy 2.5.3
environment through `uv`. The script uses Cargo offline after dependencies
are fetched. CI fetches dependencies and installs NumPy before invoking the
same gate. The saved-score fixture needs no LightGBM installation. If tools
or dependencies are unavailable, the gate fails and the report must say
which check could not run.
The link check verifies local file targets, not headings or external URLs.

## Review after the commands

- For model parsing or objective changes, verify LightGBM score and leaf
  parity on an independent fixture, including missing and zero values.
- For index changes, verify packed ordinal bounds, duplicate and unknown IDs,
  sparse and dense cohorts, and the descriptor fingerprint.
- For bundle changes, verify single-snapshot reads, atomic key-pair writes,
  version rejection, checksum errors, and round trips.
- For key namespaces, check widths 1 and 32, namespace ID and model ID limits,
  pair boundaries, and load/dump/index/predict access through CLI and Python.
- For Python or CLI changes, verify actual entry points and result shapes;
  compiling a Rust library alone does not exercise the adapter.
- For peer changes, run `tests/peer.rs` with a follower-role socket in arena
  and inline modes, then `./scripts/gate-e2e.sh` for a real leader/follower
  pair. Socket creation precedes complete bootstrap.
- For a performance claim, verify that the benchmark calls the changed code,
  measure the same workload with a relevant baseline, and record machine,
  versions, build mode, affinity, repeated timings, and correctness.
- For release changes, inspect wheel and sdist metadata and contents; the
  archive-list gate does not substitute for `cargo package`, which needs the
  published `yesno-core` dependency.

Use [.agents/docs/TESTING.md](TESTING.md) for the appropriate test layer and
[RELEASING.md](../../RELEASING.md) for publication checks. CI is configured
in `.github/workflows/ci.yml`; do not claim that hosted CI has passed until a
run has actually completed.
