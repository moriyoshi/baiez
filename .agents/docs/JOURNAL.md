# Engineering journal

Append dated findings, failed hypotheses, and decisions here. Preserve the
original evidence when correcting an entry; add a correction with its date.
Move durable conclusions to topic notes under `LTM/` when this file grows.
Keep open follow-ups in `TODO.md`.

## 2026-10-03 — Agent harness adoption

Adapted the yesno repository's agent document map, local gate, test-layer
guidance, and reusable workflows for baiez. The routine gate covers Rust,
Python, and archive hygiene. Real LightGBM score attestation remains a
fixture-driven check because the fixture and native LightGBM package are not
ordinary build dependencies. The local Rust crate cannot complete
`cargo package` until `yesno-core` 0.1.0 is published; archive listing is
checked independently.

The first gate run showed that `cargo fmt --all` traversed the path dependency
and reported formatting in the sibling yesno checkout. The local gate now
selects `--package baiez`, so its verdict concerns this project alone.
The corrected gate passed under Rust 1.97.1 and the CI-pinned Rust 1.95.0.
The four agent skills passed the local skill validator. Cargo archive listing
and a rebuilt Python source archive contained no agent-only files. The GitHub
workflow has been parsed locally but has not run on GitHub yet.

## 2026-10-04 — Python runtime gate

The first routine gate checked Python syntax and style but never imported the
built extension. A focused workflow now loads a small model through the CLI,
opens the same bundle through `Booster.from_yesno`, and checks score order,
leaf shape, dump contents, and error behavior. This covers the Python boundary
without making a trained LightGBM fixture a routine-gate prerequisite.
The gate also checks local Markdown file links so the agent document map
cannot silently point to deleted or moved files.
The expanded gate passed locally. The Python workflow also passed with the
CI-pinned Rust 1.95.0 build. Changing the fixture's right-leaf value from
4.0 to 5.0 made the workflow fail on its independent score assertion; the
fixture was restored afterward.

## 2026-10-04 — Saved-score oracle in the routine gate

The existing 100,000-row LightGBM fixture matched the hashes in
`reports/gbm_workflow.json`. A 1,024-row prefix now ships under
`tests/fixtures/lightgbm_regression_1024/`, with the exact trained model,
native raw scores, and a hash manifest. The attestation checks that manifest
before scoring. It also includes scattered and shuffled cohorts at sizes
below the full row count; this prevents a small CI fixture from exercising
only contiguous row IDs.

## 2026-10-04 — Harness work summary and findings

### Work completed

- Added `AGENTS.md`, the `CLAUDE.md` pointer, contributor guidance, and
  `.agents/docs/` notes for architecture, testing, quality checks, the backlog,
  and durable findings. Added four reusable agent skills and validated their
  metadata.
- Added `scripts/gate.sh` and a GitHub Actions workflow that checks out
  yesnodb at commit `95d130037a62059bd7e4f849dbd43a43c0a0358c` and
  runs the same local gate with Rust 1.95.0. The gate covers Rust formatting,
  Clippy, all-feature tests, Ruff, local document links, the built Python
  adapter, saved LightGBM score parity, and Cargo archive contents.
- Added a shared small JSON model fixture for the Rust CLI and Python workflow
  checks. Added a 1,024-row slice of the existing trained LightGBM regression
  fixture with a hash manifest; the model and full source fixture matched the
  hashes already recorded in `reports/gbm_workflow.json` before extraction.
  The score attestation now exercises scattered and shuffled cohorts below
  the full row count.
- Excluded agent-only files from Rust and Python release archives. A rebuilt
  Python source distribution included all six saved-fixture files and no
  `.agents/`, `.github/`, `AGENTS.md`, or `CLAUDE.md` files.

### Findings and evidence

- `cargo fmt --all` traversed the sibling yesno path dependency and produced
  findings outside baiez. `cargo fmt --package baiez -- --check` scopes the
  baiez gate correctly.
- Python lint alone left the extension and adapter unexercised. The new CLI to
  Python workflow checks score order, leaf shape, model dumping, and invalid
  input errors. Deliberately changing the fixture's right leaf from 4.0 to
  5.0 made its score assertion fail; the fixture was restored.
- The 1,024-row saved fixture has eight features, 16 trees, 80 missing values,
  and 89 zero values. CLI and Python scores matched native LightGBM raw scores
  with maximum absolute error `4.44e-16` across contiguous, scattered, and
  shuffled cohorts. Deliberately altering a saved score in a temporary copy
  was rejected by the hash check before prediction.
- The eight-step local gate passed. The earlier full gate and the new Python
  workflow also passed with Rust 1.95.0; the latest full gate passed with Rust
  1.97.1. The link checker found 41 valid local links across 17 Markdown files.
  The CI YAML parsed locally, but no hosted GitHub Actions run has been seen.

### Remaining verification

The host-built wheel and source archive have been inspected, but the new CI
workflow still needs its first hosted run. Native LightGBM C API text import
and ordinary `Booster.predict(X)` remain outside the saved-score gate. Full
Rust crate packaging remains blocked until `yesno-core` 0.1.0 is published to
crates.io. The 100,000-row fixture remains the broader measurement and
attestation reference; the small fixture is the routine correctness oracle.

## 2026-10-04 — yesno replica peer socket

Added a read-only baiez peer loader for yesnod's plugin Unix socket. It loads
bundle metadata and the packed index from one remote snapshot, decodes array,
bitmap, and inclusive-end run lanes, and prepares owned yesno sets for repeated
scoring after the socket closes. The CLI accepts `--peer-socket` for prediction
and model dump; the Python adapter exposes `Booster.from_peer` and `bind_peer`.
Follower startup `UNAVAILABLE` waits for readiness, and a generation change
restarts the entire load. A real Unix-socket regression checks follower-role
arena and inline modes, Rust and CLI score parity, and model dump. A full
TLS leader-to-follower run also passed: baiez CLI and Python
`Booster.from_peer` both scored `[9, 7]` as `[4.0, 2.0]` after a completed
replication pass. The first attempt queried while the follower was still
bootstrapping and received invalid bundle magic because the metadata key had
not replicated. The socket's existence and daemon readiness do not imply a
complete bootstrap; wait for `yesnod_follower_passes_total` before loading a
bundle. The final eight-step local gate passed, including Rust tests, Python
workflow, saved LightGBM score parity, documentation links, and Cargo archive
contents. A later model rollout remains untested. Rust publication now also
requires `yesno-plugin` 0.1.0 on crates.io alongside `yesno-core`.

## 2026-10-04 — real-process E2E harness

Adapted haiiie's scenario-per-fixture harness for baiez under `e2e/`, with a
separate `scripts/gate-e2e.sh` and CI job. Each scenario owns fresh ports and
directories, bounds readiness and subprocess calls, reaps children, and keeps
logs/configs under `.agents-workspace/tmp/e2e/`. The loopback fixture uses
yesno's explicit insecure replication mode; it is a process and replication
check, not a TLS or Kubernetes deployment check.

`replica_rollout` passed in 32.98 seconds: a seeded leader replicated its
bundle, CLI and Python scored `[9, 7]` as `[4.0, 2.0]`, a new model key pair
published in one Flight batch reached the follower and scored `[6.0, 2.0]`,
and the previously prepared Booster kept `[4.0, 2.0]`. `replica_restart`
passed in 32.60 seconds: the follower stopped, missed a new model key pair,
reopened its data directory, caught up, and served `[8.0, 2.0]` through CLI
and Python. Both scenarios wait for the follower's first completed replication
pass because `/readyz` and the peer socket become available earlier. The
full default runner later passed both scenarios in 63.46 seconds. A follow-up
role check confirmed the follower rejects Flight writes with a “read-only
replica” error; its first assertion was too narrow for that wording, and the
corrected scenario passed in 31.85 seconds. CI checkout nesting can push Unix socket
paths to Linux's length limit, so the harness now uses a short system
temporary directory for sockets and keeps logs/configs in the repository's
E2E scratch directory. The corrected rollout scenario passed again with that
layout in 33.36 seconds, and socket cleanup left no temporary directories.
The restart scenario also passed with short socket paths in 32.89 seconds,
confirming socket rebinding and WAL catch-up with the final fixture layout.
The final eight-step routine gate passed after the socket-path change,
including Python lint, saved LightGBM score parity, and Cargo archive checks.
The hosted CI E2E job has not run yet.

## 2026-10-04 — Monty E2E migration

Followed haiiie's current Rust-hosted Monty harness. Removed the CPython
`e2e/run.py` and `e2e/harness.py` runner, replacing it with the `baiez-e2e`
crate. The checked-in `.py` scenarios now execute inside Monty and can only
call Rust host verbs; they cannot directly open files, spawn processes, read
environment variables, or use the network. The host controls yesnod children,
publishes model key pairs through the yesno Flight CLI, checks baiez CLI output
and model dumps, and retains a prepared peer index to verify old snapshot
stability after rollout. Python adapter checks remain in the routine gate.

The Monty dependency required `get-size2` 0.10.1 in Cargo.lock, matching
haiiie's working resolution; 0.10.3 selected a different `compact_str`
version and failed to compile `ruff_python_ast`. The new parser test passed.
The real-process E2E gate passed both scenarios: `replica_restart` in 31.48
seconds with 12 host calls and `replica_rollout` in 31.70 seconds with 15
calls. The E2E gate now needs Rust binaries and no Python or NumPy setup.
`cargo clippy --offline -p baiez-e2e --all-targets -- -D warnings` passed.
`cargo +1.95.0 check --offline -p baiez-e2e --locked` also passed with CI's
Rust toolchain.
The eight-step routine gate passed after the migration, including Python
workflow, saved LightGBM score parity, and Cargo archive checks. The hosted
CI job has not run yet.

## 2026-10-05 — expand Monty E2E scenarios

Added `leader_peer_restart` to cover peer reads directly from the authoritative
yesnod owner, including scores, leaf indices, model dump, an owner restart, and
an already prepared snapshot surviving that restart. Added `missing_model` to
verify an absent bundle key fails closed and valid reads still work afterward.
The current CLI reports the absent key as `InvalidBundle("missing or invalid
bundle magic")`, which this scenario asserts as the expected rejection. The
full suite passed: leader peer restart 0.77 seconds (12 host calls), missing
model 0.33 seconds (5 calls), follower restart 31.73 seconds (12 calls), and
replica rollout 31.71 seconds (15 calls).

Added `lightgbm_replica`, which seeds the checked-in LightGBM 4.7.0 model and
1,024-row fixture into the leader, waits for follower bootstrap, and checks
every saved native score in contiguous, scattered, shuffled, and reversed
cohorts through the peer socket at `1e-10` tolerance. The real process scenario
passed in 31.28 seconds. The full five-scenario gate also passed: leader peer
restart 0.52 seconds, LightGBM replica 31.29 seconds, missing model 0.31
seconds, follower restart 31.47 seconds, and replica rollout 32.95 seconds.

Added `replica_fanout`, which starts two followers against one leader, verifies
both initial model reads, publishes a new key pair, and checks both followers
catch up while their original model remains readable. The scenario passed in
33.53 seconds with 15 host calls.
The full six-scenario suite passed as well: leader peer restart 0.49 seconds,
LightGBM replica 31.20 seconds, missing model 0.31 seconds, fan-out 32.10
seconds, follower restart 31.65 seconds, and rollout 32.90 seconds.

## 2026-10-05 — namespace encoded model keys

Added `KeyNamespace` to allocate high key bits to a namespace ID while keeping
the low bit for baiez's metadata/index pair. Namespace widths are configurable
from 1 to 32 bits; the remaining bits encode logical model IDs. CLI bundle
load, dump, index-build, and predict accept paired `--namespace-bits` and
`--namespace-id` options. The Python adapter accepts matching keyword options
on local and peer opens/binds. Calls without them keep existing raw-key
behavior. All users of a database must agree on the width, and deployed raw
keys that overlap encoded ranges need migration.

Namespace tests cover widths 1, 8, 16, and 32, invalid widths and IDs, model-ID
limits, and metadata/index pair boundaries. CLI tests exercise namespaced
load, index build, prediction, dump, wrong-ID rejection, and incomplete option
rejection. The Python workflow opens and binds the same namespace and checks
invalid configuration and namespace IDs.

The eight-step routine gate passed, including Python extension workflow and
saved LightGBM parity. The full seven-scenario E2E gate passed: leader peer
restart 0.50 seconds, LightGBM replica 31.41 seconds, missing model 0.31
seconds, namespace replica 31.49 seconds, fan-out 32.19 seconds, follower
restart 32.42 seconds, and rollout 33.29 seconds. CI's Rust 1.95 toolchain
passed `cargo check --offline --locked --all-targets --all-features`.

## 2026-10-05 — read bundles over Arrow Flight

Added a Flight-backed bundle loader using yesno's public `YesnoClient`. It
plans the metadata query, then plans the adjacent packed-index query at the
metadata ticket's database version before consuming either stream. This keeps
both sets on one generation and lets the server lease the two reads together.
The loader materializes the ordinal streams into owned sets and closes the
connection after preparing the packed index.

The CLI now accepts `--flight-endpoint` for model dump and indexed prediction.
The Python adapter adds `Booster.from_flight` and `bind_flight`, including the
existing namespace options. Flight is enabled in default Cargo builds and is
also required when building the Python feature. README, architecture, and test
notes describe the new path.

`tests/flight.rs` starts an in-process yesno Flight service and checks a
namespaced bundle against direct local prediction, then exercises the CLI over
the wire. The test passed. `./scripts/gate.sh` passed formatting, Clippy, all
Rust tests, Python lint/format/workflow, saved LightGBM score parity, and Cargo
archive checks. The yesno Flight client-only dependency emits three existing
dead-code warnings in its own library build; baiez's Clippy gate remains clean.

Extended the routine Python workflow to invoke both `Booster.from_flight` and
`bind_flight` against the in-process Flight server, including namespace mapping,
raw-score prediction, and leaf output. The test reuses the NumPy-enabled Python
selected by the gate, so it does not depend on the machine's system Python.
The smoke check passed under the cached NumPy 2.5.3 environment.
