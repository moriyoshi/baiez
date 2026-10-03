# Coding agents in baiez

Read this file before changing baiez. The root [README.md](README.md) describes
the user-facing product and supported features. [BENCHMARKS.md](BENCHMARKS.md)
owns published measurements, and [RELEASING.md](RELEASING.md) owns the release
procedure. Keep those documents aligned with behavior rather than copying their
contents into agent notes.

## Agent documents

- [.agents/docs/OVERVIEW.md](.agents/docs/OVERVIEW.md): project scope and current state.
- [.agents/docs/ARCHITECTURE.md](.agents/docs/ARCHITECTURE.md): module map and correctness invariants. Read it before changing `src/`, the Python adapter, or persisted formats.
- [.agents/docs/QUALITY_GATE.md](.agents/docs/QUALITY_GATE.md): checks to run and review criteria.
- [.agents/docs/TESTING.md](.agents/docs/TESTING.md): test layers, LightGBM oracle, and benchmark protocol.
- [.agents/docs/JOURNAL.md](.agents/docs/JOURNAL.md): append-only findings and decisions.
- [.agents/docs/TODO.md](.agents/docs/TODO.md): unresolved work, with evidence and scope.
- [.agents/docs/LTM/INDEX.md](.agents/docs/LTM/INDEX.md): durable topic notes distilled from the journal.

Reusable workflows live in `.agents/skills/`. The routine local gate is
`./scripts/gate.sh`; CI runs the same command. Run it before reporting a code
change as done. Do not claim a gate passed if a prerequisite prevented it from
running.
CI checks out yesnodb at a pinned commit; update that pin when baiez begins
using newer yesno-core behavior, then run the local gate against the same
revision.
Peer and replication behavior also has a separate real-process gate,
`./scripts/gate-e2e.sh`, described in [e2e/README.md](e2e/README.md).

## Working rules

- Keep the `../yesno/yesno-core` path dependency usable for local development.
  A packaged Rust crate resolves its version from crates.io; see
  [RELEASING.md](RELEASING.md) before changing dependency or release metadata.
- Preserve the model, predicate fingerprint, stride, and snapshot rules in
  [ARCHITECTURE.md](.agents/docs/ARCHITECTURE.md). Treat bundle encoding and
  Python prediction shapes as compatibility surfaces.
- Put one-off experiments and generated files under `.agents-workspace/tmp/`.
  Keep repeatable benchmarks in `examples/` or `scripts/` and their results in
  `reports/` only when the inputs, binary hashes, and timing method are recorded.
- Test behavioral changes against an independent oracle at the layer that can
  expose the failure. Do not loosen score tolerances or overwrite a saved
  LightGBM reference merely to make a check pass.
- Run `cargo fmt --package baiez` for Rust formatting and Ruff for Python. The
  package selector keeps rustfmt out of the sibling yesno checkout. Avoid
  machine-specific paths in scripts, CI, and documented commands.
- Agent notes should explain decisions and evidence. Append to `JOURNAL.md`;
  edit overview, architecture, gate, test, and LTM documents when their durable
  facts change. Keep unresolved work in `TODO.md`.
- Do not delete a user's files or make discretionary commits. Review the actual
  change set when a Git checkout is available; this directory may also be an
  uninitialized source tree.

## Verification by change type

- Rust or CLI behavior: `./scripts/gate.sh`, plus a relevant integration test.
- Peer socket or replica behavior: the routine gate and `./scripts/gate-e2e.sh`.
- Python adapter behavior: the gate and a built-extension workflow check with
  an available LightGBM fixture.
- Stored format or index arithmetic: the gate, bundle round trip, and score
  parity over missing, zero, contiguous, and scattered rows.
- Performance claim: follow `.agents/docs/TESTING.md`, save enough evidence to
  reproduce the comparison, and update `BENCHMARKS.md` only after measurement.
- Release metadata: run the gate and inspect built wheel, sdist, and Cargo
  package file lists as described in `RELEASING.md`.
