---
name: write-tests
description: Design or add baiez regression tests using the layer that can catch model, packed-index, bundle, CLI, Python, or performance failures.
---

# Write baiez tests

Read `.agents/docs/ARCHITECTURE.md` and `.agents/docs/TESTING.md`, then the
changed implementation and existing tests. State the concrete failure being
guarded and choose a layer that exercises the production path.

- Use saved official LightGBM scores for prediction semantics and objective
  parity. Cover missing and zero values when split behavior changes.
- Use `tests/packed.rs` for ordinal layout and cohort behavior, including
  contiguous and scattered order, duplicate and unknown IDs.
- Use `tests/bundle.rs` for metadata framing, version or checksum rejection,
  atomic model/index storage, and reopen behavior.
- Use `tests/cli.rs` for command contracts and actual load/dump/predict flows.
- Exercise a built extension for Python shape and binding changes. A Rust-only
  unit test cannot prove the adapter's NumPy result shape.

Prefer a focused test with an independent expected result over repeating the
same algorithm in the test. If practical, make the test fail on a temporary
implementation fault, undo the fault, then run the test and `./scripts/gate.sh`.
Record any new durable failure mode in `.agents/docs/JOURNAL.md`.
