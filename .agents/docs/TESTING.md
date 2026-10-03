# Test and measurement layers

Choose a check that could fail if the changed behavior is wrong. A test of a
helper alone does not prove its production caller uses it.

| Failure class | Existing layer |
| --- | --- |
| LightGBM parsing, objective, and score semantics | `src/model.rs` tests and independent saved LightGBM scores in `scripts/attest_workflow.py` |
| Packed ordinal layout, cohort order, duplicate IDs, and missing IDs | `tests/packed.rs` and real-fixture attestation |
| Model/index atomic storage and format rejection | `tests/bundle.rs` |
| Namespace range encoding and bundle CLI isolation | `src/namespace.rs` tests and `tests/cli.rs` |
| CLI arguments and load/dump/predict workflow | `tests/cli.rs` |
| Follower peer socket, arena/inline lanes, CLI parity | `tests/peer.rs` |
| yesno Arrow Flight snapshot consistency and namespaced CLI prediction | `tests/flight.rs` |
| Live leader/follower bootstrap, model rollout, and restart | `e2e/` via `./scripts/gate-e2e.sh` |
| Python output shape, DB/Flight adapters, and binding | `scripts/check_python_workflow.py` plus the Python-enabled `tests/flight.rs` workflow invoking `scripts/check_python_flight.py`; `scripts/attest_workflow.py` checks real LightGBM parity |
| Latency or memory claim | `examples/measure*.rs`, `scripts/measure_workflow.py`, and recorded reports |

## LightGBM oracle

`scripts/measure_lightgbm.py` creates a real model, feature rows, and native
scores. The repository includes a hash-pinned 1,024-row slice at
`tests/fixtures/lightgbm_regression_1024/`; the routine gate uses it as an
independent score oracle. `scripts/attest_workflow.py` compares the CLI and
Python indexed path with those saved scores and checks JSON round trips,
separate loading, a leaf window, and unknown-row rejection. It needs NumPy.
Generate a new fixture only with a working official LightGBM install; do not
substitute baiez-generated scores as the reference.

```console
python3 scripts/measure_lightgbm.py .agents-workspace/tmp/lightgbm-fixture
cargo build --features python --bin baiez --lib
python3 scripts/attest_workflow.py .agents-workspace/tmp/lightgbm-fixture
```

The fixture generation line needs LightGBM and NumPy. The attestation line
uses saved scores and needs NumPy. The full 100,000-row run remains a separate
check for performance or broad compatibility work; the small fixture runs in
the routine gate.
The routine gate also runs a tiny CLI-to-Python indexed workflow with
hand-checked expected scores and leaves. That check verifies the binding and
array shapes without claiming parity on a real trained model. It also opens a
namespaced bundle through the Python adapter and checks namespace validation.

## Real-process peer checks

`./scripts/gate-e2e.sh` runs checked-in Monty scenarios through a Rust host
that starts separate yesnod leader and follower processes. It waits for a
completed first replication pass before loading a model, then checks follower
write refusal, live model rollout, follower restart, authoritative owner
reopen, leaf output, missing-key failure, and replicated LightGBM score parity
against the saved native fixture, plus fan-out to two followers. The runner
also checks namespaced key lookup through a follower peer socket. It keeps logs
and generated configs under `.agents-workspace/tmp/e2e/`.
See [the E2E guide](../../e2e/README.md) for selection and prerequisites. The
routine Rust peer test covers arena and inline lane modes without starting
yesnod; the E2E gate covers process boundaries and WAL catch-up.

## Measuring

Use [BENCHMARKS.md](../../BENCHMARKS.md) for existing workload definitions and
published numbers. Run release builds for latency comparisons. Record model
and binary hashes, row count, tree and predicate counts, CPU and affinity,
software versions, warmup and repetitions, and whether index construction or
snapshot preparation is timed. Compare outputs before timing. Keep CLI
process, Python call, Rust scorer, and Flight round trip separate; they
include different work. A benchmark that does not exercise the changed path
cannot support a speed claim.
