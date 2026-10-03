# Real-process E2E scenarios

This harness follows haiiie's Rust-hosted Monty design. Checked-in `.py`
scenarios run inside Monty, without CPython, `uv`, NumPy, or Python binding
access. Monty gives scripts no filesystem, process, environment, or network
access. Rust host verbs alone start `yesnod`, call the `yesno` Flight CLI,
invoke baiez's CLI and peer reader, and verify model values.

From the baiez repository root:

```console
./scripts/gate-e2e.sh
./scripts/gate-e2e.sh --list
./scripts/gate-e2e.sh --show-output replica_rollout
./scripts/gate-e2e.sh --timeout 240 replica_restart
```

The script builds baiez, `yesnod`, and `yesno`, tests the scenario parser, and
runs all scenarios. `YESNOD_BIN`, `YESNO_BIN`, and `BAIEZ_BIN` can select
existing binaries. Every scenario gets fresh directories and ports; children
are reaped after success or failure. Logs and configs remain under
`.agents-workspace/tmp/e2e/`. Socket paths use a short system temporary
directory to fit Linux's Unix socket path limit. The default deadline is 180
seconds per scenario. Scenarios without an `assert` or a host call fail.

`replica_rollout` checks follower bootstrap and write refusal, CLI predictions
and model dump, an atomic model generation published through Flight, and the
stability of an already prepared peer snapshot. `replica_restart` stops the
follower, publishes another generation while it is down, then checks reopen
and WAL catch-up. `leader_peer_restart` checks peer reads directly from the
authoritative owner, including scores, leaf indices, model dump, owner reopen,
and a prepared snapshot surviving restart. `missing_model` checks that an
absent peer key fails closed and does not affect later reads of a valid bundle.
`lightgbm_replica` replicates the checked-in 16-tree LightGBM regression model
and indexed 1,024-row fixture, then compares peer predictions with saved native
scores across contiguous, scattered, shuffled, and reversed row orders.
`replica_fanout` starts two followers under one leader, checks their initial
model view, publishes a generation, and verifies both followers catch up while
retaining the original model. `namespace_replica` confirms namespaced keys are
read correctly after leader/follower replication. The routine gate tests the
Python binding separately.

The local fixture uses loopback HTTP and yesno's explicit insecure replication
option. It tests process boundaries and replication behavior; it does not test
TLS, deployment charts, or a remote host. The routine Rust gate separately
tests arena and inline peer transport modes.
