# baiez overview

baiez is an experimental LightGBM inference engine over yesnodb. It imports a
LightGBM model dump, indexes a fixed collection of rows by split predicate,
and scores row-ID cohorts with yesno set operations. The packed index stores
split outcomes, not feature vectors. See the root README for commands and the
current feature list.

The project exposes a Rust library, a `baiez` CLI, and a Python extension with
an opt-in `baiez.lightgbm.Booster` adapter. The Python adapter delegates
ordinary feature-matrix prediction to LightGBM and uses yesnodb for indexed
row-ID prediction. Arrow Flight is currently a benchmark example, not a
production service API. A Linux yesnod plugin peer socket can supply a
snapshot-pinned model and index from a replicating follower sidecar; scoring
then uses local owned sets.
The separate `e2e/` harness checks real leader/follower processes, rollout,
and follower restart through that socket.

The source checkout expects `../yesno/yesno-core` and
`../yesno/yesno-plugin`. The Python distribution can bundle these path
dependencies in its source archive. Publishing the Rust crate requires both
matching versions on crates.io first.

## Where to look

- [ARCHITECTURE.md](ARCHITECTURE.md): persisted layout and module boundaries.
- [TESTING.md](TESTING.md): independent score checks and measurement method.
- [QUALITY_GATE.md](QUALITY_GATE.md): local and CI checks.
- [JOURNAL.md](JOURNAL.md): dated findings and decisions.
- [TODO.md](TODO.md): open work.
- [LTM/INDEX.md](LTM/INDEX.md): consolidated topic notes as they are created.
