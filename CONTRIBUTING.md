# Contributing to baiez

Read [AGENTS.md](AGENTS.md) for the architecture, testing, and agent workflow
map. The local check for ordinary changes is:

```console
./scripts/gate.sh
```

The gate checks Rust formatting, Clippy, all-feature tests, Python formatting
and lint, a built-extension workflow, saved LightGBM score parity, and the
Cargo archive file list. Its prerequisites are Rust 1.95 or newer, the sibling
`../yesno` checkout, Python,
Ruff or `uv`, and NumPy or `uv` with cached NumPy 2.5.3.
For score or performance changes, use the real LightGBM fixture procedure in
[.agents/docs/TESTING.md](.agents/docs/TESTING.md). A pull request should state
the behavior changed, the independent test or oracle used, the commands run,
and any remaining limitation.

Do not replace saved oracle scores or raise score tolerances to make a failure
pass. Keep unrelated changes intact in shared working trees.
