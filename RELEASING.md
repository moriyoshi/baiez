# Releasing baiez

baiez is published under Apache-2.0. The source distribution includes
`LICENSE` and `NOTICE`; package metadata in `Cargo.toml` and `pyproject.toml`
must use the same license and version.

## Prerequisites

1. Publish `yesno-core` and `yesno-plugin` 0.1.0 to crates.io before publishing
   the baiez Rust crate. baiez uses the sibling checkout for local development
   and resolves these versions from crates.io when packaged. `cargo package`
   cannot complete until both registry versions exist. Python wheels and source
   distributions can be built earlier: maturin includes local dependencies in
   its source distribution. The `yesno-plugin` manifest must also give its
   `yesno-core` dependency a registry version before publishing that crate.
2. Confirm that the configured public repository exists and that the `baiez`
   name is available under the intended owners on both package registries.
3. Review the optional LightGBM C API import on a machine with a real
   LightGBM shared library. The JSON import and indexed scoring paths have
   been checked against saved LightGBM scores.

## Checks

Run the routine local and CI gate first:

```console
./scripts/gate.sh
./scripts/gate-e2e.sh
```

It scopes formatting to baiez because `cargo fmt --all` also visits the
path dependency in the sibling yesno checkout. Then perform release-specific
checks:

```console
cargo package --list
cargo package
```

`cargo package --list` should contain `LICENSE`, `NOTICE`, `README.md`,
`BENCHMARKS.md`, and the `third_party/yesno` notices, and should exclude
bytecode caches and the local reports.
The full `cargo package` command validates the archive using the published
`yesno-core` and `yesno-plugin` dependencies.

Build and inspect Python artifacts with maturin 1.9.3 or newer:

```console
maturin build --release --out dist
maturin sdist --out dist
```

Check that the wheel and source distribution contain the Apache license and
the original yesnodb notices, and that a fresh environment can install each
artifact, import `baiez`, and score indexed rows. The optional `lightgbm`
extra supplies ordinary
`Booster.predict(X)` support.
Build Linux release wheels in a manylinux environment suited to the intended
minimum glibc version; a wheel built directly on this host was tagged
`manylinux_2_34_aarch64`.

Update both package versions together before publishing. Uploads to crates.io
and PyPI are separate actions and should follow review of the built artifacts.
