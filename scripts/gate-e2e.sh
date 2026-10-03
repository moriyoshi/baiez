#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Real-process scenarios, separate from the routine library gate.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--list" ]]; then
    exec cargo run --offline -p baiez-e2e -- --list
fi

cargo build --offline --bin baiez
if [[ -z "${YESNOD_BIN:-}" || -z "${YESNO_BIN:-}" ]]; then
    cargo build --offline --manifest-path ../yesno/Cargo.toml -p yesno-server --bin yesnod --bin yesno
fi
cargo test --offline -p baiez-e2e
exec cargo run --offline -p baiez-e2e -- "$@"
