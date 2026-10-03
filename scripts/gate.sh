#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the routine baiez checks from any working directory.
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

if (( $# != 0 )); then
    printf 'usage: %s\n' './scripts/gate.sh' >&2
    exit 2
fi

fail=0
steps_run=0
expected_steps=8

step() {
    steps_run=$((steps_run + 1))
    printf '\n== %s\n' "$1"
    shift
    if "$@"; then
        printf '   ok\n'
    else
        printf '   FAILED\n'
        fail=1
    fi
}

python_quality() {
    if command -v ruff >/dev/null 2>&1; then
        ruff check python scripts && ruff format --check python scripts
    elif command -v uvx >/dev/null 2>&1; then
        uvx --offline --from ruff==0.16.10 ruff check python scripts &&
            uvx --offline --from ruff==0.16.10 ruff format --check python scripts
    else
        printf 'ruff or uvx is required\n' >&2
        return 1
    fi
}

run_with_numpy() {
    if python3 -c 'import numpy' >/dev/null 2>&1; then
        python3 "$@"
    elif command -v uv >/dev/null 2>&1; then
        uv run --offline --no-project --with numpy==2.5.3 python "$@"
    else
        printf 'NumPy or uv with cached NumPy 2.5.3 is required\n' >&2
        return 1
    fi
}

python_workflow() {
    cargo build --offline --features python --lib --bin baiez || return 1
    run_with_numpy scripts/check_python_workflow.py || return 1
    run_with_numpy scripts/run_python_flight_test.py
}

archive_list() {
    local listing
    listing=$(cargo package --offline --allow-dirty --list) || return 1
    local required
    for required in LICENSE NOTICE README.md RELEASING.md third_party/yesno/NOTICE; do
        if ! grep -Fxq "$required" <<< "$listing"; then
            printf 'missing from Cargo archive: %s\n' "$required" >&2
            return 1
        fi
    done
    if grep -Eq '(^|/)(__pycache__|\.agents|\.agents-workspace|target)/|\.pyc$|^reports/' <<< "$listing"; then
        printf 'generated or agent-only content entered Cargo archive\n' >&2
        return 1
    fi
    printf '%s\n' "$listing" | tail -n 5
}

step 'Rust formatting' cargo fmt --package baiez -- --check
step 'Rust Clippy, all targets and features' cargo clippy --offline --all-targets --all-features -- -D warnings
step 'Rust tests, all features' cargo test --offline --all-features
step 'Python lint and formatting' python_quality
step 'Local documentation links' python3 scripts/check_doc_links.py
step 'Python CLI-to-index workflow' python_workflow
step 'Saved LightGBM score parity' run_with_numpy scripts/attest_workflow.py tests/fixtures/lightgbm_regression_1024
step 'Cargo archive contents' archive_list

if (( steps_run != expected_steps )); then
    printf 'INCOMPLETE: ran %d of %d steps\n' "$steps_run" "$expected_steps" >&2
    fail=1
fi

if (( fail == 0 )); then
    printf '\ngate passed\n'
else
    printf '\ngate failed\n' >&2
fi
exit "$fail"
