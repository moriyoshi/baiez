#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the release Python and CLI workflow on a LightGBM fixture.

    cargo build --release --offline --features python --lib --bin baiez
    taskset -c 19 python3 scripts/measure_workflow.py /tmp/baiez-lightgbm-measure

Requires NumPy. Timing excludes fixture JSON generation and model training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
from attest_workflow import check_scores, cohorts, write_rows

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / "target/release/baiez"
EXTENSION = ROOT / "target/release/libbaiez.so"


def adapter() -> Any:
    spec = importlib.util.spec_from_file_location("baiez._baiez", EXTENSION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXTENSION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["baiez._baiez"] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT / "python"))
    import baiez.lightgbm as lgb

    return lgb


def run_cli(*args: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(CLI), *(str(arg) for arg in args)], check=True, capture_output=True
    )


def timing(
    call: Callable[[], Any], repetitions: int, samples: int = 7
) -> dict[str, float]:
    call()
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            call()
        values.append((time.perf_counter_ns() - start) / repetitions / 1_000.0)
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocated_bytes(path: pathlib.Path) -> int:
    return sum(
        item.stat().st_blocks * 512 for item in path.rglob("*") if item.is_file()
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: measure_workflow.py FIXTURE_DIR")
    fixture = pathlib.Path(sys.argv[1])
    model_path = fixture / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    columns = model["max_feature_idx"] + 1
    features = np.fromfile(fixture / "rows.f64", dtype="<f8").reshape((-1, columns))
    reference = np.fromfile(fixture / "native_scores.f64", dtype="<f8")
    if len(reference) != len(features):
        raise AssertionError("feature and score counts differ")
    lgb = adapter()

    with tempfile.TemporaryDirectory(prefix="baiez-measure-") as workspace:
        work = pathlib.Path(workspace)
        db = work / "db"
        rows_json = work / "rows.json"
        ids_json = work / "ids.json"
        write_rows(rows_json, features)
        build_start = time.perf_counter_ns()
        run_cli(
            "model",
            "load",
            "--db",
            db,
            "--key",
            9000,
            "--input",
            model_path,
            "--format",
            "json",
            "--rows",
            rows_json,
        )
        build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0
        prepare_start = time.perf_counter_ns()
        booster = lgb.Booster.from_yesno(str(db), 9000)
        prepare_ms = (time.perf_counter_ns() - prepare_start) / 1_000_000.0

        measurements = []
        for name, ids in cohorts(len(features)):
            encoded = np.ascontiguousarray(ids, dtype="<u8").tobytes()
            direct = booster._index.predict_ids(encoded, raw_score=True)
            direct_scores = np.frombuffer(direct[0], dtype="<f8")
            check_scores(direct_scores, reference[ids], f"direct {name}")
            check_scores(
                booster.predict_ids(ids, raw_score=True),
                reference[ids],
                f"adapter {name}",
            )
            reps = (
                1000
                if len(ids) == 1
                else 500
                if len(ids) <= 64
                else (100 if len(ids) <= 1024 else 20 if len(ids) <= 8192 else 5)
            )
            raw = timing(
                lambda encoded=encoded: booster._index.predict_ids(
                    encoded, raw_score=True
                ),
                reps,
            )
            wrapped = timing(
                lambda ids=ids: booster.predict_ids(ids, raw_score=True), reps
            )
            marked = timing(
                lambda ids=ids: booster.predict(lgb.IndexedRows(ids), raw_score=True),
                reps,
            )
            measurement: dict[str, Any] = {
                "cohort": name,
                "rows": len(ids),
                "extension": raw,
                "predict_ids": wrapped,
                "predict_indexed_rows": marked,
            }
            if len(ids) in (1, 8192, len(features)):
                ids_json.write_text(json.dumps(ids.tolist()), encoding="utf-8")
                cmd = ("predict", "--db", db, "--key", 9000, "--ids", ids_json, "--raw")
                cli_output = json.loads(run_cli(*cmd).stdout)
                cli_scores = np.array([row["scores"][0] for row in cli_output])
                check_scores(cli_scores, reference[ids], f"CLI {name}")
                measurement["cli_process"] = timing(
                    lambda cmd=cmd: run_cli(*cmd), 1, samples=7
                )
            measurements.append(measurement)

        print(
            json.dumps(
                {
                    "fixture": str(fixture),
                    "model_sha256": digest(model_path),
                    "cli_sha256": digest(CLI),
                    "extension_sha256": digest(EXTENSION),
                    "machine": platform.machine(),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "rows": len(features),
                    "trees": len(model["tree_info"]),
                    "index_build_ms": build_ms,
                    "python_prepare_ms": prepare_ms,
                    "database_allocated_bytes": allocated_bytes(db),
                    "measurements": measurements,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
