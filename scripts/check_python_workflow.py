#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the built PyO3 adapter against a CLI-loaded indexed bundle.

Run after ``cargo build --features python --lib --bin baiez`` with NumPy in
the active Python environment. This fixture needs no LightGBM installation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "target/debug/baiez"
EXTENSION = ROOT / "target/debug/libbaiez.so"
MODEL = ROOT / "tests/fixtures/simple_model.json"


def import_adapter():
    spec = importlib.util.spec_from_file_location("baiez._baiez", EXTENSION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load built extension at {EXTENSION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["baiez._baiez"] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT / "python"))
    import baiez.lightgbm as lgb

    return lgb


def expect_error(error_type, action):
    try:
        action()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def main() -> None:
    lgb = import_adapter()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = root / "rows.json"
        rows.write_text(
            json.dumps(
                [
                    {"id": 7, "features": [0.0]},
                    {"id": 9, "features": [3.0]},
                ]
            ),
            encoding="utf-8",
        )
        db = root / "db"
        subprocess.run(
            [
                str(CLI),
                "model",
                "load",
                "--db",
                str(db),
                "--key",
                "9000",
                "--input",
                str(MODEL),
                "--format",
                "json",
                "--rows",
                str(rows),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        booster = lgb.Booster.from_yesno(str(db), 9000)
        scores = booster.predict_ids(np.array([9, 7], dtype=np.int64), raw_score=True)
        np.testing.assert_array_equal(scores, np.array([4.0, 2.0]))
        assert scores.shape == (2,)
        leaves = booster.predict_ids([9, 7], pred_leaf=True)
        np.testing.assert_array_equal(leaves, np.array([[1], [0]], dtype=np.uint64))
        assert leaves.shape == (2, 1)
        np.testing.assert_array_equal(
            booster.predict(lgb.IndexedRows([7, 9])), np.array([2.0, 4.0])
        )
        assert booster.dump_model() == json.loads(MODEL.read_text(encoding="utf-8"))
        expect_error(RuntimeError, lambda: booster.predict_ids([42]))
        expect_error(ValueError, lambda: booster.predict_ids([-1]))
        expect_error(
            NotImplementedError,
            lambda: booster.predict_ids([7], pred_contrib=True),
        )

        namespaced_db = root / "namespaced-db"
        subprocess.run(
            [
                str(CLI),
                "model",
                "load",
                "--db",
                str(namespaced_db),
                "--key",
                "42",
                "--namespace-bits",
                "8",
                "--namespace-id",
                "3",
                "--input",
                str(MODEL),
                "--format",
                "json",
                "--rows",
                str(rows),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        namespaced = lgb.Booster.from_yesno(
            str(namespaced_db), 42, namespace_bits=8, namespace_id=3
        )
        np.testing.assert_array_equal(
            namespaced.predict_ids([9, 7]), np.array([4.0, 2.0])
        )
        namespaced.bind_index(str(namespaced_db), 42, namespace_bits=8, namespace_id=3)
        expect_error(
            ValueError,
            lambda: lgb.Booster.from_yesno(str(namespaced_db), 42, namespace_bits=8),
        )
        expect_error(
            RuntimeError,
            lambda: lgb.Booster.from_yesno(
                str(namespaced_db), 42, namespace_bits=8, namespace_id=4
            ),
        )
    print("Python CLI-to-index workflow passed")


if __name__ == "__main__":
    main()
