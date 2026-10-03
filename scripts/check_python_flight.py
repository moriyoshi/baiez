# SPDX-License-Identifier: Apache-2.0
"""Exercise the Python adapter against a live yesno Flight endpoint."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = Path(os.environ["BAIEZ_EXTENSION_PATH"])
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_python_flight.py ENDPOINT")
    lgb = import_adapter()
    endpoint = sys.argv[1]
    namespace = {"namespace_bits": 8, "namespace_id": 17}

    remote = lgb.Booster.from_flight(endpoint, 33, **namespace)
    np.testing.assert_array_equal(
        remote.predict_ids([127, 0, 7, 8], raw_score=True),
        np.array([4.0, 2.0, 4.0, 2.0]),
    )
    np.testing.assert_array_equal(
        remote.predict_ids([127, 0], pred_leaf=True),
        np.array([[1], [0]], dtype=np.uint64),
    )

    model: dict[str, Any] = json.loads(MODEL.read_text(encoding="utf-8"))
    fake_lightgbm = ModuleType("lightgbm")

    class NativeBooster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        @property
        def best_iteration(self) -> None:
            return None

        def dump_model(self) -> dict[str, Any]:
            return model

    fake_lightgbm.Booster = NativeBooster  # type: ignore[attr-defined]
    sys.modules["lightgbm"] = fake_lightgbm
    attached = lgb.Booster().bind_flight(endpoint, 33, **namespace)
    np.testing.assert_array_equal(
        attached.predict_ids([8, 7], raw_score=True), np.array([2.0, 4.0])
    )
    print("Python Flight binding passed")


if __name__ == "__main__":
    main()
