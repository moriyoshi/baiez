#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a real LightGBM fixture and time Booster.predict on fixed cohorts.

Run with a Python environment containing lightgbm and numpy:
    python scripts/measure_lightgbm.py /tmp/baiez-lightgbm-measure
"""

import ctypes
import json
import pathlib
import statistics
import sys
import time

import lightgbm as lgb
import lightgbm.basic as lgb_basic
import numpy as np


def cohorts(rows: int):
    for size in (1, 64, 1024, 8192, rows):
        if size > rows:
            continue
        begin = (rows - size) // 2
        yield "contiguous", size, np.arange(begin, begin + size)
        if 1 < size < rows:
            scattered = np.arange(size, dtype=np.int64) * rows // size
            yield "scattered", size, scattered
            if size == 8192:
                yield (
                    "scattered_shuffled",
                    size,
                    scattered[(np.arange(size) * 4051) % size],
                )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: measure_lightgbm.py OUTPUT_DIR")
    output = pathlib.Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    rows = 100_000
    rng = np.random.default_rng(42)
    features = rng.random((rows, 8), dtype=np.float64)
    row_index = np.arange(rows)[:, None]
    feature_index = np.arange(8)[None, :]
    features[(row_index + feature_index) % 97 == 0] = np.nan
    features[(row_index + feature_index) % 89 == 0] = 0.0
    filled = np.nan_to_num(features, nan=0.0)
    labels = (
        2.0 * filled[:, 0]
        + np.sin(4.0 * filled[:, 1])
        - 0.7 * filled[:, 2] * filled[:, 3]
        + 0.2 * filled[:, 4]
    )
    dataset = lgb.Dataset(features, label=labels)
    booster = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "learning_rate": 0.1,
            "num_threads": 1,
            "seed": 7,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=16,
    )
    booster.save_model(str(output / "model.txt"))
    with (output / "model.json").open("w", encoding="utf-8") as stream:
        json.dump(booster.dump_model(), stream, allow_nan=False)
    features.astype("<f8", copy=False).tofile(output / "rows.f64")
    native_all = booster.predict(features, raw_score=True, num_threads=1)
    native_all.astype("<f8", copy=False).tofile(output / "native_scores.f64")
    print(
        f"lightgbm={lgb.__version__} rows={rows} trees={booster.num_trees()} "
        f"features={features.shape[1]}"
    )
    print("shape,rows,native_python_us,native_c_api_us")
    for shape, size, ids in cohorts(rows):
        matrix = np.ascontiguousarray(features[ids])
        # Warm the native predictor and keep array selection out of the timer.
        predicted = booster.predict(matrix, raw_score=True, num_threads=1)
        if not np.array_equal(predicted, native_all[ids]):
            raise AssertionError("native cohort result differs from full prediction")
        c_output = np.empty(size, dtype=np.float64)
        c_length = ctypes.c_int64()
        matrix_pointer = ctypes.c_void_p(matrix.ctypes.data)
        output_pointer = c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        def c_predict(
            matrix_pointer=matrix_pointer,
            size=size,
            c_length=c_length,
            output_pointer=output_pointer,
        ):
            status = lgb_basic._LIB.LGBM_BoosterPredictForMat(
                booster._handle,
                matrix_pointer,
                lgb_basic._C_API_DTYPE_FLOAT64,
                size,
                8,
                1,
                lgb_basic._C_API_PREDICT_RAW_SCORE,
                0,
                -1,
                b"num_threads=1",
                ctypes.byref(c_length),
                output_pointer,
            )
            if status != 0 or c_length.value != size:
                raise RuntimeError("LightGBM C API prediction failed")

        c_predict()
        if not np.array_equal(c_output, predicted):
            raise AssertionError("direct C API result differs from Booster.predict")
        repetitions = (
            20 if size <= 64 else 10 if size <= 1024 else 5 if size <= 8192 else 2
        )
        python_samples = []
        c_samples = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(repetitions):
                result = booster.predict(matrix, raw_score=True, num_threads=1)
                if result.shape[0] != size:
                    raise AssertionError("wrong native result length")
            python_samples.append((time.perf_counter() - start) * 1e6 / repetitions)
            start = time.perf_counter()
            for _ in range(repetitions):
                c_predict()
            c_samples.append((time.perf_counter() - start) * 1e6 / repetitions)
        print(
            f"{shape},{size},{statistics.median(python_samples):.3f},"
            f"{statistics.median(c_samples):.3f}"
        )


if __name__ == "__main__":
    main()
