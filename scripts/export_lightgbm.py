#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a LightGBM model file to the JSON consumed by baiez.

Requires the official `lightgbm` Python package, which uses LightGBM's native
library to load and dump the model. It is an import-time tool, not a scorer.
"""

import json
import sys

import lightgbm as lgb


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_lightgbm.py MODEL.txt MODEL.json")
    booster = lgb.Booster(model_file=sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        json.dump(booster.dump_model(), output, allow_nan=False)


if __name__ == "__main__":
    main()
