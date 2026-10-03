# SPDX-License-Identifier: Apache-2.0
"""Run the Flight integration test with this NumPy-enabled Python executable."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    environment = os.environ.copy()
    environment["BAIEZ_PYTHON_EXECUTABLE"] = sys.executable
    subprocess.run(
        ["cargo", "test", "--offline", "--features", "python", "--test", "flight"],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
