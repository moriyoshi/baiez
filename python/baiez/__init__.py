# SPDX-License-Identifier: Apache-2.0
"""Python access to baiez's yesnodb backed LightGBM inference."""

from .lightgbm import Booster, IndexedRows

__all__ = ["Booster", "IndexedRows"]
