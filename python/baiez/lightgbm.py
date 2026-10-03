# SPDX-License-Identifier: Apache-2.0
"""LightGBM Booster adapter with explicit indexed-row prediction.

``Booster.predict(X)`` delegates to LightGBM for ordinary feature matrices.
``Booster.predict(IndexedRows(ids))`` uses a prepared yesnodb index.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ._baiez import Index


class IndexedRows:
    """Mark stable row IDs that were stored in a baiez packed index."""

    def __init__(self, ids: Any) -> None:
        self.ids = ids


class Booster:
    """A LightGBM Booster adapter with an opt-in yesnodb fast path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        native = importlib.import_module("lightgbm")
        self._native = native.Booster(*args, **kwargs)
        self._index: Index | None = None
        self._source_native_text: str | None = None
        if kwargs.get("model_file") is not None:
            self._source_native_text = Path(kwargs["model_file"]).read_text(
                encoding="utf-8"
            )
        elif kwargs.get("model_str") is not None:
            self._source_native_text = kwargs["model_str"]

    @classmethod
    def from_yesno(
        cls,
        db_path: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Open an indexed bundle; LightGBM is optional for ``predict_ids``."""
        obj = cls.__new__(cls)
        obj._index = Index(
            str(db_path),
            key,
            namespace_bits=namespace_bits,
            namespace_id=namespace_id,
        )
        obj._source_native_text = obj._index.native_text
        obj._native = None
        if obj._source_native_text is not None:
            try:
                native = importlib.import_module("lightgbm")
            except ImportError:
                pass
            else:
                obj._native = native.Booster(model_str=obj._source_native_text)
        return obj

    @classmethod
    def from_peer(
        cls,
        socket_path: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Load an indexed bundle through a local yesnod peer socket."""
        obj = cls.__new__(cls)
        obj._index = Index.from_peer(
            str(socket_path),
            key,
            namespace_bits=namespace_bits,
            namespace_id=namespace_id,
        )
        obj._source_native_text = obj._index.native_text
        obj._native = None
        if obj._source_native_text is not None:
            try:
                native = importlib.import_module("lightgbm")
            except ImportError:
                pass
            else:
                obj._native = native.Booster(model_str=obj._source_native_text)
        return obj

    @classmethod
    def from_flight(
        cls,
        endpoint: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Load an indexed bundle from a yesno Arrow Flight endpoint."""
        obj = cls.__new__(cls)
        obj._index = Index.from_flight(
            str(endpoint),
            key,
            namespace_bits=namespace_bits,
            namespace_id=namespace_id,
        )
        obj._source_native_text = obj._index.native_text
        obj._native = None
        if obj._source_native_text is not None:
            try:
                native = importlib.import_module("lightgbm")
            except ImportError:
                pass
            else:
                obj._native = native.Booster(model_str=obj._source_native_text)
        return obj

    def bind_index(
        self,
        db_path: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Attach an indexed bundle after checking that its model matches."""
        index = Index(
            str(db_path),
            key,
            namespace_bits=namespace_bits,
            namespace_id=namespace_id,
        )
        return self._bind_prepared(index)

    def bind_peer(
        self,
        socket_path: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Attach an indexed bundle from a local yesnod peer socket."""
        return self._bind_prepared(
            Index.from_peer(
                str(socket_path),
                key,
                namespace_bits=namespace_bits,
                namespace_id=namespace_id,
            )
        )

    def bind_flight(
        self,
        endpoint: str,
        key: int,
        *,
        namespace_bits: int | None = None,
        namespace_id: int | None = None,
    ) -> Booster:
        """Attach an indexed bundle from a yesno Arrow Flight endpoint."""
        return self._bind_prepared(
            Index.from_flight(
                str(endpoint),
                key,
                namespace_bits=namespace_bits,
                namespace_id=namespace_id,
            )
        )

    def _bind_prepared(self, index: Index) -> Booster:
        if self._source_native_text is not None and index.native_text is not None:
            if self._source_native_text != index.native_text:
                raise ValueError("yesnodb bundle contains a different LightGBM model")
        elif self._native is not None and self._native.dump_model() != json.loads(
            index.dump_json
        ):
            raise ValueError("yesnodb bundle contains a different LightGBM model")
        self._index = index
        return self

    def predict_ids(
        self,
        row_ids: Any,
        *,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
    ) -> Any:
        """Predict indexed row IDs, returning LightGBM-shaped NumPy arrays."""
        import numpy as np

        if self._index is None:
            raise RuntimeError("no yesnodb index is bound")
        if pred_contrib:
            raise NotImplementedError("indexed SHAP contributions are not supported")
        ids = np.asarray(row_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise TypeError("row IDs must be a one-dimensional integer array")
        if ids.dtype.kind == "i" and np.any(ids < 0):
            raise ValueError("row IDs must be nonnegative")
        if start_iteration < 0 or (num_iteration is not None and num_iteration < 0):
            raise ValueError("iteration values must be nonnegative")
        if num_iteration is None and start_iteration == 0 and self._native is not None:
            best = self._native.best_iteration
            if best is not None and best > 0:
                num_iteration = best
        encoded = np.ascontiguousarray(ids, dtype="<u8").tobytes()
        scores, leaves, outputs, tree_count = self._index.predict_ids(
            encoded,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
        )
        if pred_leaf:
            return np.frombuffer(leaves, dtype="<u8").reshape((ids.size, tree_count))
        result = np.frombuffer(scores, dtype="<f8").reshape((ids.size, outputs))
        return result[:, 0] if outputs == 1 else result

    def predict(
        self,
        data: Any,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> Any:
        if isinstance(data, IndexedRows):
            if data_has_header or validate_features or kwargs:
                raise ValueError("feature-matrix options do not apply to IndexedRows")
            return self.predict_ids(
                data.ids,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                raw_score=raw_score,
                pred_leaf=pred_leaf,
                pred_contrib=pred_contrib,
            )
        if self._native is None:
            raise RuntimeError(
                "ordinary predict(X) requires the lightgbm package and a native model"
            )
        return self._native.predict(
            data,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
            **kwargs,
        )

    def dump_model(self, *args: Any, **kwargs: Any) -> Any:
        if self._native is not None:
            return self._native.dump_model(*args, **kwargs)
        if args or kwargs:
            raise ValueError("dump_model options require the lightgbm package")
        if self._index is None:
            raise RuntimeError("no model is loaded")
        return json.loads(self._index.dump_json)

    def model_to_string(self, *args: Any, **kwargs: Any) -> str:
        if self._native is not None:
            return self._native.model_to_string(*args, **kwargs)
        if args or kwargs or self._index is None or self._index.native_text is None:
            raise RuntimeError("native model text is unavailable")
        return self._index.native_text

    def __getattr__(self, name: str) -> Any:
        native = self.__dict__.get("_native")
        if native is None:
            raise AttributeError(name)
        attribute = getattr(native, name)
        if name in {
            "update",
            "refit",
            "rollback_one_iter",
            "shuffle_models",
            "reset_parameter",
            "set_leaf_output",
        } and callable(attribute):

            def invalidate(*args: Any, **kwargs: Any) -> Any:
                result = attribute(*args, **kwargs)
                self._index = None
                return result

            return invalidate
        return attribute


def __getattr__(name: str) -> Any:
    """Expose the remaining LightGBM API for ordinary project imports."""
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(importlib.import_module("lightgbm"), name)
