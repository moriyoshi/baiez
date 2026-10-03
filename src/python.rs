// SPDX-License-Identifier: Apache-2.0
use std::path::Path;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use yesno_core::Db;

use crate::{load_flight_bundle, load_peer_bundle};
use crate::{KeyNamespace, ModelBundle, PredictOptions, PreparedIndex};

type PredictionBytes = (Py<PyBytes>, Option<Py<PyBytes>>, usize, usize);

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A prepared, immutable view of one model/index generation.
#[pyclass(name = "Index")]
struct PyIndex {
    prepared: PreparedIndex,
    dump_json: String,
    native_text: Option<String>,
    output_count: usize,
    num_features: usize,
    num_iterations: usize,
    trees_per_iteration: usize,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (db_path, key, *, namespace_bits=None, namespace_id=None))]
    fn new(
        db_path: &str,
        key: u64,
        namespace_bits: Option<u8>,
        namespace_id: Option<u32>,
    ) -> PyResult<Self> {
        let key = resolve_key(key, namespace_bits, namespace_id)?;
        let db = Db::open(Path::new(db_path)).map_err(py_error)?;
        let snapshot = db.snapshot().map_err(py_error)?;
        let bundle = ModelBundle::load(&snapshot, key).map_err(py_error)?;
        let descriptor = bundle
            .index()
            .ok_or_else(|| PyValueError::new_err("model bundle has no packed index"))?;
        let prepared = descriptor
            .prepare(&snapshot, bundle.model())
            .map_err(py_error)?;
        Ok(Self::from_bundle(bundle, prepared))
    }

    #[staticmethod]
    #[pyo3(signature = (socket_path, key, *, namespace_bits=None, namespace_id=None))]
    fn from_peer(
        py: Python<'_>,
        socket_path: &str,
        key: u64,
        namespace_bits: Option<u8>,
        namespace_id: Option<u32>,
    ) -> PyResult<Self> {
        let key = resolve_key(key, namespace_bits, namespace_id)?;
        let path = socket_path.to_owned();
        let (bundle, prepared) = py
            .detach(move || load_peer_bundle(path, key))
            .map_err(py_error)?;
        Ok(Self::from_bundle(bundle, prepared))
    }

    #[staticmethod]
    #[pyo3(signature = (endpoint, key, *, namespace_bits=None, namespace_id=None))]
    fn from_flight(
        py: Python<'_>,
        endpoint: &str,
        key: u64,
        namespace_bits: Option<u8>,
        namespace_id: Option<u32>,
    ) -> PyResult<Self> {
        let key = resolve_key(key, namespace_bits, namespace_id)?;
        let endpoint = endpoint.to_owned();
        let (bundle, prepared) = py
            .detach(move || load_flight_bundle(endpoint, key))
            .map_err(py_error)?;
        Ok(Self::from_bundle(bundle, prepared))
    }

    #[getter]
    fn dump_json(&self) -> &str {
        &self.dump_json
    }

    #[getter]
    fn native_text(&self) -> Option<&str> {
        self.native_text.as_deref()
    }

    #[getter]
    fn output_count(&self) -> usize {
        self.output_count
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.num_features
    }

    #[getter]
    fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    /// Accept little-endian uint64 row IDs; return little-endian float64
    /// scores and uint64 leaf indices as bytes plus their row widths.
    #[pyo3(signature = (ids, *, raw_score=false, pred_leaf=false, start_iteration=0, num_iteration=None))]
    fn predict_ids(
        &self,
        py: Python<'_>,
        ids: &[u8],
        raw_score: bool,
        pred_leaf: bool,
        start_iteration: usize,
        num_iteration: Option<usize>,
    ) -> PyResult<PredictionBytes> {
        if !ids.len().is_multiple_of(8) {
            return Err(PyValueError::new_err(
                "row ID buffer length must be a multiple of 8",
            ));
        }
        let row_ids: Vec<u64> = ids
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let options = PredictOptions {
            start_iteration,
            num_iteration,
            raw_score,
            pred_leaf,
        };
        let selected_iterations = self
            .num_iterations
            .saturating_sub(start_iteration)
            .min(num_iteration.filter(|&n| n > 0).unwrap_or(usize::MAX));
        let tree_count = selected_iterations * self.trees_per_iteration;
        let mut scores = vec![0.0; row_ids.len() * self.output_count];
        let mut leaves = pred_leaf.then(|| vec![0; row_ids.len() * tree_count]);
        py.detach(|| {
            self.prepared
                .predict_into(&row_ids, options, &mut scores, leaves.as_deref_mut())
        })
        .map_err(py_error)?;
        let mut score_bytes = Vec::with_capacity(scores.len() * 8);
        for score in scores {
            score_bytes.extend_from_slice(&score.to_le_bytes());
        }
        let leaf_bytes = leaves.map(|values| {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&(value as u64).to_le_bytes());
            }
            PyBytes::new(py, &bytes).unbind()
        });
        Ok((
            PyBytes::new(py, &score_bytes).unbind(),
            leaf_bytes,
            self.output_count,
            tree_count,
        ))
    }
}

fn resolve_key(key: u64, namespace_bits: Option<u8>, namespace_id: Option<u32>) -> PyResult<u64> {
    match (namespace_bits, namespace_id) {
        (None, None) => Ok(key),
        (Some(bits), Some(id)) => KeyNamespace::new(id, bits)
            .and_then(|namespace| namespace.model_key(key))
            .map_err(|error| PyValueError::new_err(error.to_string())),
        _ => Err(PyValueError::new_err(
            "namespace_bits and namespace_id must be provided together",
        )),
    }
}

impl PyIndex {
    fn from_bundle(bundle: ModelBundle, prepared: PreparedIndex) -> Self {
        Self {
            prepared,
            dump_json: bundle.dump_json().to_owned(),
            native_text: bundle.native_text().map(str::to_owned),
            output_count: bundle.model().output_count(),
            num_features: bundle.model().num_features(),
            num_iterations: bundle.model().num_iterations(),
            trees_per_iteration: bundle.model().output_count(),
        }
    }
}

#[pymodule]
fn _baiez(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    Ok(())
}
