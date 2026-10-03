// SPDX-License-Identifier: Apache-2.0
//! Version-consistent model bundle reads over yesno's Arrow Flight API.

use arrow_array::UInt64Array;
use futures::TryStreamExt;
use std::future::Future;
use yesno_core::OrdSet;
use yesno_flight::{SetExpr, YesnoClient};

use crate::{Error, ModelBundle, PreparedIndex, Result};

fn flight_error(error: impl std::fmt::Display) -> Error {
    Error::Flight(error.to_string())
}

fn run_flight<T: Send + 'static>(
    future: impl Future<Output = Result<T>> + Send + 'static,
) -> Result<T> {
    std::thread::Builder::new()
        .name("baiez-flight".into())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(flight_error)?
                .block_on(future)
        })
        .map_err(flight_error)?
        .join()
        .map_err(|_| Error::Flight("Flight worker thread panicked".into()))?
}

async fn collect_ordinals<S>(mut batches: S) -> Result<Vec<u64>>
where
    S: futures::TryStream<Ok = arrow_array::RecordBatch> + Unpin,
    S::Error: std::fmt::Display,
{
    let mut ordinals = Vec::new();
    while let Some(batch) = batches.try_next().await.map_err(flight_error)? {
        let column = batch
            .column_by_name("ordinal")
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| {
                Error::Flight("yesno Flight response has no UInt64 ordinal column".into())
            })?;
        ordinals.extend((0..column.len()).map(|index| column.value(index)));
    }
    Ok(ordinals)
}

/// Load a model bundle and its packed index from a yesno Flight endpoint.
///
/// The metadata key determines a database version; the adjacent index key is
/// then queried at that exact version so both pieces belong to one generation.
pub fn load_flight_bundle(
    endpoint: impl Into<String>,
    key: u64,
) -> Result<(ModelBundle, PreparedIndex)> {
    let endpoint = endpoint.into();
    run_flight(async move {
        let mut client = YesnoClient::connect(endpoint).await.map_err(flight_error)?;
        let metadata_query = client.prepare_key(key).await.map_err(flight_error)?;
        let version = metadata_query.version();
        let index_key = key
            .checked_add(1)
            .ok_or_else(|| Error::InvalidBundle("metadata key cannot be u64::MAX".into()))?;
        // Plan both reads while the metadata ticket still pins this version.
        // Fetching metadata first could let retention reclaim the version before
        // the second query is planned on a busy server.
        let index_query = client
            .prepare_query_at(&SetExpr::Key(index_key), version)
            .await
            .map_err(flight_error)?;
        let metadata =
            collect_ordinals(client.fetch(&metadata_query).await.map_err(flight_error)?).await?;
        let metadata_set = OrdSet::from_sorted_slice(&metadata);
        let bundle = ModelBundle::from_set(&metadata_set, key)?;
        let descriptor = bundle
            .index()
            .ok_or_else(|| Error::InvalidBundle("model bundle has no packed index".into()))?;
        if descriptor.key != index_key {
            return Err(Error::InvalidBundle("index key mismatch".into()));
        }
        let packed =
            collect_ordinals(client.fetch(&index_query).await.map_err(flight_error)?).await?;
        let packed_set = OrdSet::from_sorted_slice(&packed);
        let prepared = descriptor.prepare_from_set(&packed_set, bundle.model())?;
        Ok((bundle, prepared))
    })
}

/// Load only model metadata over Flight, without fetching its packed index.
pub fn load_flight_model(endpoint: impl Into<String>, key: u64) -> Result<ModelBundle> {
    let endpoint = endpoint.into();
    run_flight(async move {
        let mut client = YesnoClient::connect(endpoint).await.map_err(flight_error)?;
        let query = client.prepare_key(key).await.map_err(flight_error)?;
        let metadata = collect_ordinals(client.fetch(&query).await.map_err(flight_error)?).await?;
        ModelBundle::from_set(&OrdSet::from_sorted_slice(&metadata), key)
    })
}
