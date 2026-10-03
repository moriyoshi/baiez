// SPDX-License-Identifier: Apache-2.0
//! Batch inference for LightGBM trees using yesnodb posting lists.
//!
//! A blocked packed view under one database key holds a live-row set and one
//! left-branch set per distinct split predicate. The model and the view
//! descriptor can also be stored as a versioned model bundle in yesnodb.

mod bundle;
#[cfg(feature = "flight")]
mod flight;
mod index;
mod model;
mod namespace;
#[cfg(target_os = "linux")]
mod peer;
#[cfg(feature = "python")]
mod python;

pub use bundle::ModelBundle;
#[cfg(feature = "flight")]
pub use flight::{load_flight_bundle, load_flight_model};
pub use index::{IndexedRow, PackedIndex, PreparedIndex};
pub use model::{Model, PredictOptions, Prediction};
pub use namespace::KeyNamespace;
#[cfg(target_os = "linux")]
pub use peer::{load_peer_bundle, load_peer_model};

#[cfg(not(target_os = "linux"))]
pub fn load_peer_bundle(
    _path: impl AsRef<std::path::Path>,
    _key: u64,
) -> Result<(ModelBundle, PreparedIndex)> {
    Err(Error::Unsupported(
        "yesno peer sockets require Linux".into(),
    ))
}

#[cfg(not(target_os = "linux"))]
pub fn load_peer_model(_path: impl AsRef<std::path::Path>, _key: u64) -> Result<ModelBundle> {
    Err(Error::Unsupported(
        "yesno peer sockets require Linux".into(),
    ))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid LightGBM model: {0}")]
    InvalidModel(String),
    #[error("invalid packed index: {0}")]
    InvalidIndex(String),
    #[error("invalid model bundle: {0}")]
    InvalidBundle(String),
    #[error("invalid key namespace: {0}")]
    InvalidNamespace(String),
    #[error("unsupported LightGBM feature: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yesno(#[from] yesno_core::CodecError),
    #[error("yesno peer: {0}")]
    Peer(String),
    #[error("yesno peer status {status}: {message}")]
    PeerStatus { status: u32, message: String },
    #[error("yesno Flight: {0}")]
    Flight(String),
}

pub type Result<T> = std::result::Result<T, Error>;
