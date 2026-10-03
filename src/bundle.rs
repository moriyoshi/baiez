// SPDX-License-Identifier: Apache-2.0
//! Versioned model metadata stored as bit ordinals under one yesnodb key.

use serde::{Deserialize, Serialize};
use yesno_core::{Db, OrdSet, Snapshot};

use crate::{Error, IndexedRow, Model, PackedIndex, Result};

const MAGIC: &[u8; 8] = b"BAIEZ001";
const HEADER_LEN: usize = 24;
const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
struct StoredIndex {
    key: u64,
    stride: u64,
    num_predicates: usize,
    predicate_fingerprint: u64,
}

#[derive(Clone, Serialize, Deserialize)]
struct StoredBundle {
    version: u32,
    dump_json: String,
    native_text: Option<String>,
    index: Option<StoredIndex>,
}

/// A LightGBM dump and optional packed view descriptor read from one snapshot.
pub struct ModelBundle {
    model: Model,
    stored: StoredBundle,
    index: Option<PackedIndex>,
}

impl ModelBundle {
    pub fn from_dump_json(dump_json: String, native_text: Option<String>) -> Result<Self> {
        let model = Model::from_dump_json(&dump_json)?;
        Ok(Self {
            model,
            stored: StoredBundle {
                version: 1,
                dump_json,
                native_text,
                index: None,
            },
            index: None,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn dump_json(&self) -> &str {
        &self.stored.dump_json
    }

    pub fn native_text(&self) -> Option<&str> {
        self.stored.native_text.as_deref()
    }

    pub fn index(&self) -> Option<PackedIndex> {
        self.index
    }

    /// Store the model, optionally with a freshly built packed index. The two
    /// keys are replaced in one yesnodb batch. `key + 1` is reserved for the
    /// packed view even when no rows have been indexed yet.
    pub fn store(&mut self, db: &Db, key: u64, rows: Option<&[IndexedRow<'_>]>) -> Result<()> {
        let index_key = key
            .checked_add(1)
            .ok_or_else(|| Error::InvalidBundle("metadata key cannot be u64::MAX".into()))?;
        let built = rows
            .map(|rows| PackedIndex::build_set(&self.model, index_key, rows))
            .transpose()?;
        let index = built.as_ref().map(|(descriptor, _)| *descriptor);
        let mut stored = self.stored.clone();
        stored.index = index.map(|descriptor| StoredIndex {
            key: descriptor.key,
            stride: descriptor.stride,
            num_predicates: descriptor.num_predicates,
            predicate_fingerprint: descriptor.predicate_fingerprint,
        });
        let payload = serde_json::to_vec(&stored)?;
        let metadata = encode(&payload)?;
        let mut batch = db.batch();
        batch.store_set(key, &metadata);
        if let Some((_, set)) = &built {
            batch.store_set(index_key, set);
        } else {
            batch.delete_key(index_key);
        }
        batch.commit()?;
        self.stored = stored;
        self.index = index;
        Ok(())
    }

    pub fn load(snapshot: &Snapshot, key: u64) -> Result<Self> {
        Self::from_set(&snapshot.load(key)?, key)
    }

    pub(crate) fn from_set(set: &OrdSet, key: u64) -> Result<Self> {
        let payload = decode(set)?;
        let stored: StoredBundle = serde_json::from_slice(&payload)?;
        if stored.version != 1 {
            return Err(Error::InvalidBundle(format!(
                "unsupported bundle version {}",
                stored.version
            )));
        }
        let model = Model::from_dump_json(&stored.dump_json)?;
        let expected_key = key
            .checked_add(1)
            .ok_or_else(|| Error::InvalidBundle("metadata key cannot be u64::MAX".into()))?;
        let index = stored
            .index
            .as_ref()
            .map(|descriptor| {
                if descriptor.key != expected_key {
                    return Err(Error::InvalidBundle("index key mismatch".into()));
                }
                let index = PackedIndex::from_descriptor(
                    descriptor.key,
                    descriptor.stride,
                    descriptor.num_predicates,
                    descriptor.predicate_fingerprint,
                )?;
                if descriptor.num_predicates != model.num_predicates()
                    || descriptor.predicate_fingerprint != model.predicate_fingerprint()
                {
                    return Err(Error::InvalidBundle("index does not match model".into()));
                }
                Ok(index)
            })
            .transpose()?;
        Ok(Self {
            model,
            stored,
            index,
        })
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn encode(payload: &[u8]) -> Result<OrdSet> {
    if payload.len() > MAX_PAYLOAD {
        return Err(Error::InvalidBundle("bundle exceeds 64 MiB".into()));
    }
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&checksum(payload).to_le_bytes());
    bytes.extend_from_slice(payload);
    let bits = bytes.iter().enumerate().flat_map(|(i, byte)| {
        (0..8).filter_map(move |bit| ((byte >> bit) & 1 != 0).then_some((i as u64) * 8 + bit))
    });
    Ok(OrdSet::from_iter_unsorted(bits))
}

fn decode(set: &OrdSet) -> Result<Vec<u8>> {
    let mut header = [0_u8; HEADER_LEN];
    for ordinal in set.iter() {
        if ordinal < (HEADER_LEN * 8) as u64 {
            header[(ordinal / 8) as usize] |= 1 << (ordinal % 8);
        }
    }
    if &header[..8] != MAGIC {
        return Err(Error::InvalidBundle(
            "missing or invalid bundle magic".into(),
        ));
    }
    let length = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let length = usize::try_from(length)
        .map_err(|_| Error::InvalidBundle("bundle length is too large".into()))?;
    if length > MAX_PAYLOAD {
        return Err(Error::InvalidBundle("bundle exceeds 64 MiB".into()));
    }
    let mut payload = vec![0_u8; length];
    let end_bit = (HEADER_LEN + length) as u64 * 8;
    for ordinal in set.iter() {
        if ordinal >= end_bit {
            return Err(Error::InvalidBundle("ordinal beyond bundle length".into()));
        }
        if ordinal >= (HEADER_LEN * 8) as u64 {
            let offset = ordinal - (HEADER_LEN * 8) as u64;
            payload[(offset / 8) as usize] |= 1 << (offset % 8);
        }
    }
    let expected = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if checksum(&payload) != expected {
        return Err(Error::InvalidBundle("checksum mismatch".into()));
    }
    Ok(payload)
}
