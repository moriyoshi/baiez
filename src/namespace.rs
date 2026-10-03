// SPDX-License-Identifier: Apache-2.0
//! Map logical model IDs into disjoint yesno key ranges.

use crate::{Error, Result};

/// A namespace prefix for model bundles stored in a shared yesnodb keyspace.
///
/// `bits` chooses the number of high key bits reserved for `id`. The remaining
/// bits hold the logical model ID and its adjacent metadata/index pair. All
/// callers accessing one database must use the same namespace width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyNamespace {
    id: u32,
    bits: u8,
}

impl KeyNamespace {
    /// Create a namespace. `bits` must be between 1 and 32 inclusive, and `id`
    /// must fit in those bits.
    pub fn new(id: u32, bits: u8) -> Result<Self> {
        if !(1..=32).contains(&bits) {
            return Err(Error::InvalidNamespace(format!(
                "namespace width must be from 1 to 32 bits, got {bits}"
            )));
        }
        if (id as u64) >= (1u64 << bits) {
            return Err(Error::InvalidNamespace(format!(
                "namespace ID {id} does not fit in {bits} bits"
            )));
        }
        Ok(Self { id, bits })
    }

    /// Namespace ID selected from the high bits of physical yesno keys.
    pub fn id(self) -> u32 {
        self.id
    }

    /// Configured namespace width.
    pub fn bits(self) -> u8 {
        self.bits
    }

    /// Return the physical metadata key for a logical model ID.
    ///
    /// The low bit is reserved for baiez's adjacent pair: the returned even key
    /// contains metadata and `key + 1` contains its packed index.
    pub fn model_key(self, model_id: u64) -> Result<u64> {
        let local_bits = 64 - u32::from(self.bits);
        let model_id_bits = local_bits - 1;
        let model_id_limit = 1u64 << model_id_bits;
        if model_id >= model_id_limit {
            return Err(Error::InvalidNamespace(format!(
                "model ID {model_id} does not fit in {model_id_bits} bits"
            )));
        }
        Ok(((self.id as u64) << local_bits) | (model_id << 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_keep_metadata_and_index_pairs_in_separate_ranges() {
        for bits in [1, 8, 16, 32] {
            let max_namespace = ((1u64 << bits) - 1) as u32;
            let left = KeyNamespace::new(0, bits).unwrap().model_key(11).unwrap();
            let right = KeyNamespace::new(max_namespace, bits)
                .unwrap()
                .model_key(11)
                .unwrap();
            assert_eq!(left & 1, 0);
            assert_eq!((left + 1) >> (64 - u32::from(bits)), 0);
            assert_eq!((right + 1) >> (64 - u32::from(bits)), max_namespace as u64);
            assert_ne!(left, right);
            assert_eq!(right & 1, 0);
        }
    }

    #[test]
    fn logical_model_ids_fill_the_local_range_without_pair_overflow() {
        let namespace = KeyNamespace::new(u32::MAX, 32).unwrap();
        let key = namespace.model_key((1u64 << 31) - 1).unwrap();
        assert_eq!(key, u64::MAX - 1);
        assert_eq!(key + 1, u64::MAX);
        assert!(namespace.model_key(1u64 << 31).is_err());
    }

    #[test]
    fn rejects_invalid_namespace_width_and_id() {
        for bits in [0, 33, u8::MAX] {
            assert!(KeyNamespace::new(0, bits).is_err());
        }
        assert!(KeyNamespace::new(2, 1).is_err());
    }
}
