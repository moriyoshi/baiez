// SPDX-License-Identifier: Apache-2.0
use std::collections::{HashMap, HashSet};

use yesno_core::container::Container;
use yesno_core::ops;
use yesno_core::stream::ChunkStream;
use yesno_core::view::View;
use yesno_core::{Db, OrdSet, Snapshot, ORDINAL_MAX};

use crate::model::Node;
use crate::{Error, Model, PredictOptions, Prediction, Result};

const CHUNK_SIZE: u64 = 65_536;

/// A row to index. `id` is a compact, stable logical slot, not necessarily an
/// application's external identifier.
#[derive(Clone, Copy, Debug)]
pub struct IndexedRow<'a> {
    pub id: u64,
    pub features: &'a [f64],
}

/// Descriptor for a blocked packed view stored under one yesnodb key.
///
/// Slot zero holds live rows. Slot `predicate + 1` holds rows for which that
/// predicate takes its left branch. The caller must persist this descriptor
/// beside the exact model used to build it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedIndex {
    pub key: u64,
    pub stride: u64,
    pub num_predicates: usize,
    pub predicate_fingerprint: u64,
}

/// Predicate sets captured from one database snapshot and ready for repeated
/// predictions. The sets own their decoded containers, so the snapshot may be
/// dropped after preparation.
pub struct PreparedIndex {
    descriptor: PackedIndex,
    model: Model,
    live: OrdSet,
    predicates: Vec<OrdSet>,
}

enum PositionLookup {
    Contiguous { first: u64 },
    Dense { first: u64, positions: Vec<usize> },
    Sparse(HashMap<u64, usize>),
}

impl PositionLookup {
    fn new(row_ids: &[u64]) -> Result<Self> {
        let first = *row_ids.iter().min().expect("nonempty cohort");
        if is_contiguous(row_ids, first) {
            return Ok(Self::Contiguous { first });
        }
        let last = *row_ids.iter().max().unwrap();
        let span = last - first + 1;
        if span <= (row_ids.len() as u64).saturating_mul(4) && span <= 1_000_000 {
            let mut positions = vec![usize::MAX; span as usize];
            for (position, &id) in row_ids.iter().enumerate() {
                let slot = &mut positions[(id - first) as usize];
                if *slot != usize::MAX {
                    return Err(Error::InvalidIndex(format!(
                        "duplicate requested row ID {id}"
                    )));
                }
                *slot = position;
            }
            Ok(Self::Dense { first, positions })
        } else {
            let mut positions = HashMap::with_capacity(row_ids.len());
            for (position, &id) in row_ids.iter().enumerate() {
                if positions.insert(id, position).is_some() {
                    return Err(Error::InvalidIndex(format!(
                        "duplicate requested row ID {id}"
                    )));
                }
            }
            Ok(Self::Sparse(positions))
        }
    }

    #[inline]
    fn get(&self, row_id: u64) -> usize {
        match self {
            Self::Contiguous { first } => (row_id - first) as usize,
            Self::Dense { first, positions } => positions[(row_id - first) as usize],
            Self::Sparse(positions) => positions[&row_id],
        }
    }
}

fn is_contiguous(row_ids: &[u64], first: u64) -> bool {
    row_ids
        .iter()
        .enumerate()
        .all(|(position, &id)| first.checked_add(position as u64) == Some(id))
}

impl PackedIndex {
    pub fn new(key: u64, stride: u64, model: &Model) -> Result<Self> {
        Self::from_descriptor(
            key,
            stride,
            model.num_predicates(),
            model.predicate_fingerprint(),
        )
    }

    /// Reopen a descriptor stored beside a model. `predict` checks its
    /// fingerprint against the model before reading any predicate sets.
    pub fn from_descriptor(
        key: u64,
        stride: u64,
        num_predicates: usize,
        predicate_fingerprint: u64,
    ) -> Result<Self> {
        if stride == 0 || !stride.is_multiple_of(CHUNK_SIZE) {
            return Err(Error::InvalidIndex(
                "stride must be a positive multiple of 65,536".into(),
            ));
        }
        let slots = num_predicates
            .checked_add(1)
            .ok_or_else(|| Error::InvalidIndex("too many predicates".into()))?;
        let slots_u32 =
            u32::try_from(slots).map_err(|_| Error::InvalidIndex("too many predicates".into()))?;
        // Validate the largest possible physical ordinal, including the final
        // slot. u64::MAX is reserved by yesnodb.
        let end = u64::from(slots_u32)
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidIndex("packed view exceeds ordinal space".into()))?;
        if end == 0 || end - 1 > ORDINAL_MAX {
            return Err(Error::InvalidIndex(
                "packed view exceeds ordinal space".into(),
            ));
        }
        Ok(Self {
            key,
            stride,
            num_predicates,
            predicate_fingerprint,
        })
    }

    /// Build a new generation under `key`, replacing any previous contents of
    /// that key in one atomic yesnodb batch. Model rollout should use a fresh
    /// key so readers pinned to an earlier generation keep their descriptor.
    pub fn build(db: &Db, model: &Model, key: u64, rows: &[IndexedRow<'_>]) -> Result<Self> {
        let (index, set) = Self::build_set(model, key, rows)?;
        let mut batch = db.batch();
        batch.store_set(key, &set);
        batch.commit()?;
        Ok(index)
    }

    /// Prepare a packed view without committing it, so callers can store the
    /// view and its model metadata in the same database transaction.
    pub(crate) fn build_set(
        model: &Model,
        key: u64,
        rows: &[IndexedRow<'_>],
    ) -> Result<(Self, OrdSet)> {
        let max_id = rows.iter().map(|row| row.id).max().unwrap_or(0);
        if max_id == u64::MAX {
            return Err(Error::InvalidIndex("u64::MAX is not a row ordinal".into()));
        }
        let stride = max_id
            .checked_add(1)
            .and_then(|n| n.checked_add(CHUNK_SIZE - 1))
            .map(|n| n / CHUNK_SIZE * CHUNK_SIZE)
            .ok_or_else(|| Error::InvalidIndex("row ID range is too large".into()))?;
        let index = Self::new(key, stride, model)?;
        let mut physical = Vec::new();
        let mut seen = std::collections::HashSet::with_capacity(rows.len());
        for row in rows {
            if row.features.len() != model.num_features() {
                return Err(Error::InvalidIndex(format!(
                    "row {} has {} features; model expects {}",
                    row.id,
                    row.features.len(),
                    model.num_features()
                )));
            }
            if !seen.insert(row.id) {
                return Err(Error::InvalidIndex(format!("duplicate row ID {}", row.id)));
            }
            physical.push(row.id); // live-row slot zero
            for (predicate, split) in model.predicates.iter().enumerate() {
                if split.goes_left(row.features) {
                    physical.push(index.ordinal(predicate + 1, row.id)?);
                }
            }
        }
        Ok((index, OrdSet::from_iter_unsorted(physical)))
    }

    /// Decode all packed predicate slots from one snapshot for repeated
    /// inference. Preparation cost is paid once and can be shared by callers.
    pub fn prepare(&self, snapshot: &Snapshot, model: &Model) -> Result<PreparedIndex> {
        self.check_model(model)?;
        let high_prefix = self.stride / CHUNK_SIZE;
        let live = self.load_slot(snapshot, 0, 0, high_prefix)?;
        let mut predicates = Vec::with_capacity(self.num_predicates);
        for slot in 1..=self.num_predicates {
            predicates.push(self.load_slot(snapshot, slot, 0, high_prefix)?);
        }
        Ok(PreparedIndex {
            descriptor: *self,
            model: model.clone(),
            live,
            predicates,
        })
    }

    /// Prepare containers received from the yesno peer channel. The packed
    /// stride aligns slots to chunk boundaries, so no ordinal expansion is needed.
    pub(crate) fn prepare_from_set(&self, packed: &OrdSet, model: &Model) -> Result<PreparedIndex> {
        self.check_model(model)?;
        let chunks_per_slot = self.stride / CHUNK_SIZE;
        let mut slots = vec![Vec::new(); self.num_predicates + 1];
        for (prefix, container) in packed.chunks() {
            let slot = usize::try_from(prefix / chunks_per_slot)
                .map_err(|_| Error::InvalidIndex("packed slot exceeds model".into()))?;
            let target = slots
                .get_mut(slot)
                .ok_or_else(|| Error::InvalidIndex("packed slot exceeds model".into()))?;
            target.push((prefix % chunks_per_slot, container.clone()));
        }
        let mut sets = slots.into_iter().map(OrdSet::from_chunks);
        let live = sets.next().expect("at least the live-row slot");
        Ok(PreparedIndex {
            descriptor: *self,
            model: model.clone(),
            live,
            predicates: sets.collect(),
        })
    }

    /// Score a requested cohort from a single yesnodb snapshot. Result order
    /// matches `row_ids`; the stored set supplies row membership, while the
    /// model supplies leaf values and output transforms.
    pub fn predict(
        &self,
        snapshot: &Snapshot,
        model: &Model,
        row_ids: &[u64],
        options: PredictOptions,
    ) -> Result<Vec<Prediction>> {
        self.check_model(model)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut positions = HashMap::with_capacity(row_ids.len());
        for (position, &id) in row_ids.iter().enumerate() {
            if id >= self.stride {
                return Err(Error::InvalidIndex(format!(
                    "row ID {id} exceeds packed stride"
                )));
            }
            if positions.insert(id, position).is_some() {
                return Err(Error::InvalidIndex(format!(
                    "duplicate requested row ID {id}"
                )));
            }
        }
        let requested = OrdSet::from_iter_unsorted(row_ids.iter().copied());
        let low_prefix = row_ids.iter().copied().min().unwrap() / CHUNK_SIZE;
        let high_prefix = row_ids.iter().copied().max().unwrap() / CHUNK_SIZE + 1;
        let live = self.load_slot(snapshot, 0, low_prefix, high_prefix)?;
        let absent = requested.and_not(&live);
        if let Some(id) = absent.iter().next() {
            return Err(Error::InvalidIndex(format!("row ID {id} is not indexed")));
        }

        let tree_range = model.tree_range(options);
        let mut needed = vec![false; model.num_predicates()];
        for tree in &model.trees[tree_range.clone()] {
            mark_predicates(tree, &mut needed);
        }
        let mut predicates = Vec::with_capacity(needed.len());
        for (i, required) in needed.into_iter().enumerate() {
            predicates.push(if required {
                Some(self.load_slot(snapshot, i + 1, low_prefix, high_prefix)?)
            } else {
                None
            });
        }

        let mut result: Vec<_> = row_ids
            .iter()
            .map(|_| Prediction {
                scores: vec![0.0; model.output_count()],
                leaves: options
                    .pred_leaf
                    .then(|| Vec::with_capacity(tree_range.len())),
            })
            .collect();
        for tree_idx in tree_range.clone() {
            route(
                &model.trees[tree_idx],
                &requested,
                &predicates,
                &positions,
                &mut result,
                tree_idx % model.output_count(),
            );
        }
        for prediction in &mut result {
            model.finish_scores(&mut prediction.scores, tree_range.len(), options.raw_score);
        }
        Ok(result)
    }

    fn ordinal(&self, slot: usize, row_id: u64) -> Result<u64> {
        let view = View::blocked((self.num_predicates + 1) as u32, self.stride);
        view.ordinal_of(slot as u32, row_id)
            .ok_or_else(|| Error::InvalidIndex("unaddressable packed ordinal".into()))
    }

    fn check_model(&self, model: &Model) -> Result<()> {
        if self.num_predicates != model.num_predicates()
            || self.predicate_fingerprint != model.predicate_fingerprint()
        {
            return Err(Error::InvalidIndex(
                "index predicate layout does not match model".into(),
            ));
        }
        Ok(())
    }

    fn load_slot(
        &self,
        snapshot: &Snapshot,
        slot: usize,
        low_prefix: u64,
        high_prefix: u64,
    ) -> Result<OrdSet> {
        let chunks_per_slot = self.stride / CHUNK_SIZE;
        let base = (slot as u64) * chunks_per_slot;
        let lo = base + low_prefix;
        let hi = base + high_prefix;
        let mut stream = snapshot.key_stream_prefix_range(self.key, lo, hi)?;
        let mut chunks = Vec::new();
        while let Some((prefix, container)) = stream.next_chunk()? {
            chunks.push((prefix - base, container));
        }
        Ok(OrdSet::from_chunks(chunks))
    }
}

impl PreparedIndex {
    /// Score in caller order into reusable row-major buffers. `scores` needs
    /// one value per model output per row. With `pred_leaf`, `leaves` needs one
    /// value per selected tree per row, also in row-major order.
    pub fn predict_into(
        &self,
        row_ids: &[u64],
        options: PredictOptions,
        scores: &mut [f64],
        mut leaves: Option<&mut [usize]>,
    ) -> Result<()> {
        let tree_range = self.model.tree_range(options);
        let output_count = self.model.output_count();
        let score_len = row_ids
            .len()
            .checked_mul(output_count)
            .ok_or_else(|| Error::InvalidIndex("score buffer length overflow".into()))?;
        if scores.len() != score_len {
            return Err(Error::InvalidIndex(format!(
                "score buffer has {} entries; expected {score_len}",
                scores.len()
            )));
        }
        let leaf_len = row_ids
            .len()
            .checked_mul(tree_range.len())
            .ok_or_else(|| Error::InvalidIndex("leaf buffer length overflow".into()))?;
        if options.pred_leaf != leaves.is_some()
            || leaves.as_ref().is_some_and(|v| v.len() != leaf_len)
        {
            return Err(Error::InvalidIndex(format!(
                "leaf buffer must have {leaf_len} entries when pred_leaf is enabled"
            )));
        }
        if row_ids.is_empty() {
            return Ok(());
        }
        for &id in row_ids {
            if id >= self.descriptor.stride {
                return Err(Error::InvalidIndex(format!(
                    "row ID {id} exceeds packed stride"
                )));
            }
        }
        let first = *row_ids.iter().min().unwrap();
        let last = *row_ids.iter().max().unwrap();
        let span = last - first + 1;
        let sparse = row_ids.len() <= 2_048
            || (row_ids.len() <= 16_384 && span > (row_ids.len() as u64).saturating_mul(4));
        if sparse {
            let ordered = row_ids.windows(2).all(|pair| pair[0] < pair[1])
                || row_ids.windows(2).all(|pair| pair[0] > pair[1]);
            if !ordered {
                if span <= 1_000_000 {
                    let mut seen = vec![0u64; span.div_ceil(64) as usize];
                    for &id in row_ids {
                        let offset = id - first;
                        let word = &mut seen[(offset / 64) as usize];
                        let bit = 1u64 << (offset % 64);
                        if *word & bit != 0 {
                            return Err(Error::InvalidIndex(format!(
                                "duplicate requested row ID {id}"
                            )));
                        }
                        *word |= bit;
                    }
                } else {
                    let mut seen = HashSet::with_capacity(row_ids.len());
                    for &id in row_ids {
                        if !seen.insert(id) {
                            return Err(Error::InvalidIndex(format!(
                                "duplicate requested row ID {id}"
                            )));
                        }
                    }
                }
            }
            for &id in row_ids {
                if !self.live.contains(id) {
                    return Err(Error::InvalidIndex(format!("row ID {id} is not indexed")));
                }
            }
            scores.fill(0.0);
            let mut current_prefix = u64::MAX;
            let mut current_predicates = Vec::with_capacity(self.predicates.len());
            let first_prefix = first / CHUNK_SIZE;
            let chunk_span = last / CHUNK_SIZE - first_prefix + 1;
            let mut chunk_cache: Vec<Option<Vec<Option<&Container>>>> =
                if !ordered && chunk_span <= 4_096 {
                    vec![None; chunk_span as usize]
                } else {
                    Vec::new()
                };
            let mut wide_cache: HashMap<u64, Vec<Option<&Container>>> = HashMap::new();
            for (position, &id) in row_ids.iter().enumerate() {
                let prefix = id / CHUNK_SIZE;
                let low = id as u16;
                let predicates = if ordered {
                    if prefix != current_prefix {
                        current_predicates.clear();
                        current_predicates
                            .extend(self.predicates.iter().map(|set| find_chunk(set, prefix)));
                        current_prefix = prefix;
                    }
                    &current_predicates
                } else if !chunk_cache.is_empty() {
                    chunk_cache[(prefix - first_prefix) as usize].get_or_insert_with(|| {
                        self.predicates
                            .iter()
                            .map(|set| find_chunk(set, prefix))
                            .collect()
                    })
                } else {
                    wide_cache.entry(prefix).or_insert_with(|| {
                        self.predicates
                            .iter()
                            .map(|set| find_chunk(set, prefix))
                            .collect()
                    })
                };
                for tree_idx in tree_range.clone() {
                    let (leaf, value) =
                        route_row_chunk(&self.model.trees[tree_idx], low, predicates);
                    scores[position * output_count + tree_idx % output_count] += value;
                    if let Some(leaves) = leaves.as_deref_mut() {
                        leaves[position * tree_range.len() + tree_idx - tree_range.start] = leaf;
                    }
                }
                self.model.finish_scores(
                    &mut scores[position * output_count..(position + 1) * output_count],
                    tree_range.len(),
                    options.raw_score,
                );
            }
            return Ok(());
        }
        let positions = PositionLookup::new(row_ids)?;
        let requested = OrdSet::from_iter_unsorted(row_ids.iter().copied());
        let absent = requested.and_not(&self.live);
        if let Some(id) = absent.iter().next() {
            return Err(Error::InvalidIndex(format!("row ID {id} is not indexed")));
        }
        scores.fill(0.0);
        for (prefix, rows) in requested.chunks() {
            let predicates: Vec<_> = self
                .predicates
                .iter()
                .map(|set| find_chunk(set, prefix))
                .collect();
            for tree_idx in tree_range.clone() {
                route_chunk(
                    &self.model.trees[tree_idx],
                    rows,
                    &predicates,
                    prefix,
                    &positions,
                    scores,
                    &mut leaves,
                    output_count,
                    tree_idx % output_count,
                    tree_idx - tree_range.start,
                    tree_range.len(),
                );
            }
        }
        for row_scores in scores.chunks_exact_mut(output_count) {
            self.model
                .finish_scores(row_scores, tree_range.len(), options.raw_score);
        }
        Ok(())
    }

    pub fn predict(&self, row_ids: &[u64], options: PredictOptions) -> Result<Vec<Prediction>> {
        let output_count = self.model.output_count();
        let tree_count = self.model.tree_range(options).len();
        let score_len = row_ids
            .len()
            .checked_mul(output_count)
            .ok_or_else(|| Error::InvalidIndex("score buffer length overflow".into()))?;
        let leaf_len = row_ids
            .len()
            .checked_mul(tree_count)
            .ok_or_else(|| Error::InvalidIndex("leaf buffer length overflow".into()))?;
        let mut scores = vec![0.0; score_len];
        let mut leaves = options.pred_leaf.then(|| vec![0; leaf_len]);
        self.predict_into(row_ids, options, &mut scores, leaves.as_deref_mut())?;
        Ok((0..row_ids.len())
            .map(|i| Prediction {
                scores: scores[i * output_count..(i + 1) * output_count].to_vec(),
                leaves: leaves
                    .as_ref()
                    .map(|all| all[i * tree_count..(i + 1) * tree_count].to_vec()),
            })
            .collect())
    }
}

fn route_row_chunk(node: &Node, low: u16, predicates: &[Option<&Container>]) -> (usize, f64) {
    let mut current = node;
    loop {
        match current {
            Node::Leaf { index, value } => return (*index, *value),
            Node::Split {
                predicate,
                left,
                right,
            } => {
                current = if predicates[*predicate].is_some_and(|set| set.contains(low)) {
                    left
                } else {
                    right
                };
            }
        }
    }
}

fn find_chunk(set: &OrdSet, prefix: u64) -> Option<&Container> {
    let mut lo = 0;
    let mut hi = set.chunk_count();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (candidate, container) = set.chunk_at(mid).expect("chunk index in range");
        if candidate == prefix {
            return Some(container);
        }
        if candidate < prefix {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn route_chunk(
    node: &Node,
    rows: &Container,
    predicates: &[Option<&Container>],
    prefix: u64,
    positions: &PositionLookup,
    scores: &mut [f64],
    leaves: &mut Option<&mut [usize]>,
    output_count: usize,
    output: usize,
    tree_position: usize,
    tree_count: usize,
) {
    match node {
        Node::Leaf { index, value } => {
            for low in rows.iter() {
                let row_id = prefix * CHUNK_SIZE + u64::from(low);
                let position = positions.get(row_id);
                scores[position * output_count + output] += value;
                if let Some(leaves) = leaves.as_deref_mut() {
                    leaves[position * tree_count + tree_position] = *index;
                }
            }
        }
        Node::Split {
            predicate,
            left,
            right,
        } => {
            if let Some(split) = predicates[*predicate] {
                if let Some(left_rows) = ops::and(rows, split) {
                    route_chunk(
                        left,
                        &left_rows,
                        predicates,
                        prefix,
                        positions,
                        scores,
                        leaves,
                        output_count,
                        output,
                        tree_position,
                        tree_count,
                    );
                }
                if let Some(right_rows) = ops::and_not(rows, split) {
                    route_chunk(
                        right,
                        &right_rows,
                        predicates,
                        prefix,
                        positions,
                        scores,
                        leaves,
                        output_count,
                        output,
                        tree_position,
                        tree_count,
                    );
                }
            } else {
                route_chunk(
                    right,
                    rows,
                    predicates,
                    prefix,
                    positions,
                    scores,
                    leaves,
                    output_count,
                    output,
                    tree_position,
                    tree_count,
                );
            }
        }
    }
}

fn mark_predicates(node: &Node, needed: &mut [bool]) {
    if let Node::Split {
        predicate,
        left,
        right,
    } = node
    {
        needed[*predicate] = true;
        mark_predicates(left, needed);
        mark_predicates(right, needed);
    }
}

fn route(
    node: &Node,
    rows: &OrdSet,
    predicates: &[Option<OrdSet>],
    positions: &HashMap<u64, usize>,
    result: &mut [Prediction],
    output: usize,
) {
    match node {
        Node::Leaf { index, value } => {
            for row_id in rows.iter() {
                let prediction = &mut result[positions[&row_id]];
                prediction.scores[output] += value;
                if let Some(ref mut leaves) = prediction.leaves {
                    leaves.push(*index);
                }
            }
        }
        Node::Split {
            predicate,
            left,
            right,
        } => {
            let split = predicates[*predicate]
                .as_ref()
                .expect("all selected tree predicates were loaded");
            let left_rows = rows.and(split);
            if !left_rows.is_empty() {
                route(left, &left_rows, predicates, positions, result, output);
            }
            let right_rows = rows.and_not(split);
            if !right_rows.is_empty() {
                route(right, &right_rows, predicates, positions, result, output);
            }
        }
    }
}
