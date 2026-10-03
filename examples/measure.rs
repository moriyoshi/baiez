// SPDX-License-Identifier: Apache-2.0
//! End-to-end latency probe for the current batch scorer.
//!
//! `cargo run --release --example measure -- [rows] [trees]`
//! The scalar reference reads features already resident in memory. The packed
//! path reads a yesnodb snapshot and builds its predicate sets per request.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use baiez::{IndexedRow, Model, PackedIndex, PredictOptions, Prediction};
use serde_json::{json, Value};
use yesno_core::Db;

fn node(tree: usize, depth: usize, heap: usize, leaf: usize) -> Value {
    if depth == 0 {
        let value = (tree % 5) as f64 * 0.01 + (leaf as f64 - 3.5) * 0.04;
        return json!({"leaf_index":leaf,"leaf_value":value});
    }
    let feature = (tree * 3 + heap) % 8;
    let threshold = 0.2 + ((tree * 17 + heap * 11) % 55) as f64 / 100.0;
    json!({
        "split_index": heap - 1,
        "split_feature": feature,
        "decision_type": "<=",
        "threshold": threshold,
        "missing_type": "NaN",
        "default_left": heap.is_multiple_of(2),
        "left_child": node(tree, depth - 1, heap * 2, leaf * 2),
        "right_child": node(tree, depth - 1, heap * 2 + 1, leaf * 2 + 1)
    })
}

fn model(trees: usize) -> Result<Model, Box<dyn Error>> {
    let tree_info: Vec<_> = (0..trees)
        .map(|i| json!({"tree_index": i, "tree_structure": node(i, 3, 1, 0)}))
        .collect();
    let dump = json!({
        "max_feature_idx":7,
        "num_class":1,
        "num_tree_per_iteration":1,
        "objective":"regression",
        "tree_info":tree_info
    });
    Ok(Model::from_dump_json(&dump.to_string())?)
}

fn data(rows: usize) -> Vec<[f64; 8]> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..rows)
        .map(|row| {
            let mut values = [0.0; 8];
            for (feature, value) in values.iter_mut().enumerate() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *value = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                if (row + feature) % 97 == 0 {
                    *value = f64::NAN;
                } else if (row + feature) % 89 == 0 {
                    *value = 0.0;
                }
            }
            values
        })
        .collect()
}

fn allocated_bytes(path: &std::path::Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += allocated_bytes(&entry.path())?;
        } else {
            total += metadata.blocks() * 512;
        }
    }
    Ok(total)
}

fn median(mut timings: Vec<Duration>, repetitions: usize) -> f64 {
    timings.sort_unstable();
    timings[timings.len() / 2].as_secs_f64() * 1e6 / repetitions as f64
}

fn time_case(mut run: impl FnMut(), repetitions: usize) -> f64 {
    let mut timings = Vec::with_capacity(7);
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..repetitions {
            run();
        }
        timings.push(start.elapsed());
    }
    median(timings, repetitions)
}

fn score_scalar(model: &Model, data: &[[f64; 8]], ids: &[u64]) -> Vec<Prediction> {
    ids.iter()
        .map(|id| {
            model
                .predict_scalar(&data[*id as usize], PredictOptions::default())
                .unwrap()
        })
        .collect()
}

fn case(
    label: &str,
    ids: &[u64],
    model: &Model,
    data: &[[f64; 8]],
    index: &PackedIndex,
    snapshot: &yesno_core::Snapshot,
) -> Result<(), Box<dyn Error>> {
    let expected = score_scalar(model, data, ids);
    let actual = index.predict(snapshot, model, ids, PredictOptions::default())?;
    if expected != actual {
        return Err(format!("{label}: packed scores differ from scalar").into());
    }
    let repetitions = match ids.len() {
        0..=64 => 30,
        65..=1024 => 12,
        1025..=8192 => 5,
        _ => 2,
    };
    let scalar_us = time_case(
        || {
            black_box(score_scalar(model, data, black_box(ids)));
        },
        repetitions,
    );
    let packed_us = time_case(
        || {
            black_box(
                index
                    .predict(snapshot, model, black_box(ids), PredictOptions::default())
                    .unwrap(),
            );
        },
        repetitions,
    );
    println!(
        "{label},{},{scalar_us:.3},{packed_us:.3},{:.2}",
        ids.len(),
        packed_us / scalar_us
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let row_count: usize = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100_000);
    let tree_count: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(16);
    if row_count == 0 || tree_count == 0 {
        return Err("rows and trees must be positive".into());
    }
    let model = model(tree_count)?;
    let data = data(row_count);
    let indexed: Vec<_> = data
        .iter()
        .enumerate()
        .map(|(id, features)| IndexedRow {
            id: id as u64,
            features,
        })
        .collect();
    let directory = tempfile::tempdir()?;
    let db = Db::open(directory.path())?;
    let start = Instant::now();
    let index = PackedIndex::build(&db, &model, 1, &indexed)?;
    let build_seconds = start.elapsed().as_secs_f64();
    let checkpoint_start = Instant::now();
    db.checkpoint()?;
    let checkpoint_seconds = checkpoint_start.elapsed().as_secs_f64();
    let snapshot = db.snapshot()?;
    let members = snapshot.cardinality(1)?;
    let disk_bytes = allocated_bytes(directory.path())?;
    println!(
        "setup: rows={row_count} trees={tree_count} predicates={} packed_memberships={members} build_s={build_seconds:.3} checkpoint_s={checkpoint_seconds:.3} allocated_bytes={disk_bytes}",
        model.num_predicates()
    );
    println!("shape,rows,scalar_us,packed_us,packed_over_scalar");
    for &size in &[1usize, 64, 1024, 8192, row_count] {
        if size > row_count {
            continue;
        }
        let begin = (row_count - size) / 2;
        let contiguous: Vec<u64> = (begin..begin + size).map(|id| id as u64).collect();
        case("contiguous", &contiguous, &model, &data, &index, &snapshot)?;
        if size > 1 && size < row_count {
            let scattered: Vec<u64> = (0..size)
                .map(|i| ((i as u128 * row_count as u128) / size as u128) as u64)
                .collect();
            case("scattered", &scattered, &model, &data, &index, &snapshot)?;
        }
    }
    Ok(())
}
