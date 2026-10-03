// SPDX-License-Identifier: Apache-2.0
use baiez::{IndexedRow, Model, PackedIndex, PredictOptions};
use yesno_core::Db;

const MODEL: &str = r#"
{
  "max_feature_idx": 1,
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "objective": "binary sigmoid:2",
  "average_output": false,
  "tree_info": [
    {"tree_index": 0, "tree_structure": {
      "split_index": 0, "split_feature": 0, "decision_type": "<=",
      "threshold": 1.5, "missing_type": "NaN", "default_left": false,
      "left_child": {"leaf_index": 0, "leaf_value": 0.1},
      "right_child": {
        "split_index": 1, "split_feature": 1, "decision_type": "==",
        "threshold": "2||7", "missing_type": "None", "default_left": false,
        "left_child": {"leaf_index": 1, "leaf_value": 0.2},
        "right_child": {"leaf_index": 2, "leaf_value": -0.3}
      }
    }},
    {"tree_index": 1, "tree_structure": {
      "split_index": 0, "split_feature": 0, "decision_type": "<=",
      "threshold": 0.0, "missing_type": "Zero", "default_left": true,
      "left_child": {"leaf_index": 0, "leaf_value": -0.02},
      "right_child": {"leaf_index": 1, "leaf_value": 0.05}
    }}
  ]
}
"#;

#[test]
fn packed_scoring_matches_scalar_with_missing_and_categorical_splits() {
    let model = Model::from_dump_json(MODEL).unwrap();
    assert_eq!(model.num_predicates(), 3);
    let data = [
        [0.0, 2.0],
        [f64::NAN, 7.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [-1.0, -1.0],
        [f64::NAN, -1.0],
    ];
    let ids = [10, 11, 12, 65_540, 65_541, 131_100];
    let rows: Vec<_> = ids
        .iter()
        .zip(&data)
        .map(|(&id, features)| IndexedRow { id, features })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let index = PackedIndex::build(&db, &model, 9000, &rows).unwrap();
    assert_eq!(index.stride, 196_608);
    let snapshot = db.snapshot().unwrap();
    let prepared = index.prepare(&snapshot, &model).unwrap();
    let requested = [131_100, 10, 65_540, 12, 11, 65_541];
    let expected_raw = [-0.32, 0.08, 0.25, -0.25, 0.18, 0.08];
    let raw = index
        .predict(
            &snapshot,
            &model,
            &requested,
            PredictOptions {
                raw_score: true,
                pred_leaf: true,
                ..Default::default()
            },
        )
        .unwrap();
    for (prediction, expected) in raw.iter().zip(expected_raw) {
        assert!((prediction.scores[0] - expected).abs() < 1e-12);
    }

    for options in [
        PredictOptions {
            pred_leaf: true,
            ..Default::default()
        },
        PredictOptions {
            raw_score: true,
            pred_leaf: true,
            ..Default::default()
        },
        PredictOptions {
            start_iteration: 1,
            num_iteration: Some(1),
            raw_score: true,
            pred_leaf: true,
        },
    ] {
        let actual = index
            .predict(&snapshot, &model, &requested, options)
            .unwrap();
        let mut scores = vec![f64::NAN; requested.len()];
        let mut leaves = vec![usize::MAX; requested.len() * model.num_iterations()];
        let selected_trees = if options.start_iteration == 1 { 1 } else { 2 };
        leaves.truncate(requested.len() * selected_trees);
        prepared
            .predict_into(&requested, options, &mut scores, Some(&mut leaves))
            .unwrap();
        assert_eq!(prepared.predict(&requested, options).unwrap(), actual);
        for (position, prediction) in actual.iter().enumerate() {
            assert_eq!(scores[position], prediction.scores[0]);
            assert_eq!(
                &leaves[position * selected_trees..(position + 1) * selected_trees],
                prediction.leaves.as_ref().unwrap()
            );
        }
        for (id, prediction) in requested.iter().zip(actual) {
            let position = ids.iter().position(|candidate| candidate == id).unwrap();
            let expected = model.predict_scalar(&data[position], options).unwrap();
            assert_eq!(prediction.leaves, expected.leaves);
            assert_eq!(prediction.scores, expected.scores);
        }
    }
    assert!(index
        .predict(&snapshot, &model, &[42], PredictOptions::default())
        .is_err());
    let mut score = [0.0];
    assert!(prepared
        .predict_into(&[42], PredictOptions::default(), &mut score, None)
        .is_err());
    assert!(prepared
        .predict_into(&[10, 10], PredictOptions::default(), &mut [0.0; 2], None)
        .is_err());
    assert!(prepared
        .predict_into(&[10], PredictOptions::default(), &mut [], None)
        .is_err());
}

#[test]
fn dense_prepared_routing_preserves_row_order_across_chunks() {
    let model = Model::from_dump_json(MODEL).unwrap();
    let features: Vec<[f64; 2]> = (0..4_096)
        .map(|i| {
            [
                if i % 17 == 0 {
                    f64::NAN
                } else {
                    (i % 5) as f64 - 1.0
                },
                (i % 9) as f64,
            ]
        })
        .collect();
    let rows: Vec<_> = features
        .iter()
        .enumerate()
        .map(|(i, values)| IndexedRow {
            id: 65_000 + i as u64,
            features: values,
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let index = PackedIndex::build(&db, &model, 9, &rows).unwrap();
    let prepared = index.prepare(&db.snapshot().unwrap(), &model).unwrap();
    let requested: Vec<u64> = rows.iter().rev().map(|row| row.id).collect();
    let options = PredictOptions {
        raw_score: true,
        pred_leaf: true,
        ..Default::default()
    };
    let mut scores = vec![0.0; requested.len()];
    let mut leaves = vec![0; requested.len() * model.num_iterations()];
    prepared
        .predict_into(&requested, options, &mut scores, Some(&mut leaves))
        .unwrap();
    for (position, &id) in requested.iter().enumerate() {
        let expected = model
            .predict_scalar(&features[(id - 65_000) as usize], options)
            .unwrap();
        assert_eq!(scores[position], expected.scores[0]);
        assert_eq!(
            &leaves[position * 2..position * 2 + 2],
            expected.leaves.as_ref().unwrap()
        );
    }
}

#[test]
fn sparse_unordered_rows_with_wide_prefixes_match_scalar() {
    let model = Model::from_dump_json(MODEL).unwrap();
    let ids = [0, 150_000_000, 300_000_000];
    let features = [[0.0, 2.0], [f64::NAN, 7.0], [2.0, 1.0]];
    let rows: Vec<_> = ids
        .iter()
        .zip(&features)
        .map(|(&id, features)| IndexedRow { id, features })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let index = PackedIndex::build(&db, &model, 44, &rows).unwrap();
    let prepared = index.prepare(&db.snapshot().unwrap(), &model).unwrap();
    let requested = [300_000_000, 0, 150_000_000];
    let options = PredictOptions {
        raw_score: true,
        pred_leaf: true,
        ..Default::default()
    };
    let actual = prepared.predict(&requested, options).unwrap();
    for (i, prediction) in actual.iter().enumerate() {
        let source = ids.iter().position(|&id| id == requested[i]).unwrap();
        assert_eq!(
            prediction,
            &model.predict_scalar(&features[source], options).unwrap()
        );
    }
    assert!(prepared
        .predict(&[300_000_000, 0, 300_000_000], options)
        .is_err());
}

#[test]
fn categorical_codes_truncate_toward_zero() {
    let model_json = MODEL.replace("2||7", "0||2||7");
    let model = Model::from_dump_json(&model_json).unwrap();
    let prediction = model
        .predict_scalar(
            &[2.0, -0.5],
            PredictOptions {
                raw_score: true,
                pred_leaf: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(prediction.leaves, Some(vec![1, 1]));
}

#[test]
fn constant_tree_needs_no_feature_column() {
    let model = Model::from_dump_json(
        r#"{"max_feature_idx":-1,"num_class":1,"num_tree_per_iteration":1,
        "objective":"regression","tree_info":[{"tree_index":0,
        "tree_structure":{"leaf_index":0,"leaf_value":3.5}}]}"#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let index = PackedIndex::build(
        &db,
        &model,
        10,
        &[IndexedRow {
            id: 1,
            features: &[],
        }],
    )
    .unwrap();
    let result = index
        .predict(
            &db.snapshot().unwrap(),
            &model,
            &[1],
            PredictOptions::default(),
        )
        .unwrap();
    assert_eq!(result[0].scores, vec![3.5]);
}

#[test]
fn snapshot_keeps_old_packed_generation_after_key_replacement() {
    let model = Model::from_dump_json(MODEL).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let before = [[0.0, 2.0]];
    let after = [[2.0, 1.0]];
    let old = PackedIndex::build(
        &db,
        &model,
        9020,
        &[IndexedRow {
            id: 4,
            features: &before[0],
        }],
    )
    .unwrap();
    let old_snapshot = db.snapshot().unwrap();
    let prepared_old = old.prepare(&old_snapshot, &model).unwrap();
    let new = PackedIndex::build(
        &db,
        &model,
        9020,
        &[IndexedRow {
            id: 4,
            features: &after[0],
        }],
    )
    .unwrap();
    let new_snapshot = db.snapshot().unwrap();
    let options = PredictOptions {
        raw_score: true,
        pred_leaf: true,
        ..Default::default()
    };
    let old_score = old.predict(&old_snapshot, &model, &[4], options).unwrap();
    let new_score = new.predict(&new_snapshot, &model, &[4], options).unwrap();
    assert_eq!(
        old_score[0],
        model.predict_scalar(&before[0], options).unwrap()
    );
    assert_eq!(prepared_old.predict(&[4], options).unwrap(), old_score);
    assert_eq!(
        new_score[0],
        model.predict_scalar(&after[0], options).unwrap()
    );
    assert_ne!(old_score, new_score);
}

#[test]
fn descriptor_rejects_bad_stride_and_wrong_predicate_layout() {
    let model = Model::from_dump_json(MODEL).unwrap();
    assert!(PackedIndex::new(1, 65_535, &model).is_err());
    assert!(PackedIndex::new(1, 0, &model).is_err());
    let altered = Model::from_dump_json(&MODEL.replace("1.5", "1.25")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let row = [0.0, 0.0];
    let index = PackedIndex::build(
        &db,
        &model,
        20,
        &[IndexedRow {
            id: 0,
            features: &row,
        }],
    )
    .unwrap();
    assert!(index
        .predict(
            &db.snapshot().unwrap(),
            &altered,
            &[0],
            PredictOptions::default()
        )
        .is_err());
}

#[test]
fn multiclass_tree_order_and_iteration_window() {
    let model = Model::from_dump_json(
        r#"{
          "max_feature_idx":-1,"num_class":3,"num_tree_per_iteration":3,
          "objective":"multiclass","average_output":false,
          "tree_info":[
            {"tree_index":0,"tree_structure":{"leaf_index":0,"leaf_value":0.2}},
            {"tree_index":1,"tree_structure":{"leaf_index":0,"leaf_value":0.4}},
            {"tree_index":2,"tree_structure":{"leaf_index":0,"leaf_value":0.6}},
            {"tree_index":3,"tree_structure":{"leaf_index":0,"leaf_value":0.1}},
            {"tree_index":4,"tree_structure":{"leaf_index":0,"leaf_value":0.2}},
            {"tree_index":5,"tree_structure":{"leaf_index":0,"leaf_value":0.3}}
          ]
        }"#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let index = PackedIndex::build(
        &db,
        &model,
        100,
        &[IndexedRow {
            id: 8,
            features: &[],
        }],
    )
    .unwrap();
    let snapshot = db.snapshot().unwrap();
    let prepared = index.prepare(&snapshot, &model).unwrap();
    let options = PredictOptions {
        raw_score: true,
        pred_leaf: true,
        ..Default::default()
    };
    let prediction = index.predict(&snapshot, &model, &[8], options).unwrap();
    assert_eq!(prediction[0].leaves, Some(vec![0; 6]));
    for (got, want) in prediction[0].scores.iter().zip([0.3, 0.6, 0.9]) {
        assert!((got - want).abs() < 1e-12);
    }
    let sliced = index
        .predict(
            &snapshot,
            &model,
            &[8],
            PredictOptions {
                start_iteration: 1,
                num_iteration: Some(1),
                raw_score: true,
                pred_leaf: false,
            },
        )
        .unwrap();
    assert_eq!(sliced[0].scores, vec![0.1, 0.2, 0.3]);
    let probabilities = index
        .predict(&snapshot, &model, &[8], PredictOptions::default())
        .unwrap();
    assert!((probabilities[0].scores.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for options in [options, PredictOptions::default()] {
        assert_eq!(
            prepared.predict(&[8], options).unwrap(),
            index.predict(&snapshot, &model, &[8], options).unwrap()
        );
    }
}
