// SPDX-License-Identifier: Apache-2.0
use baiez::{IndexedRow, ModelBundle, PredictOptions};
use yesno_core::Db;

const MODEL: &str = r#"{
  "max_feature_idx": 0,
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "objective": "regression",
  "tree_info": [{"tree_index": 0, "tree_structure": {
    "split_index": 0, "split_feature": 0, "decision_type": "<=",
    "threshold": 1.5, "missing_type": "None", "default_left": true,
    "left_child": {"leaf_index": 0, "leaf_value": 2.0},
    "right_child": {"leaf_index": 1, "leaf_value": 4.0}
  }}]
}"#;

#[test]
fn bundle_round_trips_and_replaces_index_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    let mut bundle =
        ModelBundle::from_dump_json(MODEL.into(), Some("native bytes\n".into())).unwrap();
    let features = [[0.0], [3.0]];
    let rows = [
        IndexedRow {
            id: 7,
            features: &features[0],
        },
        IndexedRow {
            id: 9,
            features: &features[1],
        },
    ];
    bundle.store(&db, 9000, Some(&rows)).unwrap();
    let old_snapshot = db.snapshot().unwrap();
    let old = ModelBundle::load(&old_snapshot, 9000).unwrap();
    assert_eq!(old.dump_json(), MODEL);
    assert_eq!(old.native_text(), Some("native bytes\n"));
    let predictions = old
        .index()
        .unwrap()
        .prepare(&old_snapshot, old.model())
        .unwrap()
        .predict(&[9, 7], PredictOptions::default())
        .unwrap();
    assert_eq!(predictions[0].scores, vec![4.0]);
    assert_eq!(predictions[1].scores, vec![2.0]);

    bundle.store(&db, 9000, None).unwrap();
    let latest = ModelBundle::load(&db.snapshot().unwrap(), 9000).unwrap();
    assert!(latest.index().is_none());
    assert_eq!(latest.dump_json(), MODEL);
    // The older snapshot still has a matched model and packed view.
    let old_again = ModelBundle::load(&old_snapshot, 9000).unwrap();
    assert!(old_again.index().is_some());
    assert!(db.snapshot().unwrap().load(9001).unwrap().is_empty());
}
