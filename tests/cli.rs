// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const MODEL: &str = include_str!("fixtures/simple_model.json");

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_baiez"))
        .args(args)
        .output()
        .expect("baiez CLI should launch")
}

fn string(path: &Path) -> &str {
    path.to_str().expect("temporary path is UTF-8")
}

#[test]
fn model_bundle_and_legacy_cli_round_trip() {
    assert!(run(&["--help"]).status.success());
    assert_eq!(run(&["--version"]).stdout, b"baiez 0.1.0\n");
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.json");
    let rows = dir.path().join("rows.json");
    let ids = dir.path().join("ids.json");
    let db = dir.path().join("db");
    let dump = dir.path().join("dump.json");
    let descriptor = dir.path().join("descriptor.json");
    fs::write(&model, MODEL).unwrap();
    fs::write(
        &rows,
        r#"[{"id":7,"features":[0]},{"id":9,"features":[3]}]"#,
    )
    .unwrap();
    fs::write(&ids, "[9,7]").unwrap();

    assert!(run(&[
        "model",
        "load",
        "--db",
        string(&db),
        "--key",
        "9000",
        "--input",
        string(&model),
        "--format",
        "json",
        "--rows",
        string(&rows),
    ])
    .status
    .success());
    assert!(run(&[
        "model",
        "dump",
        "--db",
        string(&db),
        "--key",
        "9000",
        "--format",
        "json",
        "--output",
        string(&dump),
    ])
    .status
    .success());
    assert_eq!(fs::read(&dump).unwrap(), MODEL.as_bytes());
    let predicted = run(&[
        "predict",
        "--db",
        string(&db),
        "--key",
        "9000",
        "--ids",
        string(&ids),
        "--raw",
        "--leaf",
    ]);
    assert!(predicted.status.success());
    let result: serde_json::Value = serde_json::from_slice(&predicted.stdout).unwrap();
    assert_eq!(result[0]["scores"][0], 4.0);
    assert_eq!(result[1]["leaves"][0], 0);

    assert!(!run(&[
        "predict",
        "--db",
        string(&db),
        "--key",
        "9000",
        "--ids",
        string(&ids),
        "--bogus",
    ])
    .status
    .success());
    assert!(!run(&[
        "model",
        "load",
        "--db",
        string(&db),
        "--key",
        "9000",
        "--key",
        "9001",
        "--input",
        string(&model),
        "--format",
        "json",
    ])
    .status
    .success());

    assert!(run(&[
        "index",
        string(&model),
        string(&rows),
        string(&db),
        "9100",
        string(&descriptor),
    ])
    .status
    .success());
    let legacy = run(&[
        "predict",
        string(&model),
        string(&descriptor),
        string(&db),
        string(&ids),
        "--raw",
    ]);
    assert!(legacy.status.success());
    let result: serde_json::Value = serde_json::from_slice(&legacy.stdout).unwrap();
    assert_eq!(result[0]["scores"][0], 4.0);
}

#[test]
fn namespace_flags_partition_bundle_load_index_dump_and_predict() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.json");
    let rows = dir.path().join("rows.json");
    let ids = dir.path().join("ids.json");
    let db = dir.path().join("db");
    let dump = dir.path().join("dump.json");
    fs::write(&model, MODEL).unwrap();
    fs::write(
        &rows,
        r#"[{"id":7,"features":[0]},{"id":9,"features":[3]}]"#,
    )
    .unwrap();
    fs::write(&ids, "[9,7]").unwrap();

    assert!(run(&[
        "model",
        "load",
        "--db",
        string(&db),
        "--key",
        "42",
        "--input",
        string(&model),
        "--format",
        "json",
        "--rows",
        string(&rows),
        "--namespace-bits",
        "8",
        "--namespace-id",
        "3",
    ])
    .status
    .success());
    assert!(run(&[
        "index",
        "build",
        "--db",
        string(&db),
        "--key",
        "42",
        "--rows",
        string(&rows),
        "--namespace-bits",
        "8",
        "--namespace-id",
        "3",
    ])
    .status
    .success());
    let predicted = run(&[
        "predict",
        "--db",
        string(&db),
        "--key",
        "42",
        "--ids",
        string(&ids),
        "--namespace-bits",
        "8",
        "--namespace-id",
        "3",
    ]);
    assert!(predicted.status.success());
    let result: serde_json::Value = serde_json::from_slice(&predicted.stdout).unwrap();
    assert_eq!(result[0]["scores"][0], 4.0);
    assert_eq!(result[1]["scores"][0], 2.0);

    assert!(run(&[
        "model",
        "dump",
        "--db",
        string(&db),
        "--key",
        "42",
        "--namespace-bits",
        "8",
        "--namespace-id",
        "3",
        "--format",
        "json",
        "--output",
        string(&dump),
    ])
    .status
    .success());
    assert_eq!(fs::read(dump).unwrap(), MODEL.as_bytes());

    assert!(!run(&[
        "predict",
        "--db",
        string(&db),
        "--key",
        "42",
        "--ids",
        string(&ids),
        "--namespace-bits",
        "8",
        "--namespace-id",
        "4",
    ])
    .status
    .success());
    assert!(!run(&[
        "predict",
        "--db",
        string(&db),
        "--key",
        "42",
        "--ids",
        string(&ids),
        "--namespace-bits",
        "8",
    ])
    .status
    .success());
}
