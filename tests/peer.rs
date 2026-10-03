// SPDX-License-Identifier: Apache-2.0
#![cfg(target_os = "linux")]

use std::os::unix::net::UnixListener;
use std::process::Command;
use std::sync::{Arc, RwLock};

use baiez::{load_peer_bundle, IndexedRow, ModelBundle, PredictOptions};
use yesno_core::container::{Container, RunContainer};
use yesno_core::Db;
use yesno_core::OrdSet;
use yesno_plugin::abi::Role;
use yesno_plugin::channel::{send_fd, serve_blocking, Arena, Limits, Session};
use yesno_plugin::Host;

const MODEL: &str = include_str!("fixtures/simple_model.json");

#[test]
fn follower_peer_matches_direct_snapshot_in_both_transport_modes() {
    for inline in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let db = Arc::new(Db::open(directory.path().join("data")).unwrap());
        let mut bundle = ModelBundle::from_dump_json(MODEL.into(), None).unwrap();
        let features: Vec<[f64; 1]> = (0..10_000)
            .map(|id| [if id % 2 == 0 { 0.0 } else { 3.0 }])
            .collect();
        let rows: Vec<_> = features
            .iter()
            .enumerate()
            .map(|(id, features)| IndexedRow {
                id: id as u64,
                features,
            })
            .collect();
        bundle.store(&db, 9000, Some(&rows)).unwrap();
        // Preserve the same ordinals but force the live slot through the run
        // representation; the builder otherwise leaves this fixture as a bitmap.
        let original = db.snapshot().unwrap().load(9001).unwrap();
        let chunks = original
            .chunks()
            .map(|(prefix, container)| {
                if prefix == 0 {
                    (
                        prefix,
                        Container::Run(RunContainer::from_pairs(&[(0, 9999)])),
                    )
                } else {
                    (prefix, container.clone())
                }
            })
            .collect();
        let mut batch = db.batch();
        batch.store_set(9001, &OrdSet::from_chunks(chunks));
        batch.commit().unwrap();
        let snapshot = db.snapshot().unwrap();
        let direct = ModelBundle::load(&snapshot, 9000).unwrap();
        let expected = direct
            .index()
            .unwrap()
            .prepare(&snapshot, direct.model())
            .unwrap()
            .predict(
                &[9999, 0, 7, 8],
                PredictOptions {
                    pred_leaf: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let kinds: Vec<_> = snapshot
            .load(9001)
            .unwrap()
            .chunks()
            .map(|(_, container)| container.kind())
            .collect();
        assert!(kinds.contains(&yesno_core::container::ContainerKind::Bitmap));
        assert!(kinds.contains(&yesno_core::container::ContainerKind::Run));

        let path = directory.path().join("peer.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let host = Host::new(Arc::new(RwLock::new(Some(db.clone()))), 1, Role::Follower);
        let server = std::thread::spawn(move || {
            for _ in 0..3 {
                let (socket, _) = listener.accept().unwrap();
                let limits = Limits {
                    max_lanes: 2,
                    max_blocks: 4,
                    ..Default::default()
                };
                if inline {
                    let mut session = Session::new_inline(host.clone(), limits);
                    serve_blocking(&mut session, socket).unwrap();
                } else {
                    let arena = Arena::new(limits.arena_bytes()).unwrap();
                    send_fd(&socket, arena.as_fd()).unwrap();
                    let mut session = Session::new(host.clone(), arena, limits);
                    serve_blocking(&mut session, socket).unwrap();
                }
            }
        });
        let (remote, prepared) = load_peer_bundle(&path, 9000).unwrap();
        assert_eq!(remote.dump_json(), MODEL);
        assert_eq!(
            prepared
                .predict(
                    &[9999, 0, 7, 8],
                    PredictOptions {
                        pred_leaf: true,
                        ..Default::default()
                    }
                )
                .unwrap(),
            expected
        );
        let ids = directory.path().join("ids.json");
        std::fs::write(&ids, "[9999,0,7,8]").unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_baiez"))
            .args(["predict", "--peer-socket"])
            .arg(&path)
            .args(["--key", "9000", "--ids"])
            .arg(&ids)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let output: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(output[0]["scores"][0], 4.0);
        assert_eq!(output[1]["scores"][0], 2.0);
        let dump = directory.path().join("dump.json");
        let result = Command::new(env!("CARGO_BIN_EXE_baiez"))
            .args(["model", "dump", "--peer-socket"])
            .arg(&path)
            .args(["--key", "9000", "--format", "json", "--output"])
            .arg(&dump)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(std::fs::read_to_string(dump).unwrap(), MODEL);
        server.join().unwrap();
        drop(snapshot);
        drop(db);
    }
}
