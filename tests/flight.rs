use std::sync::Arc;

use baiez::{load_flight_bundle, IndexedRow, KeyNamespace, ModelBundle, PredictOptions};
use yesno_core::{Db, DbOptions};
use yesno_flight::YesnoFlightService;

const MODEL: &str = include_str!("fixtures/simple_model.json");

async fn serve(db: Arc<Db>) -> (String, tokio::sync::oneshot::Sender<()>) {
    use arrow_flight::flight_service_server::FlightServiceServer;
    use tonic::transport::Server;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        Server::builder()
            .add_service(FlightServiceServer::new(YesnoFlightService::new(db)))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = stopped.await;
                },
            )
            .await
            .unwrap();
    });
    (format!("http://{address}"), stop)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_bundle_reads_a_namespaced_model_and_index_from_one_generation() {
    let directory = tempfile::tempdir().unwrap();
    let db = Arc::new(
        Db::open_with(
            directory.path(),
            DbOptions {
                shards: 2,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let namespace = KeyNamespace::new(17, 8).unwrap();
    let key = namespace.model_key(33).unwrap();
    let features: Vec<[f64; 1]> = (0..128)
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
    let mut bundle = ModelBundle::from_dump_json(MODEL.into(), None).unwrap();
    bundle.store(&db, key, Some(&rows)).unwrap();

    let expected_snapshot = db.snapshot().unwrap();
    let expected_bundle = ModelBundle::load(&expected_snapshot, key).unwrap();
    let expected = expected_bundle
        .index()
        .unwrap()
        .prepare(&expected_snapshot, expected_bundle.model())
        .unwrap()
        .predict(
            &[127, 0, 7, 8],
            PredictOptions {
                pred_leaf: true,
                ..Default::default()
            },
        )
        .unwrap();

    let (endpoint, stop) = serve(db.clone()).await;
    let endpoint_for_load = endpoint.clone();
    let (remote, prepared) =
        tokio::task::spawn_blocking(move || load_flight_bundle(endpoint_for_load, key))
            .await
            .unwrap()
            .unwrap();
    assert_eq!(remote.dump_json(), MODEL);
    assert_eq!(
        prepared
            .predict(
                &[127, 0, 7, 8],
                PredictOptions {
                    pred_leaf: true,
                    ..Default::default()
                }
            )
            .unwrap(),
        expected
    );

    let ids_path = directory.path().join("ids.json");
    std::fs::write(&ids_path, "[127,0,7,8]").unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_baiez"))
        .args(["predict", "--flight-endpoint", &endpoint, "--key"])
        .arg("33")
        .args(["--namespace-bits", "8", "--namespace-id", "17", "--ids"])
        .arg(&ids_path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let predictions: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(predictions[0]["scores"][0], expected[0].scores[0]);

    if let Some(python) = std::env::var_os("BAIEZ_PYTHON_EXECUTABLE") {
        let extension = std::env::current_dir()
            .unwrap()
            .join("target/debug/libbaiez.so");
        let result = std::process::Command::new(python)
            .arg("scripts/check_python_flight.py")
            .arg(&endpoint)
            .env("BAIEZ_EXTENSION_PATH", extension)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("Python Flight binding passed"));
    }

    let _ = stop.send(());
    drop(expected_snapshot);
    drop(db);
}
