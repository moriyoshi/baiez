// SPDX-License-Identifier: Apache-2.0
//! OS-capable verbs exposed to otherwise isolated Monty scenarios.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use baiez::{load_peer_bundle, IndexedRow, ModelBundle, PredictOptions, PreparedIndex};
use monty_types::{ExcType, MontyException, MontyObject};
use serde_json::{json, Value};
use tempfile::TempDir;
use yesno_core::Db;

type Result<T> = std::result::Result<T, MontyException>;
const MODEL_KEY: u64 = 9000;
const VERBS: &[&str] = &[
    "seed",
    "seed_namespaced",
    "seed_fixture",
    "leader_start",
    "follower_start",
    "wait_pass",
    "reject_write",
    "predict",
    "predict_leaf",
    "predict_namespaced",
    "missing_model",
    "attest_fixture",
    "wait_predict",
    "dump_leaf",
    "publish",
    "snapshot",
    "snapshot_predict",
    "stop",
    "restart",
];

pub struct World {
    root: PathBuf,
    sockets: TempDir,
    deadline: Instant,
    yesnod: PathBuf,
    yesno: PathBuf,
    baiez: PathBuf,
    nodes: Vec<Node>,
    snapshots: Vec<PreparedIndex>,
    ports: HashSet<u16>,
}

struct Node {
    name: String,
    config: PathBuf,
    socket: PathBuf,
    flight: u16,
    metrics: u16,
    control: Option<u16>,
    starts: usize,
    process: Option<Child>,
}

impl Drop for World {
    fn drop(&mut self) {
        for node in self.nodes.iter_mut().rev() {
            stop_child(&mut node.process);
        }
    }
}

fn stop_child(process: &mut Option<Child>) {
    if let Some(mut child) = process.take() {
        if child.try_wait().ok().flatten().is_none() {
            let _ = child.kill();
        }
        let _ = child.wait();
    }
}

fn err(message: impl std::fmt::Display) -> MontyException {
    MontyException::new(ExcType::RuntimeError, Some(message.to_string()))
}

fn integer(value: &MontyObject) -> Result<u64> {
    match value {
        MontyObject::Int(number) if *number >= 0 => Ok(*number as u64),
        other => Err(err(format!(
            "expected nonnegative int, got {}",
            other.type_name()
        ))),
    }
}

fn float(value: &MontyObject) -> Result<f64> {
    match value {
        MontyObject::Float(number) => Ok(*number),
        MontyObject::Int(number) => Ok(*number as f64),
        other => Err(err(format!("expected number, got {}", other.type_name()))),
    }
}

fn string(value: &MontyObject) -> Result<&str> {
    match value {
        MontyObject::String(value) => Ok(value),
        other => Err(err(format!("expected string, got {}", other.type_name()))),
    }
}

fn index(value: &MontyObject) -> Result<usize> {
    usize::try_from(integer(value)?).map_err(err)
}

fn handle(value: usize) -> MontyObject {
    MontyObject::Int(value as i64)
}

fn scores(values: &[f64]) -> MontyObject {
    MontyObject::List(values.iter().copied().map(MontyObject::Float).collect())
}

impl World {
    pub fn is_verb(name: &str) -> bool {
        VERBS.contains(&name)
    }

    pub fn new(name: &str, timeout: Duration) -> std::io::Result<Self> {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.agents-workspace/tmp/e2e");
        fs::create_dir_all(&scratch)?;
        let root = tempfile::Builder::new()
            .prefix(name.trim_end_matches(".py"))
            .tempdir_in(scratch)?
            .keep();
        // Linux limits Unix socket paths to 107 bytes, including nested CI paths.
        let sockets = tempfile::Builder::new().prefix("baiez-e2e-").tempdir()?;
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace");
        let yesnod = std::env::var_os("YESNOD_BIN").map_or_else(
            || workspace.join("../yesno/target/debug/yesnod"),
            PathBuf::from,
        );
        let yesno = std::env::var_os("YESNO_BIN").map_or_else(
            || workspace.join("../yesno/target/debug/yesno"),
            PathBuf::from,
        );
        let baiez = std::env::var_os("BAIEZ_BIN")
            .map_or_else(|| workspace.join("target/debug/baiez"), PathBuf::from);
        for binary in [&yesnod, &yesno, &baiez] {
            if !binary.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("missing binary: {}", binary.display()),
                ));
            }
        }
        Ok(Self {
            root,
            sockets,
            deadline: Instant::now() + timeout,
            yesnod,
            yesno,
            baiez,
            nodes: Vec::new(),
            snapshots: Vec::new(),
            ports: HashSet::new(),
        })
    }

    fn remaining(&self) -> Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|duration| !duration.is_zero())
            .ok_or_else(|| err(format!("scenario timed out; logs: {}", self.root.display())))
    }

    fn port(&mut self) -> Result<u16> {
        for _ in 0..100 {
            let listener = TcpListener::bind("127.0.0.1:0").map_err(err)?;
            let port = listener.local_addr().map_err(err)?.port();
            if self.ports.insert(port) {
                return Ok(port);
            }
        }
        Err(err("could not select distinct local ports"))
    }

    fn run(&self, binary: &Path, args: &[String]) -> Result<Output> {
        let mut child = Command::new(binary)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(err)?;
        loop {
            if child.try_wait().map_err(err)?.is_some() {
                let output = child.wait_with_output().map_err(err)?;
                return Ok(output);
            }
            if self.remaining().is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(err(format!("command timed out: {}", binary.display())));
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn run_ok(&self, binary: &Path, args: &[String]) -> Result<String> {
        let output = self.run(binary, args)?;
        if !output.status.success() {
            return Err(err(format!(
                "{} {:?} exited {}; stdout: {}; stderr: {}; logs: {}",
                binary.display(),
                args,
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
                self.root.display()
            )));
        }
        String::from_utf8(output.stdout).map_err(err)
    }

    fn spawn(&self, name: &str, start: usize, args: &[String]) -> Result<Child> {
        let log = File::create(self.root.join(format!("{name}-{start}.log"))).map_err(err)?;
        let stderr = log.try_clone().map_err(err)?;
        Command::new(&self.yesnod)
            .args(args)
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(err)
    }

    fn node(&mut self, name: &str, leader_control: Option<u16>) -> Result<usize> {
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(err(
                "node name must contain only ASCII letters, digits, or underscores",
            ));
        }
        if self.nodes.iter().any(|node| node.name == name) {
            return Err(err(format!("duplicate node name: {name}")));
        }
        let data = self.root.join(name);
        let config = self.root.join(format!("{name}.toml"));
        let socket = self.sockets.path().join(format!("{name}.sock"));
        let flight = self.port()?;
        let metrics = self.port()?;
        let control = if leader_control.is_none() {
            Some(self.port()?)
        } else {
            None
        };
        let (role, extra) = if let Some(leader_control) = leader_control {
            (
                "follower",
                format!("[follower]\nleader = \"http://127.0.0.1:{leader_control}\"\nserve_reads = true\npoll_interval_secs = 1\n"),
            )
        } else {
            (
                "leader",
                format!("[server.control]\nlisten = \"127.0.0.1:{}\"\njournal_dir = \"{}/control\"\n[[auth.rule]]\nprincipal = \"all\"\naddress = \"127.0.0.1/32\"\ncapability = \"replication\"\naction = \"allow\"\n", control.expect("leader control"), self.root.display()),
            )
        };
        fs::write(&config, format!("[server]\nrole = \"{role}\"\ndata_dir = \"{}\"\n[db]\nshards = 8\n[server.flight]\nlisten = \"127.0.0.1:{flight}\"\n[server.metrics]\nlisten = \"127.0.0.1:{metrics}\"\n{extra}[plugin]\nchannel_socket = \"{}\"\nchannel_socket_mode = \"0600\"\n", data.display(), socket.display())).map_err(err)?;
        let mut args = vec!["--config".to_owned(), config.display().to_string()];
        if leader_control.is_none() {
            args.push("--insecure-replication".to_owned());
        }
        let mut check = args.clone();
        check.push("--check-config".to_owned());
        self.run_ok(&self.yesnod, &check)?;
        let process = self.spawn(name, 0, &args)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            name: name.to_owned(),
            config,
            socket,
            flight,
            metrics,
            control,
            starts: 1,
            process: Some(process),
        });
        self.wait_ready(id)?;
        Ok(id)
    }

    fn check_alive(&mut self, id: usize) -> Result<()> {
        let node = self.nodes.get_mut(id).ok_or_else(|| err("unknown node"))?;
        let child = node
            .process
            .as_mut()
            .ok_or_else(|| err("node is stopped"))?;
        if let Some(status) = child.try_wait().map_err(err)? {
            return Err(err(format!(
                "{} exited {status}; logs: {}",
                node.name,
                self.root.display()
            )));
        }
        self.remaining()?;
        Ok(())
    }

    fn wait_ready(&mut self, id: usize) -> Result<()> {
        let port = self.nodes[id].metrics;
        loop {
            self.check_alive(id)?;
            if http_get(port, "/readyz")
                .is_some_and(|response| response.starts_with("HTTP/1.1 200"))
            {
                return Ok(());
            }
            std::thread::sleep(self.remaining()?.min(Duration::from_millis(100)));
        }
    }

    fn wait_pass(&mut self, id: usize) -> Result<()> {
        let port = self
            .nodes
            .get(id)
            .ok_or_else(|| err("unknown node"))?
            .metrics;
        loop {
            self.check_alive(id)?;
            if http_get(port, "/metrics").is_some_and(|response| {
                response.lines().any(|line| {
                    line.strip_prefix("yesnod_follower_passes_total ")
                        .and_then(|value| value.trim().parse::<u64>().ok())
                        .is_some_and(|passes| passes > 0)
                })
            }) {
                return Ok(());
            }
            std::thread::sleep(self.remaining()?.min(Duration::from_millis(100)));
        }
    }

    fn seed(&self) -> Result<()> {
        fs::write(
            self.root.join("rows.json"),
            r#"[{"id":7,"features":[0.0]},{"id":9,"features":[3.0]}]"#,
        )
        .map_err(err)?;
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/simple_model.json");
        self.run_ok(
            &self.baiez,
            &[
                "model".into(),
                "load".into(),
                "--db".into(),
                self.root.join("leader").display().to_string(),
                "--key".into(),
                MODEL_KEY.to_string(),
                "--input".into(),
                model.display().to_string(),
                "--format".into(),
                "json".into(),
                "--rows".into(),
                self.root.join("rows.json").display().to_string(),
            ],
        )?;
        Ok(())
    }

    fn seed_namespaced(&self) -> Result<()> {
        fs::write(
            self.root.join("rows.json"),
            r#"[{"id":7,"features":[0.0]},{"id":9,"features":[3.0]}]"#,
        )
        .map_err(err)?;
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/simple_model.json");
        self.run_ok(
            &self.baiez,
            &[
                "model".into(),
                "load".into(),
                "--db".into(),
                self.root.join("leader").display().to_string(),
                "--key".into(),
                "42".into(),
                "--namespace-bits".into(),
                "8".into(),
                "--namespace-id".into(),
                "3".into(),
                "--input".into(),
                model.display().to_string(),
                "--format".into(),
                "json".into(),
                "--rows".into(),
                self.root.join("rows.json").display().to_string(),
            ],
        )?;
        Ok(())
    }

    fn seed_fixture(&self) -> Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/lightgbm_regression_1024");
        let dump_path = fixture.join("model.json");
        let dump: Value =
            serde_json::from_slice(&fs::read(&dump_path).map_err(err)?).map_err(err)?;
        let columns = dump["max_feature_idx"]
            .as_u64()
            .and_then(|value| usize::try_from(value + 1).ok())
            .ok_or_else(|| err("fixture model has an invalid feature count"))?;
        let feature_bytes = fs::read(fixture.join("rows.f64")).map_err(err)?;
        let row_width = columns
            .checked_mul(8)
            .ok_or_else(|| err("fixture row width overflow"))?;
        if feature_bytes.is_empty() || feature_bytes.len() % row_width != 0 {
            return Err(err("fixture row data has an invalid byte length"));
        }
        let mut rows = Vec::with_capacity(feature_bytes.len() / row_width);
        for (id, row_bytes) in feature_bytes.chunks_exact(row_width).enumerate() {
            let features: Vec<Value> = row_bytes
                .chunks_exact(8)
                .map(|bytes| {
                    let value = f64::from_le_bytes(bytes.try_into().expect("8-byte value"));
                    if value.is_finite() {
                        json!(value)
                    } else {
                        Value::Null
                    }
                })
                .collect();
            rows.push(json!({"id": id, "features": features}));
        }
        let rows_path = self.root.join("fixture-rows.json");
        fs::write(&rows_path, serde_json::to_vec(&rows).map_err(err)?).map_err(err)?;
        self.run_ok(
            &self.baiez,
            &[
                "model".into(),
                "load".into(),
                "--db".into(),
                self.root.join("leader").display().to_string(),
                "--key".into(),
                MODEL_KEY.to_string(),
                "--input".into(),
                dump_path.display().to_string(),
                "--format".into(),
                "json".into(),
                "--rows".into(),
                rows_path.display().to_string(),
            ],
        )?;
        Ok(())
    }

    fn attest_fixture(&self, id: usize) -> Result<bool> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/lightgbm_regression_1024");
        let score_bytes = fs::read(fixture.join("native_scores.f64")).map_err(err)?;
        if score_bytes.len() % 8 != 0 {
            return Err(err("fixture score data has an invalid byte length"));
        }
        let expected: Vec<f64> = score_bytes
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("8-byte score")))
            .collect();
        let cohorts = [
            ("contiguous", (480..544).collect::<Vec<_>>()),
            (
                "scattered",
                (0..64).map(|offset| offset * expected.len() / 64).collect(),
            ),
            (
                "shuffled",
                (0..64)
                    .map(|offset| (offset * 13 % 64) * expected.len() / 64)
                    .collect(),
            ),
            ("reversed", (0..expected.len()).rev().collect()),
        ];
        for (name, ids) in cohorts {
            let ids_path = self.root.join(format!("fixture-ids-{name}.json"));
            fs::write(&ids_path, serde_json::to_vec(&ids).map_err(err)?).map_err(err)?;
            let output = self.run_ok(
                &self.baiez,
                &[
                    "predict".into(),
                    "--peer-socket".into(),
                    node.socket.display().to_string(),
                    "--key".into(),
                    MODEL_KEY.to_string(),
                    "--ids".into(),
                    ids_path.display().to_string(),
                ],
            )?;
            let predictions: Value = serde_json::from_str(&output).map_err(err)?;
            let predictions = predictions
                .as_array()
                .ok_or_else(|| err("prediction output is not an array"))?;
            if predictions.len() != ids.len() {
                return Err(err(format!("{name} cohort returned the wrong row count")));
            }
            for (position, (&row_id, prediction)) in ids.iter().zip(predictions).enumerate() {
                let actual = prediction["scores"][0]
                    .as_f64()
                    .ok_or_else(|| err("fixture prediction has no score"))?;
                if !actual.is_finite() || (actual - expected[row_id]).abs() > 1e-10 {
                    return Err(err(format!(
                        "replicated LightGBM score mismatch in {name} cohort at output {position}, row {row_id}: {actual} != {}",
                        expected[row_id]
                    )));
                }
            }
        }
        Ok(true)
    }

    fn predict(&self, id: usize, key: u64) -> Result<Vec<f64>> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        fs::write(self.root.join("ids.json"), "[9,7]").map_err(err)?;
        let output = self.run_ok(
            &self.baiez,
            &[
                "predict".into(),
                "--peer-socket".into(),
                node.socket.display().to_string(),
                "--key".into(),
                key.to_string(),
                "--ids".into(),
                self.root.join("ids.json").display().to_string(),
            ],
        )?;
        let value: Value = serde_json::from_str(&output).map_err(err)?;
        value
            .as_array()
            .ok_or_else(|| err("prediction output is not an array"))?
            .iter()
            .map(|row| {
                row["scores"][0]
                    .as_f64()
                    .ok_or_else(|| err("missing score"))
            })
            .collect()
    }

    fn predict_namespaced(
        &self,
        id: usize,
        key: u64,
        bits: u8,
        namespace_id: u32,
    ) -> Result<Vec<f64>> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        fs::write(self.root.join("ids.json"), "[9,7]").map_err(err)?;
        let output = self.run_ok(
            &self.baiez,
            &[
                "predict".into(),
                "--peer-socket".into(),
                node.socket.display().to_string(),
                "--key".into(),
                key.to_string(),
                "--namespace-bits".into(),
                bits.to_string(),
                "--namespace-id".into(),
                namespace_id.to_string(),
                "--ids".into(),
                self.root.join("ids.json").display().to_string(),
            ],
        )?;
        let value: Value = serde_json::from_str(&output).map_err(err)?;
        value
            .as_array()
            .ok_or_else(|| err("prediction output is not an array"))?
            .iter()
            .map(|row| {
                row["scores"][0]
                    .as_f64()
                    .ok_or_else(|| err("missing score"))
            })
            .collect()
    }

    fn predict_leaf(&self, id: usize, key: u64) -> Result<Vec<Vec<u64>>> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        fs::write(self.root.join("ids.json"), "[9,7]").map_err(err)?;
        let output = self.run_ok(
            &self.baiez,
            &[
                "predict".into(),
                "--peer-socket".into(),
                node.socket.display().to_string(),
                "--key".into(),
                key.to_string(),
                "--ids".into(),
                self.root.join("ids.json").display().to_string(),
                "--leaf".into(),
            ],
        )?;
        let value: Value = serde_json::from_str(&output).map_err(err)?;
        value
            .as_array()
            .ok_or_else(|| err("prediction output is not an array"))?
            .iter()
            .map(|row| {
                row["leaves"]
                    .as_array()
                    .ok_or_else(|| err("missing leaves"))?
                    .iter()
                    .map(|leaf| leaf.as_u64().ok_or_else(|| err("invalid leaf index")))
                    .collect()
            })
            .collect()
    }

    fn missing_model(&self, id: usize, key: u64) -> Result<bool> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        fs::write(self.root.join("ids.json"), "[9,7]").map_err(err)?;
        let output = self.run(
            &self.baiez,
            &[
                "predict".into(),
                "--peer-socket".into(),
                node.socket.display().to_string(),
                "--key".into(),
                key.to_string(),
                "--ids".into(),
                self.root.join("ids.json").display().to_string(),
            ],
        )?;
        let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
        if output.status.success() {
            return Ok(false);
        }
        if !stderr.contains("invalidbundle(") || !stderr.contains("missing or invalid bundle magic")
        {
            return Err(err(format!(
                "missing model returned an unexpected diagnostic; stdout: {}; stderr: {}",
                String::from_utf8_lossy(&output.stdout),
                stderr
            )));
        }
        Ok(true)
    }

    fn wait_predict(&mut self, id: usize, key: u64, expected: f64) -> Result<()> {
        loop {
            self.check_alive(id)?;
            if let Ok(values) = self.predict(id, key) {
                if values == [expected, 2.0] {
                    return Ok(());
                }
            }
            std::thread::sleep(self.remaining()?.min(Duration::from_millis(100)));
        }
    }

    fn dump_leaf(&self, id: usize, key: u64) -> Result<f64> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        let path = self.root.join(format!("dump-{key}.json"));
        self.run_ok(
            &self.baiez,
            &[
                "model".into(),
                "dump".into(),
                "--peer-socket".into(),
                node.socket.display().to_string(),
                "--key".into(),
                key.to_string(),
                "--format".into(),
                "json".into(),
                "--output".into(),
                path.display().to_string(),
            ],
        )?;
        let value: Value = serde_json::from_slice(&fs::read(path).map_err(err)?).map_err(err)?;
        value["tree_info"][0]["tree_structure"]["right_child"]["leaf_value"]
            .as_f64()
            .ok_or_else(|| err("missing right leaf"))
    }

    fn publish(&self, id: usize, key: u64, leaf: f64) -> Result<()> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/simple_model.json");
        let mut dump: Value =
            serde_json::from_slice(&fs::read(model).map_err(err)?).map_err(err)?;
        dump["tree_info"][0]["tree_structure"]["right_child"]["leaf_value"] = json!(leaf);
        let offline = Db::open(self.root.join(format!("offline-{key}"))).map_err(err)?;
        let mut bundle = ModelBundle::from_dump_json(dump.to_string(), None).map_err(err)?;
        let left = [0.0];
        let right = [3.0];
        let rows = [
            IndexedRow {
                id: 7,
                features: &left,
            },
            IndexedRow {
                id: 9,
                features: &right,
            },
        ];
        bundle.store(&offline, key, Some(&rows)).map_err(err)?;
        let snapshot = offline.snapshot().map_err(err)?;
        let mut pairs = String::new();
        for stored_key in [key, key + 1] {
            for ordinal in snapshot.load(stored_key).map_err(err)?.iter() {
                pairs.push_str(&format!("{stored_key},{ordinal}\n"));
            }
        }
        let count = pairs.lines().count();
        let path = self.root.join(format!("pairs-{key}.csv"));
        fs::write(&path, pairs).map_err(err)?;
        let output = self.run_ok(
            &self.yesno,
            &[
                "--endpoint".into(),
                format!("http://127.0.0.1:{}", node.flight),
                "put".into(),
                path.display().to_string(),
            ],
        )?;
        if !output.contains(&format!("ingested {count} of {count} pairs")) {
            return Err(err(format!("unexpected Flight put result: {output}")));
        }
        Ok(())
    }

    fn reject_write(&self, id: usize) -> Result<bool> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        let path = self.root.join("refused.csv");
        fs::write(&path, "9999,1\n").map_err(err)?;
        let output = self.run(
            &self.yesno,
            &[
                "--endpoint".into(),
                format!("http://127.0.0.1:{}", node.flight),
                "put".into(),
                path.display().to_string(),
            ],
        )?;
        let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
        Ok(!output.status.success()
            && [
                "read-only replica",
                "follower",
                "wrong role",
                "failed_precondition",
            ]
            .iter()
            .any(|word| stderr.contains(word)))
    }

    fn snapshot(&mut self, id: usize, key: u64) -> Result<usize> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        let (_, prepared) = load_peer_bundle(&node.socket, key).map_err(err)?;
        let snapshot = self.snapshots.len();
        self.snapshots.push(prepared);
        Ok(snapshot)
    }

    fn snapshot_predict(&self, id: usize) -> Result<Vec<f64>> {
        let prepared = self
            .snapshots
            .get(id)
            .ok_or_else(|| err("unknown snapshot"))?;
        prepared
            .predict(&[9, 7], PredictOptions::default())
            .map_err(err)
            .map(|rows| rows.iter().map(|row| row.scores[0]).collect())
    }

    fn stop(&mut self, id: usize) -> Result<()> {
        let node = self.nodes.get_mut(id).ok_or_else(|| err("unknown node"))?;
        stop_child(&mut node.process);
        Ok(())
    }

    fn restart(&mut self, id: usize) -> Result<()> {
        let node = self.nodes.get(id).ok_or_else(|| err("unknown node"))?;
        if node.process.is_some() {
            return Err(err("node is still running"));
        }
        let mut args = vec!["--config".into(), node.config.display().to_string()];
        if node.control.is_some() {
            args.push("--insecure-replication".into());
        }
        let child = self.spawn(&node.name, node.starts, &args)?;
        self.nodes[id].process = Some(child);
        self.nodes[id].starts += 1;
        self.wait_ready(id)
    }

    pub fn call(
        &mut self,
        name: &str,
        args: &[MontyObject],
        kwargs: &[(MontyObject, MontyObject)],
    ) -> Result<MontyObject> {
        if !kwargs.is_empty() {
            return Err(err("host verbs do not accept keyword arguments"));
        }
        let arity = match name {
            "seed" | "seed_fixture" | "seed_namespaced" => 0,
            "leader_start" | "wait_pass" | "reject_write" | "snapshot_predict"
            | "attest_fixture" | "stop" | "restart" => 1,
            "follower_start" => {
                if !(1..=2).contains(&args.len()) {
                    return Err(err(format!(
                        "follower_start needs 1 or 2 arguments, got {}",
                        args.len()
                    )));
                }
                args.len()
            }
            "predict" | "dump_leaf" | "snapshot" | "predict_leaf" | "missing_model" => 2,
            "predict_namespaced" => 4,
            "wait_predict" | "publish" => 3,
            _ => return Err(err(format!("unknown host verb: {name}"))),
        };
        if args.len() != arity {
            return Err(err(format!(
                "{name} needs {arity} arguments, got {}",
                args.len()
            )));
        }
        match name {
            "seed" => {
                self.seed()?;
                Ok(MontyObject::None)
            }
            "seed_namespaced" => {
                self.seed_namespaced()?;
                Ok(MontyObject::None)
            }
            "seed_fixture" => {
                self.seed_fixture()?;
                Ok(MontyObject::None)
            }
            "leader_start" => self.node(string(&args[0])?, None).map(handle),
            "follower_start" => {
                let leader = self
                    .nodes
                    .get(index(&args[0])?)
                    .ok_or_else(|| err("unknown leader"))?;
                let control = leader.control.ok_or_else(|| err("node is not a leader"))?;
                let name = args.get(1).map_or(Ok("follower"), |value| string(value))?;
                self.node(name, Some(control)).map(handle)
            }
            "wait_pass" => {
                self.wait_pass(index(&args[0])?)?;
                Ok(MontyObject::None)
            }
            "reject_write" => self.reject_write(index(&args[0])?).map(MontyObject::Bool),
            "predict" => self
                .predict(index(&args[0])?, integer(&args[1])?)
                .map(|v| scores(&v)),
            "predict_leaf" => self
                .predict_leaf(index(&args[0])?, integer(&args[1])?)
                .map(|rows| {
                    MontyObject::List(
                        rows.into_iter()
                            .map(|row| {
                                MontyObject::List(
                                    row.into_iter()
                                        .map(|leaf| MontyObject::Int(leaf as i64))
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                }),
            "predict_namespaced" => self
                .predict_namespaced(
                    index(&args[0])?,
                    integer(&args[1])?,
                    u8::try_from(integer(&args[2])?).map_err(err)?,
                    u32::try_from(integer(&args[3])?).map_err(err)?,
                )
                .map(|values| scores(&values)),
            "missing_model" => self
                .missing_model(index(&args[0])?, integer(&args[1])?)
                .map(MontyObject::Bool),
            "attest_fixture" => self.attest_fixture(index(&args[0])?).map(MontyObject::Bool),
            "wait_predict" => {
                self.wait_predict(index(&args[0])?, integer(&args[1])?, float(&args[2])?)?;
                Ok(MontyObject::None)
            }
            "dump_leaf" => self
                .dump_leaf(index(&args[0])?, integer(&args[1])?)
                .map(MontyObject::Float),
            "publish" => {
                self.publish(index(&args[0])?, integer(&args[1])?, float(&args[2])?)?;
                Ok(MontyObject::None)
            }
            "snapshot" => self
                .snapshot(index(&args[0])?, integer(&args[1])?)
                .map(handle),
            "snapshot_predict" => self.snapshot_predict(index(&args[0])?).map(|v| scores(&v)),
            "stop" => {
                self.stop(index(&args[0])?)?;
                Ok(MontyObject::None)
            }
            "restart" => {
                self.restart(index(&args[0])?)?;
                Ok(MontyObject::None)
            }
            _ => unreachable!(),
        }
    }
}

fn http_get(port: u16, path: &str) -> Option<String> {
    let address = format!("127.0.0.1:{port}").parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(300)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_millis(300)))
        .ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response)
}
