// SPDX-License-Identifier: Apache-2.0
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use baiez::{
    load_flight_bundle, load_flight_model, load_peer_bundle, load_peer_model, IndexedRow,
    KeyNamespace, Model, ModelBundle, PackedIndex, PredictOptions,
};
use serde_json::{json, Value};
use yesno_core::Db;

#[path = "baiez/native_lightgbm.rs"]
mod native_lightgbm;

type OwnedRows = (Vec<u64>, Vec<Vec<f64>>);

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--help" | "-h" | "help") if args.len() == 2 => {
            println!("{}", usage());
            Ok(())
        }
        Some("--version" | "-V") if args.len() == 2 => {
            println!("baiez {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("model") if args.get(2).map(String::as_str) == Some("load") => model_load(&args[3..]),
        Some("model") if args.get(2).map(String::as_str) == Some("dump") => model_dump(&args[3..]),
        Some("index") if args.get(2).map(String::as_str) == Some("build") => {
            index_build(&args[3..])
        }
        Some("predict") if args.get(2).is_some_and(|arg| arg.starts_with("--")) => {
            bundle_predict(&args[2..])
        }
        Some("index") if args.len() == 7 => index(&args[2..]),
        Some("predict") if args.len() >= 6 => predict(&args[2..]),
        _ => Err(usage().into()),
    }
}

fn usage() -> &'static str {
    "usage:\n  baiez model load --db DB_DIR --key KEY --input MODEL --format json|lightgbm [--rows ROWS.json] [--lib LIGHTGBM_SO] [--namespace-bits BITS --namespace-id ID]\n  baiez model dump (--db DB_DIR | --peer-socket SOCKET | --flight-endpoint URL) --key KEY --format json|lightgbm --output MODEL [--namespace-bits BITS --namespace-id ID]\n  baiez index build --db DB_DIR --key KEY --rows ROWS.json [--namespace-bits BITS --namespace-id ID]\n  baiez predict (--db DB_DIR | --peer-socket SOCKET | --flight-endpoint URL) --key KEY --ids ROW_IDS.json [--namespace-bits BITS --namespace-id ID] [--raw] [--leaf] [--start=N] [--count=N]\nlegacy:\n  baiez index MODEL.json ROWS.json DB_DIR KEY DESCRIPTOR.json\n  baiez predict MODEL.json DESCRIPTOR.json DB_DIR ROW_IDS.json [--raw] [--leaf] [--start=N] [--count=N]"
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=");
    args.iter().enumerate().find_map(|(i, arg)| {
        arg.strip_prefix(&prefix).or_else(|| {
            (arg == name)
                .then(|| args.get(i + 1).map(String::as_str))
                .flatten()
        })
    })
}

fn required<'a>(args: &'a [String], name: &str) -> Result<&'a str, Box<dyn Error>> {
    flag(args, name).ok_or_else(|| format!("missing {name}").into())
}

/// Resolve a bundle model ID to its physical yesno key, or preserve the raw
/// key for callers that do not opt in to namespace mapping.
fn model_key(args: &[String]) -> Result<u64, Box<dyn Error>> {
    let key: u64 = required(args, "--key")?.parse()?;
    match (flag(args, "--namespace-bits"), flag(args, "--namespace-id")) {
        (None, None) => Ok(key),
        (Some(bits), Some(id)) => {
            let namespace = KeyNamespace::new(id.parse()?, bits.parse()?)?;
            Ok(namespace.model_key(key)?)
        }
        _ => Err("--namespace-bits and --namespace-id must be provided together".into()),
    }
}

fn validate_flags(
    args: &[String],
    value_flags: &[&str],
    switches: &[&str],
) -> Result<(), Box<dyn Error>> {
    let mut seen = HashSet::new();
    let mut position = 0;
    while position < args.len() {
        let argument = &args[position];
        let (name, inline_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(name, value)| {
                (name, Some(value))
            });
        if !seen.insert(name) {
            return Err(format!("duplicate option: {name}").into());
        }
        if value_flags.contains(&name) {
            if inline_value.is_none() {
                position += 1;
                if position == args.len() || args[position].starts_with("--") {
                    return Err(format!("missing value for {name}").into());
                }
            } else if inline_value == Some("") {
                return Err(format!("empty value for {name}").into());
            }
        } else if switches.contains(&name) && inline_value.is_none() {
            // Boolean switch with no value.
        } else {
            return Err(format!("unknown option: {argument}").into());
        }
        position += 1;
    }
    Ok(())
}

fn read_rows(path: &str) -> Result<OwnedRows, Box<dyn Error>> {
    let rows: Value = serde_json::from_slice(&fs::read(path)?)?;
    let entries = rows.as_array().ok_or("rows must be a JSON array")?;
    let mut ids = Vec::with_capacity(entries.len());
    let mut features = Vec::with_capacity(entries.len());
    for entry in entries {
        ids.push(
            entry
                .get("id")
                .and_then(Value::as_u64)
                .ok_or("row lacks id")?,
        );
        let values = entry
            .get("features")
            .and_then(Value::as_array)
            .ok_or("row lacks features")?;
        features.push(
            values
                .iter()
                .map(|value| {
                    if value.is_null() {
                        Ok(f64::NAN)
                    } else {
                        value.as_f64().ok_or("feature is not numeric or null")
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok((ids, features))
}

fn read_ids(path: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    let ids: Value = serde_json::from_slice(&fs::read(path)?)?;
    ids.as_array()
        .ok_or_else(|| "row IDs must be a JSON array".into())
        .and_then(|ids| {
            ids.iter()
                .map(|id| {
                    id.as_u64()
                        .ok_or_else(|| "row ID is not an unsigned integer".into())
                })
                .collect()
        })
}

fn options(args: &[String]) -> Result<PredictOptions, Box<dyn Error>> {
    let mut options = PredictOptions {
        raw_score: args.iter().any(|arg| arg == "--raw"),
        pred_leaf: args.iter().any(|arg| arg == "--leaf"),
        ..Default::default()
    };
    if let Some(value) = flag(args, "--start") {
        options.start_iteration = value.parse()?;
    }
    if let Some(value) = flag(args, "--count") {
        options.num_iteration = Some(value.parse()?);
    }
    Ok(options)
}

fn model_load(args: &[String]) -> Result<(), Box<dyn Error>> {
    validate_flags(
        args,
        &[
            "--db",
            "--key",
            "--input",
            "--format",
            "--rows",
            "--lib",
            "--namespace-bits",
            "--namespace-id",
        ],
        &[],
    )?;
    let db = Db::open(required(args, "--db")?)?;
    let key = model_key(args)?;
    let path = required(args, "--input")?;
    let format = required(args, "--format")?;
    let (dump_json, native_text) = match format {
        "json" => (fs::read_to_string(path)?, None),
        "lightgbm" => {
            let native = fs::read_to_string(path)?;
            let dump = native_lightgbm::dump_model(Path::new(path), flag(args, "--lib"))?;
            (dump, Some(native))
        }
        _ => return Err(format!("unknown model format: {format}").into()),
    };
    let mut bundle = ModelBundle::from_dump_json(dump_json, native_text)?;
    if let Some(path) = flag(args, "--rows") {
        let (ids, features) = read_rows(path)?;
        let rows: Vec<_> = ids
            .iter()
            .zip(&features)
            .map(|(&id, features)| IndexedRow { id, features })
            .collect();
        bundle.store(&db, key, Some(&rows))?;
    } else {
        bundle.store(&db, key, None)?;
    }
    Ok(())
}

fn model_dump(args: &[String]) -> Result<(), Box<dyn Error>> {
    validate_flags(
        args,
        &[
            "--db",
            "--peer-socket",
            "--flight-endpoint",
            "--key",
            "--format",
            "--output",
            "--namespace-bits",
            "--namespace-id",
        ],
        &[],
    )?;
    let key = model_key(args)?;
    let bundle = match (
        flag(args, "--db"),
        flag(args, "--peer-socket"),
        flag(args, "--flight-endpoint"),
    ) {
        (Some(path), None, None) => ModelBundle::load(&Db::open(path)?.snapshot()?, key)?,
        (None, Some(path), None) => load_peer_model(path, key)?,
        (None, None, Some(endpoint)) => load_flight_model(endpoint, key)?,
        _ => return Err("provide exactly one of --db, --peer-socket, or --flight-endpoint".into()),
    };
    let contents = match required(args, "--format")? {
        "json" => bundle.dump_json(),
        "lightgbm" => bundle
            .native_text()
            .ok_or("bundle has no native LightGBM text")?,
        other => return Err(format!("unknown model format: {other}").into()),
    };
    fs::write(required(args, "--output")?, contents)?;
    Ok(())
}

fn index_build(args: &[String]) -> Result<(), Box<dyn Error>> {
    validate_flags(
        args,
        &[
            "--db",
            "--key",
            "--rows",
            "--namespace-bits",
            "--namespace-id",
        ],
        &[],
    )?;
    let db = Db::open(required(args, "--db")?)?;
    let key = model_key(args)?;
    let mut bundle = ModelBundle::load(&db.snapshot()?, key)?;
    let (ids, features) = read_rows(required(args, "--rows")?)?;
    let rows: Vec<_> = ids
        .iter()
        .zip(&features)
        .map(|(&id, features)| IndexedRow { id, features })
        .collect();
    bundle.store(&db, key, Some(&rows))?;
    Ok(())
}

fn bundle_predict(args: &[String]) -> Result<(), Box<dyn Error>> {
    validate_flags(
        args,
        &[
            "--db",
            "--peer-socket",
            "--flight-endpoint",
            "--key",
            "--ids",
            "--start",
            "--count",
            "--namespace-bits",
            "--namespace-id",
        ],
        &["--raw", "--leaf"],
    )?;
    let key = model_key(args)?;
    let ids = read_ids(required(args, "--ids")?)?;
    let predictions = match (
        flag(args, "--db"),
        flag(args, "--peer-socket"),
        flag(args, "--flight-endpoint"),
    ) {
        (Some(path), None, None) => {
            let db = Db::open(path)?;
            let snapshot = db.snapshot()?;
            let bundle = ModelBundle::load(&snapshot, key)?;
            let index = bundle.index().ok_or("model bundle has no index")?;
            index
                .prepare(&snapshot, bundle.model())?
                .predict(&ids, options(args)?)?
        }
        (None, Some(path), None) => load_peer_bundle(path, key)?
            .1
            .predict(&ids, options(args)?)?,
        (None, None, Some(endpoint)) => load_flight_bundle(endpoint, key)?
            .1
            .predict(&ids, options(args)?)?,
        _ => return Err("provide exactly one of --db, --peer-socket, or --flight-endpoint".into()),
    };
    let output: Vec<_> = predictions
        .into_iter()
        .map(|p| json!({"scores": p.scores, "leaves": p.leaves}))
        .collect();
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn index(args: &[String]) -> Result<(), Box<dyn Error>> {
    let model = Model::from_dump_json(&fs::read_to_string(&args[0])?)?;
    let (ids, features) = read_rows(&args[1])?;
    let rows: Vec<_> = ids
        .iter()
        .zip(&features)
        .map(|(&id, values)| IndexedRow {
            id,
            features: values,
        })
        .collect();
    let key: u64 = args[3].parse()?;
    let db = Db::open(&args[2])?;
    let descriptor = PackedIndex::build(&db, &model, key, &rows)?;
    let contents = serde_json::to_vec_pretty(&json!({
        "key": descriptor.key,
        "stride": descriptor.stride,
        "num_predicates": descriptor.num_predicates,
        "predicate_fingerprint": descriptor.predicate_fingerprint
    }))?;
    fs::write(&args[4], contents)?;
    Ok(())
}

fn predict(args: &[String]) -> Result<(), Box<dyn Error>> {
    let model = Model::from_dump_json(&fs::read_to_string(&args[0])?)?;
    let descriptor: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let field = |name: &str| -> Result<u64, Box<dyn Error>> {
        Ok(descriptor
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("descriptor lacks {name}"))?)
    };
    let index = PackedIndex::from_descriptor(
        field("key")?,
        field("stride")?,
        usize::try_from(field("num_predicates")?)?,
        field("predicate_fingerprint")?,
    )?;
    let row_ids = read_ids(&args[3])?;
    validate_flags(&args[4..], &["--start", "--count"], &["--raw", "--leaf"])?;
    let options = options(&args[4..])?;
    let db = Db::open(&args[2])?;
    let snapshot = db.snapshot()?;
    let prepared = index.prepare(&snapshot, &model)?;
    let result = prepared.predict(&row_ids, options)?;
    let output: Vec<_> = result
        .into_iter()
        .map(|prediction| json!({"scores": prediction.scores, "leaves": prediction.leaves}))
        .collect();
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
