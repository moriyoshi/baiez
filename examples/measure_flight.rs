// SPDX-License-Identifier: Apache-2.0
//! Benchmark a prepared baiez scorer over an Apache Arrow Flight DoExchange RPC.
//!
//! Reuses the fixture from `scripts/measure_lightgbm.py` and runs a Flight
//! server over loopback TCP. The server, connection, model, and index are ready
//! before any request is timed.
//!
//! cargo run --release --offline --features flight-benchmark --example measure_flight -- \
//!   /tmp/baiez-lightgbm-measure

use std::error::Error;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightClient, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use baiez::{IndexedRow, Model, PackedIndex, PredictOptions, PreparedIndex};
use futures::stream::BoxStream;
use futures::{stream, StreamExt, TryStreamExt};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Endpoint, Server};
use tonic::{Request, Response, Status, Streaming};
use yesno_core::Db;

type BenchResult<T> = Result<T, Box<dyn Error>>;
static MODEL: OnceLock<Model> = OnceLock::new();

#[derive(Clone)]
struct InferenceService {
    prepared: Arc<PreparedIndex>,
    last_inference_ns: Arc<AtomicU64>,
}

#[tonic::async_trait]
impl FlightService for InferenceService {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;

    async fn handshake(
        &self,
        _: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake"))
    }

    async fn list_flights(
        &self,
        _: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }

    async fn get_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    async fn do_get(&self, _: Request<Ticket>) -> Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented("do_get"))
    }

    async fn do_put(
        &self,
        _: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put"))
    }

    async fn do_action(
        &self,
        _: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action"))
    }

    async fn list_actions(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let decoder =
            FlightRecordBatchStream::new_from_flight_data(request.into_inner().map_err(Into::into));
        let batches: Vec<RecordBatch> = decoder.try_collect().await.map_err(Status::from)?;
        if batches.len() != 1 || batches[0].num_columns() != 1 {
            return Err(Status::invalid_argument("expected one row_id batch"));
        }
        let row_ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Status::invalid_argument("row_id must be UInt64"))?;
        if row_ids.null_count() != 0 {
            return Err(Status::invalid_argument("row_id must not contain nulls"));
        }

        let mut scores = vec![0.0; row_ids.len()];
        let start = Instant::now();
        self.prepared
            .predict_into(
                row_ids.values(),
                PredictOptions {
                    raw_score: true,
                    ..Default::default()
                },
                &mut scores,
                None,
            )
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        self.last_inference_ns
            .store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let output = RecordBatch::try_from_iter(vec![(
            "score",
            Arc::new(Float64Array::from(scores)) as ArrayRef,
        )])
        .map_err(|error| Status::internal(error.to_string()))?;
        let encoded = FlightDataEncoderBuilder::new().build(stream::iter(vec![Ok(output)]));
        Ok(Response::new(encoded.map_err(Status::from).boxed()))
    }
}

fn read_f64(path: &std::path::Path) -> BenchResult<Vec<f64>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(8) {
        return Err("partial f64 value".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|part| f64::from_le_bytes(part.try_into().unwrap()))
        .collect())
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

async fn exchange(client: &mut FlightClient, batch: &RecordBatch) -> BenchResult<Vec<RecordBatch>> {
    let encoded = FlightDataEncoderBuilder::new().build(stream::iter(vec![Ok(batch.clone())]));
    Ok(client.do_exchange(encoded).await?.try_collect().await?)
}

async fn measure_case(
    label: &str,
    ids: Vec<u64>,
    native: &[f64],
    prepared: &PreparedIndex,
    client: &mut FlightClient,
    server_ns: &AtomicU64,
) -> BenchResult<()> {
    let batch = RecordBatch::try_from_iter(vec![(
        "row_id",
        Arc::new(UInt64Array::from(ids.clone())) as ArrayRef,
    )])?;
    let response = exchange(client, &batch).await?;
    if response.len() != 1 || response[0].num_columns() != 1 || response[0].num_rows() != ids.len()
    {
        return Err("Flight response has the wrong shape".into());
    }
    let scores = response[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("Flight score column is not Float64")?;
    let max_error = ids
        .iter()
        .enumerate()
        .map(|(position, &id)| (scores.value(position) - native[id as usize]).abs())
        .fold(0.0f64, f64::max);
    if max_error > 1e-10 {
        return Err(format!("{label}: maximum error {max_error}").into());
    }

    let options = PredictOptions {
        raw_score: true,
        ..Default::default()
    };
    let mut local_scores = vec![0.0; ids.len()];
    let repetitions = match ids.len() {
        0..=64 => 30,
        65..=1024 => 12,
        1025..=8192 => 5,
        _ => 2,
    };
    let mut local_samples = Vec::new();
    let mut flight_samples = Vec::new();
    let mut server_samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..repetitions {
            prepared.predict_into(&ids, options, black_box(&mut local_scores), None)?;
            black_box(&local_scores);
        }
        local_samples.push(start.elapsed().as_secs_f64() * 1e6 / repetitions as f64);

        let mut inference_ns = 0u64;
        let start = Instant::now();
        for _ in 0..repetitions {
            let response = exchange(client, &batch).await?;
            black_box(response);
            inference_ns += server_ns.load(Ordering::Relaxed);
        }
        flight_samples.push(start.elapsed().as_secs_f64() * 1e6 / repetitions as f64);
        server_samples.push(inference_ns as f64 / 1e3 / repetitions as f64);
    }
    println!(
        "{label},{},{:.3},{:.3},{:.3},{max_error:.3e}",
        ids.len(),
        median(local_samples),
        median(server_samples),
        median(flight_samples)
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> BenchResult<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: measure_flight FIXTURE_DIR")?;
    let directory = std::path::Path::new(&directory);
    let loaded_model =
        Model::from_dump_json(&std::fs::read_to_string(directory.join("model.json"))?)?;
    MODEL.set(loaded_model).expect("model initialized once");
    let model = MODEL.get().unwrap();
    if model.num_features() != 8 || model.output_count() != 1 {
        return Err("fixture must have eight features and one output".into());
    }
    let values = read_f64(&directory.join("rows.f64"))?;
    let features: Vec<[f64; 8]> = values
        .chunks_exact(8)
        .map(|part| part.try_into().unwrap())
        .collect();
    let native = read_f64(&directory.join("native_scores.f64"))?;
    if features.len() != native.len() {
        return Err("feature and score row counts differ".into());
    }
    let rows: Vec<_> = features
        .iter()
        .enumerate()
        .map(|(id, features)| IndexedRow {
            id: id as u64,
            features,
        })
        .collect();
    let directory_db = tempfile::tempdir()?;
    let db = Db::open(directory_db.path())?;
    let index = PackedIndex::build(&db, model, 1, &rows)?;
    db.checkpoint()?;
    let snapshot = db.snapshot()?;
    let prepared = Arc::new(index.prepare(&snapshot, model)?);
    let server_ns = Arc::new(AtomicU64::new(0));
    let service = InferenceService {
        prepared: Arc::clone(&prepared),
        last_inference_ns: Arc::clone(&server_ns),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    // `serve_with_incoming` does not apply tonic's server TCP_NODELAY setting.
    let incoming = TcpListenerStream::new(listener).map(|result| {
        result.and_then(|socket| {
            socket.set_nodelay(true)?;
            Ok(socket)
        })
    });
    let server = tokio::spawn(
        Server::builder()
            .add_service(FlightServiceServer::new(service))
            .serve_with_incoming(incoming),
    );
    let channel = Endpoint::from_shared(format!("http://{address}"))?
        .connect()
        .await?;
    let mut client = FlightClient::new(channel);

    println!(
        "transport=Arrow Flight DoExchange loopback_tcp rows={}",
        features.len()
    );
    println!("shape,rows,local_us,server_inference_us,flight_roundtrip_us,max_abs_error");
    for &size in &[1usize, 64, 1_024, 8_192, features.len()] {
        if size > features.len() {
            continue;
        }
        let begin = (features.len() - size) / 2;
        let contiguous = (begin..begin + size).map(|id| id as u64).collect();
        measure_case(
            "contiguous",
            contiguous,
            &native,
            &prepared,
            &mut client,
            &server_ns,
        )
        .await?;
        if size > 1 && size < features.len() {
            let scattered: Vec<u64> = (0..size)
                .map(|i| ((i as u128 * features.len() as u128) / size as u128) as u64)
                .collect();
            measure_case(
                "scattered",
                scattered.clone(),
                &native,
                &prepared,
                &mut client,
                &server_ns,
            )
            .await?;
            if size == 8_192 {
                let shuffled = (0..size).map(|i| scattered[(i * 4_051) % size]).collect();
                measure_case(
                    "scattered_shuffled",
                    shuffled,
                    &native,
                    &prepared,
                    &mut client,
                    &server_ns,
                )
                .await?;
            }
        }
    }
    server.abort();
    Ok(())
}
