# baiez

An experimental LightGBM inference engine backed by yesnodb set arithmetic.
It scores cohorts of rows that have already been indexed. LightGBM's native
library can be used to export a model dump; prediction itself runs in Rust.

## Layout

Each packed index occupies one yesnodb key as an aligned blocked view.
Constituent zero is the set of live row slots. Constituent `p + 1` contains the
slots whose feature values take predicate `p`'s left branch. For stride `S`, a
membership bit is stored at physical ordinal `(p + 1) * S + row_slot`. `S` is a
multiple of 65,536, so selecting a constituent only shifts chunk prefixes.
A versioned model bundle can store the dump and descriptor in a neighboring
metadata key. The bundle encodes bytes as bit ordinals because yesnodb stores
ordinal sets. The metadata key is `K`; the packed view is `K + 1`. Loading a
model with rows commits both keys in one batch and a reader opens one snapshot.

For dense cohorts, the prepared scorer partitions one 65,536-row container at
a time with yesno's `AND` and `AND NOT` operations. Small or scattered cohorts
check membership along each tree path. Leaf values are accumulated in original
tree order. `PackedIndex::prepare` captures all predicate containers from one
yesnodb snapshot for repeated calls.

## Try it

The export helper needs the official Python `lightgbm` package. Building from
this checkout needs Rust 1.95 or newer and the sibling `../yesno` checkout.

```console
python3 scripts/export_lightgbm.py model.txt model.json
cargo run -- model load --db ./db --key 9000 --input model.json --format json --rows rows.json
cargo run -- predict --db ./db --key 9000 --ids ids.json --raw --leaf
cargo run -- model dump --db ./db --key 9000 --format json --output exported.json
```

The model and index can also be loaded separately with `baiez model load`
followed by `baiez index build --db ./db --key 9000 --rows rows.json`.
`--format lightgbm` imports native LightGBM text through its C API when the
shared library is installed; pass `--lib /path/to/lib_lightgbm.so` if needed.
The original text is preserved for `model dump --format lightgbm`. A JSON-only
import can be dumped as JSON, but has no native text to return. The original
positional `index` and `predict` commands remain available.

To share one yesnodb keyspace, opt in to a namespace on every bundle command:

```console
baiez model load --db ./db --key 42 --namespace-bits 12 --namespace-id 37 --input model.json --format json --rows rows.json
baiez predict --db ./db --key 42 --namespace-bits 12 --namespace-id 37 --ids ids.json
baiez predict --flight-endpoint http://127.0.0.1:50051 --key 42 --namespace-bits 12 --namespace-id 37 --ids ids.json
```

The width must be from 1 to 32 bits, and the namespace ID must fit that width.
The remaining key bits hold the logical model ID and its two-key bundle pair:
metadata uses an even key and the packed index uses the following odd key.
Every client of a database must use the same namespace width. Without both
namespace options, `--key` keeps its existing meaning as a raw yesno key.
Namespaced keys occupy encoded ranges, so migrate or re-key any existing data
that overlaps those ranges before enabling namespaces in a populated database.
Rust callers can use `baiez::KeyNamespace::new(namespace_id, bits)?.model_key(model_id)`
to obtain the same even metadata key before calling `ModelBundle::store` or a
peer loader. The Python adapter accepts the matching `namespace_bits` and
`namespace_id` keyword arguments on `from_yesno`, `from_peer`, `from_flight`,
`bind_index`, `bind_peer`, and `bind_flight`.

`rows.json` is an array of `{"id": 10, "features": [1.2, null, 3]}` objects.
`null` denotes NaN. IDs are compact, stable nonnegative row slots. `ids.json`
is an array of slots, such as `[10, 11]`. `predict` returns an array in the
requested order, with scores and optional leaf indices.

For repeated calls in Rust, prepare once and reuse the output buffer:

```rust
let snapshot = db.snapshot()?;
let prepared = index.prepare(&snapshot, &model)?;
let mut scores = vec![0.0; row_ids.len() * model.output_count()];
prepared.predict_into(&row_ids, options, &mut scores, None)?;
```

Scores are row-major. With `options.pred_leaf = true`, pass a leaf buffer with
one entry per selected tree per row. The prepared scorer owns its decoded
predicate sets, so the snapshot may be dropped after preparation.

Use a fresh key pair for each new model generation when readers should retain
the old one. The bundle commands replace both keys atomically at the selected
pair. Keep the feature data elsewhere if rows will be updated or a new model
will be indexed; yesnodb stores the model and split memberships, not features.

## Replica sidecar

baiez can read a bundle through a local yesnod plugin peer socket. Configure a
yesno follower with `server.role = "follower"`, `follower.leader` pointing at the
leader's replication endpoint, `follower.serve_reads = true`, and
`plugin.channel_socket` on a Unix path shared with baiez. The follower's TLS and
authorization settings still need to match the leader's replication setup.
baiez does not open the follower's data directory.

```console
baiez predict --peer-socket /run/yesno/baiez.sock --key 9000 --ids ids.json
baiez model dump --peer-socket /run/yesno/baiez.sock --key 9000 --format json --output model.json
```

In Python, use `lgb.Booster.from_peer("/run/yesno/baiez.sock", 9000)` or attach
the same bundle to an existing Booster with `bind_peer(socket_path, key)`.
baiez loads model metadata and the packed index from one follower snapshot,
then scores against owned yesno sets. The socket is closed after preparation.
Reopen the bundle to observe later replication or model updates. A follower may
lag the leader; during startup or rebootstrap, baiez waits up to 120 seconds
for the database to become available. On initial bootstrap, wait for the
follower's first completed replication pass (`yesnod_follower_passes_total`)
before loading a bundle; the socket can exist while shards are still copying.
The peer socket must allow the baiez
process's user ID to connect; see yesno's plugin channel configuration.

## Arrow Flight

For a yesno server exposing Arrow Flight, use its HTTP or HTTPS endpoint:

```console
baiez predict --flight-endpoint http://127.0.0.1:50051 --key 9000 --ids ids.json
baiez model dump --flight-endpoint http://127.0.0.1:50051 --key 9000 --format json --output model.json
```

Flight reads the model metadata first, then fetches the packed index at the
version carried by that metadata response. This keeps both keys on one
consistent snapshot while the server is receiving updates. Python callers can
use `lgb.Booster.from_flight(endpoint, key)` or attach the same bundle with
`bind_flight(endpoint, key)`. Namespace arguments work with these commands and
methods too.

## Python binding

The optional PyO3 extension builds with `pip install -e '.[lightgbm]'` from
this checkout. The official `lightgbm` package handles ordinary feature-matrix
predictions and training; indexed row IDs use Rust and yesnodb:

```python
import baiez.lightgbm as lgb

booster = lgb.Booster(model_file="model.txt").bind_index("./db", 9000)
ordinary = booster.predict(X)
indexed = booster.predict(lgb.IndexedRows(row_ids))
# Or open a stored bundle directly:
indexed = lgb.Booster.from_yesno("./db", 9000).predict_ids(row_ids)
# A namespaced bundle uses its logical model ID and the same width/ID pair:
indexed = lgb.Booster.from_yesno(
    "./db", 42, namespace_bits=12, namespace_id=37
).predict_ids(row_ids)
# Or read an indexed bundle from a yesno replica sidecar:
indexed = lgb.Booster.from_peer("/run/yesno/baiez.sock", 9000).predict_ids(row_ids)
```

`predict_ids` accepts a one-dimensional integer array and returns a NumPy
array with LightGBM's score or leaf shape. It batches IDs into one Rust call;
the Rust scorer releases the Python interpreter lock while predicting. The
adapter rejects a mismatched model when binding an index and clears the index
after model mutation methods. `pred_contrib` requires feature data and is not
available for indexed IDs. Other LightGBM names, including sklearn estimators,
are exposed from the official package; they currently use its predictor.

## Current scope

The importer supports ordinary numeric and categorical tree splits, regression
and ranking identity outputs, binary sigmoid, and multiclass softmax. It
rejects unsupported objectives and linear leaves. Feature contributions and
dynamic row updates remain to be implemented. The current builder holds the
packed membership set in memory while constructing it. Preparation loads all
predicate slots, so its memory use grows with the packed index size. The
one-shot `PackedIndex::predict` API reads only the needed predicate ranges;
the CLI uses the prepared path.

## Validation and performance

The [benchmark and attestation notes](BENCHMARKS.md) include score parity on a
100,000-row LightGBM fixture, Python and CLI latency, and loopback Flight
measurements. The scripts and machine-readable run reports are in `scripts/`
and `reports/`.

The [real-process E2E harness](e2e/README.md) checks a running yesnod leader
and follower, including live model rollout and follower restart:

```console
./scripts/gate-e2e.sh
```

## License

baiez is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). The yesno-core and yesno-plugin notices distributed with Python source
packages are preserved in [third_party/yesno](third_party/yesno).
Release checks and the yesno dependency publishing prerequisites are documented in
[RELEASING.md](RELEASING.md).

For contributions and the local coding gate, see
[CONTRIBUTING.md](CONTRIBUTING.md).
