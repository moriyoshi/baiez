# Architecture and invariants

## Module map

| Path | Responsibility |
| --- | --- |
| `src/model.rs` | Parse LightGBM dump JSON, normalize predicates, and evaluate tree and objective semantics. |
| `src/index.rs` | Build the packed predicate set and score row-ID cohorts from snapshots. |
| `src/bundle.rs` | Persist versioned model metadata and optional index descriptor. |
| `src/namespace.rs` | Map logical model IDs into configurable high-bit yesno key namespaces. |
| `src/peer.rs` | Read snapshot-pinned model and packed containers over the yesnod plugin peer socket. |
| `src/flight.rs` | Fetch yesno sets over Arrow Flight, pinning both bundle keys to the metadata ticket's version. |
| `src/bin/baiez.rs` | CLI for model load/dump, index build, and prediction. |
| `src/bin/baiez/native_lightgbm.rs` | Optional LightGBM C API text import. |
| `src/python.rs` | PyO3 indexed scoring interface. |
| `python/baiez/lightgbm.py` | Optional LightGBM-compatible Booster adapter. |
| `e2e/` | Managed leader/follower process fixtures and scenario checks for peer rollout and restart. |
| `baiez-e2e/`, `e2e/`, `scripts/gate-e2e.sh` | Rust-hosted Monty scenarios against real yesnod processes. |
| `examples/measure*.rs` | Local, real-fixture, and Flight benchmark probes. |

## Model and packed index

`Model` compiles a LightGBM JSON dump into trees and deduplicated split
predicates. `PackedIndex` stores one yesnodb ordinal set under one key. For
stride `S`, physical ordinal `row_id` represents a live row; physical ordinal
`(predicate + 1) * S + row_id` represents that predicate taking its left
branch. `S` is a positive multiple of 65,536 so each predicate maps to whole
yesno containers. Every arithmetic operation that constructs a physical
ordinal must respect yesnodb's reserved `u64::MAX` ordinal.

Row IDs are compact, stable slots, not arbitrary external IDs. An index must
reject duplicate row IDs and missing or out-of-range requested IDs. The
descriptor records stride, predicate count, and predicate fingerprint. Model
and index must agree before a predicate set is read or scored.

Dense cohorts partition container sets with yesno `AND` and `AND NOT`;
small or scattered cohorts can check membership along each tree path. Leaf
values accumulate in model tree order. An optimized path must preserve score,
leaf-index order, selected iteration window, missing-value semantics, and
output shape. `PreparedIndex` owns decoded predicate sets captured from one
snapshot, so repeated predictions may outlive that snapshot.

## Bundle and rollout

`ModelBundle` stores versioned metadata as bit ordinals under key `K` and
reserves `K + 1` for the packed index. A load with rows writes both keys in one
yesnodb batch; a reader opens one snapshot. The metadata includes model dump,
optional native text, and index descriptor. Changes to the magic, framing,
checksum, version, or key mapping are persisted-format changes and need
round-trip and compatibility review. Use a fresh key pair for a model
generation when existing readers need to keep using the old one.

`KeyNamespace` reserves 1–32 high key bits for a namespace ID. The remaining
bits encode a logical model ID plus the low metadata/index selector bit. The
same width must be used by all clients of one database; the namespace mapping
is deployment configuration, and namespaced data must not overlap legacy raw
keys. Namespace prefixes partition yesno keys only, not row ordinals.

The peer loader reads both keys from one remote snapshot. A follower may be
behind its leader, but a single load must never mix model and index versions.
The Unix socket closes after preparation; owned sets can be scored repeatedly.
On follower rebootstrap, a generation change invalidates the load and starts
it again. The peer path decodes array, bitmap, and inclusive-end run lanes.
The Flight loader reads the metadata key first and uses its ticket version for
the packed-index query, then owns the decoded ordinal sets after the streams
close. This preserves bundle consistency over a network endpoint.

## Python boundary

`baiez.lightgbm.Booster.predict(X)` uses the official LightGBM package for
ordinary feature matrices. `predict(IndexedRows(ids))` and `predict_ids(ids)`
use the Rust index. NumPy result shapes must match the LightGBM modes baiez
supports. The adapter checks model agreement when binding an index and clears
the binding after model mutation. Indexed prediction needs NumPy; ordinary
prediction needs LightGBM. Unsupported objectives, linear leaves, and indexed
feature contributions must fail explicitly.

## Dependency boundary

yesno-core owns set containers, views, database batches, snapshots, and
durability. baiez owns LightGBM parsing, split semantics, row layout, model
metadata, and output formatting. Keep experiments out of production APIs.
