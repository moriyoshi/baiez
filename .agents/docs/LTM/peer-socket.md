# Replica peer socket

baiez reads a yesnod follower through the plugin Unix socket, not the
replication control endpoint. It opens one remote snapshot and reads bundle
metadata under key `K` and the packed set under `K + 1` from that snapshot.
The peer channel transports container lanes; baiez decodes them into owned
yesno sets and closes the socket. Rebinding is required to see later changes.

The sidecar socket appears before bootstrap is complete. `UNAVAILABLE` means
the database slot is empty, but a partially bootstrapped follower can also
answer with an empty metadata key. Orchestrate initial loading after the
follower's first completed replication pass. A follower is asynchronous and
may still lag the leader after that pass.

The real-socket regression uses a follower-role channel in both arena and
inline modes. A separate live two-daemon TLS run confirmed that the CLI and
Python `Booster.from_peer` both scored the replicated fixture correctly. The
checked-in E2E harness now also verifies a live new key pair, stability of an
older prepared Booster, and a follower restart that catches up on a missed
model generation. The E2E fixture uses a short system temporary path for Unix
sockets because a nested CI checkout can exceed the socket path limit; its
configs and logs remain in the repository scratch area.

Source: JOURNAL.md, “2026-10-04 — yesno replica peer socket”.
