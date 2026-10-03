# SPDX-License-Identifier: Apache-2.0
# A follower bootstraps, rejects writes, and serves an atomic model rollout.
seed()
leader = leader_start("leader")
follower = follower_start(leader)
wait_pass(follower)
assert reject_write(follower)
assert predict(follower, 9000) == [4.0, 2.0]
assert dump_leaf(follower, 9000) == 4.0
old = snapshot(follower, 9000)
assert snapshot_predict(old) == [4.0, 2.0]
publish(leader, 9010, 6.0)
wait_predict(follower, 9010, 6.0)
assert predict(follower, 9010) == [6.0, 2.0]
assert dump_leaf(follower, 9010) == 6.0
assert snapshot_predict(old) == [4.0, 2.0]
assert predict(follower, 9000) == [4.0, 2.0]
print("bootstrap, write refusal, CLI score and dump, Flight rollout, and snapshot stability passed")
