# SPDX-License-Identifier: Apache-2.0
# A stopped follower reopens the same directory and catches up on missed WAL.
seed()
leader = leader_start("leader")
follower = follower_start(leader)
wait_pass(follower)
assert predict(follower, 9000) == [4.0, 2.0]
stop(follower)
publish(leader, 9020, 8.0)
restart(follower)
wait_predict(follower, 9020, 8.0)
assert predict(follower, 9020) == [8.0, 2.0]
assert predict(follower, 9000) == [4.0, 2.0]
assert dump_leaf(follower, 9020) == 8.0
print("follower restart, WAL catch-up, and peer-backed CLI reads passed")
