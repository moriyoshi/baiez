# SPDX-License-Identifier: Apache-2.0
# The authoritative yesnod owner serves baiez reads before and after restart.
seed()
leader = leader_start("leader")
old = snapshot(leader, 9000)
assert predict(leader, 9000) == [4.0, 2.0]
assert predict_leaf(leader, 9000) == [[1], [0]]
assert dump_leaf(leader, 9000) == 4.0
stop(leader)
assert snapshot_predict(old) == [4.0, 2.0]
restart(leader)
assert predict(leader, 9000) == [4.0, 2.0]
assert predict_leaf(leader, 9000) == [[1], [0]]
assert dump_leaf(leader, 9000) == 4.0
print("leader peer scores, leaf output, model dump, and reopen passed")
