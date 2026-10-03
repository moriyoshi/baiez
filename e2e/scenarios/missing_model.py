# SPDX-License-Identifier: Apache-2.0
# Missing peer keys fail closed while valid models remain readable.
seed()
leader = leader_start("leader")
assert missing_model(leader, 999999)
assert predict(leader, 9000) == [4.0, 2.0]
assert dump_leaf(leader, 9000) == 4.0
print("missing model key rejection and subsequent valid reads passed")
