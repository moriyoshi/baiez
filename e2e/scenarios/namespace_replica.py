# SPDX-License-Identifier: Apache-2.0
# Namespace encoded keys replicate and remain addressable over the peer socket.
seed_namespaced()
leader = leader_start("leader")
follower = follower_start(leader)
wait_pass(follower)
assert predict_namespaced(follower, 42, 8, 3) == [4.0, 2.0]
print("namespace-prefixed model and index keys replicated to peer reader")
