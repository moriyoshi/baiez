# SPDX-License-Identifier: Apache-2.0
# Replicated real LightGBM fixture predictions match its saved native scores.
seed_fixture()
leader = leader_start("leader")
follower = follower_start(leader)
wait_pass(follower)
assert attest_fixture(follower)
print("replicated LightGBM scores match native reference across four row orders")
