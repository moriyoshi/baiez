# SPDX-License-Identifier: Apache-2.0
# Multiple followers bootstrap and receive the same atomic model generation.
seed()
leader = leader_start("leader")
first = follower_start(leader, "follower_a")
second = follower_start(leader, "follower_b")
wait_pass(first)
wait_pass(second)
assert predict(first, 9000) == [4.0, 2.0]
assert predict(second, 9000) == [4.0, 2.0]
publish(leader, 9030, 7.0)
wait_predict(first, 9030, 7.0)
wait_predict(second, 9030, 7.0)
assert predict(first, 9030) == [7.0, 2.0]
assert predict(second, 9030) == [7.0, 2.0]
assert predict(first, 9000) == [4.0, 2.0]
assert predict(second, 9000) == [4.0, 2.0]
print("two followers bootstrapped and received the same model rollout")
