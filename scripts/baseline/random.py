"""
Baseline solution consisting of cycling through all phases with a fixed duration.
"""

import numpy as np
from tlrl.sim.env import SumoEnv

env = SumoEnv(
    network="sumo/eval/0/intersection.net.xml",
    config="sumo-env.cfg",
    render=True,
    verbose=False,
    rnd_seed=58736,
)
env.reset()

# Random action on the agent's identical action window
while True:
    action = np.random.randint(0, 4)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print("total reward:", sum(env.reward))
        print("episode disutility:", info["disutility_sum"])
        print("vehicle disutility:", info["disutility_avg"])
        break
with open("agg_results.csv", "a") as file:
    file.write(
        f"random_5,{int(info["disutility_avg"])},{int(info["disutility_sum"])}\n"
    )
    file.close()
