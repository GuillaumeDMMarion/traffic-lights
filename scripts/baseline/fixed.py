"""
Baseline solution consisting of cycling through all phases with a fixed duration.
"""

import numpy as np

from tlrl.sim.env import SumoEnv


### USER INPUT ###############
RENDER = False
VERBOSE = False
SEED = 58736
##############################


def main():
    env = SumoEnv(
        network="sumo/eval/0/intersection.net.xml",
        config="sumo-env.cfg",
        render=RENDER,
        verbose=VERBOSE,
        rnd_seed=SEED,
    )
    env.reset()

    # Constant uniform length cycling
    n_samples = 1
    for timing in range(3, 30):
        disutility_sums = []
        disutility_avgs = []
        for sample in range(n_samples):
            info = env.cycle(timing)
            disutility_sums.append(info["disutility_sum"])
            disutility_avgs.append(info["disutility_avg"])
        with open("agg_results.csv", "a") as file:
            vehicle_disutility = int(np.mean(disutility_avgs))
            episode_disutility = int(np.mean(disutility_sums))
            print(timing, vehicle_disutility, episode_disutility)
            file.write(f"uniform_{timing},{vehicle_disutility},{episode_disutility}\n")
            file.close()

    # Constant non-uniform length cycling
    ...


if __name__ == "__main__":
    main()
