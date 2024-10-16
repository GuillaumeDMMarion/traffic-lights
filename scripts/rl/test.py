import numpy as np
import stable_baselines3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from tlrl.sim.env import SumoEnv
from tlrl.sim.wrapper import VecFrameStack3D


### USER INPUT ###############
MODEL = "PPO"
ARCH = "att_3d_cnn"
FRAMESTACK = 3
RENDER = False
VERBOSE = False
SEED = 58736
BEST_MODEL_PATH = "models/final_model.zip"
##############################


def main():
    # Initialize SumoEnv environment
    env = SumoEnv(
        network="sumo/eval/0/intersection.net.xml",
        config="sumo-env.cfg",
        render=RENDER,
        verbose=VERBOSE,
        rnd_seed=SEED,
    )

    # Wrap env with dummyvec
    env = DummyVecEnv([lambda: env])

    # Framestack wrapping
    if FRAMESTACK:
        env = VecFrameStack3D(env, n_stack=FRAMESTACK)

    # Load the model
    model = getattr(stable_baselines3, MODEL).load(BEST_MODEL_PATH)

    # Run the simulation
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        disutility_dict = env._get_target_envs(0)[0].disutility_dict
        obs, reward, done, truncated = env.step(action)
        total_reward += reward
    disutility_sum = np.sum(list(disutility_dict.values()))
    disutility_avg = np.mean(list(disutility_dict.values()))

    # Print metrics
    print(disutility_sum)
    print(disutility_avg)
    print(total_reward)


if __name__ == "__main__":
    main()
