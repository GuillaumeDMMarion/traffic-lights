import glob
import yaml

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from tlrl.mod import extractor
from tlrl.mod.callbacks import (
    CurriculumCallback,
    CustomEvalCallback,
    # MonitoringCallback,
    # SchedulerCallback,
)
from tlrl.sim.env import SumoEnvFactory
from tlrl.sim.utils import get_available_port
from tlrl.sim.wrapper import VecFrameStack3D


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(train_cfg, sumo_cfg):
    # Check for sumo folders
    train_networks = glob.glob("sumo/train/*/*.net.xml")
    train_ports = [get_available_port() for _ in range(len(train_networks))]
    eval_networks = glob.glob("sumo/eval/*/*.net.xml")
    eval_ports = [get_available_port() for _ in range(len(eval_networks))]

    # Extract and derive times
    episode_seconds = sumo_cfg["end_time"]
    seconds_per_envstep = sumo_cfg["seconds_per_envstep"]
    steps_per_episode = episode_seconds // seconds_per_envstep
    num_envs = len(train_networks) - 1 if train_cfg["vectorize"] else 1
    total_timesteps = steps_per_episode * train_cfg["train_episodes"] * num_envs
    save_freq_intelligible = steps_per_episode * 500
    save_freq = save_freq_intelligible // num_envs
    eval_freq_intelligible = steps_per_episode * 50
    eval_freq = eval_freq_intelligible // num_envs
    curriculum_length = int(train_cfg["train_episodes"] * train_cfg["curriculum"])

    # Env factory
    fact = SumoEnvFactory(
        config="sumo-env.cfg",
        render=False,
        verbose=train_cfg["verbose"],
    )

    # Wrap for vectorization
    if train_cfg["vectorize"]:
        env = SubprocVecEnv(
            [
                lambda network=network, port=port: fact.make_env(network, port)
                for network, port in zip(train_networks, train_ports)
            ]
        )
        eval_env = SubprocVecEnv(
            [
                lambda network=network, port=port, seed=58736: fact.make_env(
                    network, port, rnd_seed=seed
                )  # Monitor()
                for network, port in zip(eval_networks, eval_ports)
            ]
        )

    else:
        env = DummyVecEnv([lambda: fact.make_env()])
        eval_env = DummyVecEnv(
            [
                lambda network=eval_networks[0],
                port=eval_ports[0],
                seed=58736: fact.make_env(network, port, rnd_seed=seed)  # Monitor()
            ]
        )

    if train_cfg["framestack"]:
        env = VecFrameStack3D(env, n_stack=train_cfg["framestack"])
        eval_env = VecFrameStack3D(eval_env, n_stack=train_cfg["framestack"])

    # Define the custom model using the custom extractor
    features_extractor = getattr(
        getattr(extractor, train_cfg["arch"]), "TrafficlightFeatureExtractor"
    )
    policy_kwargs = dict(
        features_extractor_class=features_extractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # Agent setup
    if train_cfg["checkpoint_path"]:
        print()
        print(
            f">>> Picking up training from checkpoint: {train_cfg['checkpoint_path']}"
        )
        load_kwargs = {
            "path": train_cfg["checkpoint_path"],
            "env": env,
            "tensorboard_log": f"logs/{train_cfg['model']}",
        }
        model = globals()[train_cfg["model"]].load(**load_kwargs)
    else:
        print()
        print(">>> Training new model")
        tensorboard_log = (
            f"logs/{train_cfg['model']}" if train_cfg["callback"] else None
        )
        default_model_kwargs = {
            "PPO": dict(
                policy="MultiInputPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,  # 5e-4
                n_steps=2048,  # 512
                batch_size=256,  # 256
                n_epochs=10,  # 10
                gamma=0.99,  # 0.95
                gae_lambda=0.90,  # 0.95
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.02,  # 0.001
                vf_coef=0.5,  # 0.5
                max_grad_norm=0.5,  # 5
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=None,
                tensorboard_log=tensorboard_log,
                device="cuda",
                verbose=2,
            ),
            "DQN": dict(
                policy="MultiInputPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=0.00001,  # 0.001
                buffer_size=300000,  # 300000
                learning_starts=10000,  # 20000
                batch_size=256,  # 256
                gamma=0.99,  # 0.98
                target_update_interval=1000,  # 500
                exploration_fraction=0.4,  # 0.5
                exploration_final_eps=0.1,  # 0.15
                max_grad_norm=10,
                tensorboard_log=tensorboard_log,
                device="cuda",
                verbose=2,
            ),
            "A2C": dict(
                policy="MultiInputPolicy",
                env=env,
                tensorboard_log=tensorboard_log,
                device="cuda",
                verbose=2,
            ),
        }
        model_kwargs = {
            **default_model_kwargs[train_cfg["model"]],
            **train_cfg["model_kwargs"],
        }
        model = globals()[train_cfg["model"]](**model_kwargs)

    # Callbacks
    if train_cfg["callback"]:
        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./models/",
            name_prefix=f"{train_cfg['model']}_checkpoint",
        )
        callbacks = [eval_callback, checkpoint_callback]
        if train_cfg["checkpoint_path"]:
            env.set_attr("w1", train_cfg["target_w1"])
            eval_env.set_attr("w1", train_cfg["target_w1"])
        else:
            curr_callback = CurriculumCallback(
                eval_env,
                curriculum_length=curriculum_length,
                target_w1=train_cfg["target_w1"],
            )
            callbacks += [curr_callback]
    else:
        callbacks = None

    # Train the model (and save a last checkpoint if interrupted)
    try:
        reset_num_timesteps = False if train_cfg["checkpoint_path"] else True
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=train_cfg["name"],
        )
    except KeyboardInterrupt:
        print()
        print(">>> KeyboardInterrupt: Saving model one last time...")
        # model.save(f"models/last_checkpoint")
        print()
        print(">>> Model saved.")


if __name__ == "__main__":
    train_cfg = load_config("train.cfg")
    sumo_cfg = load_config("sumo-env.cfg")
    print()
    print(">>> Training parameters")
    print(*["{}: {}".format(k, v) for k, v in train_cfg.items()], sep="\n")
    print()
    print(">>> Sumo parameters")
    print(*["{}: {}".format(k, v) for k, v in sumo_cfg.items()], sep="\n")
    main(train_cfg=train_cfg, sumo_cfg=sumo_cfg)
