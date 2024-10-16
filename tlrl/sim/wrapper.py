"""
Stacks multiple frames together for vectorized environments.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces


class VecFrameStack3D(VecEnvWrapper):
    def __init__(self, venv, n_stack):
        super(VecFrameStack3D, self).__init__(venv)
        self.n_stack = n_stack

        # Get the original observation space (assume it's a spaces.Box)
        orig_space = venv.observation_space
        self.channels, self.height, self.width = orig_space.shape

        # Add a new "depth" dimension after the channels (axis=1)
        low = np.expand_dims(
            orig_space.low, axis=1
        )  # Shape: (channels, 1, height, width)
        high = np.expand_dims(orig_space.high, axis=1)

        # Repeat along this new depth axis (axis=1)
        self.observation_space = spaces.Box(
            low=np.repeat(
                low, n_stack, axis=1
            ),  # Shape: (channels, n_stack, height, width)
            high=np.repeat(high, n_stack, axis=1),
            shape=(self.channels, n_stack, self.height, self.width),  # Desired shape
            dtype=orig_space.dtype,
        )

        # Initialize storage for stacked observations
        self.stackedobs = None

    def reset(self):
        # Reset the environment and get the initial observation
        obs = self.venv.reset()  # shape (batch_size, channels, height, width)

        # Initialize the stack with n_stack copies of the first observation along the depth axis
        self.stackedobs = np.repeat(
            obs[:, :, None, :, :], self.n_stack, axis=2
        )  # Add depth axis after channels

        return self.stackedobs

    def step_wait(self):
        # Step the environment and get the new observation
        obs, rewards, dones, infos = self.venv.step_wait()

        # Roll the stack along the depth axis (axis=2), making room for the new observation
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=2)

        # Replace the last frame with the new observation along the depth axis
        self.stackedobs[:, :, -1, :, :] = obs  # Update last frame in the stack

        return self.stackedobs, rewards, dones, infos


class VecFrameStackDict(VecEnvWrapper):
    def __init__(self, venv, n_stack):
        super(VecFrameStackDict, self).__init__(venv)
        self.n_stack = n_stack

        # Adapt the observation space: For each key, increase the number of channels by n_stack
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=np.repeat(space.low, n_stack, axis=0),
                    high=np.repeat(space.high, n_stack, axis=0),
                    shape=(
                        space.shape[0] * n_stack,
                        *space.shape[1:],
                    ),  # Multiply channels by n_stack
                    dtype=space.dtype,
                )
                for key, space in venv.observation_space.spaces.items()
            }
        )
        print(self.observation_space)

        # Initialize storage for stacked observations
        self.stackedobs = {key: None for key in venv.observation_space.spaces.keys()}

    def reset(self):
        # Reset the environment and get the initial observations (OrderedDict of arrays)
        observations = self.venv.reset()

        # Initialize the stack for each key by repeating the first observation n_stack times
        for key, obs in observations.items():
            # Shape of observations[key]: (n_envs, channels, height, width)
            # Stack the first observation along the channel axis (axis=1) for all environments
            self.stackedobs[key] = np.repeat(obs, self.n_stack, axis=1)

        return self.stackedobs

    def step_wait(self):
        # Step the environment with the given actions
        observations, rewards, dones, infos = self.venv.step_wait()

        # Update stacked observations for each key
        for key, obs in observations.items():
            # Shift the existing frames to make room for the new one
            self.stackedobs[key] = np.roll(
                self.stackedobs[key], shift=-obs.shape[1], axis=1
            )
            # Replace the last frame with the new observation
            self.stackedobs[key][:, -obs.shape[1] :, :, :] = obs

        return self.stackedobs, rewards, dones, infos
