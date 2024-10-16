"""
Custom callbacks
"""

import torch
import numpy as np
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class CurriculumCallback(BaseCallback):
    """
    A callback to control the curriculum (i.e., the w1 hyperparameter balancing myopic shaped and long-term unshaped reward).
    during training.
    """

    def __init__(self, eval_env, curriculum_length, target_w1=0.5, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.target_w1 = target_w1
        self.curriculum_length = curriculum_length

    def _on_step(self):
        if self.curriculum_length:
            # Access the train_env from self.locals
            train_env = self.locals["self"].env

            # Get episode progress from train_env (assuming constant episode length across environments)
            episode_counter = np.min(train_env.get_attr("episode"))

            # Calculate w1 based on training progress
            w1 = min(episode_counter / self.curriculum_length, self.target_w1)

            # # Set w1 in all sub-environments in train_env
            # train_env.env_method('advance_curriculum', w1=w1) # Droped Monitor() usage because I couldn't make this work.
            train_env.set_attr("w1", w1)

            # # Set w1 in all sub-environments in eval_env
            # self.eval_env.env_method('advance_curriculum', w1=w1) # Droped Monitor() usage because I couldn't make this work.
            self.eval_env.set_attr("w1", w1)

        return True


class CustomMetricsCallback(BaseCallback):
    """
    Callback to log custom metrics for the exact same episodes as the evaluation runs. Added mean_total_reward as comparison.
    """

    def __init__(
        self,
        episode_disutils,
        vehicle_disutils,
        shaped_rewards,
        unshaped_rewards,
        total_rewards,
        w1s,
        verbose=0,
    ):
        super().__init__(verbose)
        # self.logger = logger
        self.episode_disutils = episode_disutils
        self.vehicle_disutils = vehicle_disutils
        self.shaped_rewards = shaped_rewards
        self.unshaped_rewards = unshaped_rewards
        self.total_rewards = total_rewards
        self.w1s = w1s

    def _on_step(self) -> bool:
        # Calculate and log averages after evaluation

        if len(self.episode_disutils) > 0:
            avg_episode_disutility = np.nanmean(self.episode_disutils)
            self.logger.record("avg_episode_disutility", avg_episode_disutility)

        if len(self.vehicle_disutils) > 0:
            avg_vehicle_disutility = np.nanmean(self.vehicle_disutils)
            self.logger.record("avg_vehicle_disutility", avg_vehicle_disutility)

        if len(self.shaped_rewards) > 0:
            mean_shaped_reward = np.nanmean(self.shaped_rewards)
            self.logger.record("eval/mean_shaped_reward", mean_shaped_reward)

        if len(self.unshaped_rewards) > 0:
            mean_unshaped_reward = np.nanmean(self.unshaped_rewards)
            self.logger.record("eval/mean_unshaped_reward", mean_unshaped_reward)

        if len(self.total_rewards) > 0:
            mean_total_reward = np.nanmean(self.total_rewards)
            self.logger.record("eval/mean_total_reward", mean_total_reward)

        if len(self.w1s) > 0:
            mean_w1 = np.mean(self.w1s)
            self.logger.record("eval/mean_w1", mean_w1)

        self.episode_disutils.clear()
        self.vehicle_disutils.clear()
        self.shaped_rewards.clear()
        self.unshaped_rewards.clear()
        self.total_rewards.clear()
        self.w1s.clear()

        return True


class CustomEvalCallback(EvalCallback):
    """
    Callback to log custom metrics for the exact same episodes as the evaluation runs. Added mean_total_reward as comparison.
    """

    def __init__(self, *args, **kwargs):
        self.episode_disutils = []
        self.vehicle_disutils = []
        self.shaped_rewards = []
        self.unshaped_rewards = []
        self.total_rewards = []
        self.w1s = []

        callback_after_eval = CustomMetricsCallback(
            # logger=self.logger,
            episode_disutils=self.episode_disutils,
            vehicle_disutils=self.vehicle_disutils,
            shaped_rewards=self.shaped_rewards,
            unshaped_rewards=self.unshaped_rewards,
            total_rewards=self.total_rewards,
            w1s=self.w1s,
            verbose=kwargs.get("verbose", 0),
        )

        super().__init__(callback_after_eval=callback_after_eval, *args, **kwargs)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Collect metrics from the `info` dict during evaluation episodes.
        """
        for i, done in enumerate(locals_["dones"]):
            if done:
                # Collect disutility_sum from the info dict if it exists
                info = locals_["infos"][i]
                self.episode_disutils.append(info.get("disutility_sum", np.nan))
                self.vehicle_disutils.append(info.get("disutility_avg", np.nan))
                self.shaped_rewards.append(info.get("shaped_reward_sum", np.nan))
                self.unshaped_rewards.append(info.get("unshaped_reward_sum", np.nan))
                self.total_rewards.append(info.get("total_reward_sum", np.nan))
                self.w1s.append(info.get("w1", np.nan))

        # Call the parent callback for success metrics
        super()._log_success_callback(locals_, globals_)


class SchedulerCallback(BaseCallback):
    """
    Callback to update the learning rate on each step.
    """

    def __init__(self, initial_lr, decay_factor, step_size, verbose=0):
        super(SchedulerCallback, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.learning_rate = initial_lr

    def _on_step(self) -> bool:
        # Update the learning rate every step_size steps
        if self.num_timesteps % self.step_size == 0:
            self.learning_rate *= self.decay_factor
            self.model.learning_rate = self.learning_rate
            if self.verbose > 0:
                print(f"Updated learning rate to: {self.learning_rate}")
        return True


class MonitoringCallback(BaseCallback):
    """
    Debuggin Callback to print high and low gradient norms.
    """

    def __init__(
        self, upper_threshold=5.0, lower_threshold=0.001, log_freq=5000, verbose=0
    ):
        super(MonitoringCallback, self).__init__(verbose)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            model = self.model
            total_grad_norm = 0.0
            grad_count = 0
            with torch.no_grad():
                for name, param in model.policy.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += grad_norm
                        grad_count += 1
                        # Print if above upper threshold
                        if grad_norm > self.upper_threshold:
                            print(f"High gradient norm for {name}: {grad_norm}")
                        # Print if below lower threshold
                        elif grad_norm < self.lower_threshold:
                            print(f"Low gradient norm for {name}: {grad_norm}")

            if grad_count > 0:
                avg_grad_norm = total_grad_norm / grad_count
                print(f"Average gradient norm: {avg_grad_norm:.6f}")

        return True
