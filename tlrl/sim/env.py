"""
SUMO gymnasium environment and factory.
"""

# default
import warnings
import logging
import yaml
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, List, Tuple, Union
from types import SimpleNamespace
from subprocess import Popen
from xml.dom import minidom
from pathlib import Path

# gymnasium
from gymnasium import Env, spaces
from gymnasium.utils import seeding

# sumo
import traci
from traci.constants import (
    VAR_LANE_ID,
    VAR_POSITION,
    VAR_NEXT_TLS,
    VAR_ACCUMULATED_WAITING_TIME,
    VAR_SPEED,
    VAR_ACCEL,
)
from sumolib import checkBinary

# other
import numpy as np

# tlrl
from tlrl.sim.trip import generate, delete


class SumoEnv(Env):
    """
        Description:
            Gymnasium wrapper for SUMO (Simulation of Urban MObility) runs.

        Observation:
            A dictionary of arrays representing the traffic light phase and vehicle attributes (speed, acceleration, etc.).

        Actions:
            Discrete actions indicating the desired next phase.

        Reward:
            A weigthed combination of 2 components:
            a) 1 - the increase in normalized accumulated disutility for all vehicles at the intersection.
            b) total (scaled) episode disutility.

        Episode termination:
            When end_time is reached.

    Args:
        network: The simulation's network filepath, other filepaths will be derived from it.
        config : The environment's config filepath.
        binary: Allows specifying the binary name, overrides potential render values.
        render: Enables sumo vizualization. Will utilize gui-settings.cfg if located in the same folder as the network.
        verbose: Enables debug printing.
        conn_retries: Number of times to retry connecting to SUMO.
        port: The port that will be used to connect to SUMO.
    """

    YELLOW_PHASE_SECONDS = 3  # Duration of each yellow phases in seconds. This shouldn't be changed, or at most be set between [3, 6].

    MAX_VEHICLE_DISUTILITY = 6000  # 3000 TODO: investigate if we can set this dynamically. CRITICAL VALUE SETTING!
    MAX_WAIT = 600  # 120.0 # TODO: investigate if we can set this dynamically. CRITICAL VALUE SETTING!

    MAX_SPEED = 16.36  # 13.89 # TODO: investigate if we can set this dynamically.
    MIN_ACCEL = -4.5
    MAX_ACCEL = 2.6

    def __init__(
        self,
        network: str,
        config: str,
        binary: Optional[str] = None,
        render: bool = True,
        verbose: bool = False,
        conn_retries: int = 10,
        port: int = 8813,
        **kwargs,
    ):
        # Define network and derived filepaths.
        network = Path(network).resolve()
        self.path = SimpleNamespace(root=None, prefix=None)
        self.path.root = network.parent
        self.path.prefix = network.stem.split(".")[0]

        # Load sumo env configuration.
        with open("sumo-env.cfg", "r") as f:
            cfg = yaml.safe_load(f)

        # Assert required parameters.
        required_keys = [
            "obs_center",
            "obs_length",
            "obs_nrows",
            "tls_id",
            "tls_lanes",
            "tls_phases",
            "rnd_src",
            "rnd_dst",
        ]
        for key in required_keys:
            assert key in cfg, f"{key} required for initializing SumoEnv."

        # Update cfg with default None values.
        cfg.setdefault("rnd_seed", None)
        cfg.setdefault("rnd_state", None)
        cfg.setdefault("rnd_scale", (10, 10))
        cfg.setdefault("end_time", 600)
        cfg.setdefault("seconds_per_sumostep", 1.0)
        cfg.setdefault("seconds_per_envstep", SumoEnv.YELLOW_PHASE_SECONDS + 1)
        cfg.setdefault("delay_exp", 2)

        # Update cfg with any additional kwargs.
        cfg.update(kwargs)

        # Assert them. # TODO: make assertions exhaustive .
        assert (
            cfg["seconds_per_envstep"] >= cfg["seconds_per_sumostep"]
        ), "cannot have a sumostep that last longer than an envstep."
        assert (
            cfg["seconds_per_envstep"] >= SumoEnv.YELLOW_PHASE_SECONDS + 1
        ), "envstep must last at least 1 second longer than the yellow phase."
        assert (
            cfg["seconds_per_envstep"] % 1 == 0
        ), "no decimals allowed for seconds_per_envstep."
        assert cfg["seconds_per_envstep"] > 0, "seconds_per_envstep must be positive."
        assert cfg["seconds_per_sumostep"] > 0, "seconds_per_sumostep must be positive."
        assert (
            cfg["end_time"] >= cfg["seconds_per_sumostep"] * 10
        ), "end_time must be at least an order of magnitude larger than seconds_per_sumostep."

        # Assign them.
        self.obs_center = cfg["obs_center"]
        self.obs_length = cfg["obs_length"]
        self.obs_nrows = cfg["obs_nrows"]
        self.tls_id = cfg["tls_id"]
        self.tls_lanes = cfg["tls_lanes"]
        self.tls_phases = cfg["tls_phases"]
        self.rnd_src = cfg["rnd_src"]
        self.rnd_dst = cfg["rnd_dst"]
        self.rnd_seed = cfg["rnd_seed"]
        self.rnd_state = cfg["rnd_state"]
        self.rnd_scale = cfg["rnd_scale"]
        self.end_time = cfg["end_time"]
        self.seconds_per_sumostep = cfg["seconds_per_sumostep"]
        self.seconds_per_envstep = cfg["seconds_per_envstep"]
        self.delay_exp = cfg["delay_exp"]

        # Create loggers.
        self.loggers = SimpleNamespace(init=None, reset=None, step=None)
        log_filepath = self.path.root / f"{self.path.prefix}.env.log"
        self.loggers.init = SumoEnv._create_logger(
            "sumo_env.init_logger", log_filepath, verbose
        )
        self.loggers.reset = SumoEnv._create_logger(
            "sumo_env.reset_logger", log_filepath, verbose
        )
        self.loggers.step = SumoEnv._create_logger(
            "sumo_env.step_logger", log_filepath, verbose
        )

        # Assert network file exists.
        self.loggers.init.info("[ASSERTING] network file existence... .")
        assert network.is_file(), f"{network} required for initializing SumoEnv."
        self.loggers.init.info("[SUCCESS] assserting network file existence !")

        # Verify default phase durations.
        tree = ET.parse(network)
        root = tree.getroot()
        durations = []
        for tlLogic in root.findall("tlLogic"):
            for phase in tlLogic.findall("phase"):
                durations.append(float(phase.get("duration")))
        if any([duration < cfg["end_time"] for duration in durations]):
            warnings.warn(
                f"Default `phase duration` ({network}) < `self.end_time`. Correct phase action cannot be guaranteed."
            )

        # Defining observation variables.
        self.obs_xmin, self.obs_xmax, self.obs_ymin, self.obs_ymax = (
            SumoEnv._get_boundaries(self.obs_center, self.obs_length, self.obs_nrows)
        )

        # Defining remainder of variables.
        self.binary = binary
        if self.binary is None:
            self.loggers.init.info(
                f"Binary not specified: using default and render={render} arg ..."
            )
            self.binary = "sumo" + ["", "-gui"][int(render)]
            self.loggers.init.info(f"Binary set to: {self.binary}")
        self.verbose = verbose
        self.conn_retries = conn_retries
        self.port = port

        # Defining inter-episode attributes (not to be reset).
        self.episode = 0
        self.sumo_process = None
        self._w2 = 1
        self._w1 = 0

        # Defining gymnasium variables.
        self.action_space = spaces.Discrete(len(self.tls_phases))
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4, self.obs_nrows, self.obs_nrows),
            dtype=np.float64,
        )
        self.observation_space_DEPRECATED = spaces.Dict(
            {
                "phase": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1, self.obs_nrows, self.obs_nrows),
                    dtype=np.float64,
                ),
                "wait": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1, self.obs_nrows, self.obs_nrows),
                    dtype=np.float64,
                ),
                "speed": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1, self.obs_nrows, self.obs_nrows),
                    dtype=np.float64,
                ),
                "accel": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1, self.obs_nrows, self.obs_nrows),
                    dtype=np.float64,
                ),
            }
        )

        # Create config file.
        gui = "gui" in self.binary
        self.loggers.init.info("[CREATING] config file... .")
        SumoEnv._create_config_file(
            self.path.root,
            self.path.prefix,
            end_time=self.end_time,
            gui=gui,
            step_length=self.seconds_per_sumostep,
        )
        self.loggers.init.info(f"[SUCCESS] creating config file at {self.path.root} !")

    @property
    def sumo_time(self) -> float:
        """
        Returns:
            The current SUMO simulation time in seconds.
        """
        return traci.simulation.getTime()

    @property
    def connected(self) -> bool:
        """
        Returns:
            Indication if a connection with SUMO is established.
        """
        try:
            traci.getConnection()
            return True
        except traci.exceptions.TraCIException:
            return False

    @property
    def terminated(self) -> bool:
        """
        Returns:
            Indication if the current episode is terminated.
        """
        if self.end_time is None:
            if self.env_step < 10:  # NOTE: Leave. Tried to hack around some weird bug.
                return False
            else:
                return traci.simulation.getMinExpectedNumber() == 0
        else:
            return self.sumo_time >= self.end_time

    @property
    def w1(self) -> float:
        return self._w1

    @w1.setter
    def w1(self, value: float) -> None:
        self._w1 = value

    @property
    def w2(self) -> float:
        return 1 - self.w1

    def seed(
        self, seed: int, state: Optional[np.random.RandomState] = None
    ) -> List[int]:
        """
        Returns:
            Float of the seed used.
        """
        self.rng, seed = seeding.np_random(seed)
        if state is not None:
            self.rng.set_state(state)
        return [seed]

    @staticmethod
    def _create_logger(name: str, filename: str, to_console: bool) -> logging.Logger:
        """
        Creates a logger with the given name that logs into a file and
        optionally into the console.

        Args:
            name: The name of the logger.
            filename: The name of the file to log into.
            to_console: If True, also log to the console.

        Returns:
            logging.Logger: The created logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create file handler and set level to debug
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create console handler and set level to debug
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def _create_config_file(
        root: Path, prefix: str, end_time: float, gui: bool, step_length: float
    ) -> None:
        """
        Creates a SUMO simulation configuration file.

        Args:
            root: The root directory of the simulation.
            prefix: The prefix of the simulation files.
            end_time: The end time of the simulation in seconds.
            gui: If True, include the GUI settings file.
            step_length: The time step of the SUMO simulation in seconds.
        """

        # Create the root element
        configuration = ET.Element("configuration")

        # Create the input section
        input_element = ET.SubElement(configuration, "input")
        ET.SubElement(input_element, "net-file", {"value": f"{prefix}.net.xml"})
        ET.SubElement(input_element, "route-files", {"value": f"{prefix}.rou.xml"})
        ET.SubElement(input_element, "waiting-time-memory", {"value": "10000000"})
        if gui:
            gui_filename = "gui-settings.cfg"
            if Path(root / gui_filename).is_file():
                ET.SubElement(
                    input_element, "gui-settings-file", {"value": gui_filename}
                )

        # Create the time section
        time_element = ET.SubElement(configuration, "time")
        ET.SubElement(
            time_element, "begin", {"value": "0"}
        )  # Start time of the simulation in seconds.
        ET.SubElement(
            time_element, "end", {"value": str(end_time)}
        )  # End time of the simulation in seconds.
        ET.SubElement(time_element, "step-length", {"value": str(step_length)})

        # Create the report section
        report_element = ET.SubElement(configuration, "report")
        ET.SubElement(report_element, "no-step-log", {"value": "true"})

        # Create the tree and write to the file
        xml_string = ET.tostring(configuration, encoding="utf-8")
        pretty_xml_string = minidom.parseString(xml_string).toprettyxml(indent="    ")
        cfg_file = Path(root / f"{prefix}.sumo.cfg")
        with open(cfg_file, "w") as f:
            f.write(pretty_xml_string)

    @staticmethod
    def _get_boundaries(
        obs_center: Tuple[float, float],
        obs_length: int,
        obs_nrows: int,
    ) -> Tuple[float, float, float, float]:
        """
        Compute coordinate boundaries given the observation matrix' center and size.

        Args:
            obs_center: The x/y coordinates of the intersection's centerpoint.
            obs_length: Length in meters of one unit in the position & speed matrix, i.e. precision.
            obs_nrows: Desired number of rows of the position & speed matrix.

        Returns:
            Floats of the x/y boundaries for the position & speed matrix.
        """
        dx = dy = obs_nrows * obs_length
        deltas = (
            dx,
            dy,
        )  # In case x/y have different lenghts.
        xmin, xmax, ymin, ymax = [
            min_max_offset
            for coord_offsets in [
                (coord - delta / 2, coord + delta / 2)
                for (coord, delta) in zip(obs_center, deltas)
            ]
            for min_max_offset in coord_offsets
        ]
        return xmin, xmax, ymin, ymax

    @staticmethod
    def _get_obs_matrix_index(
        positions: np.ndarray,
        obs_length: int,
        obs_xmin: float,
        obs_ymin: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps vehicle positions to their corresponding indices in the observation matrix.

        Args:
            positions: Tuples of x/y coordinates.
            obs_length: Length in meters of one unit in the position & speed matrix, i.e. precision.
            obs_xmin: Lower bound x value of the observation matrix.
            obs_ymin: Lower bound y value of the observation matrix.

        Returns:
            Tuple of the x/y index in the vehicle observation matrix.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        x_index = np.maximum(0, (x - obs_xmin) // obs_length).astype(np.int32)
        y_index = np.maximum(0, (y - obs_ymin) // obs_length).astype(np.int32)
        return x_index, y_index

    @staticmethod
    def _scale_neg_pos(x: np.ndarray, X_min: float, X_max: float) -> np.ndarray:
        """
        Scales negative and positive values from 0 to their respective min and max values.

        Args:
            x: The input array.
            X_min: The minimum value to which to scale to.
            X_max: The maximum value to which to scale to.

        Returns:
            The scaled array.
        """
        neg_scaled = 0.5 * (x - X_min) / (0 - X_min)
        pos_scaled = 0.5 + 0.5 * (x / X_max)
        return np.where(x < 0, neg_scaled, pos_scaled)

    def _get_vehicle_matrix(self, real_data: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Compute the vehicle observation matrix for the current step.

        Args:
            real_data: Whether to include real data in the observation matrix, or dummy data.

        Returns:
            Tuple of the three Nd-arrays for the vehicle attributes.
        """
        # Get info for matrix-receptacle creation.
        obs_nrows = obs_ncols = self.obs_nrows
        pos_matrix = np.zeros((1, obs_nrows, obs_ncols), dtype=np.float64)

        tls_matrix = np.full((1, obs_nrows, obs_ncols), -1.0, dtype=np.float64)
        speed_matrix = np.full((1, obs_nrows, obs_ncols), -1.0, dtype=np.float64)
        wait_matrix = np.full((1, obs_nrows, obs_ncols), -1.0, dtype=np.float64)
        accel_matrix = np.full((1, obs_nrows, obs_ncols), -1.0, dtype=np.float64)
        # angle_matrix = np.full((1, obs_nrows, obs_ncols), -1.0, dtype=np.float64)

        # Compute observation.
        if len(self.subscr_results) > 0 and real_data:
            num_entries = len(self.subscr_results)
            position = np.empty((num_entries, 2))
            tls = np.empty(num_entries, dtype=object)
            wait = np.empty(num_entries)
            speed = np.empty(num_entries)
            accel = np.empty(num_entries)
            for i, v in enumerate(self.subscr_results.values()):
                position[i] = v[VAR_POSITION]
                tls_info = v[VAR_NEXT_TLS]
                tls[i] = tls_info[0][3] if tls_info else "N"
                wait[i] = v[VAR_ACCUMULATED_WAITING_TIME]
                speed[i] = v[VAR_SPEED]
                accel[i] = v[VAR_ACCEL]
            tls = (tls == "G").astype(int)
            wait /= SumoEnv.MAX_WAIT
            wait = np.minimum(wait, 1.0)
            speed /= SumoEnv.MAX_SPEED
            accel = self._scale_neg_pos(accel, SumoEnv.MIN_ACCEL, SumoEnv.MAX_ACCEL)
            x_index, y_index = self._get_obs_matrix_index(
                position, self.obs_length, self.obs_xmin, self.obs_ymin
            )
            mask = (y_index < pos_matrix.shape[1]) & (x_index < pos_matrix.shape[2])
            x_index = x_index[mask]
            y_index = y_index[mask]
            tls = tls[mask]
            wait = wait[mask]
            speed = speed[mask]
            accel = accel[mask]
            y_index_flipped = pos_matrix.shape[1] - 1 - y_index
            pos_matrix[0, y_index_flipped, x_index] = 1.0
            tls_matrix[0, y_index_flipped, x_index] = tls
            wait_matrix[0, y_index_flipped, x_index] = wait
            speed_matrix[0, y_index_flipped, x_index] = speed
            accel_matrix[0, y_index_flipped, x_index] = accel

        return tls_matrix, wait_matrix, speed_matrix, accel_matrix

    def _update_subscriptions(self) -> None:
        """
        Updates the subscriptions of the vehicle attributes.
        """
        active_veh_ids = traci.vehicle.getIDList()
        new_veh_ids = set(active_veh_ids) - set(self.subscr_results.keys())
        for veh_id in new_veh_ids:
            traci.vehicle.subscribe(
                veh_id,
                [
                    VAR_LANE_ID,
                    VAR_POSITION,
                    VAR_NEXT_TLS,
                    VAR_ACCUMULATED_WAITING_TIME,
                    VAR_SPEED,
                    VAR_ACCEL,
                ],
            )
        self.subscr_results = traci.vehicle.getAllSubscriptionResults()

    @contextmanager
    def _update_reward(self, steps: int) -> None:
        """
        Computes and stores the reward based on the change in normalized disutility between the current and the next step.

        Args:
            steps: The number of steps taken.

        Yields:
            None
        """
        initial_disutil = np.sum(list(self.disutility_dict.values()))
        yield
        final_disutil = np.sum(list(self.disutility_dict.values()))
        penalty = initial_disutil - final_disutil
        self.penalty.append(penalty)
        penalty /= steps
        shaped_reward = max(1 - penalty / self.max_penalty, 0)
        self.shaped_reward.append(shaped_reward)
        unshaped_reward = 0
        if self.terminated:
            avg_disutility = np.mean(list(self.disutility_dict.values()))
            unshaped_reward = max(1 - avg_disutility / self.MAX_VEHICLE_DISUTILITY, 0)
            unshaped_reward *= len(self.shaped_reward)
            self.unshaped_reward.append(unshaped_reward)
        reward = self.w2 * shaped_reward + self.w1 * unshaped_reward
        self.reward.append(reward)

    def _step_and_track_sumo(self) -> None:
        """
        Performs a sumo simulation step and updates the delay & disutility dictionaries.
        """
        # Advance sumo simulation with 1 step.
        traci.simulationStep()
        self._update_subscriptions()
        for veh_id, results in self.subscr_results.items():
            lane_id = results[VAR_LANE_ID]
            if lane_id in self.tls_lanes:
                speed = results[VAR_SPEED]
                self.delay_dict[veh_id] = self.delay_dict[veh_id] + (speed <= 0.1)
                self.disutility_dict[veh_id] = self.delay_dict[veh_id] ** self.delay_exp

    def _advance_sim(self, action: int, steps: int = 1) -> None:
        """
        Advances the simulation by `steps` steps while taking the given action.
        Tracks and records observation and reward.

        Args:
            action: Action to be taken.
            steps: Number of steps to advance.
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        desired_phase = self.tls_phases[action]
        yellow_countdown = 0

        for step in range(steps):
            if yellow_countdown > 0:  # walrus operators don't jam well with sumo.
                yellow_countdown -= 1
                if yellow_countdown == 0:
                    traci.trafficlight.setPhase(self.tls_id, desired_phase)
                    current_phase = desired_phase
            elif desired_phase != current_phase:
                traci.trafficlight.setPhaseDuration(self.tls_id, 0)
                current_phase = None
                yellow_countdown = (
                    SumoEnv.YELLOW_PHASE_SECONDS // self.seconds_per_sumostep
                )
            self._step_and_track_sumo()

    def step(
        self, action: int, seconds: Optional[int] = None, observe: bool = True
    ) -> Tuple[dict, float, bool, bool, dict]:
        """
        Advances the environment by the specified number of seconds, takes the given action, and returns the observation, reward, done, and info.

        Args:
            action: Action to be taken.
            seconds: Number of seconds to advance.
            observe: Indicates whether to compute real observations in the returned data.

        Returns:
            A tuple of the observation, reward, done, truncated, and info.
        """
        if self.env_step == 0:
            self.loggers.step.info("[START] stepping... .")

        seconds = seconds if seconds else self.seconds_per_envstep
        if seconds < SumoEnv.YELLOW_PHASE_SECONDS + 1:
            seconds = SumoEnv.YELLOW_PHASE_SECONDS + 1  # at least 1 second green.
            warnings.warn(
                f"step() seconds should be at least SumoEnv.YELLOW_PHASE_SECONDS+1 = {seconds}s."
            )
        sumosteps_per_envstep = int(seconds / self.seconds_per_sumostep)

        with self._update_reward(steps=sumosteps_per_envstep):
            self._advance_sim(action=action, steps=sumosteps_per_envstep)
        self.env_step += 1
        self.sumo_step += sumosteps_per_envstep

        tls, wait, speed, accel = self._get_vehicle_matrix(real_data=observe)
        obs = np.stack((tls[0], wait[0], speed[0], accel[0]), axis=0)

        truncated = False
        info = {}

        if self.terminated:
            info["w1"] = self.w2
            info["disutility_sum"] = np.sum(list(self.disutility_dict.values()))
            info["disutility_avg"] = np.mean(list(self.disutility_dict.values()))
            info["shaped_reward_sum"] = np.sum(self.shaped_reward)
            info["unshaped_reward_sum"] = np.sum(self.unshaped_reward)
            info["total_reward_sum"] = np.sum(self.reward)
            self.loggers.step.info("[FINISHED] stepping !")
            self.loggers.reset.info("[DELETING] .xml files... .")
            delete(prefix=self.path.prefix, cwd=self.path.root)
            self.loggers.reset.info(
                f"[SUCCESS] deleting {self.path.prefix}.*.xml files !"
            )

        return obs, self.reward[-1], self.terminated, truncated, info

    def cycle(
        self, timing: Optional[Union[int, Tuple[int, int, int, int]]] = None
    ) -> dict:
        """
        Repeats a cycle of actions until the episode is finished.

        Args:
            timing: Seconds per action. None uniformly uses self.seconds_per_action.

        Returns:
            Total delay
        """
        actions = range(self.action_space.n)
        timing = [timing] * len(actions) if isinstance(timing, int) else timing
        timing = timing if timing is not None else [None] * len(actions)

        if self.env_step > 0:
            self.reset()
        done = False
        while not done:
            for action, seconds in zip(actions, timing):
                obs, reward, done, truncated, info = self.step(
                    action=action, seconds=seconds, observe=False
                )
                if done:
                    break
        return info

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """
        Resets the environment.

        Returns:
            The observation and info dictionaries.
        """
        # env_type = ["eval_env", "train_env"][self.rnd_seed is None]
        # print(f"{env_type}: w1={self.w1}, w2={self.w2}, w1+w2={self.w1+self.w2}.")

        ### Some printing.
        self.loggers.reset.info("newline\n")
        self.loggers.reset.info("[STARTING] reset.")

        ### Increase episode counter.
        self.episode += 1

        # Seed
        self.seed(seed=self.rnd_seed, state=self.rnd_state)

        ### Defining intra-episode trackers (to be reset).
        self.env_step = 0  # To keep track of current environment step.
        self.sumo_step = 0  # To keep track of current SUMO simulation step.
        self.delay_dict = defaultdict(lambda: 0)  # Accumulated delay for each vehicle.
        self.disutility_dict = defaultdict(
            lambda: 0
        )  # Accumulated disutility for each vehicle.
        self.penalty = []  # Penalty for each step.
        self.reward = []  # Reward for each step.
        self.shaped_reward = []  # Shaped reward for each step.
        self.unshaped_reward = []  # Unshaped reward for each step.
        self.max_penalty = -1000  # TODO: make dynamic... ?
        self.subscr_results = {}  # For tracking the traci subscription results.

        if self.connected:
            self.loggers.reset.info("[CLOSING] traci connection... .")
            traci.close()
            self.loggers.reset.info("[SUCCESS] closing traci connection !")
            self.loggers.reset.info("[STOPPING] SUMO process... .")
            self.sumo_process.terminate()
            self.loggers.reset.info("[SUCCESS] stopping SUMO process !")
            # sys.stdout.flush()

        ### Delete and Generate random trips.
        self.loggers.reset.info("[DELETING] .xml files... .")
        delete(prefix=self.path.prefix, cwd=self.path.root)
        self.loggers.reset.info(f"[SUCCESS] deleting {self.path.prefix}.*.xml files !")
        self.loggers.reset.info("[GENERATING] .xml files... .")
        generate(
            prefix=self.path.prefix,
            src=self.rnd_src,
            dst=self.rnd_dst,
            rng=self.rng,
            scale=self.rnd_scale,
            cwd=self.path.root,
        )
        self.loggers.reset.info(
            f"[SUCCESS] generating {self.path.prefix}.*.xml files !"
        )

        ### Start connection.
        sumoBinary = checkBinary(self.binary)

        sumo_cmd = [
            sumoBinary,
            "-c",
            Path(self.path.root / f"{self.path.prefix}.sumo.cfg"),
            "--remote-port",
            str(self.port),
            "--end",
            str(3),
            "--no-warnings",
        ]
        self.loggers.reset.info("[STARTING] SUMO process... .")
        self.sumo_process = Popen(sumo_cmd)
        self.loggers.reset.info("[SUCCESS] in starting SUMO process !")

        for attempt in range(self.conn_retries):
            self.loggers.reset.info("[OPENING] traci connection... .")
            try:
                traci.init(
                    port=self.port,
                    numRetries=10,
                    label="default",
                    host="localhost",
                    doSwitch=True,
                )
                if self.connected:
                    self.loggers.reset.info("[SUCCESS] opening traci connection !")
                break
            except Exception as e:
                self.loggers.reset.error(
                    f"[FAILED] opening traci connection after {attempt+1} attempt(s): {e}."
                )
                if attempt == self.conn_retries - 1:
                    self.sumo_process.terminate()
                    self.loggers.reset.error(
                        "[FAILED] opening traci connection: SUMO process stopped."
                    )
                    raise e
                time.sleep(1)

        ### Define starting observation.
        tls, wait, speed, accel = self._get_vehicle_matrix(real_data=False)
        obs = np.stack((tls[0], wait[0], speed[0], accel[0]), axis=0)

        ### Define info
        info = {}

        self.loggers.reset.info("[FINISHED] reset.")

        ### Return starting obss.
        return obs, info


class SumoEnvFactory:
    """
    A factory for creating SumoEnv instances. Handy for multi-threaded training.

    Attributes:
        config: The filepath of the SumoEnv configuration file.
        render: Whether to render the simulation.
        verbose: Whether to print debug messages.
        conn_retries: The number of times to retry connecting to SUMO.
    """

    def __init__(
        self,
        config: str,
        render: bool = True,
        verbose: bool = False,
        conn_retries: int = 10,
    ):
        self.config = config
        self.render = render
        self.verbose = verbose
        self.conn_retries = conn_retries

    def make_env(self, network: str, port: int, **kwargs) -> SumoEnv:
        env = SumoEnv(
            network=network,
            config=self.config,
            render=self.render,
            verbose=self.verbose,
            conn_retries=self.conn_retries,
            port=port,
            **kwargs,
        )
        return env
