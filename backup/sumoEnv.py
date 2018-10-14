"""
SUMO environment for reinforcment learning with keras-rl
"""

# Gym
import gym
from gym import spaces
from gym.utils import seeding

# Sumo
import sys
import traci
from sumolib import checkBinary

# Misc
import numpy as np
import os

# Trip generator
from trips import generate, delete


class SumoEnv(gym.Env):
	"""
	Description: 
		A SUMO simulation is run with a single traffic light system. The goal is to end with an as low as possible cumulated waiting time by setting the duration of the individual traffic light phases.

	Observation:
		A box of values containing:
		- Current phase
		- Number of vehicles
		- Mean speed
		- Number of vehicles stopped
		For each incoming lane.

	Actions:
		Discrete values representing the phase's duration to be set. 

	Reward:
		The (non-positive) difference between previous acc. waiting time and new acc. waiting time (corrected for the phase's duration).

	Starting State:
		No cars, zero speed, no vehicles stopped.

	Episode Termination:
		When the simulation is ended, i.e. when all the vehicle-routes are depleted.

	"""

	def __init__(self):

		# Define Spaces
		self.action_space = spaces.Discrete(61)
		self.low = np.array([[0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0],
												 [0, 0, 0, 0]])
		high = np.array([[3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf],
										 [3, 20, 20, np.inf]])
		self.observation_space = spaces.Box(self.low, high, dtype=np.float32)

		# Define SUMO constants
		self.phases = [0,2,4,6]
		self.nextphase_dic = {0:2, # Next phases are mapped with current phases
													2:4,
													4:6,
													6:0}
		self.lanes_dic = {0:["523773486.24.12_0","523773486.24.12_1","-14362916#1_0"], # Lanes are mapped on the phase which lets them pass.
										  2:["523773486.24.12_2"										,"-14362916#1_1"],
										  4:["-524150338_0","510492454#0_0"],
										  6:["-524150338_1","510492454#0_1"]}
		self.tl_id = "26085303"

		# Step tracker (BUG: SUMO output obfuscates keras-rl output, which includes the number of steps. So we implement it here to obfucate what SUMO obscates. Probably a better way to do this, e.g. SUMO-verbose = False?)
		self.nstep = 0

		# Define tracker for reward
		self.waiting_dic = {}

		# Define default sumo method
		self.binary = "sumo" # We default on the command line for increased simulation speed. It can then be overwritten in the rl script to vizualize the dqn.test().

		# Define default config file
		self.config = "trone.sumo.cfg" # This way, one .cfg file can be set up with a high time-step for quick learning and one .cfg can be set up with low time-step for smooth vizualization.

		# Seed
		self.seed()

	def update(self, dic):
		for key in dic:
			self.waiting_dic[key] = dic[key]

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):		
		# Store the accumulated waiting time until the beginning of the phase.
		acc_waiting_prev = sum(self.waiting_dic.values())
		# Previous rl-step will end on the phase which we want to adapt; We then extract the next phase.
		current_phase = traci.trafficlight.getPhase(self.tl_id)
		next_phase = self.nextphase_dic[current_phase]
		# Adapt duration corresponding to the action taken.
		phase_dur = action
		traci.trafficlight.setPhaseDuration(self.tl_id, phase_dur)
		# Run simulation until the phase is finished (as well as the mandatory yellow one).
		while current_phase != next_phase :
			traci.simulationStep()
			current_phase = traci.trafficlight.getPhase(self.tl_id)
		current_phase_index = [i for i,p in enumerate(self.phases) if p==current_phase][0]
		current_phase_index_arr = np.array([current_phase_index]*9).reshape(9,1)
		# Get all vehicle ids & accumulated waiting times
		v_ids = traci.vehicle.getIDList()
		v_waits = [traci.vehicle.getAccumulatedWaitingTime(v_id) for v_id in v_ids]
		# Make a dictionary and update the dictionary within self. An update is necessary since vehicle ids are removed once the vehicle's route is over.
		new_waiting_dic = dict(zip(v_ids,v_waits))
		self.update(new_waiting_dic)
		# Compute the reward
		reward = (acc_waiting_prev - sum(self.waiting_dic.values()))/max(phase_dur,1)
		# Create an observation box to return.
		metrics = np.zeros([9,3])
		row = 0
		for sublist in self.lanes_dic.values():
			for lane in sublist:
				nr_vehicles = traci.lane.getLastStepVehicleNumber(lane)
				nr_halts = traci.lane.getLastStepHaltingNumber(lane)
				waiting_time = traci.lane.getWaitingTime(lane)
				metrics[row,:] = np.array([nr_vehicles, nr_halts, waiting_time])
				row += 1
		# Done
		done = traci.simulation.getMinExpectedNumber()==0
		# Observation box
		observation = np.append(current_phase_index_arr, metrics, axis=1)
		# Print info inbetween SUMO output
		self.nstep += 1
		print("||", self.nstep, "--", action, "--", -reward, "||") ###sum(self.waiting_dic.values()) - acc_waiting_prev
		# Return
		return observation, reward, done, {}

	def reset(self):
		# Print overall episode's accumulated waiting time
		print("---------------------------------------->", round(sum(self.waiting_dic.values()),1))
		# Reset tracker for reward
		self.waiting_dic = {}
		# Try to close connection
		try:
			traci.close()
			sys.stdout.flush()
		except:
			pass
		# Delete/Generate trips
		### delete()
		### generate()
		# Start connection
		sumoBinary = checkBinary(self.binary) # sumo-gui
		traci.start([sumoBinary, "-c", self.config])
		# Observation at start
		observation = self.low
		return observation