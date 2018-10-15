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
		- Number of vehicles stopped
		For each incoming lane.

	Actions:
		Discrete values representing the phase's duration to be set. 

	Reward:
		The (non-positive) difference between previous acc. waiting time and new acc. waiting time.

	Starting State:
		No cars, zero speed, no vehicles stopped.

	Episode Termination:
		When the simulation is ended, i.e. when all the vehicle-routes are depleted.

	"""

	def __init__(self):

		# Define Spaces
		self.action_space = spaces.Discrete(2)
		self.low = np.array([[0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0],
												 [0, 0, 0]])
		self.high = np.array([[3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50],
												  [3, 50, 50]])
		self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

		# Define SUMO constants
		self.phases = [0,2,4,6] # The non-yellow phases, i.e. the phases of which the duration can be shortened.
		self.nextphase_dic = {0:2, # The phase next-in-line, for each phase.
													2:4,
													4:6,
													6:0}
		self.previousphase_dic = {0:6, # The phase previous-in-line, for each phase.
															2:0,
															4:2,
															6:4}
		self.lanes_dic = {0:["523773486.24.12_0","523773486.24.12_1","-14362916#1_0"], # The lanes which has traffic flowing, for each phase.
										  2:["523773486.24.12_2"										,"-14362916#1_1"],
										  4:["-524150338_0","510492454#0_0"],
										  6:["-524150338_1","510492454#0_1"]}
		self.tl_id = "26085303" # The id of the traffic-light system.

		# Step tracker.
		self.lstep = 0 # Learning step
		self.sstep = 0 # SUMO simulation step

		# Define tracker for reward
		self.waiting_dic = {}
		self.stops_dic = {}

		# Define default sumo method
		self.binary = "sumo" # We default on the command line for increased simulation speed. It can then be overwritten in the reinforcement learning script to vizualize the dqn.test().

		# Define default config file
		self.config = "trone.sumo.cfg" # This way, for example, one .cfg file can be set up with a high time-step for quick learning and one .cfg can be set up with low time-step for smooth vizualization.

		# Seed
		self.seed()

	def seed(self, seed=None):
		'''
		Returns:
			Seed used
		'''
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def remainingTime(self):
		'''
		Returns:
			The remaining time in seconds until the end of the current phase.
		'''
		simu_time = traci.simulation.getCurrentTime()/1000
		next_switch = traci.trafficlight.getNextSwitch(self.tl_id)/1000
		remaining_time = next_switch - simu_time
		return remaining_time

	def getIds(self, phases=[0,2,4,6]):
		'''
		Args:
			phases: The phase which lets traffic flow through specific lanes (of which the vehicle id's will be returned).  
		Returns:
			All the vehicle id's for the lanes specified by the phase.
		'''
		v_ids = []
		for phase, sublist in zip(self.lanes_dic.keys(), self.lanes_dic.values()):
			if phase in phases:
				for lane in sublist:
					v_ids.extend(traci.lane.getLastStepVehicleIDs(lane))
		return v_ids

	def waitUpdate(self):
		'''
		Returns:
			None; Updates the waiting times for all vehicle id's with the waiting times of all vehicles currently in circulation.   
		'''
		v_ids = traci.vehicle.getIDList()
		v_waits = [traci.vehicle.getAccumulatedWaitingTime(v_id) for v_id in v_ids]
		v_waits_dic = dict(zip(v_ids, v_waits))
		# Update dic.
		for v_id in v_ids:
			self.waiting_dic[v_id] = v_waits_dic[v_id]

	def stopUpdate(self, phases=[0,2,4,6], increment=True):
		'''
		Returns:
			None; Updates the number of times each vehicle has stopped at the crossing.
		'''
		# Get all the id's of the vehicles which are in the specified lane at the crossing.
		v_ids = self.getIds(phases=phases)
		#stop_ids = [v_id for v_id in v_ids if traci.vehicle.getSpeed(v_id)<5] # So that we don't account a stop for vehicles entering the lane just when the light hits yellow.
		stop_ids = v_ids
		# Update dictionary.
		for v_id in stop_ids: #instead of: v_ids
			try:
				self.stops_dic[v_id] = (self.stops_dic[v_id] + 1) if increment else (max(self.stops_dic[v_id],1))
			except KeyError:
				self.stops_dic[v_id] = 1
		# Add zero-values for all vehicles who have never stopped yet.
		for v_id in self.waiting_dic.keys():
			if v_id not in self.stops_dic.keys():
				self.stops_dic[v_id] = 0

	def step(self, action):
		'''
		Returns:
			observation: The observation space after the action has been taken.
			reward: The reward the action taken has resulted in.
			done: Boolean indicating if the current episode is done and thus a reset needs to be performed.
		'''
		# We extract the number of total stops untill the current time period (for monitoring purposes).
		previous_nstops = sum(self.stops_dic.values())
		# We extract the current phase.
		current_phase = traci.trafficlight.getPhase(self.tl_id)
		previous_phase = current_phase
		# It's possible that the we are on the yellow phase if no action was taken during the whole previous phase.
		#  If so, we loop out of the yellow phase; We perform a similar 'looping out' if the remaining duration of the current phase is smaller than 3 seconds (since if no action is taken, 3 seconds of the current phase are needed for the reward calculations).
		#  If the above 'looping out' was performed we set a new starting point at the end of the 'looping out' (i.e. at the beginning of the next phase) and calculate the according current accumulated waiting time. This update is needed if eventually no action is taken.
		if (current_phase not in self.phases) or (self.remainingTime()<3):
			while (current_phase not in self.phases) or (self.remainingTime()<3):
				traci.simulationStep()
				current_phase = traci.trafficlight.getPhase(self.tl_id)
			previous_phase = self.previousphase_dic[current_phase]
			self.waitUpdate()
			self.stopUpdate([previous_phase], increment=True)
			self.stopUpdate([phase for phase in self.phases if phase != previous_phase], increment=False)
		# Store number of current stops
		previous_stops_dic = self.stops_dic.copy()
		# Take an active action (or not).
		#  If so, we loop out of the yellow phase.
		#  If the above 'looping out' was performed we set a new starting point at the end of this 'looping out' (i.e. at the beginning of the next phase) and calculate the according current accumulated waiting time. This update is needed since an action was taken.
		ids_flowing_stopped = []
		if action == 1:
			previous_phase = current_phase
			traci.trafficlight.setPhaseDuration(self.tl_id, 0)
			traci.simulationStep()
			current_phase =  traci.trafficlight.getPhase(self.tl_id)
			while current_phase not in self.phases:
				traci.simulationStep()
				current_phase = traci.trafficlight.getPhase(self.tl_id)
			self.waitUpdate()
			self.stopUpdate([previous_phase], increment=True)
			self.stopUpdate([phase for phase in self.phases if phase != previous_phase], increment=False)
			ids_flowing_stopped = self.getIds([previous_phase])
		# Store the next phase's index
		if self.remainingTime()<3:
			next_phase = self.nextphase_dic[current_phase]
		else:
			next_phase = current_phase
		next_phase_index = [i for i,p in enumerate(self.phases) if p==next_phase][0]
		next_phase_index_arr = np.array([next_phase_index]*9).reshape(9,1)
		# Store the starting info for the 3-seconds reward calculations.
		acc_waiting_prev = sum([self.waiting_dic[v_id] for v_id in self.waiting_dic.keys()])
		# Step 3 seconds and then update accumualted waiting time and add all existing vehicle id's to the stopping dictionary if they haven't been added yet.
		target_remaining = self.remainingTime() - 3
		while self.remainingTime() > target_remaining:
			traci.simulationStep()
		self.waitUpdate()
		self.stopUpdate([-1])
		# Compute the number of extra stops caused by the action.
		extra_stops_dic = {}
		for v_id in self.stops_dic.keys():
			if v_id in ids_flowing_stopped:
				try:
					extra_stops_dic[v_id] = int((self.stops_dic[v_id]>1) and ((self.stops_dic[v_id]-previous_stops_dic[v_id])>0))
				except KeyError:
					extra_stops_dic[v_id] = 0
			else:
					extra_stops_dic[v_id] = 0
		# Compute the updated accumulated waiting and the according reward when taking into account an extra cost for stops caused by the action.
		acc_waiting =  sum([self.waiting_dic[v_id]*(1+(extra_stops_dic[v_id])) for v_id in self.waiting_dic.keys()])
		reward = acc_waiting_prev - acc_waiting
		# Create an observation box to return.
		metrics = np.zeros([9,2])
		row = 0
		for sublist in self.lanes_dic.values():
			for lane in sublist:
				# Lane metrics
				nr_vehicles = traci.lane.getLastStepVehicleNumber(lane)
				nr_halts = traci.lane.getLastStepHaltingNumber(lane)
				metrics[row,:] = np.array([nr_vehicles, nr_halts])
				row += 1
		# Done
		done = traci.simulation.getMinExpectedNumber()==0
		# Observation box
		observation = np.append(next_phase_index_arr, metrics, axis=1)
		# Print info inbetween SUMO output.
		self.lstep += 1
		print("||", self.lstep, "--", action, "--", reward, "--", sum(extra_stops_dic.values()), "||")
		# Return
		return observation, reward, done, {}

	def reset(self):
		# Print overall episode's accumulated waiting time.
		print("-------------------------------------")
		print(round(sum(self.waiting_dic.values()),1))
		print("-------------------------------------")
		# Reset trackers for reward.
		self.waiting_dic = {}
		self.stops_dic = {}
		# Try to close connection.
		try:
			traci.close()
			sys.stdout.flush()
		except:
			pass
		# Delete and Generate trips.
		#delete()
		#generate()
		# Start connection.
		sumoBinary = checkBinary(self.binary) # sumo-gui
		traci.start([sumoBinary, "-c", self.config])
		# Return starting observation.
		observation = self.low
		return observation