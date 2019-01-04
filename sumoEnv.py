'''
SUMO gym environment.
'''

# Gym
from gym import Env, spaces
from gym.utils import seeding

# Sumo
import traci
from sumolib import checkBinary

# Random trip generator
from trips import generate, delete

# Misc
import pandas as pd
import numpy as np
import sys
import os



class SumoEnv(Env):
	"""
	Description: 
		A SUMO simulation is run with a single traffic light system to be optimized. 
		The goal is to end with a minimized disutility of accumulated delay.
		The goal is achieved by choosing the next phase each 3 seconds interval.

	Observation:
		A position and speed matrix of all the vehicles currently approaching the traffic light system.
		An array indicating the phase for which the next action will have an impact on, plus its history.

	Actions:
		4 discrete actions indicating the desired next phase.

	Reward:
		The negative difference between the previous disutility of accumulated delay and subsequent disutility of accumulated delay for all vehicles near the intersection, over a 3 seconds interval.
	
	Starting State:
		A zero-car position & speed matrix, and a (faulty) zero-phase array (solved/redefined at the start of the first step).

	Episode Termination:
		At the end_time if given, else when all the routes are depleted.
	"""

	def __init__(self, obs_center, obs_nrows, obs_length, lanes, tls, phases, binary, config, prefix, src, dst, scale, seed=None, state=None, verbose=False, end_time=None):

		# Definition of Spaces.
		self.action_space = spaces.Discrete(len(phases))
		self.observation_space = spaces.Dict({'phase':spaces.Box(low=0, high=1, shape=(len(phases),1,3), dtype=np.float32), 
																					'vehicle':spaces.Box(low=0, high=14, shape=(obs_nrows,obs_nrows,2), dtype=np.float32)})

		# Definition of general constants.
		self.prefix = prefix
		self.src = src
		self.dst = dst
		self.scale = scale

		# Definition of SUMO constants.
		self.obs_center = obs_center # x & y coordinates of the intersection's centerpoint.
		self.obs_nrows =  obs_nrows # Desired number of rows of the position & speed matrix.
		self.obs_length = obs_length # Length in meters of one unit in the position & speed matrix, i.e. precision.
		self.obs_xmin, self.obs_xmax, self.obs_ymin, self.obs_ymax = self._getBoundaries(obs_center, obs_nrows, obs_length) # Boundaries derived from previous three constants.
		self.lanes = lanes # The lanes' ids.
		self.tls = tls # The traffic lights' id.
		self.phases = phases # Keys: The non-yellow phases, i.e. those of which the duration can be shortened. Values: non-yellow phases which are next- and previous-in-line.
		self.binary = binary # Sumo method: Allows specifying whether or not vizualizations are necessary (n.b. at the cost of processing speed).
		self.config = config # Sumo config: Allows specifying parameters in a .cfg file, such as a high time-step for better processing speed at the cost of smooth(er) vizualizations.
		self.end_time = end_time # Controls the end time of each episode. If None, the episode ends when all routes are depleted.

		# Definition of intra-episode tracker (to be reset).
		self.delay = {} # Accumulated delay for each vehicle within an episode.
		self.reward = [] # Reward for each step within an episode.
		self.phase_hist = [] # History of previous phases.

		# Definition of inter-episode tracker (not to be reset).
		self.episode = 0
		
		# Verbose.
		self.verbose = verbose # Boolean for per-step console print.

		# Seed.
		self.seed(seed, state)

	def seed(self, seed, state):
		'''
		Returns:
			Float of the seed used.
		'''
		self.rng, seed = seeding.np_random(seed)
		if state != None:
			self.rng.set_state(state)
		return [seed]

	@staticmethod
	def _elapsedTime():
		'''
		Returns:
			Float of the elapsed simulation time in seconds.		
		'''
		return traci.simulation.getCurrentTime()/1000

	@staticmethod
	def _getBoundaries(obs_center, obs_nrows, obs_length):
		'''
		Returns:
			Floats of the x and y boundaries for the position & speed matrix.
		'''
		dx = dy = obs_nrows*obs_length
		deltas = (dx,dy) # In case we ever consider having different lengths for x & y.
		xmin, xmax, ymin, ymax = [_ for l in [(coord-delta/2,coord+delta/2) for (coord,delta) in zip(obs_center,deltas)] for _ in l]
		return xmin, xmax, ymin, ymax

	@staticmethod
	def _relPos(position, obs_length, obs_xmin, obs_ymin):
		'''
		Args:
			position: Tuple of x & y coordinates.
		Returns:
			Tuple of the relative x & y indices in the position matrix.
		'''
		x,y = position
		fltrNeg = lambda x: x if x>=0 else .5 # To trigger IndexError.
		relx = round((x-obs_xmin)/obs_length)
		rely = round((y-obs_ymin)/obs_length)
		relx, rely = map(fltrNeg, [relx,rely])
		return (rely,relx)

	def getVehicleInfo(self):
		'''
		Returns:
			Nd-array of the current vehicles' (within user-defined limits) positions and speeds.
		'''
		# Get info for matrix-receptacle creation.
		obs_nrows = obs_ncols = self.obs_nrows
		posMat = np.zeros((obs_nrows, obs_ncols))
		speedMat = np.zeros((obs_nrows, obs_ncols))

		# Get info for relative position measurement.
		obs_length, obs_xmin, obs_ymin = self.obs_length, self.obs_xmin, self.obs_ymin

		# Compute positions and speeds.
		v_ids = traci.vehicle.getIDList()
		positions = [(v_id, traci.vehicle.getPosition(v_id)) for v_id in v_ids]
		for (v_id, position) in positions:
			rel_position = self._relPos(position, obs_length, obs_xmin, obs_ymin)
			try:
				posMat[rel_position] = 1
				speed = traci.vehicle.getSpeed(v_id)
				speedMat[rel_position]
			except IndexError:
				pass
		vflip = lambda m: m[::-1,:]
		posMat, speedMat = map(vflip, [posMat, speedMat])
		return np.dstack([posMat, speedMat])

	def storePhase(self, phase):
		'''
		Returns:
			Always None.
		'''
		self.phase_hist = [phase]+self.phase_hist[:2]

	def updateDelay(self):
		'''
		Returns:
			Always None; Updates the delay times, for all vehicles currently on the intersection's incoming lanes. 
		'''
		v_currentIds = []
		for lane in self.lanes:
			v_currentIds += traci.lane.getLastStepVehicleIDs(lane)
		for v_id in v_currentIds:
			self.delay[v_id] = 1 if v_id not in self.delay.keys() else 1+self.delay[v_id]

	@staticmethod
	def disutilities(delays, exp=2):
		'''
		Args:
			delays: List of delays.
			exp: The exponent of increasing marginal disutility.
		Returns:
			Float of the disutility of delays.
		'''
		disutilities = [delay**exp for delay in delays]
		return disutilities

	def updateSim(self):
		'''
		Returns:
			Always None; Performs a sumo simulation step and updates the delay dictionary.
		'''
		traci.simulationStep()
		self.updateDelay()

	def trackSim(self, sec=3):
		'''
		Args:
			sec: Number of seconds to update.
		Returns:
			Performs the update over the given interval.
			A dictionary of updated info.
		'''
		# Record info pre-update.
		previous_delay = sum(self.delay.values())
		previous_disutil = sum(self.disutilities(self.delay.values()))
		# Perform the update over the interval.
		target_elapsed = self._elapsedTime()+sec
		while self._elapsedTime() < target_elapsed:
			self.updateSim()
			updated_phase = traci.trafficlight.getPhase(self.tls)
		# Record info post-update.
		updated_delay = sum(self.delay.values())
		updated_disutil = sum(self.disutilities(self.delay.values()))
		# Return info dictionary.
		return {'added_delay':updated_delay-previous_delay, 'added_disutility':previous_disutil-updated_disutil}

	def isDone(self):
		'''
		Returns:
			Boolean indicating if the current episode is done. 
		'''
		if not self.end_time:
			done = traci.simulation.getMinExpectedNumber()==0
		else:
			done = self._elapsedTime() >= self.end_time
		return done

	def step(self, action, observe=True):
		'''
		Args:
			action: Index of the desired phase's id.

		Returns:
			observation: Dictionary of the observation space after the action has been taken.
			reward: Float of the reward the action taken has resulted in.
			done: Boolean indicating if the current episode is done.
		'''
		### We derive the desired phase from the action.
		desired_phase = int(list(self.phases)[action])
		### We extract the current phase.
		current_phase = traci.trafficlight.getPhase(self.tls)
		### The next phase is still unknown.
		next_phase = None
		### It's possible that the we previously stepped into a yellow phase if the phase remained unchanged.
		### (Though it is practically impossible considering the 3600s default duration of non-yellow phases.)
		### If so, we step out of it, into the next non-yellow, i.e. adaptable, phase. This results in supplementary delay not accounted for.
		while (current_phase not in self.phases.keys()):
			self.updateSim()
			current_phase = traci.trafficlight.getPhase(self.tls)

		### Simulation progress depending on action taken.
		#  If an active action was taken...
		if desired_phase != current_phase:
			# ...we set the current phase's duration to zero.
			traci.trafficlight.setPhaseDuration(self.tls, 0)
			# ...the next phase (for which an action shall be taken in the next step) is then defined as the desired phase.
			next_phase = desired_phase

		### Simulation progress for reward calculations.
		sim_info = self.trackSim(sec=3)
		reward = sim_info['added_disutility']
		updated_phase = traci.trafficlight.getPhase(self.tls)

		### Phase changing depending on action taken.
		#  If an active action was taken...
		if desired_phase != current_phase:
			# ...change the phase according to the desired phase.
			traci.trafficlight.setPhase(self.tls, desired_phase)

		### Determine if the episode is done.
		done = self.isDone()

		if observe:
			### Compute the obervation.
			## Compute the position & speed matrix.
			position_speed_mat = self.getVehicleInfo()
			## Compute the phase array.
			# Determine the phase on which the next action will have an impact on.
			if next_phase==None:
				if updated_phase == current_phase:
					next_phase = current_phase
				else:
					next_phase = self.phases[current_phase][0]
			next_phase_index = int(next_phase/2)
			next_phase_arr = np.zeros((len(self.phases),1))
			next_phase_arr[next_phase_index] = 1
			# Store it in the history.
			self.storePhase(next_phase_arr)
			# Retrieve the full history.
			phase_hist = np.dstack(self.phase_hist)
			## Combine position & speed matrix and phase array into an observation dictionary.
			observation = {'phase':phase_hist, 'vehicle':position_speed_mat}
		else:
			### Don't compute the observation.
			observation = {'phase':None, 'vehicle':None}

		### Final reward tweaks.
		# Decrease reward if no consecutive two phases were the same in the last 3 phases (to discourage useless phase changes).
		two_consecutive_same_phases = any([np.all(item==next_item) for item,next_item in zip(self.phase_hist[0:-1], self.phase_hist[1:])])
		penalty = not(two_consecutive_same_phases)
		reward -= penalty*10000
		self.reward.append(reward)

		### Info
		added_delay = sim_info['added_delay']
		info_dic = {'added_delay':added_delay}

		### Debug information.
		debug = (self.episode, desired_phase, reward)
		if self.verbose==True:
			print(debug)

		return observation, reward, done, info_dic

	def cycle(self, action_seq=[0]*6+[1]*4+[2]*6+[3]*4):
		'''
		Args:
			action_seq: List of the sequence of actions to be taken.
		Returns:
			List of rewards and delays.
		'''
		self.reset()
		rewards,delays = [],[]
		done = False
		index = 0
		while not done:
			action = action_seq[index]
			observation, reward, done, info_dic = self.step(action=action, observe=False)
			rewards.append(reward)
			delays.append(info_dic['added_delay'])
			if index==len(action_seq)-1:
				index = 0
			else:
				index += 1
		return rewards, delays

	def reset(self):
		'''
		Returns:
			Nd.array of starting observation.
		'''
		### Increase episode counter.
		self.episode += 1

		### Reset trackers for reward calculations.
		self.delay = {}
		self.reward = []
		
		### Try to close the connection.
		try:
			traci.close()
			sys.stdout.flush()
		except KeyError:
			pass
		
		### Delete and Generate random trips.
		delete(prefix=self.prefix)
		generate(prefix=self.prefix, src=self.src, dst=self.dst, rng=self.rng, scale=self.scale)
		
		### Start connection.
		sumoBinary = checkBinary(self.binary)
		traci.start([sumoBinary, "-c", self.config])

		### Provide dummy history.
		self.phase_hist = [np.zeros((len(self.phases),1))]*3
		
		### Define starting observation.
		phase_start = self.observation_space.spaces['phase'].low
		position_speed_start = self.observation_space.spaces['vehicle'].low
		observation = {'phase':phase_start, 'vehicle':position_speed_start}	
		
		### Return starting observations.
		return observation