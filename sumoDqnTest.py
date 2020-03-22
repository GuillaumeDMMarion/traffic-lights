'''
Deep RL tester (from saved model & weights).
'''

from keras.models import load_model
from keras.optimizers import Adam

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from trafficl.rlc.dqn import DQNAgent # Local import for dictionary compatibility (see line 322). #from rl.agents.dqn import DQNAgent 
from trafficl.rlc.multiInputProcessor import MultiInputProcessor # Local import for dictionary compatibility (see all lines). #from rl.processors import MultiInputProcessor 

from trafficl.sumorl.sumoEnv import SumoEnv
from trafficl.sumorl.trips import delete



### Set up the environment.
env = SumoEnv(obs_center = (430,475),
							obs_nrows = 29,
							obs_length = 3,
							lanes = ['edge_in_0_0',
													'edge_in_0_1',
													'edge_in_0_2',
													'edge_in_1_0',
													'edge_in_1_1',
													'edge_in_2_0',
													'edge_in_2_1',
													'edge_in_3_0',
													'edge_in_3_1'],
							tls = 'tl_1',
							phases = {0:[2,6],
												2:[4,0],
												4:[6,2],
												6:[0,4]},
							binary = 'sumo-gui',
							config = 'trone.sumo.cfg',
							prefix = 'trone',
							src = ["src_"+str(n) for n in range(5)],
							dst = ["dst_"+str(n) for n in range(9)],
							scale = (10,10),
							seed = 7273970335329779073,
							state = None,
							verbose = True,
							end_time = 3600
							)

### Extract action space and shapes from the environment.
nb_actions = env.action_space.n
shape_phase = env.observation_space.spaces['phase'].shape
shape_vehicle = env.observation_space.spaces['vehicle'].shape

### Input/Output suffix
model = load_model("model/reward_-1019065.0.h5f")

### Policy, Memory & Agent set-up.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.01, nb_steps=100000)
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, batch_size=64, gamma=.95, nb_steps_warmup=2000, target_model_update=.001)
dqn.processor = MultiInputProcessor(2)	
dqn.compile(optimizer=Adam(lr=.001))

# Test the loaded model & weights.
hist = dqn.test(env, nb_episodes=20, visualize=False)

# Delete sumo files.
delete(prefix='trone')