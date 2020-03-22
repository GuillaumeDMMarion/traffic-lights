'''
Deep RL trainer.
'''

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Flatten, LeakyReLU, Conv2D, MaxPooling2D, Activation, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from trafficl.rlc.dqn import DQNAgent # Local import for dictionary compatibility (see line 322). #from rl.agents.dqn import DQNAgent 
from trafficl.rlc.callbacks import TrainEpisodeLogger # Local import with addition for on-episode-end model checkpoint and .csv file export (see lines 105:276). #from rl.callbacks import TrainEpisodeLogger 
from trafficl.rlc.multiInputProcessor import MultiInputProcessor # Local import for dictionary compatibility (see all lines). #from rl.processors import MultiInputProcessor 

from trafficl.sumorl.sumoEnv import SumoEnv
from trafficl.sumorl.trips import delete



### State from checkpoint
state = None
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
							binary = 'sumo',
							config = 'trone.sumo.cfg',
							prefix = 'trone',
							src = ["src_"+str(n) for n in range(5)],
							dst = ["dst_"+str(n) for n in range(9)],
							scale = (10,10),
							seed = 13511883520433707069,
							state = state,
							verbose = True,
							end_time = 3600
							)

### Extract action space and shapes from the environment.
nb_actions = env.action_space.n
shape_phase = env.observation_space.spaces['phase'].shape
shape_vehicle = env.observation_space.spaces['vehicle'].shape

### Load checkpoint Model & Weights...
try:
	model = load_model("none")
	print("Checkpoint model loaded, continuing training...")

### ...or create
except OSError:
	print("Checkpoint model not found, creating one...")

	### Phase model & input.
	model_phase = Sequential()
	model_phase.add(Flatten(data_format='channels_last', input_shape=shape_phase))
	model_phase_input = Input(shape=shape_phase, name='phase')
	model_phase_encoded = model_phase(model_phase_input)

	### Vehicle model & input.
	model_vehicle = Sequential()
	model_vehicle.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), data_format='channels_last', input_shape=shape_vehicle)) # padding='same'
	model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_last'))
	model_vehicle.add(LeakyReLU()) # model_vehicle.add(Activation(activation='relu'))
	model_vehicle.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), data_format='channels_last'))
	model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_last'))
	model_vehicle.add(LeakyReLU()) # model_vehicle.add(Activation(activation='relu'))
	'''
	model_vehicle.add(Conv2D(128, kernel_size=(2,2), strides=(1,1),data_format='channels_last'))
	model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_last'))
	model_vehicle.add(Activation(activation='relu'))
	'''
	model_vehicle.add(Flatten(data_format='channels_last'))
	model_vehicle_input = Input(shape=shape_vehicle, name='vehicle')
	model_vehicle_encoded = model_vehicle(model_vehicle_input)

	### Concatenation and final model.
	conc = concatenate([model_phase_encoded, model_vehicle_encoded])
	hidden = Dense(128)(conc) # activation='relu'
	hidden = LeakyReLU()(hidden)
	hidden = Dense(64)(hidden) # activation='relu'
	hidden = LeakyReLU()(hidden)
	output = Dense(nb_actions, activation='linear')(hidden)
	model = Model(inputs=[model_phase_input, model_vehicle_input], outputs=output)

	### Model info.
	print(model.summary())
	plot_model(model_phase, to_file='model/model_phase.png', show_layer_names=True, show_shapes=True) # expand_nested=True, dpi=96
	plot_model(model_vehicle, to_file='model/model_vehicle.png', show_layer_names=True, show_shapes=True) # expand_nested=True, dpi=96
	plot_model(model, to_file='model/model.png', show_layer_names=True, show_shapes=True) # expand_nested=True, dpi=96

### Policy, Memory & Agent set-up.
# Use a linearly decreasing epsilon-greedy policy, i.e. start with high exploration and gradually move to exploitation.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.01, nb_steps=100000) # value_min=.05, value_test=.05, nb_steps=40000
# Use a sequential memory for experience replay, from which the random sample will be taken for fitting the network.
memory = SequentialMemory(limit=50000, window_length=1) # limit=20000
# batch_size: number of random samples per batch.
# gamma: low discount rate to take into account future rewards on widest horizon.
# nb_steps_warmup: storing the experience in the memory buffer for the first nb_steps_warmup (if < batch_size, sampling w/ replacement will be done).
# target_model_update: controls how often the fixed target model is updated (<1: soft update, >1: hard update every _ steps), e.g. set to 1 equals 'chasing-tail'.
# enable_double_dqn: uses two networks to decouple the action selection from the target Q value generation.
# enable_dueling_network: seperates the estimation of value and advantage.
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, batch_size=64, gamma=.95, nb_steps_warmup=2000, target_model_update=.001, enable_double_dqn=False, enable_dueling_network=False) #batch_size=32, gamma=.99, nb_steps_warmup=1000 # dueling_type='avg'
dqn.processor = MultiInputProcessor(2)
dqn.compile(optimizer=Adam(lr=.001))

### Define the checkpoint
checkpoint = TrainEpisodeLogger()
callbacks_list = [checkpoint]

### Fit.
hist = dqn.fit(env, nb_episodes=450, verbose=0, callbacks=callbacks_list)

### Delete sumo files.
delete(prefix='trone')

### Print rng state.
print(env.rng.get_state())
