from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from sumoEnvQ import SumoEnv
from trips import delete

# Get the environment and extract the number of actions available in the SUMO problem
env = SumoEnv()
#np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n

# Set the model struct
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Policy & Agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Fit
env.binary = "sumo"
env.config = "trone-slow.sumo.cfg"
dqn.fit(env, nb_steps=100000, verbose=2)

# Test
env.binary = "sumo-gui"
env.config = "trone-slow.sumo.cfg"
dqn.test(env, nb_episodes=100, visualize=False)

# Delete files
delete()