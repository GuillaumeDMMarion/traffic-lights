'''
Naive cycler.
'''

from sumoEnv import SumoEnv
from trips import delete
import numpy as np
import pandas as pd



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
							state = None,
							verbose = True,
							end_time = 3600
							)


with open("debug_naive.csv", "a") as file:
	file.write("reward_total,reward_mean,delay_total,delay_mean")
	file.close()

nb_episodes = 450
# action_seq = [0]*6+[1]*4+[2]*6+[3]*4
action_seq = [0]*7+[1]*7+[2]*7+[3]*7
for i in range(nb_episodes):
	rewards, delays = env.cycle(action_seq=action_seq)
	with open("debug_naive.csv", "a") as file:
		file.write(str("\n")+str(np.sum(rewards))+str(",")+str(np.mean(rewards))+str(",")+str(np.sum(list(delays)))+str(",")+str(np.mean(list(delays))))
		file.close()

delete('trone')