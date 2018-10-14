import os
import sys
import numpy as np

import traci
from sumolib import checkBinary

def run():
	edges_dic = {"510492454#0":"N",
							 "-14362916#1":"E",
							 "-524150338":"S",
							 ":26085301_0":"S", # junction
							 "523773486.24.12":"W",	
							 ":gneJ8_0":"W", #junction
							 "244422544#0.19.0.112":"W",
							 ":gneJ14_0":"W", #junction
							 "244422544#0.19.0.0.88":"W",
							 ":gneJ16_0":"W", #junction
							 "244422544#0.19.0.0.0":"W"
							 }
	edges_dic = {"N":["510492454#0"],
							 "E":["-14362916#1"],
							 "S":["-524150338",":26085301_0"],
							 "W":["523773486.24.12","244422544#0.19.0.112",":gneJ14_0","244422544#0.19.0.0.88",":gneJ16_0","244422544#0.19.0.0.0"]}
	tl_id = "26085303"
	step = 0
	while traci.simulation.getMinExpectedNumber()>0:
		
		# Aggregated information #
		observations = np.zeros([4,3])
		for i,key in enumerate(edges_dic.keys()):
			nr_vehicles = 0
			mean_speed = 0
			nr_halts = 0
			for edge in edges_dic[key]:
				nr_vehicles += traci.edge.getLastStepVehicleNumber(edge)
				mean_speed += traci.edge.getLastStepMeanSpeed(edge)
				nr_halts += traci.edge.getLastStepHaltingNumber(edge)
			mean_speed = mean_speed/len(edges_dic[key])
			observations[i,:] = np.array([nr_vehicles, mean_speed, nr_halts])
		print(observations)

		''' 
		# Individual Pos & Speed : by translating vehicle positions into a matrix
		if step%10 == 0:
			posMat = np.zeros([260,235])
			v_ids = traci.vehicle.getIDList()
			v_poss = [(np.array(traci.vehicle.getPosition(v_id))/2).round(0) for v_id in v_ids]
			v_poss = [v_pos for v_pos in v_poss if (v_pos[0]<235 and v_pos[1]<260)]
			for v_pos in v_poss:
				posMat[260-int(v_pos[1]),int(v_pos[0])] = 1
			print(posMat[10:30,200:220])
			print(" ")
		'''

		''' 
		# Individual Pos & Speed : by extracting edge / lane information
		nr = traci.edge.getLastStepVehicleNumber("244422544#0.19.0.0.0")
		nr2 = traci.edge.getLastStepVehicleNumber("244422544#0.19.0.0.88")
		nr3_0 = traci.lane.getLastStepVehicleNumber(":gneJ14_0_0")
		nr3_1 = traci.lane.getLastStepVehicleNumber(":gneJ14_0_1")
		nr4_0 = traci.lane.getLastStepVehicleNumber("244422544#0.19.0.112_0")
		nr4_1 = traci.lane.getLastStepVehicleNumber("244422544#0.19.0.112_1")
		nr5_0 = traci.lane.getLastStepVehicleNumber(":gneJ8_0_0")
		nr5_1 = traci.lane.getLastStepVehicleNumber(":gneJ8_0_1")
		nr5_2 = traci.lane.getLastStepVehicleNumber(":gneJ8_0_2")
		nr6_0 = traci.lane.getLastStepVehicleNumber("523773486.24.12_0")
		nr6_1 = traci.lane.getLastStepVehicleNumber("523773486.24.12_1")
		nr6_2 = traci.lane.getLastStepVehicleNumber("523773486.24.12_2")
		print([nr],[nr2],[nr3_1,nr3_0],[nr4_1,nr4_0],[nr5_2,nr5_1,nr5_0],[nr6_2,nr6_1,nr6_0])
		'''

		step+=1
		traci.simulationStep()
	traci.close()
	sys.stdout.flush()




if __name__ == "__main__":
	sumoBinary = checkBinary('sumo-gui')
	traci.start([sumoBinary, "-c", "trone.sumo.cfg"])
	run()