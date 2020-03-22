.. image:: images/lights.png
    :width: 200
=================

This repository illustrates a basic example of reinforcement learning on a traffic light system.
For this, Keras is used on top of Tensorflow in conjunction with SUMO for the traffic simulation.

	
Main Dependencies
============

1. Keras 2.2.2
2. Numpy 1.14.5
3. Tensorflow-gpu 1.10.0
4. Gym 0.10.5
	

Functionalities
===============

A number of python files are needed to train and test the desired deep reinforcement learning:

trafficl.rlc
------------

A collection of customized keras-rl objects for handling multimodel neural networks.

trafficl.sumorl.trips.py
------------------------

A couple of objects for deleting and generating new trips with random source/destination weights on each episode.

trafficl.sumorl.sumoEnv.py
--------------------------

An environment class handling the simulation advancement, observation parsing and reward calculation.

sumoDqn.py
------------

Script for initiating the sumo environment and training the reinforcement learning model through Keras and Tensorflow.

sumoDqnTest.py
------------

Script for testing the saved model(s), by default through the sumo-gui.

sumoNaive.py
------------

Script for testing naive strategies, i.e. fixed phase-duration programs.


