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

A number of scripts are needed to achieve the desired reinforcement learning:

trips.py
--------

For generating new trips with random source/destination weights on each episode.

sumoEnv.py
-----------

For providing the environment class with reset() and step() methods.

sumoDqn.py
------------

For instantiation of the class and for initiating the reinforcement learning through Keras+Tensorflow.

sumoDqnTest.py
------------

For testing the saved models.

sumoNaive.py
------------

For testing naive strategies, i.e. fixed phase-duration programs.


