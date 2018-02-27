import tensorflow as tf
import numpy as np
import os
import time
import random

class Agent:

	def __init__(self,num_actions,model,max_memory_size):
		self.exploration = 1
		self.num_actions = num_actions
		self.model = model		
		self.replay_memory = []
		self.max_memory_size = max_memory_size
		self.explore_more = True

	def update_relay_memory(self,game_details):
		print("Current replay memory size: " + str(len(self.replay_memory)))
		if len(self.replay_memory) == self.max_memory_size:
			self.replay_memory.pop(0)
		self.replay_memory.append(game_details)

	def update_exploration_rate(self,iteration_numer):
		if self.explore_more==True and iteration_numer>0 and iteration_numer%100==0:
			self.exploration -= 0.075
		if self.exploration<=0:
			self.explore_more = False
			self.exploration = 0.03
		

	def predict_next_action(self,state,count):
		print("Exploration number: " + str(self.exploration))
		prob = random.uniform(0, 1)
		if self.exploration >= prob or count<4:
			
			val = random.randint(0, 9)
			print "Random Action taken : " + str(int(val/5)) + ' ' + str(prob)
			return int(val/5)


		return self.model.predict_action(state)



