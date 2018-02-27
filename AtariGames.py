import zbarlight
import gym
import gym_ple
from model import Model
from agent import Agent
import json
import utils
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np

"""
	FLOW:
		AtariGames Object will be created.
			*	__init__ will be called with 2 paramters 
			* 	__init__ calls play_games 
			* play_games will play MAX_ITERATIONS number of games and will keep updating the model 
"""

class GameDetails:
	def __init__(self,state,action,reward,next_state,done):
		self.state = state
		self.next_state = next_state
		self.action = action
		self.reward = reward
		self.result = done


class AtariGames:

	def __init__(self,rl_parameters_dir,hyperparameters_dir):
		"""
			Model instance created, graph made and session initialized
		"""
		self.model = Model(hyperparameters_dir)
		self.model.init()

		print(rl_parameters_dir)
		rl_parameters = json.loads(open(rl_parameters_dir).read())
		print(rl_parameters)

		#self.env_name = rl_parameters['env_name']
		self.max_memory_size = rl_parameters['max_memory_size']
		self.number_of_actions = rl_parameters['number_of_actions']
		self.agent = Agent(self.number_of_actions,self.model,self.max_memory_size)
		self.MAX_ITERATIONS = rl_parameters['MAX_ITERATIONS']
		self.GAMMA = rl_parameters['GAMMA']

		self.play_games()

	def play_games(self):
		load_trained_model = True
		for i in range(self.MAX_ITERATIONS):
			print("Playing game number: " + str(i))
			self.launch_game()
			self.play_one_game(i)
			self.model.update_model(self.agent.replay_memory,load_trained_model)
			load_trained_model = False
    	

	def launch_game(self):
		self.game_state = game.GameState()
		print self.game_state
		actions = np.zeros(self.number_of_actions,dtype='int32')
		print actions
		actions[0] = 1
		self.initial_state, reward, done = self.game_state.frame_step(actions)

		#self.env = gym.make(self.env_name)
		#self.initial_state = self.env.reset()
			

	def play_one_game(self,iteration_number):
		
		current_state = self.initial_state
		count = 0
		cumulative_state = []
		while(True):
			#self.env.render()

			
			if count==0:
				for i in range(3):
					cumulative_state.append(current_state)
				count=4
							
			cumulative_state.append(current_state)
			if len(cumulative_state)>4:
				cumulative_state.pop(0)

			if count>=4:
				count = 4

			if count==4:
				cumulative_state_numpy = utils.form_cumulative_state(cumulative_state,self.model.image_rows,self.model.image_columns,self.model.image_channels)

				action = self.agent.predict_next_action(cumulative_state_numpy,count)
				actions = np.zeros(self.number_of_actions,dtype='int32')
				actions[action] = 1
				next_state, reward, done = self.game_state.frame_step(actions)
				#next_state, reward, done, info = self.env.step(action)

				
				if done==True:
					reward-=0
				else: reward+=0.1

				
				print "AtariGame Reward: " + str(reward)

				cumulative_next_state = cumulative_state[1:]
				cumulative_next_state.append(next_state)
				cumulative_next_state_numpy = utils.form_cumulative_state(cumulative_next_state,self.model.image_rows,self.model.image_columns,self.model.image_channels)
				
				if done==False:		
					next_state_Q_value = self.model.predict_Q_value(cumulative_next_state_numpy)
					reward = reward + self.GAMMA*(next_state_Q_value)


				self.agent.update_relay_memory(GameDetails(cumulative_state_numpy, action, reward, cumulative_next_state_numpy, done))

				#agent.store_relay_buffer(GameDetails(current_state, action, reward, next_state, done))

			current_state = next_state
			if done:
				self.agent.update_exploration_rate(iteration_number)
				break


