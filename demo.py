import zbarlight
import gym
import gym_ple 

env = gym.make('FlappyBird-v0')
observation_n = env.reset()

while True:
  #action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
  #observation_n, reward_n, done_n, info = env.step(action_n)
  observation_n, reward_n, done_n, info = env.step(env.action_space.sample())
  print(env.action_space)
  print(env.observation_space)
  print(env.observation_space.low)
  print(env.observation_space.high)

  env.render()
  
  print(info)


  if done_n:
  	env.reset()