from functions import *

experiments = 10; MAX_EPISODES = 250
map_size = '8x8'; slippery = False; alpha = 0.9; gamma = 1; advice = True

env = gym.make('FrozenLake-v1', map_name = map_size, is_slippery = slippery)
num_states = env.observation_space.n
num_actions = env.action_space.n

evaluate(experiments, MAX_EPISODES, num_states, num_actions, map_size, slippery, alpha, gamma)