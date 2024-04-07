import gymnasium as gym
import logging
import numpy as np
import os
import sl
import map_parser
from datetime import datetime
from matplotlib import pyplot as plt
from parser import Parser
from sklearn.preprocessing import normalize
from model import Grid

#Constants
FILES_PATH = 'src/files'
RESULTS_PATH = 'src/results'
NUM_EXPERIMENTS = 10
MAX_EPISODES = 250
MAP_SIZE = 6
SEED = 10
MAP_NAME = f'{MAP_SIZE}x{MAP_SIZE}'
SLIPPERY = False
ALPHA = 0.9
GAMMA = 1
file_name = f'{MAP_SIZE}x{MAP_SIZE}-seed{SEED}'
map_desc = map_parser.parse_map(f'lake-{file_name}')
#ENVIRONMENT = gym.make('FrozenLake-v1', map_name=MAP_NAME, is_slippery=SLIPPERY)
ENVIRONMENT = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=SLIPPERY)



logging.basicConfig(format='[%(levelname)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


def get_default_policy(world):
    default_policy = np.zeros((ENVIRONMENT.observation_space.n, ENVIRONMENT.action_space.n)) # TODO: not zeros but 1/[number of neighbors]
    
    for cell_number in range(ENVIRONMENT.observation_space.n):
        cell = world.cells[cell_number]
        neighbors = cell.get_neighbors()
        
        num_neighbors = len([c for c in neighbors if c is not None])
        neighbor_exists = [0 if n is None else 1 for n in neighbors]
        probabilities = [n * (1/num_neighbors) for n in neighbor_exists]
        default_policy[cell_number] = probabilities
    
    return default_policy

def get_human_input():
    file = os.path.abspath(f'{FILES_PATH}/opinions-{file_name}.txt')
    parser = Parser()
    
    return parser.parse(file)

def shapePolicy(policy, human_input):
    for hint in human_input.hints:
        cell = hint.cell
        logging.debug(cell)
        for sap in cell.get_actions_to_me_from_all_neighbors():
            neighbor_row, neighbor_col = sap[0]
            neighbors_sequence_number = neighbor_row*cell.edge_size + neighbor_col
            action_number = sap[1].value
            
            base_action_probability = policy[neighbors_sequence_number][action_number]
            hinted_opinion = hint.get_binomial_opinion(base_action_probability)
            fused_opinion = sl.beliefConstraintFusion(sl.probability_to_opinion(base_action_probability), hinted_opinion)
            fused_probability = sl.opinion_to_probability(fused_opinion)
            logging.debug(f'fusing action {sap[1]}({action_number}) of policy[{neighbor_row}][{neighbor_col}]=({base_action_probability}) with {hinted_opinion} --> {fused_opinion} -> P={fused_probability}')
            
            policy[neighbors_sequence_number][action_number] = fused_probability
            
    return policy


def get_action_probabilities(state, policy):
    logits = np.zeros(ENVIRONMENT.action_space.n)
    for action in range(ENVIRONMENT.action_space.n):
        logit = np.exp(policy[state, action])
        logits[action] = logit
        
    return logits / np.sum(logits)  # TODO: this might be incorrect. Cf. the action probabilityies in get_default_policy()
    
def calculate_return(rewards):
    # https://stackoverflow.com/questions/65233426/discount-reward-in-reinforce-deep-reinforcement-learning-algorithm
    ep_rewards = np.asarray(rewards)
    t_steps = np.arange(ep_rewards.size)
    ep_returns = ep_rewards * GAMMA**t_steps
    ep_returns = ep_returns[::-1].cumsum()[::-1] / GAMMA**t_steps
    return ep_returns.tolist()
    
def update_policy(policy, ep_states, ep_actions, ep_probs, ep_returns):
    for t in range(0, len(ep_states)):
        state = ep_states[t]
        action = ep_actions[t]
        prob = ep_probs[t]
        action_return = ep_returns[t]

        phi = np.zeros([1, ENVIRONMENT.action_space.n])
        phi[0, action] = 1

        score = phi - prob
        policy[state, :] = policy[state, :] + ALPHA * action_return * score

    return policy

def discrete_policy_grad(initial_policy):
    policy = initial_policy

    total_reward, total_successes = [], 0
    for episode in range(MAX_EPISODES):
        state = ENVIRONMENT.reset()[0]
        ep_states, ep_actions, ep_probs, ep_rewards, total_ep_rewards = [], [], [], [], 0
        terminated, truncated = False, False

        # gather trajectory
        while not terminated and not truncated:
            ep_states.append(state)         # add state to ep_states list
            
            action_probs = get_action_probabilities(state, policy) # pass state thru policy to get action_probs
            ep_probs.append(action_probs)   # add action probabilities to action_probs list
            
            action = np.random.choice(np.array([0, 1, 2, 3]), p=action_probs)   # choose an action
            ep_actions.append(action)       # add action to ep_actions list
            
            state, reward, terminated, truncated, __ = ENVIRONMENT.step(action) # take step in environment
            ep_rewards.append(reward)       # add reward to ep_rewards list
            
            total_ep_rewards += reward
            if reward == 1:
                total_successes += 1

        ep_returns = calculate_return(ep_rewards) # calculate episode return & add total episode reward to totalReward
        total_reward.append(sum(ep_rewards))

        # update policy
        policy = update_policy(policy, ep_states, ep_actions, ep_probs, ep_returns)

    ENVIRONMENT.close()

    # success rate
    success_rate = (total_successes / MAX_EPISODES) * 100

    return success_rate

def evaluate(initial_policy):
    success_rates = []
    for i in range(NUM_EXPERIMENTS):
        iteration = discrete_policy_grad(initial_policy)
        success_rates.append(iteration)
    return success_rates

def get_file_name():
    now = datetime.now()
    return f'{MAP_NAME}-e{MAX_EPISODES}-{now.strftime("%Y%m%d-%H%M%S")}'

def save_data(success_rates, advice):
    file_name = f'{get_file_name()}-advice.csv' if advice else f'{get_file_name()}-no-advice.csv'
    np.savetxt(f'{RESULTS_PATH}/{file_name}', success_rates, delimiter=",")
    
def plot(no_advice_success_rates, advice_success_rates):
    plt.plot(no_advice_success_rates, label='No advice')
    plt.plot(advice_success_rates, label='Advice')
    plt.title(f'Training on a {MAP_NAME} map for {str(MAX_EPISODES)} episodes; is_slippery = {str(SLIPPERY)}.')
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate %')
    plt.legend()
    
    plt.savefig(f'{RESULTS_PATH}/{get_file_name()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    

'''''''''''''''''''''''''''''''''''''''''''''
Main
'''''''''''''''''''''''''''''''''''''''''''''
world = Grid(MAP_SIZE)
logging.debug([cell.get_neighbors() for cell in world.cells])
default_policy = get_default_policy(world)
logging.debug(default_policy)

human_input = get_human_input()
assert human_input.map_size == MAP_SIZE
shapedPolicy = shapePolicy(default_policy, human_input)

# evaluate without advice
logging.info('running evaluation without advice')
no_advice_success_rates = evaluate(default_policy)
save_data(no_advice_success_rates, advice=False)

# evaluate with advice
logging.info('running evaluation with advice')
#initial_policy = np.loadtxt(f'{FILES_PATH}/human_advised_policy', delimiter=",")
advice_success_rates =  evaluate(shapedPolicy)
save_data(advice_success_rates, advice=True)

plot(no_advice_success_rates, advice_success_rates)
