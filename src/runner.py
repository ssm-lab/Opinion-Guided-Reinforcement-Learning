import argparse
import gymnasium as gym
import logging
import numpy as np
import os
import sl
from deprecated import deprecated
from map_tools import MapTools
from datetime import datetime
from matplotlib import pyplot as plt
from parser import Parser
from sklearn.preprocessing import normalize
from model import Grid

class Runner():

    def __init__(self, size, seed, numexperiments, maxepisodes, log_level=logging.INFO):
        self._SIZE = size
        self._SEED = seed
        self._NUM_EXPERIMENTS = 10
        self._MAX_EPISODES = 250
        self._MAP_DESC = MapTools().parse_map(size, seed)
        
        #Hyperparameters
        self._SLIPPERY = False
        self._ALPHA = 0.9
        self._GAMMA = 1
        
        #Environment
        self._ENVIRONMENT = gym.make('FrozenLake-v1', desc=self._MAP_DESC, is_slippery=self._SLIPPERY)
        
        #File paths
        self._FILES_PATH = 'src/files'
        self._RESULTS_PATH = 'src/results'
        self._FILE_PATTERN = f'{size}x{size}-seed{seed}'
        self._MAP_NAME = f'{size}x{size}'
        
        #Logging
        logging.basicConfig(format='[%(levelname)s] %(message)s')
        logging.getLogger().setLevel(log_level)
    
    @deprecated(reason="Obsolete due to self.get_default_policy()")
    def get_weighted_default_policy(self, world):
        num_states = self._ENVIRONMENT.observation_space.n
        num_actions = self._ENVIRONMENT.action_space.n
        
        default_policy = np.zeros((num_states, num_actions))
        
        for cell_number in range(num_states):
            cell = world.cells[cell_number]
            neighbors = cell.get_neighbors()
            
            num_neighbors = len([c for c in neighbors if c is not None])
            neighbor_exists = [0 if n is None else 1 for n in neighbors]
            probabilities = [n * (1/num_neighbors) for n in neighbor_exists]
            default_policy[cell_number] = probabilities
        
        return default_policy
        
    def get_default_policy(self):
        num_states = self._ENVIRONMENT.observation_space.n
        num_actions = self._ENVIRONMENT.action_space.n
        
        default_policy = np.full((num_states, num_actions), 1/num_actions)
        
        return default_policy
        

    def get_human_input(self):
        file = os.path.abspath(f'{self._FILES_PATH}/opinions-{self._FILE_PATTERN}.txt')
        parser = Parser()
        
        return parser.parse(file)

    def shapePolicy(self, policy, human_input):
        for hint in human_input.hints:
            cell = hint.cell
            logging.debug(cell)
            for sap in cell.get_actions_to_me_from_all_neighbors():
                neighbor_row, neighbor_col = sap[0]
                neighbors_sequence_number = neighbor_row*cell.edge_size + neighbor_col
                action_number = sap[1].value
                
                base_action_probability = policy[neighbors_sequence_number][action_number]
                base_action_opinion = sl.probability_to_opinion(base_action_probability)
                
                hinted_opinion = hint.get_binomial_opinion(base_rate = base_action_probability) #base rate is set from the environment
                
                fused_opinion = sl.beliefConstraintFusion(base_action_opinion, hinted_opinion)
                fused_probability = sl.opinion_to_probability(fused_opinion)
                
                logging.debug(f'fusing action {sap[1]}({action_number}) of policy[{neighbor_row}][{neighbor_col}]=({base_action_probability}) with {hinted_opinion} --> {fused_opinion} -> P={fused_probability}')
                
                policy[neighbors_sequence_number][action_number] = fused_probability
                
        return policy


    def get_action_probabilities(self, state, policy):
        logits = np.zeros(self._ENVIRONMENT.action_space.n)
        for action in range(self._ENVIRONMENT.action_space.n):
            logit = np.exp(policy[state, action])
            logits[action] = logit
            
        return logits / np.sum(logits)  # TODO: this might be incorrect. Cf. the action probabilities in get_default_policy()
        
    def calculate_return(self,rewards):
        # https://stackoverflow.com/questions/65233426/discount-reward-in-reinforce-deep-reinforcement-learning-algorithm
        ep_rewards = np.asarray(rewards)
        t_steps = np.arange(ep_rewards.size)
        ep_returns = ep_rewards * self._GAMMA**t_steps
        ep_returns = ep_returns[::-1].cumsum()[::-1] / self._GAMMA**t_steps
        return ep_returns.tolist()
        
    def update_policy(self,policy, ep_states, ep_actions, ep_probs, ep_returns):
        for t in range(0, len(ep_states)):
            state = ep_states[t]
            action = ep_actions[t]
            prob = ep_probs[t]
            action_return = ep_returns[t]

            phi = np.zeros([1, self._ENVIRONMENT.action_space.n])
            phi[0, action] = 1

            score = phi - prob
            policy[state, :] = policy[state, :] + self._ALPHA * action_return * score

        return policy

    def discrete_policy_grad(self, initial_policy):
        policy = initial_policy

        total_reward, total_successes = [], 0
        for episode in range(self._MAX_EPISODES):
            state = self._ENVIRONMENT.reset()[0]
            ep_states, ep_actions, ep_probs, ep_rewards, total_ep_rewards = [], [], [], [], 0
            terminated, truncated = False, False

            # gather trajectory
            while not terminated and not truncated:
                ep_states.append(state)         # add state to ep_states list
                
                action_probs = self.get_action_probabilities(state, policy) # pass state thru policy to get action_probs
                ep_probs.append(action_probs)   # add action probabilities to action_probs list
                
                action = np.random.choice(np.array([0, 1, 2, 3]), p=action_probs)   # choose an action
                ep_actions.append(action)       # add action to ep_actions list
                
                state, reward, terminated, truncated, __ = self._ENVIRONMENT.step(action) # take step in environment
                ep_rewards.append(reward)       # add reward to ep_rewards list
                
                total_ep_rewards += reward
                if reward == 1:
                    total_successes += 1

            ep_returns = self.calculate_return(ep_rewards) # calculate episode return & add total episode reward to totalReward
            total_reward.append(sum(ep_rewards))

            # update policy
            policy = self.update_policy(policy, ep_states, ep_actions, ep_probs, ep_returns)

        #self._ENVIRONMENT.close()

        # success rate
        success_rate = (total_successes / self._MAX_EPISODES) * 100

        return success_rate

    def evaluate(self, initial_policy):
        success_rates = []
        for i in range(self._NUM_EXPERIMENTS):
            iteration = self.discrete_policy_grad(initial_policy)
            success_rates.append(iteration)
        return success_rates

    def get_file_name(self):
        now = datetime.now()
        return f'{self._RESULTS_PATH}/{self._MAP_NAME}-e{self._MAX_EPISODES}-{now.strftime("%Y%m%d-%H%M%S")}'

    def save_data(self, success_rates, advice):
        file_name = f'{self.get_file_name()}-advice.csv' if advice else f'{self.get_file_name()}-no-advice.csv'
        np.savetxt(f'{file_name}', success_rates, delimiter=",")
        
    def plot(self, no_advice_success_rates, advice_success_rates):
        logging.getLogger().setLevel(logging.INFO)
        plt.plot(no_advice_success_rates, label='No advice')
        plt.plot(advice_success_rates, label='Advice')
        plt.title(f'Training on a {self._MAP_NAME} map for {str(self._MAX_EPISODES)} episodes; is_slippery = {str(self._SLIPPERY)}.')
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate %')
        plt.legend()
        
        plt.savefig(f'{self.get_file_name()}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    
    def run(self):
        logging.info('run()')
        
        world = Grid(self._SIZE)
        #logging.debug([cell.get_neighbors() for cell in world.cells])
        default_policy = self.get_default_policy()
        logging.debug(default_policy)

        human_input = self.get_human_input()
        assert human_input.map_size == self._SIZE #sanity check
        shaped_policy = self.shapePolicy(default_policy, human_input)
        logging.debug(shaped_policy)

        # evaluate without advice
        logging.info('running evaluation without advice')
        no_advice_success_rates = self.evaluate(default_policy)
        self.save_data(no_advice_success_rates, advice=False)

        # evaluate with advice
        logging.info('running evaluation with advice')
        advice_success_rates =  self.evaluate(shaped_policy)
        self.save_data(advice_success_rates, advice=True)

        self.plot(no_advice_success_rates, advice_success_rates)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-size')
    parser.add_argument('-seed')
    parser.add_argument('-numexperiments')
    parser.add_argument('-maxepisodes')
    parser.add_argument(
        "-log",
        "--log",
        default="warning",
        help=("Provide logging level. "
              "Example '--log debug', default='warning'."
              )
        )
        
    options = parser.parse_args()
    
    assert options.size
    assert options.seed
    size = int(options.size)
    seed = int(options.seed)
    
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    level = levels.get(options.log.lower())
    
    numexperiments = int(options.numexperiments) if(options.numexperiments) else 10
    maxepisodes = int(options.maxepisodes) if(options.maxepisodes) else 250
    
    runner = Runner(size, seed, numexperiments, maxepisodes, level)
    runner.run()