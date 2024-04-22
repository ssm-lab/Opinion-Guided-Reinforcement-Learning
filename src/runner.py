import argparse
import gymnasium as gym
import logging
import numpy as np
import os
import pandas as pd
import sl
from deprecated import deprecated
from map_tools import MapTools
from datetime import datetime
from matplotlib import pyplot as plt
from parser import Parser
from scipy.stats import wilcoxon
from scipy.special import softmax
from sklearn.preprocessing import normalize

class Runner():

    def __init__(self, size, seed, numexperiments, maxepisodes, log_level=logging.INFO):
        self._SIZE = size
        self._SEED = seed
        self._NUM_EXPERIMENTS = numexperiments
        self._MAX_EPISODES = maxepisodes
        self._MAP_DESC = MapTools().parse_map(size, seed)
        
        #Hyperparameters
        self._SLIPPERY = False
        self._ALPHA = 0.9
        self._GAMMA = 1
        
        #File paths
        self._FILES_PATH = 'src/files'
        self._RESULTS_PATH = 'src/results'
        self._FILE_PATTERN = f'{size}x{size}-seed{seed}'
        self._MAP_NAME = f'{size}x{size}'
        
        results_folder = os.path.abspath(self._RESULTS_PATH)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        #Logging
        logging.basicConfig(format='[%(levelname)s] %(message)s')
        logging.getLogger().setLevel(log_level)
        
    def get_default_policy(self, environment):
        num_states = environment.observation_space.n
        num_actions = environment.action_space.n
        
        default_policy = np.full((num_states, num_actions), 1/num_actions)
        
        return default_policy

    def get_human_input(self):
        file = os.path.abspath(f'{self._FILES_PATH}/opinions-{self._FILE_PATTERN}.txt')
        parser = Parser()
        
        return parser.parse(file)

    def shape_policy(self, policy, human_input):
        for hint in human_input.hints:
            cell = hint.cell
            #logging.debug(cell)
            for sap in cell.get_actions_to_me_from_all_neighbors():
                neighbor_row, neighbor_col = sap[0]
                neighbors_sequence_number = neighbor_row*cell.edge_size + neighbor_col
                action_number = sap[1].value
                
                base_action_probability = policy[neighbors_sequence_number][action_number]
                base_action_opinion = sl.probability_to_opinion(base_action_probability)
                
                hinted_opinion = hint.get_binomial_opinion(base_rate = base_action_probability) #base rate is set from the environment
                
                fused_opinion = sl.beliefConstraintFusion(base_action_opinion, hinted_opinion)
                fused_probability = sl.opinion_to_probability(fused_opinion)
                
                #logging.debug(f'fusing action {sap[1]}({action_number}) of policy[{neighbor_row}][{neighbor_col}]=({base_action_probability}) with {hinted_opinion} --> {fused_opinion} -> P={fused_probability}')
                
                policy[neighbors_sequence_number][action_number] = fused_probability
        
        policy = normalize(policy, axis=1, norm='l1')
        
        return policy
    
    def policy_to_numerical_preferences(self, policy, environment):
        num_states = environment.observation_space.n
        num_actions = environment.action_space.n

        theta = np.zeros((num_states, num_actions))

        for state in range(num_states):
            mu = policy[state]
            log_sum = 0 
            for action in range(num_actions):
                log_sum += np.log(mu[action])
                c = (-1 / num_actions) * log_sum

            for action in range(num_actions):
                theta[state, action] = np.log(mu[action]) + c

        return theta

    def get_action_probabilities(self, environment, state, policy):
        logits = np.zeros(environment.action_space.n)
        for action in range(environment.action_space.n):
            logit = np.exp(policy[state, action])
            logits[action] = logit
            
        return logits / np.sum(logits)
        
    def calculate_return(self,rewards):
        # https://stackoverflow.com/questions/65233426/discount-reward-in-reinforce-deep-reinforcement-learning-algorithm
        ep_rewards = np.asarray(rewards)
        t_steps = np.arange(ep_rewards.size)
        ep_returns = ep_rewards * self._GAMMA**t_steps
        ep_returns = ep_returns[::-1].cumsum()[::-1] / self._GAMMA**t_steps
        return ep_returns.tolist()
        
    def update_policy(self, policy, ep_states, ep_actions, ep_probs, ep_returns, environment):
        for t in range(0, len(ep_states)):
            state = ep_states[t]
            action = ep_actions[t]
            prob = ep_probs[t]
            action_return = ep_returns[t]

            phi = np.zeros([1, environment.action_space.n])
            phi[0, action] = 1

            score = phi - prob
            policy[state, :] = policy[state, :] + self._ALPHA * action_return * score

        return policy

    def discrete_policy_grad(self, human_input=None):
        #Environment
        environment = gym.make('FrozenLake-v1', desc=self._MAP_DESC, is_slippery=self._SLIPPERY)
        
        logging.debug('Generating default policy')
        policy = self.get_default_policy(environment)
        if human_input:
            logging.debug('Shaping policy with human input')
            policy = self.shape_policy(policy, human_input)
        
        #logging.debug('Initial policy:')
        #logging.debug(policy)
        
        logging.debug('Policy initialized. Exploring now.')

        policy = self.policy_to_numerical_preferences(policy, environment)

        total_reward = []
        steps_taken = []
        for episode in range(self._MAX_EPISODES):
            state = environment.reset()[0]
            ep_states, ep_actions, ep_probs, ep_rewards, total_ep_rewards = [], [], [], [], 0
            terminated, truncated = False, False
            
            i = 0

            # gather trajectory
            while not terminated and not truncated:
                i += 1
                ep_states.append(state)         # add state to ep_states list
                
                action_probs = self.get_action_probabilities(environment, state, policy) # pass state thru policy to get action_probs
                ep_probs.append(action_probs)   # add action probabilities to action_probs list
                
                action = np.random.choice(np.array([0, 1, 2, 3]), p=action_probs)   # choose an action
                ep_actions.append(action)       # add action to ep_actions list
                
                state, reward, terminated, truncated, __ = environment.step(action) # take step in environment
                ep_rewards.append(reward)       # add reward to ep_rewards list
                
                total_ep_rewards += reward

            ep_returns = self.calculate_return(ep_rewards) # calculate episode return & add total episode reward to totalReward
            total_reward.append(sum(ep_rewards))
            
            steps_taken.append((i, sum(total_reward)))

            # update policy
            policy = self.update_policy(policy, ep_states, ep_actions, ep_probs, ep_returns, environment)
            
        #logging.debug(f'Final policy after {self._MAX_EPISODES} episodes')
        #logging.debug(softmax(policy, axis = 1))

        environment.close()

        # success rate
        success_rate = (sum(total_reward) / self._MAX_EPISODES) * 100

        # cumulative reward
        cumulative_reward = np.cumsum(total_reward)

        return success_rate, steps_taken, cumulative_reward

    def evaluate(self, human_input=None):
        success_rates = []
        steps = []
        cumulative_rewards = []
        for i in range(self._NUM_EXPERIMENTS):
            logging.info(f'running experiment #{i+1}')
            success_rate, steps_taken, cumulative_reward = self.discrete_policy_grad(human_input)
            success_rates.append(success_rate)
            steps.append(steps_taken)
            cumulative_rewards.append(cumulative_reward)
        return success_rates, steps, cumulative_rewards
    
    def get_file_name(self, extension, advice_explicit=False, u_explicit=False, human_input=None, extra = None):
        now = datetime.now()
        
        file_name = f'{self._RESULTS_PATH}/{self._MAP_NAME}-seed{self._SEED}-ep{self._MAX_EPISODES}-ex{self._NUM_EXPERIMENTS}'
        
        file_name = '-'.join([file_name, now.strftime("%Y%m%d-%H%M%S")])
        
        #data files might want to make this explicit, but plots that combine advised and vanilla agents might not
        if advice_explicit:
            if human_input is not None:
                file_name = '-'.join([file_name, 'advice'])
            else:
                file_name = '-'.join([file_name, 'no-advice'])
                
        if u_explicit and human_input is not None:
            u = round(human_input.u, 4)
            file_name = '-'.join([file_name, f'u-{u}'])
        
        if extra is not None:
            file_name = '-'.join([file_name, extra])
        
        file_name = '.'.join([file_name, extension])
        
        return file_name
    
    def save_data(self, success_rates, human_input=None):
        file_name = f'{self.get_file_name(extension="csv", advice_explicit=True, u_explicit=True, human_input=human_input)}'

        np.savetxt(f'{file_name}', success_rates, delimiter=",")
    
    def plot_success_rate(self, no_advice_success_rates, advice_success_rates, human_input):
        logging.getLogger().setLevel(logging.INFO)
        plt.plot(no_advice_success_rates, label='No advice')
        plt.plot(advice_success_rates, label='Advice')
        plt.title(f'Map: {self._MAP_NAME}; eps={str(self._MAX_EPISODES)}; exps={str(self._NUM_EXPERIMENTS)}; u={round(human_input.u, 4)}.')
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate %')
        plt.legend()
        
        filename = f'{self.get_file_name(extension="pdf", advice_explicit=False, u_explicit=True, human_input=human_input, extra="SUCCESSRATE")}'
        
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        #plt.show()
        
    def plot_steps(self, all_steps, human_input):
        logging.debug(f'all_steps: {all_steps}')
        w = wilcoxon(all_steps.iloc[:, 0], all_steps.iloc[:, 1])
        logging.debug(f'Wilcoxon: {w}')
        
        logging.getLogger().setLevel(logging.INFO)
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.boxplot(all_steps)
        
        ax.set_title(f'Map: {self._MAP_NAME}; eps={str(self._MAX_EPISODES)}; exps={str(self._NUM_EXPERIMENTS)}; ; u={round(human_input.u, 4)}; p={round(w.pvalue, 4)}.')
        ax.set_xlabel('Mode')
        ax.set_ylabel('Steps')
        ax.set_xticklabels(all_steps.columns)
        
        filename = f'{self.get_file_name(extension="pdf", advice_explicit=False, u_explicit=True, human_input=human_input, extra="STEPS")}'
        
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        #plt.show()
        
    def plot_cumulative_rewards(self, all_cumulative_rewards, human_input):
        for i in range(self._NUM_EXPERIMENTS):
            start = i * self._MAX_EPISODES
            end = (i + 1) * self._MAX_EPISODES
            df = all_cumulative_rewards.iloc[start:end]

            plt.figure()
            plt.plot(df.index.values, df['noadvice'], label = 'No advice')
            plt.plot(df.index.values, df['advice'], label = 'Advice')
            plt.title(f'Map: {self._MAP_NAME}; eps={str(self._MAX_EPISODES)}; exps={str(self._NUM_EXPERIMENTS)}; u={round(human_input.u, 4)}.')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.legend()

        filename = f'{self.get_file_name(extension="pdf", advice_explicit=False, u_explicit=True, human_input=human_input, extra="CUMULATIVEREWARD")}'
        plt.savefig(filename, format='pdf', bbox_inches='tight')

        plt.show()


    def run(self):
        logging.info('run()')
        
        human_input = self.get_human_input()
        assert human_input.map_size == self._SIZE #sanity check
        
        # evaluate without advice
        logging.info('running evaluation without advice')
        no_advice_success_rates, no_advice_steps, no_advice_cumulative_rewards = self.evaluate()
        self.save_data(no_advice_success_rates)
        
        # evaluate with advice
        logging.info('running evaluation with advice')
        advice_success_rates, advice_steps, advice_cumulative_rewards =  self.evaluate(human_input)
        self.save_data(advice_success_rates, human_input)
        
        #logging.debug(f'No advice step stats: {no_advice_steps}')
        #logging.debug(f'Advised step stats: {advice_steps}')

        #logging.debug(f'No advice cumulative rewards: {no_advice_cumulative_rewards}')
        #logging.debug(f'Advised cumulative rewards: {advice_cumulative_rewards}')
        
        u = human_input.u
        
        self.plot_success_rate(no_advice_success_rates, advice_success_rates, human_input)
        
        all_steps = pd.DataFrame({
            'noadvice': [steps for x in no_advice_steps for (steps, reward) in x],
            'advice': [steps for x in advice_steps for (steps, reward) in x]})

        self.plot_steps(all_steps, human_input)

        all_cumulative_rewards = pd.DataFrame({
            'noadvice': [rew for x in no_advice_cumulative_rewards for rew in x],
            'advice': [rew for x in advice_cumulative_rewards for rew in x]})
        
        self.plot_cumulative_rewards(all_cumulative_rewards, human_input) 
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