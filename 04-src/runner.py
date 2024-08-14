import argparse
import gym as gym
import logging
import numpy as np
import os
import pandas as pd
import shutil
import sl
from model import SyntheticAdvisorOpinions, HumanAdvisorOpinions
from map_tools import MapTools
from datetime import datetime
from matplotlib import pyplot as plt
from advice_parser import AdviceParser
from scipy.stats import wilcoxon
from scipy.special import softmax
from sklearn.preprocessing import normalize

class Runner():

    def __init__(self, size, seed, numexperiments, maxepisodes, log_level=logging.INFO):
        self._SIZE = size
        self._SEED = seed
        self._BASERATE = 0.25 # TODO
        self._NUM_EXPERIMENTS = numexperiments
        self._MAX_EPISODES = maxepisodes
        
        #Hyperparameters
        self._SLIPPERY = False
        self._ALPHA = 0.9
        self._GAMMA = 1
        
        #File paths
        self._INPUT_PATH = './03-input'
        self._reward_results_PATH = './05-experiments-output'
        self._FILE_PATTERN = f'{size}x{size}-seed{seed}'
        self._MAP_NAME = f'{size}x{size}'
        
        #Map
        self._MAP_DESC = MapTools(self._INPUT_PATH).parse_map(size, seed)
        
        reward_results_folder = os.path.abspath(self._reward_results_PATH)
        if not os.path.exists(reward_results_folder):
            os.makedirs(reward_results_folder)
        
        #Logging
        logging.basicConfig(format='[%(levelname)s] %(message)s')
        logging.getLogger().setLevel(log_level)
        
    def get_default_policy(self, environment):
        num_states = environment.observation_space.n
        num_actions = environment.action_space.n
        
        default_policy = np.full((num_states, num_actions), 1/num_actions)
        
        return default_policy
    
    def get_advisor_input(self, quota=None, advisor_id=None, position=None):
        if advisor_id: #supported for coop mode only
            file = os.path.abspath(f'{self._INPUT_PATH}/advice-{self._FILE_PATTERN}-{quota}-{advisor_id}-{position}.txt')
        else:
            file = os.path.abspath(f'{self._INPUT_PATH}/advice-{self._FILE_PATTERN}-{quota}.txt')
        logging.info(f'Parsing advice file {file}')
        advice_parser = AdviceParser()
        
        return advice_parser.parse(file)

    def shape_policy(self, policy, advisor_opinions):
        for advisor_opinion in advisor_opinions.opinion_list:
            cell = advisor_opinion.cell
            #logging.debug(cell)
            for sap in cell.get_actions_to_me_from_all_neighbors():
                neighbor_row, neighbor_col = sap[0]
                neighbors_sequence_number = neighbor_row*cell.edge_size + neighbor_col
                action_number = sap[1].value
                
                base_action_probability = policy[neighbors_sequence_number][action_number]
                base_action_opinion = sl.probability_to_opinion(cell, base_action_probability)
                
                fused_opinion = sl.beliefConstraintFusion(advisor_opinion, base_action_opinion)
                fused_probability = sl.opinion_to_probability(fused_opinion)
                
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

    def discrete_policy_grad(self, max_episodes, advice=None, is_random=False):
        #Environment
        environment = gym.make('FrozenLake-v1', desc=self._MAP_DESC, is_slippery=self._SLIPPERY)

        if is_random == True:
            logging.debug('Agent policy is random')
            policy = np.zeros((environment.observation_space.n, environment.action_space.n))

            total_reward = []
            steps_taken = []
            for episode in range(max_episodes):
                state = environment.reset()[0]
                ep_states, ep_actions, ep_probs, ep_rewards, total_ep_rewards = [], [], [], [], 0
                terminated, truncated = False, False
                
                i = 0

                # gather trajectory
                while not terminated and not truncated:
                    i += 1
                    ep_states.append(state)         # add state to ep_states list
                    
                    action = environment.action_space.sample() # choose an action randomly
                    ep_actions.append(action)       # add action to ep_actions list
                    
                    state, reward, terminated, truncated, __ = environment.step(action) # take step in environment
                    ep_rewards.append(reward)       # add reward to ep_rewards list
                    
                    total_ep_rewards += reward

                ep_returns = self.calculate_return(ep_rewards) # calculate episode return & add total episode reward to totalReward
                total_reward.append(sum(ep_rewards))
                
                steps_taken.append((i, sum(total_reward)))

        else:
            logging.debug('Agent policy is not random')
            logging.debug('Generating default policy')
            policy = self.get_default_policy(environment)
            if advice:
                original_policy = policy
                #logging.info(f'\t\t\t Shaping policy with advisor 03-input at u={advice.u}') #TODO does not work with coop
                policy = self.shape_policy(policy, advice)
                #if advice.u==1.0: # TODO does not work with coop
                #    assert np.array_equal(original_policy, policy)
                #    print(np.array_equal(original_policy, policy))
            
            logging.debug('Initial policy:')
            logging.debug(policy)
            
            logging.debug('Policy initialized. Exploring now.')

            policy = self.policy_to_numerical_preferences(policy, environment)

            total_reward = []
            steps_taken = []
            for episode in range(max_episodes):
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

        environment.close()

        # success rate
        success_rate = (sum(total_reward) / max_episodes) * 100

        # cumulative reward
        cumulative_reward = np.cumsum(total_reward)

        # final policy
        final_policy = softmax(policy, axis = 1)
        logging.debug("Final Policy")
        logging.debug(final_policy)

        return success_rate, steps_taken, cumulative_reward, final_policy

    def evaluate(self, max_episodes, advice=None, is_random=False):  
        success_rates = []
        steps = []
        cumulative_rewards = []
        final_policies = []
        for i in range(self._NUM_EXPERIMENTS):
            logging.info(f'\t\t running experiment #{i+1}')
            success_rate, steps_taken, cumulative_reward, final_policy= self.discrete_policy_grad(max_episodes, advice=advice, is_random=is_random)
            success_rates.append(success_rate)
            steps.append(steps_taken)
            cumulative_rewards.append(cumulative_reward)
            final_policies.append(final_policy)
        return success_rates, steps, cumulative_rewards, final_policies

    def prepare_folder(self, experiment_name=None):
        logging.info(f'Preparing output folder')
        main_folder_name = experiment_name if experiment_name is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
        complete_folder_name = f'{self._reward_results_PATH}/{main_folder_name}'
        self.create_folder(complete_folder_name)
        
        return complete_folder_name
        
    def run_experiment_random(self, max_episodes):
        logging.info(f'====== RANDOM AGENT WITH {max_episodes} EPISODES ======')
        
        success_rates, steps, cumulative_rewards, final_policies = self.evaluate(max_episodes, is_random=True)
        reward_results = cumulative_rewards
        policy_results = self.preprocess_policy_data(final_policies)
        
        return reward_results, policy_results
            
    def run_experiment_noadvice(self, max_episodes):
        logging.info(f'====== NO ADVICE AGENT WITH {max_episodes} EPISODES ======')
        
        success_rates, steps, cumulative_rewards, final_policies = self.evaluate(max_episodes)
        reward_results = cumulative_rewards
        policy_results = self.preprocess_policy_data(final_policies)
        
        return reward_results, policy_results
    
    def run_experiment_synthetic(self, max_episodes, quota, u):
        logging.info(f'====== SYNTHETIC-ADVISED AGENT WITH {max_episodes} EPISODES AT u={u} ======')
        
        advisor_input = self.get_advisor_input(quota)
        assert advisor_input.map_size == self._SIZE #sanity check
        
        synthetic_opinions = SyntheticAdvisorOpinions(advisor_input, u, self._BASERATE)
  
        success_rates, steps, cumulative_rewards, final_policies = self.evaluate(max_episodes, advice=synthetic_opinions)
        reward_results = cumulative_rewards
        policy_results = self.preprocess_policy_data(final_policies)
        
        return reward_results, policy_results
    
    """
    def run_experiment_realhuman(self, maxepisodes, quota, position):
        logging.info(f'====== SYNTHETIC-ADVISED AGENT WITH {max_episodes} EPISODES AT u={u} ======')
        
        advisor_id='A'
        advisor_input = self.get_advisor_input(quota, advisor_id) # TODO: parser needs to be able to handle advisor ID (for coop mode too)
        assert advisor_input.map_size == self._SIZE #sanity check
        
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # This is the part here that needs to be adopted
        # calibrate u
        for i in advisor_input:
            advice = Advice(advisor_input, u) # todo: this needs to be adopted to accommodate piece-by-piece compilation
        
        #~~~~~~~~~~~~~~~~~~~~~~~~
        success_rates, steps, cumulative_rewards, final_policies = self.evaluate(max_episodes, advice=advice)
        reward_results = cumulative_rewards
        policy_results = self.preprocess_policy_data(final_policies)
        
        return reward_results, policy_results
    """

    
    def run_experiment_coop(self, max_episodes, quota, advisor1_position, advisor2_position):
        logging.info(f'====== COOP ADVISED AGENT WITH {max_episodes} EPISODES ======')
        logging.info(f'QUOTA: {quota} ** ADVISOR1: {advisor1_position} ** ADVISOR2: {advisor2_position}')

        # get advisor 03-input
        advisor1_id = "A1"
        advisor2_id = "A2"

        advisor1_input = self.get_advisor_input(quota, advisor1_id, advisor1_position)
        advisor2_input = self.get_advisor_input(quota, advisor2_id, advisor2_position)

        assert (advisor1_input.map_size and advisor2_input.map_size) == self._SIZE #sanity check
        
        #transform advice into opinions
        advisor1_opinions = HumanAdvisorOpinions(advisor1_input, advisor1_position, self._BASERATE)
        advisor2_opinions = HumanAdvisorOpinions(advisor2_input, advisor2_position, self._BASERATE)

        #fuse advice
        fused_opinions = sl.fuse_advisor_opinions(advisor1_opinions, advisor2_opinions)

        success_rates, steps, cumulative_rewards, final_policies = self.evaluate(max_episodes, advice=fused_opinions)
        reward_results = cumulative_rewards
        policy_results = self.preprocess_policy_data(final_policies)
        
        return reward_results, policy_results
    
    def run_experiment(self, mode, experiment_name=None):
        complete_folder_name = self.prepare_folder(experiment_name)
        
        for max_episodes in self._MAX_EPISODES:
            reward_data_folder_name = f'{complete_folder_name}/{max_episodes}/reward_data'
            self.create_folder(reward_data_folder_name)
            policy_data_folder_name = f'{complete_folder_name}/{max_episodes}/policy_data'
            self.create_folder(policy_data_folder_name)
            
            if mode=='random':
                reward_results, policy_results = self.run_experiment_random(max_episodes)
                self.save_experiment_data(reward_results, reward_data_folder_name, 'random')
                self.save_experiment_data(policy_results, policy_data_folder_name, 'random')
            elif mode=='noadvice':
                reward_results, policy_results = self.run_experiment_noadvice(max_episodes)
                self.save_experiment_data(reward_results, reward_data_folder_name, 'noadvice')
                self.save_experiment_data(policy_results, policy_data_folder_name, 'noadvice')
            elif mode=='synthetic':
                for quota in ['all', 'holes', 'human10', 'human5']:
                    for u in [0.01, 0.2, 0.4, 0.6, 0.8]:
                    #for u in [0.01]:
                        reward_results, policy_results = self.run_experiment_synthetic(max_episodes, quota=quota, u=u)
                        self.save_experiment_data(reward_results, reward_data_folder_name, f'advice-synthetic-{quota}', file_suffix = ('u', u))
                        self.save_experiment_data(policy_results, policy_data_folder_name, f'advice-synthetic-{quota}', file_suffix = ('u', u))
            elif mode=='realhuman':
                pass
                #for quota in ['human10', 'human5']:
                #    for position in ['topleft', 'bottomright', 'topright', 'bottomleft']:                    
                #        reward_results, policy_results = self.run_experiment_realhuman(max_episodes, quota=quota, position=position)
                #        self.save_experiment_data(reward_results, reward_data_folder_name, f'advice-real-{quota}', file_suffix = ('position', position))
                #        self.save_experiment_data(policy_results, policy_data_folder_name, f'advice-real-{quota}', file_suffix = ('position', position))
            elif mode=='coop':
                for quota in ['coop10', 'coop5']:
                    for position in [['topleft', 'bottomright'], ['topright', 'bottomleft']]:   
                        advisor1_position, advisor2_position =  position[0], position[1]       
                        reward_results, policy_results = self.run_experiment_coop(max_episodes, quota=quota, advisor1_position=(advisor1_position), advisor2_position=(advisor2_position))
                        self.save_experiment_data(reward_results, reward_data_folder_name, f'advice-{quota}-{advisor1_position}-{advisor2_position}')
                        self.save_experiment_data(policy_results, policy_data_folder_name, f'advice-{quota}-{advisor1_position}-{advisor2_position}')
            else:
                raise Exception(f'Unknown mode {mode} selected')
            
            logging.info(f'======EXPERIMENT DONE======\n')
            
    def create_folder(self, folder_name):
        folder = os.path.abspath(folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    def save_data(self, data, file_name):
        np.savetxt(f'{file_name}', data, delimiter=",")
    
    def save_experiment_data(self, data, root_folder, agent, file_suffix=None):
        folder_name = f'{root_folder}/{agent}'
        self.create_folder(folder_name)
        
        file_name = f'{folder_name}/{self._FILE_PATTERN}'
        if file_suffix is not None:
            file_name = '-'.join([file_name, f'{file_suffix[0]}-{file_suffix[1]}'])
            
        file_name = '.'.join([file_name, 'csv'])
        
        self.save_data(data, file_name)

    def preprocess_policy_data(self, policies_list):
        policies_arr = np.empty(((len(policies_list)), (self._SIZE**2) * 4)) # maybe find better way to set this 

        for i in range(len(policies_list)):
            policy = policies_list[i].reshape((policies_list[i].size))
            policies_arr[i] = policy
        
        return policies_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', required=True, type=str)
    
    parser.add_argument('--name', required=False, type=str)

    parser.add_argument(
        "-log",
        "--log",
        default="warning",
        help=("Provide logging level. "
              "Example '--log debug', default='warning'."
              )
        )
    options = parser.parse_args()
    
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    level = levels.get(options.log.lower())

    size = 12
    seed = 63
    numexperiments = 30
    #maxepisodes = [5000, 7500, 10000]
    maxepisodes = [10000]
    
    experiment_name = None
    if options.name is not None:
        experiment_name = options.name.lower()
        
    runner = Runner(size, seed, numexperiments, maxepisodes, level)
    
    mode = options.mode.lower()
    if mode not in ['random', 'noadvice', 'synthetic', 'human', 'coop']:
        raise Exception('Unknown mode selected')
    
    runner.run_experiment(mode, experiment_name)