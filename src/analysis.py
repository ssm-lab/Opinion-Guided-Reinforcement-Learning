import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from enum import Enum
from map_tools import MapTools
import argparse
import os
import shutil
import logging
from datetime import datetime

#episodes = [5000, 7500, 10000]
episodes = [10000]
size = 12
seed = 63
name = 'july17'

filename = f'{size}x{size}-seed{seed}'
inputFolder = f'./experiments/{name}'
resultsPath = './analysis-output'
experiments_input_path = './input'

class DataKind(Enum):
    REWARD = 'reward'
    POLICY = 'policy'
    
class ExperimentKind(Enum):
    SYNTHETIC_ALL = 'all'
    SYNTHETIC_HOLES = 'holes'
    HUMAN_5 = 'human5'
    HUMAN_10 = 'human10'
    COOPERATIVE_5 = 'coop5'
    COOPERATIVE_10 = 'coop10'

def loadSyntheticData(experiment_kind, episode_number, data_kind):
    df_random = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/random/{filename}.csv', header=None)
    df_no_advice = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/noadvice/{filename}.csv', header=None)
    df_advice_00 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.01.csv', header=None)
    df_advice_02 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.2.csv', header=None)
    df_advice_04 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.4.csv', header=None)
    df_advice_06 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.6.csv', header=None)
    df_advice_08 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.8.csv', header=None)
    #df_advice_10 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice/{filename}-u-1.0.csv', header=None)
    
    return {
        'random': df_random,
        'no_advice': df_no_advice,
        'advice_00': df_advice_00,
        'advice_02': df_advice_02,
        'advice_04': df_advice_04,
        'advice_06': df_advice_06,
        'advice_08': df_advice_08,
        #'advice_10': df_advice_10
    }

def loadHumanData(experiment_kind, episode_number, data_kind):
    df_random = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/random/{filename}.csv', header=None)
    df_no_advice = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/noadvice/{filename}.csv', header=None)
    df_advice_coop_topleft_bottomright = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-{experiment_kind}-topleft-bottomright/{filename}.csv', header=None)
    df_advice_coop_topright_bottomleft = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-{experiment_kind}-topright-bottomleft/{filename}.csv', header=None)
    
    return {
        'random': df_random,
        'no_advice': df_no_advice,
        'coop_topleft_bottomright': df_advice_coop_topleft_bottomright,
        'coop_topright_bottomleft':df_advice_coop_topright_bottomleft
    }
    
def savefig(plot_name):
    plt.gcf().tight_layout()
    plt.savefig(f'{resultsPath}/{plot_name}.pdf', bbox_inches='tight', pad_inches=0.01)

def print_rewards():
    for experiment_kind in ExperimentKind:
        experiment_kind = experiment_kind.value
        
        print(experiment_kind)
        for episode_number in episodes:
            
            dfs = loadData(experiment_kind, episode_number, DataKind.REWARD)
            
            for df_name, df in dfs.items():
                mean = df.mean()
                print(f'{df_name} mean: {round(mean.iloc[-1], 3)}')
            
            print('')

def cumulative_reward():
    folder_name = f'cumulative_reward-{name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.mkdir(f'{resultsPath}/{folder_name}')
    
    for experiment_kind in ExperimentKind:
        experiment_kind = experiment_kind.value
        
        logging.info(f'Running analysis cumulative_reward with experiment kind {experiment_kind}')
        
        os.mkdir(f'{resultsPath}/{folder_name}/{experiment_kind}')
    
        for episode_number in episodes:
            logging.info(f'Running analysis cumulative_reward with episode_number {episode_number}')
            
            if experiment_kind in ['all', 'holes', 'human5', 'human10']:
                dfs = loadSyntheticData(experiment_kind, episode_number, DataKind.REWARD)

            else:
                dfs = loadHumanData(experiment_kind, episode_number, DataKind.REWARD)
            
            fig = plt.figure()
            ax = plt.gca()
            
            for df_name, df in dfs.items():
                mean = df.mean()
                x = np.arange(len(mean))
                plt.plot(x, mean, label=df_name)
                if df_name=='no_advice':
                    std = df.std()
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
                    maxvalue = df.max()
                    plt.plot(x, maxvalue, label='noadvice-max')                    
                    minvalue = df.min()
                    plt.plot(x, minvalue, label='noadvice-min')                    
            
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            ax.set_ylim([0, 10000])

            if experiment_kind in ['all', 'holes', 'human5', 'human10']:
                legend_labels = ['Random', 'No advice', 'No advice sigma', 'No advice max', 'No advice min', 'Advice@u=0.0', 'Advice@u=0.2', 'Advice@u=0.4', 'Advice@u=0.6', 'Advice@u=0.8']
            else: 
                legend_labels = ['Random', 'No advice', 'No advice sigma', 'No advice max', 'No advice min', 'Top Left Bottom Right', 'Top Right Bottom Left']


            plt.legend(labels = legend_labels, fontsize='14', loc = 'upper left')
            #plt.show()
            
            logging.info('\tSave linear plot')
            plt.title(f'Cumulative Reward {experiment_kind}-{episode_number}-linear') # remove for final plots
            savefig(f'{folder_name}/{experiment_kind}/cumulative_reward-{experiment_kind}-{episode_number}-linear')
            
            logging.info('\tSave log plot')
            plt.legend(labels = legend_labels, fontsize='14', loc = 'lower right')
            plt.yscale('log')
            ax.autoscale()
            plt.title(f'Cumulative Reward {experiment_kind}-{episode_number}-log')
            savefig(f'{folder_name}/{experiment_kind}/cumulative_reward-{experiment_kind}-{episode_number}-log')

def heatmap(): #TODO update for coop mode
    folder_name = 'heatmaps'
    os.mkdir(f'{resultsPath}/{folder_name}')
    
    for experiment_kind in ExperimentKind:
        experiment_kind = experiment_kind.value
        
        logging.info(f'Running analysis heatmaps with experiment kind {experiment_kind}')
        
        os.mkdir(f'{resultsPath}/{folder_name}/{experiment_kind}')
    
        for episode_number in episodes:
            logging.info(f'Running analysis heatmap with episode_number {episode_number}')
            
            dfs = loadData(experiment_kind, episode_number, DataKind.POLICY)
            
            #logging.debug(dfs['advice_00'])
            
            for advice_type, data_frame in dfs.items():
                print(advice_type)
            
                #df = dfs['advice_00'].mean().to_frame() # TODO: should process every dataframe (rand, unadvised, and every advised (every level of u))
                df = data_frame.mean().to_frame()
                
                jss = []
                for i in range(0, 144):
                    jss.append([i, i, i, i])
                cellids = [j for js in jss for j in js]
                
                df.insert(0, 'cellid', cellids)
                df.columns.values[1] = 'prob'
                
                dss = []
                for d in range(0, 144):
                    dss.append(['←', '↓', '→', '↑'])
                    #dss.append(['L', 'D', 'R', 'U'])
                directions = [d for ds in dss for d in ds]
                
                df['direction'] = directions
                
                logging.debug(df)
                
                df = df.sort_values('prob', ascending=False).drop_duplicates(['cellid'])
                df = df.sort_values('cellid', ascending=True).reset_index(drop=True)
                #df = df.drop(['cellid'], axis=1)
                
                size = 12
                seed = 63
                
                df = df.assign(row = lambda x: (x['cellid'] // 12))
                df = df.assign(col = lambda x: (x['cellid'] % 12))
                
                map_description = MapTools(experiments_input_path).parse_map(size, seed)
                logging.debug(map_description)
                
                terminals = []
                for rid, row in enumerate(map_description):
                    for cell in range(0, len(row)):
                        #logging.debug(row[cell])
                        if row[cell] in ('H', 'G'):
                            terminals.append(size*rid+cell)
                            
                for t in terminals:
                    df.loc[t, 'prob'] = 0.0
                    df.loc[t, 'direction'] = ''
                    
                df = df.loc[df['prob'] !=0.25]
                df = df.loc[df['prob'] !=0.0]
                
                logging.debug(df)
                
                result = df.pivot(index='row', columns='col', values='prob')
                logging.debug(result)
                
                directions = df.pivot(index='row', columns='col', values='direction')
                logging.debug(directions)
                
                plt.clf()
                
                ax = sns.heatmap(
                    result,
                    linewidths=0.001,
                    linecolor='gray',
                    square=True,
                    annot=directions,
                    fmt='',
                    cmap=sns.color_palette("Blues", as_cmap=True),
                    vmin=0.0,
                    vmax=1.0,
                    xticklabels=[],
                    yticklabels=[],
                    annot_kws={"fontsize": "x-large"}
                )
                ax.axis('off')
                plt.xlabel('')
                plt.ylabel('')
                #plt.show()
                logging.info('\tSave heatmap')
                savefig(f'{folder_name}/{experiment_kind}/heatmap-{experiment_kind}-{advice_type}-{episode_number}')
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str)
    parser.add_argument('-s','--stash', help='Stash results folder.', )
    
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
    
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    logging.getLogger().setLevel(level)
        
    if options.stash:
        logging.info('Previous analysis output folder will be stashed.')
        if os.path.exists(resultsPath) and os.path.isdir(resultsPath):
            shutil.rmtree(resultsPath)
        os.mkdir(resultsPath)
    
    all_analyses = ['cumulative_reward', 'heatmap']
    
    if not options.a:
        logging.info('Running all')
        for analysis in all_analyses:
            exec(f'{analysis}()')
        
    else:
        logging.info(f'Running analysis {options.a}.')
        exec(f'{options.a}()')


"""
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
"""