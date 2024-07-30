import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

episodes = [10000]
size = 12
seed = 63
name = '04'

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
    df_advice_00 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.01.csv', header=None)
    df_advice_02 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.2.csv', header=None)
    df_advice_04 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.4.csv', header=None)
    df_advice_06 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.6.csv', header=None)
    df_advice_08 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-synthetic-{experiment_kind}/{filename}-u-0.8.csv', header=None)
    #df_advice_10 = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice/{filename}-u-1.0.csv', header=None),
    df_no_advice = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/noadvice/{filename}.csv', header=None)
    df_random = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/random/{filename}.csv', header=None)
    
    return {
        'advice_00': df_advice_00,
        'advice_02': df_advice_02,
        'advice_04': df_advice_04,
        'advice_06': df_advice_06,
        'advice_08': df_advice_08,
        #'advice_10': df_advice_10
        'no_advice': df_no_advice,
        'random': df_random
    }


def loadCoopData(experiment_kind, episode_number, data_kind):
    df_advice_coop_sequential = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-{experiment_kind}-topleft-bottomright/{filename}.csv', header=None)
    df_advice_coop_parallel = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/advice-{experiment_kind}-topright-bottomleft/{filename}.csv', header=None)
    df_no_advice = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/noadvice/{filename}.csv', header=None)
    df_random = pd.read_csv(f'{inputFolder}/{episode_number}/{data_kind.value}_data/random/{filename}.csv', header=None)
    
    return {
        'coop_sequential': df_advice_coop_sequential,
        'coop_parallel': df_advice_coop_parallel,
        'no_advice': df_no_advice,
        'random': df_random
    }


def savefig(plot_name):
    plt.gcf().tight_layout()
    plt.savefig(f'{resultsPath}/{plot_name}.pdf', bbox_inches='tight', pad_inches=0.01)


def print_rewards():
    for experiment_kind in ExperimentKind:
        experiment_kind = experiment_kind.value
        
        print(experiment_kind)
        for episode_number in episodes:

            if experiment_kind in ['all', 'holes', 'human5', 'human10']:
                dfs = loadSyntheticData(experiment_kind, episode_number, DataKind.REWARD)

            else:
                dfs = loadCoopData(experiment_kind, episode_number, DataKind.REWARD)
            
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
                dfs = loadCoopData(experiment_kind, episode_number, DataKind.REWARD)
            
            fig = plt.figure()
            ax = plt.gca()
            
            for df_name, df in dfs.items():

                mean = df.mean()
                x = np.arange(len(mean))
                if df_name == 'coop_sequential':
                    plt.plot(x, mean, label=df_name, color='tomato')
                elif df_name == 'coop_parallel':
                    plt.plot(x, mean, label=df_name, color='lightseagreen')
                elif df_name == 'advice_00':
                    plt.plot(x, mean, label=df_name, color='orange')
                elif df_name == 'advice_02':
                    plt.plot(x, mean, label=df_name, color='yellowgreen')
                elif df_name == 'advice_04':
                    plt.plot(x, mean, label=df_name, color='dodgerblue')
                elif df_name == 'advice_06':
                    plt.plot(x, mean, label=df_name, color='darkviolet')
                elif df_name == 'advice_08':
                    plt.plot(x, mean, label=df_name, color='hotpink')
                elif df_name == 'no_advice':
                    plt.plot(x, mean, label=df_name, color='black', linestyle='dashed')
                elif df_name == 'random':
                    plt.plot(x, mean, label=df_name, color='black', linestyle='dotted')
                else:
                    plt.plot(x, mean, label=df_name)
                # Used to plot standard deviation
                #if df_name=='no_advice':
                #    std = df.std()
                #    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
                #    maxvalue = df.max()
                #    plt.plot(x, maxvalue, label='noadvice-max')
                #    minvalue = df.min()
                #    plt.plot(x, minvalue, label='noadvice-min')
            
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            ax.set_ylim([0, 10000])

            if experiment_kind in ['all', 'holes', 'human5', 'human10']:
                #legend_labels = ['Random', 'No advice', 'No advice sigma', 'No advice max', 'No advice min', 'Advice@u=0.0', 'Advice@u=0.2', 'Advice@u=0.4', 'Advice@u=0.6', 'Advice@u=0.8']
                legend_labels = ['Advice@u=0.0', 'Advice@u=0.2', 'Advice@u=0.4',
                                 'Advice@u=0.6', 'Advice@u=0.8', 'No advice', 'Random']
            else:
                #legend_labels = ['Random', 'No advice', 'No advice sigma', 'No advice max', 'No advice min', 'Top Left Bottom Right', 'Top Right Bottom Left']
                legend_labels = ['Coop - Sequential', 'Coop - Parallel', 'No advice', 'Random']

            plt.legend(labels=legend_labels, fontsize='14', loc='upper left')
            #plt.show()
            
            logging.info('\tSave linear plot')
            savefig(f'{folder_name}/{experiment_kind}/cumulative_reward-{experiment_kind}-{episode_number}-linear')
            
            logging.info('\tSave log plot')
            plt.legend(labels=legend_labels, fontsize='14', loc='lower right')
            plt.yscale('log')
            ax.autoscale()
            savefig(f'{folder_name}/{experiment_kind}/cumulative_reward-{experiment_kind}-{episode_number}-log')


def heatmap():
    folder_name = 'heatmaps'
    os.mkdir(f'{resultsPath}/{folder_name}')
    
    for experiment_kind in ExperimentKind:
        experiment_kind = experiment_kind.value
        
        logging.info(f'Running analysis heatmaps with experiment kind {experiment_kind}')
        
        os.mkdir(f'{resultsPath}/{folder_name}/{experiment_kind}')
    
        for episode_number in episodes:
            logging.info(f'Running analysis heatmap with episode_number {episode_number}')
            
            if experiment_kind in ['all', 'holes', 'human5', 'human10']:
                dfs = loadSyntheticData(experiment_kind, episode_number, DataKind.POLICY)

            else:
                dfs = loadCoopData(experiment_kind, episode_number, DataKind.POLICY)
            
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
