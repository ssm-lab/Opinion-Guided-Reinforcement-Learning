import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from enum import Enum
from map_tools import MapTools

episodes = 5000
size = 12
seed = 63
filename = f'{size}x{size}-seed{seed}'
inputFolder = './experiments/02-pilot-full/human5'
outputFolder = './analysis-output'
experiments_input_path = './input'

class DataKind(Enum):
    REWARD = 'reward'
    POLICY = 'policy'


def loadData(data_kind):
    df_random = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/random/{filename}.csv', header=None)
    df_no_advice = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/noadvice/{filename}.csv', header=None)
    df_advice_00 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-0.01.csv', header=None)
    df_advice_02 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-0.2.csv', header=None)
    df_advice_04 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-0.4.csv', header=None)
    df_advice_06 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-0.6.csv', header=None)
    df_advice_08 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-0.8.csv', header=None)
    df_advice_10 = pd.read_csv(f'{inputFolder}/{episodes}/{data_kind.value}_data/advice/{filename}-u-1.0.csv', header=None)
    
    return {
        'random': df_random,
        'no_advice': df_no_advice,
        'advice_00': df_advice_00,
        'advice_02': df_advice_02,
        'advice_04': df_advice_04,
        'advice_06': df_advice_06,
        'advice_08': df_advice_08,
        'advice_10': df_advice_10
    }
    
    
def cumulative_reward():
    dfs = loadData(DataKind.REWARD)
    
    fig = plt.figure()
    ax = plt.gca()
    
    for df_name, df in dfs.items():
        mean = df.mean()
        std = df.std()
        x = np.arange(len(mean))
        plt.plot(x, mean, label=df_name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        
    #plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def heatmap():
    dfs = loadData(DataKind.POLICY)
    
    #print(dfs['advice_00'])
    df = dfs['advice_00'].mean().to_frame()
    
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
    
    print(df)
    
    df = df.sort_values('prob', ascending=False).drop_duplicates(['cellid'])
    df = df.sort_values('cellid', ascending=True).reset_index(drop=True)
    #df = df.drop(['cellid'], axis=1)
    
    size = 12
    seed = 63
    
    df = df.assign(row = lambda x: (x['cellid'] // 12))
    df = df.assign(col = lambda x: (x['cellid'] % 12))
    
    map_description = MapTools(experiments_input_path).parse_map(size, seed)
    print(map_description)
    
    terminals = []
    for rid, row in enumerate(map_description):
        for cell in range(0, len(row)):
            #print(row[cell])
            if row[cell] in ('H', 'G'):
                terminals.append(size*rid+cell)
                
    for t in terminals:
        df.loc[t, 'prob'] = 0.0
        df.loc[t, 'direction'] = ''
        
    df = df.loc[df['prob'] !=0.25]
    df = df.loc[df['prob'] !=0.0]
    
    print(df)
    
    result = df.pivot(index='row', columns='col', values='prob')
    print(result)
    
    directions = df.pivot(index='row', columns='col', values='direction')
    print(directions)

    sns.heatmap(
        result,
        linewidths=0.001,
        linecolor='gray',
        annot=directions,
        fmt='',
        cmap=sns.color_palette("Blues", as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "x-large"}
    )
    plt.show()
    

#cumulative_reward()
heatmap()


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
    
    

