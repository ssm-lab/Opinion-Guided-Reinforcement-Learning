import matplotlib.pyplot as plt
import pandas as pd

episodes = 10000
size = 12
seed = 63
filename = f'{size}x{size}-seed{seed}'
inputFolder = './experiments/human5'
outputFolder = './analysis-output'

def loadData():
    df_random = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/random/{filename}.csv', header=None)
    df_no_advice = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/noadvice/{filename}.csv', header=None)
    df_advice_00 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-0.01.csv', header=None)
    df_advice_02 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-0.2.csv', header=None)
    df_advice_04 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-0.4.csv', header=None)
    df_advice_06 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-0.6.csv', header=None)
    df_advice_08 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-0.8.csv', header=None)
    df_advice_10 = pd.read_csv(f'{inputFolder}/{episodes}/reward_data/advice/{filename}-u-1.0.csv', header=None)
    return df_random, df_no_advice, df_advice_00, df_advice_02, df_advice_04, df_advice_06, df_advice_08, df_advice_10
    
    
def cumulative_reward():
    dfs = loadData()
    
    
    fig = plt.figure()
    ax = plt.gca()
    
    dfs_names = ['random', 'no advice', 'advice-0.0', 'advice-0.2', 'advice-0.4', 'advice-0.6', 'advice-0.8', 'advice-1.0']
    
    for idx, df in enumerate(dfs):
        ax = df.mean().plot(label=dfs_names[idx])
        
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()
    
    
    """
    years = df['Publication year'].value_counts().sort_index().to_frame().reset_index()
    years.columns = ['year', 'papers']

    #print(years)

    bins = [2007, 2009, 2014, 2019, 2024]
    labels = ['-2009', '2010-2014', '2015-2019', '2020-2024']

    years['yearbins'] = pd.cut(years['year'], bins, labels=labels)

    #print(years)

    #print(years.groupby('yearbins').sum()['papers'])

    fig = plt.figure(figsize = (10, 5))
    
    ax = years.groupby('yearbins').sum()['papers'].plot(kind='bar', rot=0, color='#43b82c')
    
    ax.set_ylabel('Papers', fontsize=15)
    ax.set_xlabel('Years', fontsize=15)
    ax.bar_label(ax.containers[0], fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(f'{outputFolder}/papers-per-year.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    """
    
    
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
"""
    
    
    
cumulative_reward()
