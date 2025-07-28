from enum import Enum
import pandas as pd
from scipy.stats import ttest_ind


class ExperimentType(Enum):
    ADVISED_ALL = f'advice-synthetic-all'
    ADVISED_HOLES_AND_GOAL = f'advice-synthetic-holes'
    ADVISED_HUMAN_10 = f'advice-synthetic-human10'
    ADVISED_HUMAN_5 = f'advice-synthetic-human5'

    COOP10_PARALLEL = 'advice-coop10-topright-bottomleft'
    COOP10_SEQUENTIAL = 'advice-coop10-topleft-bottomright'
    COOP5_PARALLEL = 'advice-coop5-topright-bottomleft'
    COOP5_SEQUENTIAL= 'advice-coop5-topleft-bottomright'

    RANDOM = 'random'
    UNADVISED = 'noadvice'

    def is_synthetic(self):
        return self in {
            ExperimentType.ADVISED_ALL, ExperimentType.ADVISED_HOLES_AND_GOAL, ExperimentType.ADVISED_HUMAN_10, ExperimentType.ADVISED_HUMAN_5
        }

    def is_unadvised(self):
        return self in {
            ExperimentType.RANDOM, ExperimentType.UNADVISED
        }

    def is_coop(self):
        return self in {
            ExperimentType.COOP10_PARALLEL, ExperimentType.COOP10_SEQUENTIAL, ExperimentType.COOP5_PARALLEL, ExperimentType.COOP5_SEQUENTIAL
        }

def load_synthetic_data(experiment_type):
    df = pd.DataFrame()
    for u in U_VALUES:
        sub_df = pd.read_csv(f'{input_folder}/{experiment_type.value}/{filename}-u-{u}.csv', header=None).iloc[:, -1:]
        df = pd.concat([df, sub_df], ignore_index=True)
    return df

def load_data(experiment_type):
    return pd.read_csv(f'{input_folder}/{experiment_type.value}/{filename}.csv', header=None).iloc[:, -1:]

def get_processed_data(experiment_type):
    if experiment_type.is_synthetic():
        return load_synthetic_data(experiment_type)
    else:
        return load_data(experiment_type)

input_folder = './05-experiments-output/final-experiment/10000/reward_data'
filename = f'12x12-seed63'
U_VALUES = ['0.01', '0.2', '0.4', '0.6', '0.8']
NUMBER_OF_EXPERIMENTS = 30



if __name__ == "__main__":

    # LOAD DATAFRAMES
    random = get_processed_data(ExperimentType.RANDOM)
    unadvised = get_processed_data(ExperimentType.UNADVISED)

    advised_all = get_processed_data(ExperimentType.ADVISED_ALL)
    advised_holes_goal = get_processed_data(ExperimentType.ADVISED_HOLES_AND_GOAL)
    advised_human10 = get_processed_data(ExperimentType.ADVISED_HUMAN_10)
    advised_human5 = get_processed_data(ExperimentType.ADVISED_HUMAN_5)

    coop10_parallel = get_processed_data(ExperimentType.COOP10_PARALLEL)
    coop10_sequential = get_processed_data(ExperimentType.COOP10_SEQUENTIAL)
    coop5_parallel = get_processed_data(ExperimentType.COOP5_PARALLEL)
    coop5_sequential = get_processed_data(ExperimentType.COOP5_SEQUENTIAL)

    advised_mode = {
        "advised_all": advised_all,
        "advised_holes_and_goal": advised_holes_goal,
        "human_10": advised_human10,
        "human_5": advised_human5
    }

    coop_mode = {
        "coop10_parallel" : coop10_parallel,
        "coop10_sequential" : coop10_sequential,
        "coop5_parallel" : coop5_parallel,
        "coop5_sequential" : coop5_sequential
    }

    unadvised_mode = {
        "random": random,
        "unadvised": unadvised
    }

    # MEAN & VARIANCES
    print("MEAN CUMULATIVE REWARD & VARIANCE\n")
    for unadvised_name, unadvised_df in unadvised_mode.items():
        mean = unadvised_df.mean().item()
        std = unadvised_df.std().item()
        print(f"{unadvised_name.title()}: {mean:.3f} ± {std:.3f}")
        print()

    for advised_name, advised_df in advised_mode.items():
        for index, u in enumerate(U_VALUES):
            start = index * NUMBER_OF_EXPERIMENTS
            end = start + NUMBER_OF_EXPERIMENTS
            advised_data = advised_df.iloc[start:end, 0].values
            mean = advised_data.mean().item()
            std = advised_data.std().item()
            print(f"{advised_name.title()}  @u={u}: {mean:.3f} ± {std:.3f}")
        print()

    for coop_name, coop_df in coop_mode.items():
        mean = coop_df.mean().item()
        std = coop_df.std().item()
        print(f"{coop_name.title()}: {mean:.3f} ± {std:.3f}")
        print()

    print("*" * 75)

    print("CUMULATIVE REWARD T-TEST\n")
    # UNADVISED VS RANDOM
    print("Unadvised vs Random")
    random_data = random.iloc[:, 0].values.astype(float)
    unadvised_data = unadvised.iloc[:, 0].values.astype(float)
    t_stat, p_val = ttest_ind(random_data, unadvised_data, equal_var=False)
    print(f"  p = {p_val:.4f}\n")


    # SYNTHETIC VS UNADVISED & RANDOM
    for advised_name, advised_df in advised_mode.items():
        for unadvised_name, unadvised_df in unadvised_mode.items():
            print(f"{advised_name.replace('_', ' ').title()} vs {unadvised_name.title()}")

            for index, u in enumerate(U_VALUES):
                start = index * NUMBER_OF_EXPERIMENTS
                end = start + NUMBER_OF_EXPERIMENTS

                advised_data = advised_df.iloc[start:end, 0].values.astype(float)
                unadvised_data = unadvised_df.iloc[:, 0].values.astype(float)
                _stat, p_val = ttest_ind(advised_data, unadvised_data, equal_var=False)
                print(f"  @u={u}: p = {p_val:.4f}")
            print()


    #SYNTHETIC VS COOP
    for advised_name, advised_df in advised_mode.items():
        for coop_name, coop_df in coop_mode.items():
            print(f"{advised_name.replace('_', ' ').title()} vs {coop_name.title().replace('_', ' ').title()}")

            for index, u in enumerate(U_VALUES):
                start = index * NUMBER_OF_EXPERIMENTS
                end = start + NUMBER_OF_EXPERIMENTS

                advised_data = advised_df.iloc[start:end, 0].values.astype(float)
                coop_data = coop_df.iloc[:, 0].values.astype(float)

                _stat, p_val = ttest_ind(advised_data, coop_data, equal_var=False)
                print(f"  @u={u}: p = {p_val:.4f}")
            print()

    # COOP VS UNADVISED & RANDOM 16
    for coop_name, coop_df in coop_mode.items():
        for unadvised_name, unadvised_df in unadvised_mode.items():
            print(f"{coop_name.replace('_', ' ').title()} vs {unadvised_name.title()}")

            coop_data = coop_df.iloc[:, 0].values.astype(float)
            unadvised_data = unadvised_df.iloc[:, 0].values.astype(float)

            _stat, p_val = ttest_ind(coop_data, unadvised_data, equal_var=False)
            print(f"  p = {p_val:.4f}")
            print()
