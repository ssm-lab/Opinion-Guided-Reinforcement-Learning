from enum import Enum
import pandas as pd

input_folder = './05-experiments-output/final-experiment/10000/reward_data'
filename = f'12x12-seed63'
U_VALUES = ['0.01', '0.2', '0.4', '0.6', '0.8']
NUMBER_OF_EXPERIMENTS = 30



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

def load_all_data():
    all_data = {}
    for experiment_type in ExperimentType:
        all_data[experiment_type.name.lower()] = get_processed_data(experiment_type)
    return all_data

def get_advised_df_for_u(df, u_str, u_values=None, num_experiments=NUMBER_OF_EXPERIMENTS):
    if u_values is None:
        u_values = U_VALUES
    index = u_values.index(u_str)
    start = index * num_experiments
    end = start + num_experiments
    return df.iloc[start:end, -1]