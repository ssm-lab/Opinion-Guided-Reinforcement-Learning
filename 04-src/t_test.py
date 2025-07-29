from scipy.stats import ttest_ind
from data_loader import  load_all_data, NUMBER_OF_EXPERIMENTS, U_VALUES


if __name__ == "__main__":

    all_data = load_all_data()
    advised_mode = {
        "advised_all": all_data['advised_all'],
        "advised_holes_and_goal": all_data['advised_holes_and_goal'],
        "human_10": all_data['advised_human_10'],
        "human_5": all_data['advised_human_5']
    }
    coop_mode = {
        "coop10_parallel" : all_data['coop10_parallel'],
        "coop10_sequential" : all_data['coop10_sequential'],
        "coop5_parallel" : all_data['coop5_parallel'],
        "coop5_sequential" : all_data['coop5_sequential']
    }
    unadvised_mode = {
        "random": all_data['random'],
        "unadvised": all_data['unadvised']
    }

    print("CUMULATIVE REWARD T-TEST\n")
    # UNADVISED VS RANDOM
    print("Unadvised vs Random")
    random_data = all_data['random'].iloc[:, 0].values.astype(float)
    unadvised_data = all_data['unadvised'].iloc[:, 0].values.astype(float)
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
