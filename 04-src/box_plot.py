import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from data_loader import  load_all_data, get_advised_df_for_u, U_VALUES


def get_color(label):
    if 'sequential' in label and 'coop' in label:
        return 'tomato'
    elif 'parallel' in label and 'coop' in label:
        return 'lightseagreen'
    elif '@u=0.01' in label:
        return 'orange'
    elif '@u=0.2' in label:
        return 'yellowgreen'
    elif '@u=0.4' in label:
        return 'dodgerblue'
    elif '@u=0.6' in label:
        return 'darkviolet'
    elif '@u=0.8' in label:
        return 'hotpink'
    else:
        return 'lightgray'

if __name__ == "__main__":

    all_data = load_all_data()
    advised_mode = {
        "advised_all": all_data['advised_all'],
        "advised_holes_and_goal": all_data['advised_holes_and_goal'],
        "human_10": all_data['advised_human_10'],
        "human_5": all_data['advised_human_5']
    }
    coop_mode = {
        "coop10_sequential": all_data['coop10_sequential'],
        "coop10_parallel" : all_data['coop10_parallel'],
        "coop5_sequential": all_data['coop5_sequential'],
        "coop5_parallel" : all_data['coop5_parallel']
    }
    unadvised_mode = {
        "random": all_data['random'],
        "unadvised": all_data['unadvised']
    }

    data = []
    labels = []
    color_keys = []

    # UNADVISED
    data.append(unadvised_mode['random'].iloc[:, -1])
    labels.append('Random')
    color_keys.append("unadvised")
    data.append(unadvised_mode['unadvised'].iloc[:, -1])
    labels.append('No advice')
    color_keys.append("unadvised")

    # ADVISED
    for mode_name, df in advised_mode.items():
        for u in U_VALUES:
            data.append(get_advised_df_for_u(df, u))
            display_name = {
                'advised_all': 'Oracle 100%',
                'advised_holes_and_goal': 'Oracle 20%',
                'human_10': 'Human 10%',
                'human_5': 'Human 5%'
            }.get(mode_name, mode_name.replace('_', ' '))
            labels.append(display_name)
            color_keys.append(f"@u={u}")

    # COOP
    for key in coop_mode:
        data.append(coop_mode[key].iloc[:, -1])
        display_name = {
            'coop10_sequential': 'Coop 10%`',
            'coop10_parallel': 'Coop 10%',
            'coop5_sequential': 'Coop 5%',
            'coop5_parallel': 'Coop 5%'
        }.get(key, key.replace('_', ' '))
        labels.append(display_name)
        color_keys.append(key)

    # PLOT
    plt.figure(figsize=(16, 10))
    box = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.65)
    for median in box['medians']:
        median.set_color('black')

    for patch, key in zip(box['boxes'], color_keys):
        patch.set_facecolor(get_color(key))

    legend_elements = [
        Patch(facecolor='orange', label='u = 0.0'),
        Patch(facecolor='yellowgreen', label='u = 0.2'),
        Patch(facecolor='dodgerblue', label='u = 0.4'),
        Patch(facecolor='darkviolet', label='u = 0.6'),
        Patch(facecolor='hotpink', label='u = 0.8'),
        Patch(facecolor='tomato', label='coop sequential'),
        Patch(facecolor='lightseagreen', label='coop parallel')
    ]

    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.20),
               ncol=8, frameon=False, fontsize=12)

    plt.xticks(rotation=70, fontsize = 14)
    plt.ylabel('Cumulative Reward')
    plt.tight_layout()
    plt.savefig("06-analysis-output/box_plot.pdf", format='pdf', bbox_inches='tight')
    plt.show()