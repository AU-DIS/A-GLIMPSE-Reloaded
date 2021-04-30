import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


# Filthy code to be cleaned up latur
def plot_combined_theoretical(substring, ratio):
    sizes = [(1 * 1.1)**i for i in range(0, 30)][::-1]
    # markers = ['.', ',', 'o', 'v', '<', '>']
    markers = ['-', '--', '-.', ':']
    ax = None
    labels = []
    i = 0
    ys = []

    for d in os.walk(files_dir):
        for fi in os.listdir(d[0]):
            if "csv" in fi and '.DS_Store' not in fi and '.png' not in fi and substring in fi:
                try:
                    df = pd.read_csv(f"{d[0]}/{fi}", skiprows=[0])
                except:
                    print(f"{d[0]}/{fi}")
                if 'k' not in df.columns:
                    df['k'] = 1

                df[f"cumsum"] = df["regret"].cumsum()

                stripped_s = fi.strip('.csv')
                p_start = fi.index('p')

                labels.append(f"Proportion = {stripped_s[p_start+1:]}")

                ys.append(df["cumsum"])

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = np.argsort([len(x) for x in ys])[::-1]
    labels = np.array(labels)[order]
    plt.grid(linestyle='--', linewidth=0.5)
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(ys))]
    #ax.set_prop_cycle('color', colors)
    for y in np.array(ys)[order]:
        s = sizes[i]
        i += 1
        ax.plot(y, alpha=0.5,
                linestyle=markers[i % len(markers)], markersize=5)

    ax.set_ylim([0, len(ys[0])])
    ax.legend(labels)
    plt.tight_layout()
    plt.savefig(f"{files_dir}/ratio{ratio}.png")


files_dir = "exp3_binary_kg_reward"
fixed_ratio_combinations = ["r1000000-ent100-",
                            "r1000000-ent1000-", "r1000000-ent10000-"]
labels = ["10k", "1k", "100"]

for filename, label in zip(fixed_ratio_combinations, labels):
    plot_combined_theoretical(filename, label)
