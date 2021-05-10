import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


# Filthy code to be cleaned up latur
def plot_combined_theoretical(output_path, filenames, no_rounds):
    ys = []
    markers = ['-', '--', '-.', ':']
    ax = None
    labels = []
    i = 0
    for filename in filenames:
        sizes = [(1 * 1.1)**i for i in range(0, 30)][::-1]
    # markers = ['.', ',', 'o', 'v', '<', '>']
        df = pd.read_csv(filename, skiprows=[0])
        if 'k' not in df.columns:
            df['k'] = 1
        df[f"cumsum"] = df["regret"].cumsum()
        labels.append(f"Number of rounds: {no_rounds[filename]}")
        ys.append(df["cumsum"])
        del df

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = np.argsort([len(x) for x in ys])[::-1]
    labels = np.array(labels)[order]
    plt.grid(linestyle='--', linewidth=0.5)
    colormap = plt.cm.nipy_spectral
    # colors = [colormap(i) for i in np.linspace(0, 1, len(ys))]
    # ax.set_prop_cycle('color', colors)
    for y in np.array(ys)[order]:
        s = sizes[i]
        i += 1
        ax.plot(y, alpha=0.5,
                linestyle=markers[i % len(markers)], markersize=5)

    max_ar = 0
    for ar in ys:
        max_ar = max(len(ar), max_ar)

    ax.set_ylim([0, len(ys[0])])
    ax.legend(labels)
    plt.tight_layout()
    plt.savefig(f"{output_path}.png")


# files_dir = "exp3_subgraph"
# filenames = ["regret_follow-cut-low-level.csv",
 #            "regret_happen-close-political-kid.csv", "regret_pay-physical-left-company.csv"]
# labels = {"regret_follow-cut-low-level.csv": "20000",
 #         "regret_happen-close-political-kid.csv": "10000", "regret_pay-physical-left-company.csv": "30000"}

# plot_combined_theoretical(files_dir, filenames, labels)
