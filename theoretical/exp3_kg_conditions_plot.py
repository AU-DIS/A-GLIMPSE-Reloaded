import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


def plot_combined():

    filename = "r10000000-ent6500000-notuples47000000-nocorrect10000.csv"
    file_dirs = ["exp3_kg_conditions_binary", "exp3_kg_conditions_kg_reward"]

    sizes = [(1 * 1.1)**i for i in range(0, 30)][::-1]

    markers = ['-', '--', '-.', ':']
    ax = None
    labels = []
    i = 0
    ys = []

    df1 = pd.read_csv(f"{file_dirs[0]}/{filename}", skiprows=[0])
    df2 = pd.read_csv(f"{file_dirs[1]}/{filename}", skiprows=[0])
    df1[f"cumsum"] = df1["regret"].cumsum()
    df2[f"cumsum"] = df2["regret"].cumsum()
    ys.append(df1["cumsum"])
    ys.append(df2["cumsum"])
    labels.append("Binary reward function")
    labels.append("KG reward function")

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
    plt.savefig(f"KG_Conditions.png")


plot_combined()
