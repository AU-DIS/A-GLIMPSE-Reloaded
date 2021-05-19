import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


def plot_combined(output_path, filenames, xlabel=""):
    markers = ['-', '--', '-.', ':']
    ax = None
    i = 0

    dfs = [pd.read_csv(filename, skiprows=[0])
           for filename in filenames]
    bandit = dfs[0][" bandit_accuracy"]
    for df in dfs[1:]:
        bandit += df[" bandit_accuracy"]

    bandit = [x/len(dfs) for x in bandit]

    glimpse = dfs[0][" glimpse_accuracy"]
    for df in dfs[1:]:
        glimpse += df[" glimpse_accuracy"]
    glimpse = [x/len(dfs) for x in glimpse]

    random = dfs[0][" random_accuracy"]
    for df in dfs[1:]:
        random += df[" random_accuracy"]
    random = [x/len(dfs) for x in random]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.grid(linestyle='--', linewidth=0.5)

    ax.plot(range(len(bandit)), bandit, alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.plot(range(len(glimpse)), glimpse, alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.plot(range(len(random)), random, alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.set_ylim([0, 1])
    ax.legend(["Bandit accuracy", "Glimpse accuracy",
              "Uniform random sample accuracy"])
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    print(f"Saving {output_path}.png")
    plt.savefig(f"{output_path}.png")
