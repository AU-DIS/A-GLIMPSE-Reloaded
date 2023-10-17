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
    print(f"Saving {output_path}.png based on {len(filenames)} files")
    plt.savefig(f"{output_path}.png")
    plot_unique_over_time(f"{output_path}_unique", filenames)


def plot_unique_over_time(output_path, filenames):
    markers = ['-', '--', '-.', ':']
    ax = None
    i = 0
    dfs = [pd.read_csv(filename, skiprows=[0])
           for filename in filenames]
    uniques = dfs[0][" bandit_no_unique_entities"]
    for df in dfs[1:]:
        uniques += df[" bandit_no_unique_entities"]

    uniques = [x/len(dfs) for x in uniques]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.grid(linestyle='--', linewidth=0.5)

    ax.plot(range(len(uniques)), uniques, alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.set_ylabel("Number of unique entities")
    ax.set_xlabel("Rounds")
    plt.tight_layout()
    print(f"Saving {output_path}.png based on {len(filenames)} files")
    plt.savefig(f"{output_path}.png")


def plot_combined_regret(output_path, filenames, xlabel=""):
    markers = ['-', '--', '-.', ':']
    ax = None
    i = 0
    dfs = [pd.read_csv(filename, skiprows=[0])
           for filename in filenames]
    regret = dfs[0]["regret"]
    for df in dfs[1:]:
        regret += df["regret"]

    regret = np.cumsum(regret)
    #regret = regret/len(regret)

    xrange = pd.Series(range(len(regret)))
    yrange = pd.Series(regret)
    correlation = xrange.corr(yrange)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.grid(linestyle='--', linewidth=0.5)

    ax.plot(range(len(regret)), regret, alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    #ax.set_ylim([0, 1])
    ax.legend(["Regret"])
    ax.set_xlabel(f"{xlabel}\nPearson correlation coefficient {correlation}")
    ax.set_ylabel("Regret")
    plt.tight_layout()
    print(f"Saving {output_path}.png based on {len(filenames)} files")
    plt.savefig(f"{output_path}.png")


def plot_bandit_weights(bandit, top_k=100, output="bandit"):
    markers = ['-', '--', '-.', ':']
    ax = None
    i = 0

    colours = []
    for i, _ in enumerate(bandit.weights):
        if i in bandit.debug_been_chosen:
            colours.append('red')
        else:
            colours.append('blue')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.grid(linestyle='--', linewidth=0.5)
    ax.scatter(range(len(bandit.weights)), bandit.weights, c=colours)
    #ax.set_ylim([0, 1])
    ax.set_xlabel("Choice")
    ax.set_ylabel("Weight")
    plt.tight_layout()
    print(f"Saving {output}.png")
    plt.savefig(f"{output}.png")
