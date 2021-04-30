import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


def plot_accuracy(input_path, output_path):
    df = pd.read_csv(input_path, skiprows=[0])
    df.plot(x='round', y="accuracy", color="Red")
    plt.savefig(output_path)


def plot_regret(input_path, output_path):
    df = pd.read_csv(input_path, skiprows=[0])

    if 'k' not in df.columns:
        df['k'] = 1

    df["cumsum"] = df["regret"].cumsum()
    d = np.polyfit(df["k"]+df["round"]*df["k"].max(), df["cumsum"], 1)
    f = np.poly1d(d)
    df.insert(4, "trend", f(df["k"]+df["round"]*df["k"].max()))
    ax = df.plot.line(y="cumsum", marker='o')
    df.plot(y="trend", color="Red", ax=ax)
    ax.legend(["Cumulative regret", "Trendline"])
    plt.savefig(output_path)


def plot_theoretical_regret(input_path, output_path):
    df = pd.read_csv(input_path, skiprows=[0])

    if 'k' not in df.columns:
        df['k'] = 1

    df["cumsum"] = df["regret"]
    # df["cumsum"] = (df["cumsum"]-df.min()["cumsum"]) / \
    #    (df.max()["cumsum"]-df.min()["cumsum"])
    d = np.polyfit(df["k"]+df["round"]*df["k"].max(), df["cumsum"], 1)
    f = np.poly1d(d)
    df.insert(4, "trend", f(df["k"]+df["round"]*df["k"].max()))
    ax = df.plot.line(y="cumsum", marker='o')
    df.plot(y="trend", color="Red", ax=ax)
    ax.legend(["Cumulative regret", "Trendline"])
    plt.savefig(output_path)


def plot_all__experiment_regrets():
    for d in os.walk("bandit_results"):
        for f in os.listdir(d[0]):
            if "regret" in f and '.DS_Store' not in f and '.png' not in f and not os.path.exists(f"{d[0]}/{f.strip('.csv')}.png"):
                p = Process(target=plot_regret, args=(
                    f"{d[0]}/{f}", f"{d[0]}/{f.strip('.csv')}.png",))
                p.start()


def plot_all_accuracy():
    for d in os.walk("bandit_results"):
        for f in os.listdir(d[0]):
            if "accuracy" in f and 'Store' not in f and '.png' not in f and not os.path.exists(f"{d[0]}/{f.strip('.csv')}.png"):
                p = Process(target=plot_accuracy, args=(
                    f"{d[0]}/{f}",  f"{d[0]}/{f.strip('.csv')}.png",))
                p.start()


def plot_all_theoretical_regrets():
    for d in os.walk("theoretical"):
        for f in os.listdir(d[0]):
            if 'csv' in f and '.DS_Store' not in f and '.png' not in f and not os.path.exists(f"{d[0]}/{f.strip('.csv')}.png"):
                p = Process(target=plot_theoretical_regret, args=(
                    f"{d[0]}/{f}", f"{d[0]}/{f.strip('.csv')}.png",))
                p.start()


# Filthy code to be cleaned up latur
def plot_combined_theoretical(substring):
    sizes = [(1 * 1.1)**i for i in range(0, 30)][::-1]
    # markers = ['.', ',', 'o', 'v', '<', '>']
    markers = ['-', '--', '-.', ':']
    ax = None
    labels = []
    i = 0
    ys = []

    for d in os.walk("theoretical/exp3_kg_conditions"):
        for fi in os.listdir(d[0]):
            if "csv" in fi and '.DS_Store' not in fi and '.png' not in fi and substring in fi:
                try:
                    df = pd.read_csv(f"{d[0]}/{fi}", skiprows=[0])
                except:
                    print(f"{d[0]}/{fi}")
                if 'k' not in df.columns:
                    df['k'] = 1

                df[f"cumsum"] = df["regret"]
                labels.append(
                    f"{fi.replace('r', 'Rounds:').replace('-ent', 'Entities')}")
                ys.append(df["cumsum"])

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    order = np.argsort([len(x) for x in ys])[::-1]
    labels = np.array(labels)[order]

    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(ys))]
    ax.set_prop_cycle('color', colors)
    for y in np.array(ys)[order]:
        s = sizes[i]
        i += 1
        ax.plot(y, alpha=0.5,
                linestyle=markers[i % len(markers)], markersize=5)

    ax.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"theoretical/exp3_kg_conditions/{substring}.png")


def plot_all_combined_theoretical():
    fixed_ratio_combinations = combinations = ["r1000000-ent100-",
                                               "r1000000-ent1000-", "r1000000-ent10000-", ]

    proportion_combinations = ["p0.001", "p0.01",
                               "p0.1", "p0.2", "p0.3", "p0.4", "p0.5"]

    kg_scale_combinations = ["r1000"]

    for combination in fixed_ratio_combinations:
        p = Process(target=plot_combined_theoretical, args=(combination,))
        p.start()


# theoretical/exp3_kg/r100000-ent100-gs110p0.1.csv
# 100002
plot_all_combined_theoretical()

# LAV EN TRUE BINÆR
# Med at least one is in
# Lav fixed seed (?)
# Gem random ting til grafen?
