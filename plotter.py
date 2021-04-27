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
