import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Process


def plot_combined(output_path, filename):
    markers = ['-', '--', '-.', ':']
    ax = None
    i = 0
    df = pd.read_csv(filename, skiprows=[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.grid(linestyle='--', linewidth=0.5)

    ax.plot(df["bandit_accuracy"], alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.plot(df["glimpse_accuracy"], alpha=0.5,
            linestyle=markers[i % len(markers)], markersize=5)

    ax.set_ylim([0, 1])
    ax.legend(["Bandit accuracy", "Glimpse accuracy"])

    plt.tight_layout()
    plt.savefig(f"{output_path}.png")
