import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_accuracy(input_path, output_path):
    df = pd.read_csv(input_path, skiprows=[0])
    print(df)
    df.plot.line(x="round", y="accuracy", ylim=(0,1))
    plt.show()

def plot_regret(input_path, output_path):
    df = pd.read_csv(input_path)
    df["cumsum"] = df["regret"].cumsum()
    df.plot.line(y="cumsum")
    plt.show()