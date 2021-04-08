import csv
import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy(input_path, output_path):
    df = pd.read_csv(input_path, skiprows=[0])
    print(df)
    df.plot.line(x="round", y="accuracy", ylim=(0,1))
    plt.show()