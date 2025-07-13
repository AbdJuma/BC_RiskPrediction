import matplotlib.pyplot as plt
import pandas as pd

def plot_execution_time(df):
    plt.figure(figsize=(6,4))
    for i,row in df.iterrows():
        plt.scatter(row['Model'], row['Time'], s=100)
    plt.plot(df['Model'], df['Time'], linestyle='--', alpha=0.7)
    plt.title("Execution time per model")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sensitivity_pie(df):
    sizes = df['Recall'] * 1684  # recall fraction × total positive
    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=df['Model'], autopct='%1.1f%%', startangle=140)
    plt.title("Sensitivity of truly detected to the entirely infected")
    plt.tight_layout()
    plt.show()

def plot_precision_bar(df):
    plt.figure(figsize=(6,4))
    plt.bar(df['Model'], df['Precision'], color='C2')
    plt.title("Precision of truly detected with respect to all predicted as positive")
    plt.ylabel("Precision")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()
