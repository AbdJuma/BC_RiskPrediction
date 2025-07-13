from train import run_training
from evaluate import plot_execution_time, plot_sensitivity_pie, plot_precision_bar
import pandas as pd

if __name__ == "__main__":
    results = run_training('sample.csv')
    df = pd.DataFrame(results).sort_values('F1', ascending=False)
    print(df)
    plot_execution_time(df)
    plot_sensitivity_pie(df)
    plot_precision_bar(df)
