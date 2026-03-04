import pandas as pd

df = pd.read_csv('runs/tarware_mappo_baseline/20260106_161334/metrics.csv')

print(df.head())

print(df['train/deliveries'].describe())
print(df['train/explained_var'].info())