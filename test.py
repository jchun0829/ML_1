import pandas as pd

df = pd.read_csv('used_cars.csv')

print("data head\n")
print(df.head())

print("\ndata info\n")
print(df.info())

print("\ndata summary\n")
print(df.describe())

print("\nerro num:")
print(df.isnull().sum())