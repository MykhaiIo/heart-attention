import pandas as pd

data = pd.read_csv('data\healthcare-dataset-stroke-data.csv')

train = data.sample(frac=.8, random_state=42)
valid = data.drop(train.index).sample(frac=1, random_state=42)
del train

train.to_csv('data\\train.csv')
valid.to_csv('data\\val.csv')
