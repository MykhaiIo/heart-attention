import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data\healthcare-dataset-stroke-data.csv')

train, valid = train_test_split(data, train_size=.8, stratify=data.iloc[:,-1])

train.to_csv('data\\train.csv')
valid.to_csv('data\\val.csv')
del train, valid