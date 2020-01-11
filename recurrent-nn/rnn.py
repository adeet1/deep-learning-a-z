import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the training set
train = pd.read_csv("Google_Stock_Price_Train.csv").loc[:, "Open"].values
train = train.reshape(-1, 1) # convert from dim (1258,) to (1258, 1)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # normalization
train_sc = sc.fit_transform(train) # normalized training set
