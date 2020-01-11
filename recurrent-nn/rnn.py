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

# Create a data structure with 60 timesteps and 1 output
#
# 60 timesteps --> at each time T, the RNN is going to look at the 60 stock
# prices before time T, and based on the trends it captures during these steps,
# it will try to predict the next output
#
# X_train contains the previous 60 days' stock prices, and Y_train contains the
# next day's stock price
X_train = []
Y_train = []
for i in range(60, len(train)):
    # We start the loop at i = 60 because we need the previous 60 days' stock
    # prices
    X_train.append(train_sc[i - 60:i].reshape(-1,))
    Y_train.append(train_sc[i, 0].reshape(-1,))

X_train, Y_train = np.array(X_train), np.array(Y_train)
