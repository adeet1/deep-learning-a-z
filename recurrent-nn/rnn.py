import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================================
# Part 1 - Data Preprocessing
# ==========================================================================

# Import the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # normalization
training_set_scaled = sc.fit_transform(training_set) # normalized training set

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
for i in range(60, len(training_set)):
    # We start the loop at i = 60 because we need the previous 60 days' stock
    # prices
    X_train.append(training_set_scaled[i - 60:i, 0])
    Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape the data (add more dimensionality to the data structure created above)
X_train = np.reshape(X_train,
                     (X_train.shape[0], # of observations
                      X_train.shape[1], # number of time steps
                      1) # of indicators (predictors)
                     )

# ==========================================================================
# Part 2 - Build the RNN (LSTM)
# ==========================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize RNN
regressor = Sequential()

# Add the first LSTM layer
# We need a high dimensionality to be able to pick up on sophisticated stock
# price trends
regressor.add(LSTM(units = 50, # 50 neurons
                   return_sequences = True, # needs to be True if adding multiple LSTM layers
                   input_shape = (X_train.shape[1], 1)
                   )
             )

# Add dropout regularization (to prevent overfitting)
regressor.add(Dropout(rate = 0.2)) # ignore 20% of neurons of the LSTM layer during training

# Add a second LSTM layer and dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add a third LSTM layer and dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add a fourth LSTM layer and dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))

# Add the output layer
regressor.add(Dense(units = 1))

# Compile the RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")
