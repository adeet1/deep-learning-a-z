# Part 1 - identify the frauds with the self-organizing map

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# Part 2 - going from unsupervised to supervised deep learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable from the SOM results
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    # For each customer, check if his/her customer ID is in the list of frauds
    # If it is, then set is_fraud = 1
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling (mandatory for most deep learning models)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

import keras # uses the TensorFlow backend (i.e. uses TensorFlow to create the neural network)
from keras.models import Sequential # we use this to initialize the neural network
from keras.layers import Dense # we use this to create the layers in the neural network

from keras.layers import Dropout

classifier = Sequential()

# Add the first hidden layer
classifier.add(Dense(output_dim = 2, init = "uniform", activation = "relu", input_dim = 15))

# Add the output layer
classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fit the ANN to the training set
classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 2)

# Predict the probabilities of frauds
y_pred = classifier.predict(customers)

# Link the customer ID with the predicted probability
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# Sort the probabilities in descending order
y_pred = y_pred[y_pred[:, 1].argsort()]