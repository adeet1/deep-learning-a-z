## Self-Organizing Map

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset taken from the UCI Machine Learning Repository
# Statlog (Australian Credit Approval) Data Set
dataset = pd.read_csv("Credit_Card_Applications.csv")

# Split the dataset
#
# Note that we're not doing this in order to create a supervised learning model
# (we're not trying to predict 0 or 1). We're only doing it so we can
# distinguish between the customers whose application was approved and those
# who weren't.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling (normalization, i.e. get all features between 0 and 1)
#
# fit() only stores the normalized values in memory without modifying X
# transform() will actually modify X to be the normalized values.
# fit_transform() does both of these things
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Train the SOM
# A sklearn implementation of a SOM doesn't currently exist, so we need an
# implementation from another developer
from minisom import MiniSom

# The MiniSom object is the self-organizing map itself.
#
# x, y : The dimensions of the map/grid.
# input_len : The # of features in X.
# sigma : The radius of the different neighborhoods in the grid.
# learning_rate : Decides by how much the weights are updated during each
#                 iteration.
# decay_function : This can be used to improve the model's convergence.
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# We need to initialize the weights before training the SOM on X
som.random_weights_init(X)

# Train the SOM on X (not X and y) because we're doing unsupervised learning
# (the dependent variable is not considered).
# This is step 4 of 9
#
# num_iteration : The number of times we want to repeat steps 4 to 9.
som.train_random(data = X, num_iteration = 100)som.train_random(data = X, num_iteration = 100)

# Visualize the results
# We will color the winning nodes in such a way that the larger the MID is, the closer to white the color will be
# We will need to plot the self-organizing map somewhat from scratch (we can't use matplotlib since this is a very specific type of plot)
from pylab import bone, pcolor, colorbar, plot, show

# Initialize the figure (the window that will contain the map)
bone()

# Put the various winning nodes on the map
# We'll do this by putting on the map the information of the MID (Mean Interneuron Distance) for all the winning nodes that the SOM identified
# We will not add the values of all these MIDs, but instead we will use colors (different colors will correspond to different range values of the MIDs).
# The distance_map() method of the SOM object will return all of the MIDs in one matrix.
pcolor(som.distance_map().T)

# We want to add a legend so that we can see what the different colors on the map represent
# The white colors on the map correspond to the fraudulent cases (because this color represents high MID values)
colorbar()