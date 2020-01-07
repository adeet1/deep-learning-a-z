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