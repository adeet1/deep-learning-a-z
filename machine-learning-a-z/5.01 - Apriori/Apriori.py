"""
IMPORTANT

Apriori.py is the name of this file, which contains the algorithm.
apyori.py is essentially the Python module we will use to write the algorithm.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# This dataset contains 7,500 transactions
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Converting the dataset from a DataFrame to a list of lists
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
#for i in range(0, len(dataset)):
#    transactions.append(dataset.values[i, :])

# Training the Apriori model to the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.0028, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the results
results = list(rules)

print([i[0] for i in results])