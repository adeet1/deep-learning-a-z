import numpy as np
import pandas as pd
import torch
import torch.nn as nn # neural networks
import torch.nn.parallel as parallel # for parallel computations
import torch.optim as optim # optimizer
import torch.utils.data
from torch.autograd import Variable # for stochastic gradient descent

# Import movies dataset
#
# sep : The separator (delimeter). Note that we cannot use a comma (',')
#       because columns won't be comma-separated (e.g. movie titles may contain
#       commas). In the movies.dat file, the movies are separated by their
#       ratings and their other features by a double colon.
#
# header : The names of columns.
#
# engine : Ensures that the dataset gets imported correctly.
#
# encoding : We need to put in a different encoding than usual, because some
#            movie titles contain special characters that can't be treated
#            properly with the classic encoding UTF-8.
movies = pd.read_csv("ml-1m/movies.dat", sep = "::", header = None,
                     engine = "python", encoding = "latin-1")

# Import users dataset
users = pd.read_csv("ml-1m/users.dat", sep = "::", header = None,
                    engine = "python", encoding = "latin-1")

# Import ratings dataset
ratings = pd.read_csv("ml-1m/ratings.dat", sep = "::", header = None,
                      engine = "python", encoding = "latin-1")

# Prepare the training and test sets
#
# Multiple sets of train-test splits have been provided in order to facilitate
# k-fold cross validation

training_set = pd.read_csv("ml-100k/u1.base", delimiter = "\t", header = None)
training_set = np.array(training_set, dtype = "int") # convert the dataframe to an array of integers

test_set = pd.read_csv("ml-100k/u1.test", delimiter = "\t", header = None)
test_set = np.array(test_set, dtype = "int") # convert the dataframe to an array of integers

# Get the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # the maximum user ID
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # the maximum movie ID
