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

# Convert the data into a list of lists, with users in rows and movies in
# columns
def convert(data):
    new_data = []
    
    # Create a list for each user
    for id_users in range(1, nb_users + 1):
        # For each user, we want to obtain all the ratings they gave
        id_movies = data[:, 1][data[:, 0] == id_users] # all of the movies that were rated by this user
        id_ratings = data[:, 2][data[:, 0] == id_users] # all of the movies that were rated by this user
        
        # Initialize ratings vector
        ratings = np.zeros(nb_movies)
        
        # Fill ratings vector with the appropriate ratings (we want to access
        # the index values of ratings that exist, i.e. id_movies)
        #
        # Subtract 1 because the first rating in the vector (the rating of
        # movie ID 1) is the first element of the vector, which is at index 0
        ratings[id_movies - 1] = id_ratings
        
        # Add the list of ratings to new_data
        new_data.append(list(ratings))
    
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert the data into Torch tensors
#
# Tensors are simply arrays that contain elements of a single data type
#
# A tensor is a multi-dimensional matrix, but instead of being a numpy array,
# it's a pytorch array.
#
# Building a neural network with numpy arrays is less efficient
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings into binary ratings 1 (liked) and 0 (disliked)

# Replace all ratings of unrated movies with -1 (a sentinel value)
training_set[training_set == 0] = -1
test_set[test_set == 0] = -1

# 1 or 2 star rating --> user didn't like the movie
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0

# 3, 4, or 5 star rating --> user liked the movie
training_set[training_set >= 3] = 1
test_set[test_set >= 3] = 1

# Create the architecture of the neural network (the RBM)
# The RBM is a probabilistic graphical model
class RBM():
    # nv = number of visible nodes
    # nh = number of hidden nodes
    def __init__(self, nv, nh):
        # Initialize weights and biases, which will be optimized during training
        
        # Initialize all weights in a torch tensor (the weights are the
        # probabilities of the visible nodes, given the hidden nodes). All
        # weights need to be initialized randomly according to a normal
        # distribution.
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh) # bias for the probabilities of the hidden nodes given the visible nodes
        self.b = torch.randn(1, nv) # bias for the probabilities
