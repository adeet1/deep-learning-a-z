import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Prepare the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Get the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert the data into an array with users in rows and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Convert the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Create the architecture of the neural network
# The stacked autoencoder class will be a child class of Module (inheritance)
class SAE(nn.Module): # inherit from nn.Module
    def __init__(self, ):
        super(SAE, self).__init__()
        
        # parameter 1: number of features --> nb_movies
        # parameter 2: number of nodes in the first hidden layer --> 20
        self.fc1 = nn.Linear(nb_movies, 20) # full connection 1
        
        # Connection between the 1st and 2nd hidden layers
        # 20 nodes in the 1st hidden layer, 10 nodes in the 2nd hidden layer
        self.fc2 = nn.Linear(20, 10) # full connection 2
        
        # Here, we start to decode (reconstruct the original input vector)
        self.fc3 = nn.Linear(10, 20) # full connection 3
        
        # Output layer
        self.fc4 = nn.Linear(20, nb_movies) # full connection 4
        
        # Define an activation function
        self.activation = nn.Sigmoid()
        
    # Returns the output of the network
    def forward(self, x): # forward propagation, where encoding and decoding takes place
        # To do encoding, we apply the activation function on the first full connection
        x = self.activation(self.fc1(x)) # returns the encoded vector
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) # decode the 10-element vector into a 20-element vector
        x = self.fc4(x) # when reconstructing the input vector, you don't use the activation function
        return x

sae = SAE()
criterion = nn.MSELoss() # criterion for the loss function
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # lr = learning rate

# Train the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone() # the input and target are now identical
        
        # Optimize memory
        # If the observation contains at least one non-zero rating
        if torch.sum(target.data > 0) > 0: 
            output = sae(input) # get the vector of predicted ratings (the output of the model)
            target.require_grad = False
            output[target == 0] = 0
            
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # add 1e-10 to prevent division by zero
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()
    
    print("epoch:", epoch, " ", "loss:", train_loss/s)
    
# Test the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    
    # Optimize memory
    # If the observation contains at least one non-zero rating
    if torch.sum(target.data > 0) > 0: 
        output = sae(input) # get the vector of predicted ratings (the output of the model)
        target.require_grad = False
        output[target == 0] = 0
        
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # add 1e-10 to prevent division by zero
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
        optimizer.step()
        
    print("test loss:", test_loss/s)