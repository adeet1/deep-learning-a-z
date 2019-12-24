# ==========================================================================
# Part 1 - Data Preprocessing
# ==========================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data (country and gender)
# We need 2 instances of LabelEncoder(), one for each categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # country
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # remove one dummy variable to prevent multicollinearity

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (mandatory for most deep learning models)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==========================================================================
# Part 2 - Making the ANN
# ==========================================================================
import keras # uses the TensorFlow backend (i.e. uses TensorFlow to create the neural network)
from keras.models import Sequential # we use this to initialize the neural network
from keras.layers import Dense # we use this to create the layers in the neural network

# In order to initialize the neural network, we define it as a sequence of
# layers. As an alternative approach, we could also define it as a graph.
classifier = Sequential()

# We need to add the input layer and first hidden layer. The first step in
# building the neural network is to randomly initialize all weights to small
# numbers. The Dense module will allow us to do both of these tasks.

# output_dim : Number of nodes we want to add in our hidden layer. The add()
#              function doesn't really add the input layer and the first hidden
#              layer. What it really does is that it adds just the first hidden
#              layer, and by adding the hidden layer, we're specifying the # of
#              inputs in the previous layer, which is the input layer.
#
#              Generally, this parameter value is the average of the # of nodes
#              in the input layer and the # of nodes in the output layer (in
#              this case, the average of 11 and 1 is 6). In order to fine-tune
#              this value, you would need to use a technique called PARAMETER
#              TUNING (which involves using a validation set and doing model
#              selection and ensembles).
#
# init : Corresponds to the first step of building the neural network
#        (initializing all weights to small numbers close to 0, but not equal to
#        0). Specifying "uniform" will initialize all weights according to a
#        uniform distribution.
#
# activation : The activation function we want to choose in our hidden layer.
#
# input_dim : The # of independent variables in our dataset, i.e. the # of nodes
#             in our input layer. This parameter is required when adding the
#             first hidden layer, because at this stage, the neural network is
#             only initialized (we haven't created any layers yet). Otherwise,
#             keras will not know which nodes the hidden layer we want to add is
#             expecting as inputs. But when adding additional hidden layers,
#             this parameter is NOT required because since the first hidden
#             layer will already have been created, the next hidden layer will
#             know what to expect.

# We'll choose the rectifier activation function for the hidden layers, since
# research and experiments have shown it to be the best one.
#
# The sigmoid function is great for our output layer (it will give us the
# probability that the customer leaves the bank). We can see what customers have
# the highest probability of leaving the bank, and we can rank the customers
# based on this.
#
# Side Note: The softmax function is really just the sigmoid function applied to
# a dependent variable with more than 2 categories.

# Add the first hidden layer
classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu", input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))

# Add the output layer
classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))

# optimizer : The algorithm you want to use to find the optimal set of weights
#             in the neural network (at this stage, the weights are still only
#             initialized, so we need to apply an algorithm to find the best
#             weights that will make the neural network the most powerful. Adam
#             is one very efficient stochastic gradient descent algorithm.
#
# loss : The loss function we use in the stochastic gradient descent algorithm.
#        The algorithm will optimize the weights based on this loss function.
#        For stochastic gradient descent, the loss function is not the sum of
#        squared errors like for linear regression, but is instead a
#        logarithmic function, called the LOGARITHMIC LOSS. The logarithmic loss
#        function has two names depending on the # of categories in the
#        dependent variable. If the dependent variable has a binary outcome,
#        the function is called binary_crossentropy, and if it has more than 2
#        outcomes, the function is called categorical_crossentropy.
#
# metrics : The criterion you choose to evaluate your model. The algorithm will
#           use these metrics to improve the model's performance. As the network
#           is trained, we will see these metrics gradually increase until they
#           reach a maximum value.

# Compile the neural network (compile means to apply stochastic gradient descent
# to the entire neural network)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# batch_size : The number of observations after which we want to update the
#              weights. There is no rule of thumb for this; rather we need to
#              experiment to find an optimal value.
# nb_epoch : The number of epochs. An EPOCH is simply a round during which the
#            entire training set is passed through the ANN.

# Fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
print("")

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # if probability > 0.5, predict 1 (0 otherwise)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
