# ==========================================================================
# Part 1 - Building the CNN
# ==========================================================================

# Importing the Keras libraries
from keras.models import Sequential # we use this to initialize the neural network
from keras.layers import Convolution2D # step 1 (the convolutional step)
from keras.layers import MaxPooling2D # step 2 (the pooling step)
from keras.layers import Flatten # convert the pooled feature maps into a large feature vector
from keras.layers import Dense # add the fully connected layers in a classic ANN

# Data preprocessing is not necessary.
#
# There are no categorical variables, so no encoding is necessary. The data
# has already been split into a training and testing set, based on the directory
# structure of the image files.
#
# We will take care of feature scaling later.

# Initializing the CNN
classifier = Sequential()