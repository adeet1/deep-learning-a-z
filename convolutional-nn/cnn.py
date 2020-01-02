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

# Step 1 - Convolution
# nb_filter : The number of feature detectors to apply on the input image. This
#             is equal to the number of feature maps we want to create. The
#             default value is 64, but most CNN architectures typically start
#             with 32.
#
# nb_row : The number of rows in the feature detector. The Keras documentation
#          uses the term "convolution kernel," which is another term for feature
#          detector or filter. The default value is 3.
#
# nb_col : The number of columns in the feature detector. The default value is
#          3.
#
# border_mode : Specifies how the feature detectors will handle the borders of
#               the input image. The default value is "same".
#
# input_shape : The shape of the input image, on which we want to apply the
#               feature detectors through the convolution operation. This is
#               important because all of our images don't have the same size,
#               and so we need to "force" the images to all have the same size.
#               We do this step in the image preprocessing part, just before
#               fitting the CNN to our images.
#
#               The default value is (3, 256, 256). In this tuple, the first
#               value is the number of channels in the image. This is 3 for
#               color images (1 for red, 1 for green, 1 for blue), and 1 for
#               black-and-white images. The last two values are the dimensions,
#               in pixels, of the image.
#
#               IMPORTANT: The order of the values is critical. This parameter
#               is formatted (rows, cols, channels) in the Tensorflow backend,
#               and (channels, rows, cols) in the Theano backend. We need to use
#               the former format because we're using the Tensorflow backend.
#
# activation : The activation function to use. We'll use the rectifier function
#              to eliminate any negative pixels (we need non-linearity in image
#              classification).
classifier.add(Convolution2D(nb_filter = 32, nb_row = 3, nb_col = 3, input_shape = (64, 64, 3), activation = "relu"))