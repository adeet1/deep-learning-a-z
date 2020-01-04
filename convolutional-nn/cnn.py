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

# Step 2 - Pooling
# We apply max pooling on each of the feature maps we created in the previous
# step.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a second convolutional layer
# The input of the second convolutional layer is not the 64 x 64 images, but
# rather the pooled feature maps of the previous layer. So we will apply the
# convolution and max pooling not on the images, but on the pooled feature
# maps. Therefore, the shape of our input will come from the previous layer,
# and so we don't need to specify an input_shape parameter.
classifier.add(Convolution2D(nb_filter = 32, nb_row = 3, nb_col = 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Our input layer will be very large. That's because even if we reduce the size
# of the feature maps, we still have many feature maps.
#
# We don't need to specify any parameters here, because keras will know that the
# previous layer needs to be flattened.
classifier.add(Flatten())

# Step 4 - Full Connection
# output_dim : The number of nodes in the fully connected layer (hidden layer).
#              We have chosen 128, although this is arbitrary. An ideal value is
#              (# of nodes in input layer + # of nodes in output layer) / 2, but
#              it is impractical to count the number of input nodes because it's
#              too large.
#
#              Our goal is to choose a value that is large enough to make the
#              classifier a good model, but not too large (or else the model
#              will be very computationally intensive).

# Add a fully connected layer
classifier.add(Dense(output_dim = 128, activation = "relu"))

# Add an output layer
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# ==========================================================================
# Part 2 - Fitting the CNN to the images
# ==========================================================================
from keras.preprocessing.image import ImageDataGenerator

# This code was taken from the image preprocessing section in the Keras
# documentation (https://keras.io/preprocessing/image/)
train_datagen = ImageDataGenerator(
        rescale=1./255, # rescale pixel values from range 0-255 to range 0-1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# target_size : The size of your images that is expected in the CNN.
#
# batch_size : The size of the batches in which some random samples of our
#              images will be included, and that contains the number of images
#              that will go through the CNN, after which the weight will be
#              updated.
#
# class_mode : Whether your dependent variable is binary or has more than two
#              categories.
train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)