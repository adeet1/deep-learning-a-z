# DATA PREPROCESSING ##########################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the data set into a test set and training set
from sklearn.cross_validation import train_test_split
#            model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

###############################################################################

# Fit Simple Linear Regression to the train set
from sklearn.linear_model import LinearRegression
y_train = y_train.reshape(-1,1)
X_train = X_train.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set results
X_test = X_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_pred = regressor.predict(X_test)

# Plot the train set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

# Plot the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

"""
Note: In line 43, we plot X_train and not X_test. This is because the model
learns correlations from the training set, applies its knowledge to the test
set, and compares the results.

The machine in this case is the simple linear regression model. The learning
means that we trained the model to find correlations in preexisting data (the
training set), and apply this knowledge to new observations (the test set).
"""