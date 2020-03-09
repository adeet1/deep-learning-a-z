import numpy as np
import matplotlib.pyplot as plt

# Create dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
Y = np.array([-3, 1, 0, 4, 7, 9, 10, 9, 13, 16, 17, 19, 20, 21, 21, 19, 24, 26, 27, 29])

class LinearRegression():
    def __init__(self):
        self.w = [0, 0] # weights: y = mx + b --> [m, b]
    
    # Trains the model on the training set
    def train(self, X_train, Y_train):
        alpha = 0.001 # learning rate
        n = len(X_train)
        num_iter = 1000
        
        for i in range(num_iter):
            Y_actual = Y_train
            Y_pred = self.w[0] * X_train + self.w[1]        # y = mx + b
            
            # Gradient descent algorithm
            Dm = -2/n * sum(X_train * (Y_actual - Y_pred))  # dy/dm
            Db = -2/n * sum(Y_actual - Y_pred)              # dy/db
            
            self.w[0] -= alpha * Dm
            self.w[1] -= alpha * Db
    
    # Returns the weights
    def eq(self):
        return self.w
        
    # Predicts values for the specified test set
    def predict(self, X_test):
        Y_pred = self.w[0] * X_test + self.w[1]
        return Y_pred

# Train/test split
ind = int(0.8 * len(X))
X_train = X[:ind]
X_test = X[ind:]
Y_train = Y[:ind]
Y_test = Y[ind:]

# Initialize and train model
lin = LinearRegression()
lin.train(X_train, Y_train)

# Print best-fit equation
print("y =", lin.eq()[0], "* x +", lin.eq()[1])

# Make predictions on the training and test sets
Y_train_pred = lin.predict(X_train)
Y_test_pred = lin.predict(X_test)

# Plot the training set results
plt.figure()
plt.scatter(X_train, Y_train)
plt.plot(X_train, Y_train_pred)

# Plot the test set results
plt.figure()
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_test_pred)
plt.show()