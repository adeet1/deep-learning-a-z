# Polynomial Regression

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
1 would work for X instead of 1:2, but then X would be an array instead of a
matrix. It's always best if X is a matrix and y is a vector in order to avoid
warnings or errors when we run the code.
"""

"""
Unlike in previous exercises, we don't split the data into a training and test
set. This is because it's impractical, as we have only 10 observations.
"""

# Fit a linear regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X.reshape(-1, 1)
y.reshape(-1, 1)
lin_reg.fit(X, y)

# Fit a polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualize the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()

# Visualize the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()

# Predict a new result with linear regression
print(lin_reg.predict(6.5))

# Predict a new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))
