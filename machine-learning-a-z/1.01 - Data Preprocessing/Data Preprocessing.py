# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replaces the missing data with actual data
print(X)

# Encode categorical data

"""
The problem with regular encoding is that it assigns relational orders to the
countries (France = 0, Germany = 1, Spain = 2). However, in this situation, it
doesn't make sense for Spain to be greater than Germany or for Germany to be
greater than France.

It would make sense in other situations involving
categorical variables, such as shirt size (small, medium, large), where large
is greater than medium and medium is greater than small. But it doesn't make
sense for countries.

To deal with this problem, we use dummy variables. In our data set, the number
of columns we create is equal to the number of countries (in this case 3). The
values in the data set would be 1 if the country is XXXX and 0 if the country
is not XXXX.
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data set into a test set and training set
"""
This is necessary to make sure that the machine learning models don't rely
too much on correlations in one scenario. It needs to adapt to other scenarios
as well.

The 0.2 means that 20% of the observations will be in the test set and 80% in
the training set.
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
The machine learning modes will learn the correlation between the independent
and dependent variables in the train set, and apply its knowledge to the test set
by predicting whether the customers in the test set will buy the product.

But we have to make sure the model doesn't learn the train set by heart. Otherwise
it will end up memorizing the set and not really understand it, and thus it won't
be able to make good predictions for the test set. This phenomenon is called
OVERFITTING.
"""

# Feature scaling
"""
Feature scaling is necessary because in our dataset, the age and salary are not
at the same scale. Salary has a much wider range of values than age. Machine
learning models are based on the concept of Euclidean distance, so the salary
will dominate the age to the extent that the age practically won't exist.

Thus, we need to standardize the age and salary variables.
"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
