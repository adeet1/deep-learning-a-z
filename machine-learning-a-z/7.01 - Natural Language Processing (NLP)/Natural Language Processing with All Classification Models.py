# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_table('Restaurant_Reviews.tsv')

# Cleaning the texts
# This replaces non-alphabetical characters with spaces
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

##########################################################################################

# Fitting the model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy1 = (cm1[1][1] + cm1[0][0]) / (cm1[1][1] + cm1[0][0] + cm1[0][1] + cm1[1][0])
precision1 = cm1[1][1] / (cm1[1][1] + cm1[0][1])
recall1 = cm1[1][1] / (cm1[1][1] + cm1[1][0])
f1score1 = 2 * precision1 * recall1 / (precision1 + recall1)

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy2 = (cm2[1][1] + cm2[0][0]) / (cm2[1][1] + cm2[0][0] + cm2[0][1] + cm2[1][0])
precision2 = cm2[1][1] / (cm2[1][1] + cm2[0][1])
recall2 = cm2[1][1] / (cm2[1][1] + cm2[1][0])
f1score2 = 2 * precision2 * recall2 / (precision2 + recall2)

# Fitting the classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy3 = (cm3[1][1] + cm3[0][0]) / (cm3[1][1] + cm3[0][0] + cm3[0][1] + cm3[1][0])
precision3 = cm3[1][1] / (cm3[1][1] + cm3[0][1])
recall3 = cm3[1][1] / (cm3[1][1] + cm3[1][0])
f1score3 = 2 * precision3 * recall3 / (precision3 + recall3)

# Fitting the classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy4 = (cm4[1][1] + cm4[0][0]) / (cm4[1][1] + cm4[0][0] + cm4[0][1] + cm4[1][0])
precision4 = cm4[1][1] / (cm4[1][1] + cm4[0][1])
recall4 = cm4[1][1] / (cm4[1][1] + cm4[1][0])
f1score4 = 2 * precision4 * recall4 / (precision4 + recall4)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy5 = (cm5[1][1] + cm5[0][0]) / (cm5[1][1] + cm5[0][0] + cm5[0][1] + cm5[1][0])
precision5 = cm5[1][1] / (cm5[1][1] + cm5[0][1])
recall5 = cm5[1][1] / (cm5[1][1] + cm5[1][0])
f1score5 = 2 * precision5 * recall5 / (precision5 + recall5)

# Fitting the classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy6 = (cm6[1][1] + cm6[0][0]) / (cm6[1][1] + cm6[0][0] + cm6[0][1] + cm6[1][0])
precision6 = cm6[1][1] / (cm6[1][1] + cm6[0][1])
recall6 = cm6[1][1] / (cm6[1][1] + cm6[1][0])
f1score6 = 2 * precision6 * recall6 / (precision6 + recall6)

# Fitting the classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test, y_pred)

# Evaluating the model's performance
accuracy7 = (cm7[1][1] + cm7[0][0]) / (cm7[1][1] + cm7[0][0] + cm7[0][1] + cm7[1][0])
precision7 = cm7[1][1] / (cm7[1][1] + cm7[0][1])
recall7 = cm7[1][1] / (cm7[1][1] + cm7[1][0])
f1score7 = 2 * precision7 * recall7 / (precision7 + recall7)

##########################################################################################
##########################################################################################
##########################################################################################
## COMPARISON OF MODELS
##########################################################################################
##########################################################################################
##########################################################################################

df = [['Logistic Regression', accuracy1, precision1, recall1, f1score1],
      ['K-Nearest Neighbor', accuracy2, precision2, recall2, f1score2],
      ['Support Vector Machine (SVM)', accuracy3, precision3, recall3, f1score3],
      ['Kernel SVM', accuracy4, precision4, recall4, f1score4],
      ['Naive Bayes', accuracy5, precision5, recall5, f1score5],
      ['Decision Tree', accuracy6, precision6, recall6, f1score6],
      ['Random Forest', accuracy7, precision7, recall7, f1score7]]

comparison = pd.DataFrame(df, columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])