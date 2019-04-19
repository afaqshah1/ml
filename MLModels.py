# This code has been written by Dr Syed Afaq Ali Shah to demonstrate the performance of different machine learning techniques
# Syed is a computer scientist by profession and founder of Intelligaroo (www.intelligaroo.com)

# Prerequisites - Python and scikit learn installation required
# The code is well commented.
# Models tested include: Linear regression, Ridge regression, Naive Bayes Classifier, Decision Tree, Bagging,
# Adaboost, Random Forest and Artificial Neural Network

# Training and testing has been done on digits datasets which is available in scikit learn library.

from sklearn import linear_model
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
import itertools
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np


iris = datasets.load_digits() # Digits Dataset
dt = iris.data
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state = 0) 

#Linear Regression
lr = linear_model.LinearRegression()
#Train the model using the training sets
lr.fit (X_train, y_train)
#Predict the response for test dataset
lr.predict(X_test)
print("Linear Regression:", round(lr.score(X_test, y_test),3))


# Ridge Regression
clf = Ridge(alpha=3)
clf.fit (X_train, y_train)
clf.predict(X_test)
print("Ridge Regression:",round(clf.score(X_test, y_test),3))


# Naive Bayes
#Create a Gaussian Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Naive Bayes:",round(metrics.accuracy_score(y_test, y_pred),3))

# Decision Tree with Gini
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
print("Decision Tree:",round(clf_gini.score(X_test, y_test),3))

# Bagging Example
tree = DecisionTreeClassifier(criterion= 'entropy')
bag = BaggingClassifier(base_estimator= tree, n_estimators= 50)
bag.fit(X_train, y_train) 
y_pred = bag.predict(X_test)
print("Bagging:",round(metrics.accuracy_score(y_test, y_pred),3))

#Adaboost
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, y_train)
y_pred  = bdt.predict((X_test))
print("Adaboost:",round(metrics.accuracy_score(y_test, y_pred),3))

#Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest:",round(metrics.accuracy_score(y_test, y_pred),3))


#Artificial Neural Networks
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='logistic', max_iter = 1000)

# Train the classifier with the traning data
mlp.fit(X_train_scaled,y_train)

y_pred = mlp.predict(X_test_scaled)
print("ANN:",round(metrics.accuracy_score(y_test, y_pred),3))

