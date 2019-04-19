# k-folds run demonstration

# This code is written by Dr Syed Afaq Ali Shah to demonstrate the performance of different machine learning techniques
# Syed is a computer scientist by profession and founder of Intelligaroo (www.intelligaroo.com)

# Prerequisites - Python and scikit learn installation required

# Training and testing has been done on digits datasets which is available in scikit learn library.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits() # Digits Dataset
dt = digits.data
X = digits['data']
y = digits['target']

results = []

folds = range(1,11)

for i in folds:
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state = 42) 
    # Decision Tree
    clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth=5)
    clf_gini.fit(X_train, y_train)

    y_pred = clf_gini.predict(X_test)
    accuracy = clf_gini.score(X_test, y_test)

    results.append(accuracy)

    print("Fold {0} accuracy: {1}".format(i, round(accuracy,5)))     
mean_result = np.mean(results)
print("Mean Accuracy: {0}".format(round(mean_result,5))) 
