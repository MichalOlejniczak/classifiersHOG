import numpy as np
import timeit

from sklearn import tree
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positives", required=True, help="Path to positives")
ap.add_argument('-n', '--negatives', required=True, help="Path to negatives")
ap.add_argument('-c', '--saveAs', required=True, help="Save as")
args = vars(ap.parse_args())

positives = np.load(args['positives'])
negatives = np.load(args['negatives'])

ones = np.ones(len(positives))
zeros = np.zeros(len(negatives))

x = np.concatenate((positives, negatives), axis=0)
y = np.concatenate((ones, zeros), axis=0)

x = x.reshape(len(x), -1)

decisionTreeParamGrid = [{'splitter': ['best'], 'max_depth': [10, 50, 100, None], 'min_samples_split': [2, 3, 6]},
                         {'splitter': ['random'], 'max_depth': [10, 50, 100, None], 'min_samples_split': [2, 3, 6]}]

startTimeDecisionTree = timeit.default_timer()

decisionTreeClassifier = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=decisionTreeParamGrid, cv=5, n_jobs=-1)
fitResult = decisionTreeClassifier.fit(x, y)

elapsedDecisionTree = timeit.default_timer() - startTimeDecisionTree

with open(args['saveAs'], "wb") as f:
    joblib.dump(decisionTreeClassifier.best_estimator_, f + ".pkl", compress=3)
    joblib.dump(x, "samples_" + args['saveAs'] + ".dat")
    joblib.dump(y, "labels_" + args['saveAs'] + ".dat")

print()
print("Time taken: ", elapsedDecisionTree)
print()

print("Best parameters set found on development set:")
print()
print(decisionTreeClassifier.best_params_)
print()
