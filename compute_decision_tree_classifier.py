import numpy as np
import timeit

from sklearn import tree
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="Path to samples")
ap.add_argument('-l', '--labels', required=True, help="Path to labels")
ap.add_argument('-c', '--saveAs', required=True, help="Save as")
args = vars(ap.parse_args())

samples = np.load(args['samples'])
labels = np.load(args['labels'])

decisionTreeParamGrid = [{'splitter': ['best'], 'max_depth': [10, 50, 100, None], 'min_samples_split': [2, 3, 6]},
                         {'splitter': ['random'], 'max_depth': [10, 50, 100, None], 'min_samples_split': [2, 3, 6]}]

startTimeDecisionTree = timeit.default_timer()

decisionTreeClassifier = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=decisionTreeParamGrid, cv=5, n_jobs=-1)
fitResult = decisionTreeClassifier.fit(samples, labels)

elapsedDecisionTree = timeit.default_timer() - startTimeDecisionTree

with open(args['saveAs'] + ".pkl", "wb") as f:
    joblib.dump(decisionTreeClassifier.best_estimator_, f, compress=3)

print()
print("Time taken: ", elapsedDecisionTree)
print()

print("Best parameters set found on development set:")
print()
print(decisionTreeClassifier.best_params_)
print()

summary = open('summary_decision_tree_' + args['saveAs'] + '.txt', 'wb')

print >> summary, "Time taken: " + str(elapsedDecisionTree)
print >> summary, "Best parameters set found on development set:"
print >> summary, decisionTreeClassifier.best_params_
print >> summary, "Best score found on development set: " + str(decisionTreeClassifier.best_score_)
