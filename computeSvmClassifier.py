import numpy as np
import timeit

from sklearn import svm
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

svmParamGrid = [{'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [2, 3, 4]}]

startTimeSvm = timeit.default_timer()

svmClassifier = GridSearchCV(svm.SVC(), param_grid=svmParamGrid, cv=5, n_jobs=4)
svmClassifier.fit(samples, labels)

elapsedSvm = timeit.default_timer() - startTimeSvm

with open(args['saveAs'] + ".pkl", "wb") as f:
    joblib.dump(svmClassifier.best_estimator_, f, compress=3)

print()
print("Time taken: ", elapsedSvm)
print()

print("Best parameters set found on development set:")
print()
print(svmClassifier.best_params_)
print()

summary = open('summary_decision_tree_' + args['saveAs'] + '.txt', 'wb')

print >> summary, "Time taken: " + str(elapsedSvm)
print >> summary, "Best parameters set found on development set:"
print >> summary, svmClassifier.best_params_
print >> summary, "Best score found on development set: " + str(svmClassifier.best_score_)