import numpy as np
import timeit

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positives", required=True, help="Path to positives")
ap.add_argument('-n', '--negatives', required=True, help="Path to negatives")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")

args = vars(ap.parse_args())

positives = np.load(args['positives'])
negatives = np.load(args['negatives'])

ones = np.ones(len(positives))
zeros = np.zeros(len(negatives))

x = np.concatenate((positives, negatives), axis=0)
y = np.concatenate((ones, zeros), axis=0)

x = x.reshape(len(x), -1)

svmParamGrid = [{'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']},
                {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [2, 3, 4]}]

startTimeSvm = timeit.default_timer()

svmClassifier = GridSearchCV(svm.SVC(), param_grid=svmParamGrid, cv=5, n_jobs=4)
svmClassifier.fit(x, y)

elapsedSvm = timeit.default_timer() - startTimeSvm

with open(args['saveAs'], "wb") as f:
    joblib.dump(svmClassifier.best_estimator_, f, compress=3)

print()
print("Time taken: ", elapsedSvm)
print()

print("Best parameters set found on development set:")
print()
print(svmClassifier.best_params_)
print()
