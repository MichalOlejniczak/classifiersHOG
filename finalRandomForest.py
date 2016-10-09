import numpy as np
import timeit

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample", required=True, help="Path to positives")
ap.add_argument('-l', '--labels', required=True, help="Path to negatives")
ap.add_argument('-c', '--classifier', required=True, help="Save as")
args = vars(ap.parse_args())

samples = np.load(args['sample'])
labels = np.load(args['labels'])

randomForestParamGrid = [
    {'n_estimators': [5, 10, 15, 30, 40, 50, 60], 'max_depth': [10, 50, 100, None],
     'min_samples_split': [2, 3, 6, 10, 12]}]

startTimeRandomForest = timeit.default_timer()

randomForestClassifier = GridSearchCV(RandomForestClassifier(), param_grid=randomForestParamGrid, cv=5, n_jobs=4)
randomForestClassifier.fit(samples, labels)

elapsedRandomForest = timeit.default_timer() - startTimeRandomForest
with open(args['classifier'] + ".pkl", "wb") as f:
    joblib.dump(randomForestClassifier.best_estimator_, f, compress=3)

print()
print("Time taken: ", elapsedRandomForest)
print()

print("Best parameters set found on development set:")
print()
print(randomForestClassifier.best_params_)
print()
