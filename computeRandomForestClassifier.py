import numpy as np
import timeit

from sklearn.ensemble import RandomForestClassifier
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

randomForestParamGrid = [
    {'n_estimators': [5, 10, 15, 30, 40, 50, 60], 'max_depth': [10, 50, 100, None],
     'min_samples_split': [2, 3, 6, 10, 12]}]

startTimeRandomForest = timeit.default_timer()

randomForestClassifier = GridSearchCV(RandomForestClassifier(), param_grid=randomForestParamGrid, cv=5, n_jobs=4)
randomForestClassifier.fit(x, y)

elapsedRandomForest = timeit.default_timer() - startTimeRandomForest
joblib.dump(randomForestClassifier, args['saveAs'] + '.pkl', compress=3)


print()
print("Time taken: ", elapsedRandomForest)
print()

print("Best parameters set found on development set:")
print()
print(randomForestClassifier.best_params_)
print()


