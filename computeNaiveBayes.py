import argparse
import numpy as np
import timeit

from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

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

startTimeNaiveBayes = timeit.default_timer()

naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(x, y)

elapsedTimeNaiveBayes = timeit.default_timer() - startTimeNaiveBayes
joblib.dump(naiveBayesClassifier, args['saveAs'] + '.pkl', compress=3)

print()
print("Time taken: ", elapsedTimeNaiveBayes)
print()

print("Best parameters set found on development set:")
print()
print(naiveBayesClassifier.get_params())
print()
