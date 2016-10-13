import argparse
import numpy as np
import timeit

from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="Path to samples")
ap.add_argument('-l', '--labels', required=True, help="Path to labels")
ap.add_argument('-c', '--saveAs', required=True, help="Save as")
args = vars(ap.parse_args())

samples = np.load(args['samples'])
labels = np.load(args['labels'])

startTimeNaiveBayes = timeit.default_timer()

naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(samples, labels)

elapsedTimeNaiveBayes = timeit.default_timer() - startTimeNaiveBayes

with open(args['saveAs'] + ".pkl", "wb") as f:
    joblib.dump(naiveBayesClassifier, f, compress=3)

print()
print("Time taken: ", elapsedTimeNaiveBayes)
print()

print("Parameters:")
print()
print(naiveBayesClassifier.get_params())
print()
summary = open('summary_naive_bayes_' + args['saveAs'] + '.txt', 'wb')

print >> summary, "Time taken: " + str(elapsedTimeNaiveBayes)
print >> summary, "Parameters:"
print >> summary, naiveBayesClassifier.get_params()