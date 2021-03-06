import argparse
import numpy as np
import timeit

from sklearn.externals import joblib
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="Path to samples")
ap.add_argument('-l', '--labels', required=True, help="Path to labels")
ap.add_argument('-c', '--classifier', required=True, help="Path to classifier")
args = vars(ap.parse_args())

with open(args['classifier'], "rb") as f:
    classifier = joblib.load(f)
    samples = joblib.load(args['samples'])
    labels = joblib.load(args['labels'])

x = samples.reshape(len(samples), -1)

startTimeSvm = timeit.default_timer()
yTrue, yPred = labels, classifier.predict(x)
elapsedSvm = timeit.default_timer() - startTimeSvm
print(classification_report(yTrue, yPred))

summary = open('validation_report' + '.txt', 'wb')

print >> summary, classification_report(yTrue, yPred)
print >> summary, "Time taken: " + str(elapsedSvm)
