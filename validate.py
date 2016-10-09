import argparse
import numpy as np

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

yTrue, yPred = labels, classifier.predict(x)
print(classification_report(yTrue, yPred))
