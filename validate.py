import argparse
import numpy as np

from sklearn.externals import joblib
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positives", required=True, help="Path to positives")
ap.add_argument('-n', '--negatives', required=True, help="Path to negatives")
ap.add_argument('-c', '--classifier', required=True, help="Path to classifier")
args = vars(ap.parse_args())

positives = np.load(args['positives'])
negatives = np.load(args['negatives'])

ones = np.ones(len(positives))
zeros = np.zeros(len(negatives))

x = np.concatenate((positives, negatives), axis=0)
y = np.concatenate((ones, zeros), axis=0)

x = x.reshape(len(x), -1)

with open(args['classifier'], "rb") as f:
    classifier = joblib.load(f)

yTrue, yPred = y, classifier.predict(x)
print(classification_report(yTrue, yPred))
