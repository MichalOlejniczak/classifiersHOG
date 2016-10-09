import argparse

import numpy as np
from imblearn.under_sampling import ClusterCentroids

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positives", required=True, help="Path to positives")
ap.add_argument('-n', '--negatives', required=True, help="Path to negatives")
ap.add_argument('-h', '--hardExamples', required=True, help="Path to hard examples")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")

args = vars(ap.parse_args())

# Generate the dataset
positives = np.load(args['positives'])
negatives = np.load(args['negatives'])
hardExamples = np.load(args['hardExamples'])

ones = np.ones(len(positives))
zeros = np.zeros(len(negatives))
hardZeros = np.zeros(len(hardExamples))

x = np.concatenate((positives, negatives, hardExamples), axis=0)
y = np.concatenate((ones, zeros, hardZeros), axis=0)

x = x.reshape(len(x), -1)

X = x
y = y
#
# pca = PCA(n_components=2)
# X_vis = pca.fit_transform(X)

constantRatio = float(len(positives) / len(negatives))

rus = ClusterCentroids(ratio=constantRatio)
X_resampled, y_resampled = rus.fit_sample(X, y)

np.array(X_resampled).dump('samples' + args['saveAs'])
np.array(y_resampled).dump('labels' + args['saveAs'])
