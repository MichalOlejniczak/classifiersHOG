import argparse

import numpy as np
from imblearn.under_sampling import ClusterCentroids
from sklearn.externals import joblib

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="Path to samples")
ap.add_argument('-l', '--labels', required=True, help="Path to labels")
ap.add_argument('-he', '--hardExamples', required=True, help="Path to hard examples")
ap.add_argument('-ss', '--saveAs', required=True, help="Save as")

args = vars(ap.parse_args())

# Generate the dataset

samples = joblib.load(args['samples'])
labels = joblib.load(args['labels'])
hardExamples = joblib.load(args['hardExamples'])

hardZeros = np.zeros(len(hardExamples))

x = np.concatenate((samples, hardExamples), axis=0)
y = np.concatenate((labels, hardZeros), axis=0)

x = x.reshape(len(x), -1)

X = x
y = y
#
# pca = PCA(n_components=2)
# X_vis = pca.fit_transform(X)


rus = ClusterCentroids(ratio=0.5)
X_resampled, y_resampled = rus.fit_sample(X, y)

np.array(X_resampled).dump('sub_samples_' + args['saveAs'] + '.dat')
np.array(y_resampled).dump('sub_labels' + args['saveAs'] + '.dat')
