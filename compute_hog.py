import numpy as np

import argparse
from skimage.feature import hog

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positives", required=True, help="Path to positives")
ap.add_argument("-n", "--negatives", required=True, help="Path to negatives")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")
args = vars(ap.parse_args())

positives = np.load(args["positives"])
negatives = np.load(args["negatives"])

samples = []
labels = []

orientation = 9
pixelsPerCell = (6, 6)
cellsPerBlock = (3, 3)

for i in range(0, len(positives), 1):
    currentHog = hog(positives[i], orientations=orientation, pixels_per_cell=pixelsPerCell,
                     cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)
    samples.append(currentHog)
    labels.append(1)

for j in range(0, len(negatives), 1):
    currentHog = hog(negatives[j], orientations=orientation, pixels_per_cell=pixelsPerCell,
                     cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)
    samples.append(currentHog)
    labels.append(0)

np.array(samples).dump('samples_' + args["saveAs"] + '.dat')
np.array(labels).dump('labels_' + args["saveAs"] + '.dat')
