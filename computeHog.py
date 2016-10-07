import numpy as np

import argparse
from skimage.feature import hog

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="Path to the file with data")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")
args = vars(ap.parse_args())

loadedData = np.load(args["data"])
computedHog = []
orientation = 9
pixelsPerCell = (6, 6)
cellsPerBlock = (3, 3)

for i in range(0, len(loadedData), 1):
    currentHog = hog(loadedData[i], orientations=orientation, pixels_per_cell=pixelsPerCell,
                     cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)
    computedHog.append(currentHog)

np.array(computedHog).dump(args["saveAs"] + '.dat')
