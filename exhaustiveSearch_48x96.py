import argparse
import multiprocessing
from joblib import Parallel

import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.externals.joblib import delayed

from helpers import sliding_window, pyramid

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True, help="Path to classifier")
ap.add_argument('-b', '--backgroundImagesAsSingleDat', required=True, help="Path to background images")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")

args = vars(ap.parse_args())

(winW, winH) = (48, 96)

with open(args['classifier'], "rb") as f:
    model = joblib.load(f)

back = np.load(args['backgroundImagesAsSingleDat'])
hardExamples = []
orientation = 9
pixelsPerCell = (6, 6)
cellsPerBlock = (3, 3)


def processInput(i):
    windows = []
    for resized in pyramid(back[i], scale=1.2):
        for (x, y, window) in sliding_window(resized, stepSize=15, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            checkHog = hog(window, orientations=orientation, pixels_per_cell=pixelsPerCell,
                           cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)
            if model.predict(checkHog.reshape(1, -1)) == [1]:
                windows.append(checkHog)

    return windows


num_cores = multiprocessing.cpu_count()
xx = len(back)
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(0, len(back), 1))

refined = []
for v in range(0, len(results), 1):
    if len(results[v]) == 0:
        continue
    for b in range(0, len(results[v]), 1):
        refined.append(results[v][b])

np.array(refined).dump(args['saveAs'] + '.dat')
