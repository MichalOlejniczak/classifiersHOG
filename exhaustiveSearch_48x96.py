import argparse
import multiprocessing
from joblib import Parallel

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.externals.joblib import delayed
from sklearn.naive_bayes import GaussianNB

from helpers import sliding_window, pyramid

# Path /home/michalo/Phd/DataBases/CVC03-Virtual-Pedestrian/train/background-frames/backgroundsCvc03.txt

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True, help="Path to classifier")
ap.add_argument("-nb", "--naiveBayes", required=True, help="Path to Bayes")
ap.add_argument('-b', '--images', required=True, help="Path to background images")
ap.add_argument('-s', '--saveAs', required=True, help="Save as")

args = vars(ap.parse_args())

(winW, winH) = (48, 96)

with open(args['classifier'], "rb") as f:
    model = joblib.load(f)
    bayes = joblib.load(args["naiveBayes"])

path = args["images"]

backgroundFile = open(path)
backgroundLines = backgroundFile.readlines()
splitPath = path.rsplit('/', 1)

hardExamples = []
orientation = 9
pixelsPerCell = (6, 6)
cellsPerBlock = (3, 3)


def process_input(i):
    windows = []
    background_image = cv2.imread(splitPath[0] + "/" + backgroundLines[i].rstrip("\r\n"), 0)
    for resized in pyramid(background_image, scale=1.2):
        for (x, y, window) in sliding_window(resized, stepSize=15, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            check_hog = hog(window, orientations=orientation, pixels_per_cell=pixelsPerCell,
                            cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)

            c = bayes.predict_proba(check_hog.reshape(1, -1))
            if c[0][1] > 0.88:
                if model.predict(check_hog.reshape(1, -1)) == [1]:
                    windows.append(check_hog)
    return windows


num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in range(0, len(backgroundLines), 1))

refined = []
for v in range(0, len(results), 1):
    if len(results[v]) == 0:
        continue
    for b in range(0, len(results[v]), 1):
        refined.append(results[v][b])

np.array(refined).dump(args['saveAs'] + '.dat')
print len(refined)
