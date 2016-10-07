import numpy as np
import random

import cv2
import argparse

# Path /home/michalo/Phd/DataBases/CVC03-Virtual-Pedestrian/train/background-frames/backgorundsCvc03.txt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the file with image list")
args = vars(ap.parse_args())

path = args["images"]

backgroundFile = open(path)
backgroundLines = backgroundFile.readlines()

negatives = []
splitPath = path.rsplit('/', 1)

for j in range(0, len(backgroundLines), 1):
    backgroundImage = cv2.imread(splitPath[0] + "/" + backgroundLines[j].rstrip("\r\n"), 0)
    for z in range(0, 7, 1):
        xRand = random.randint(1, 530)
        yRand = random.randint(1, 300)
        xx = xRand + 48
        yy = yRand + 96
        croppedImage = backgroundImage[yRand:yy, xRand:xx]  # crops a fragment
        negatives.append(croppedImage)

negatives = np.array(negatives)
name = splitPath[1].rsplit('.', 1)[0]

negatives.dump(name + 'negative_patches_48x96.dat')
print str(len(negatives)) + " negatives were loaded"
