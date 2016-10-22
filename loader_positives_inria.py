import numpy as np
import cv2
import argparse

# Positives /home/michalo/Phd/DataBases/MIT_128x64/positives.txt

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="Path to the file with image list")
# args = vars(ap.parse_args())

path = "/home/michalo/Phd/DataBases/INRIAPerson/Train/annotations.lst"

annotationFile = open(path)
annotationsLines = annotationFile.readlines()

splitPath = path.rsplit('/', 2)
an = open(splitPath[0] + "/" + annotationsLines[0].rstrip("\r\n"))
anlin = an.readlines()
print anlin[10]
# for i in range(0, len(annotationsLines), 1):

# pedestrianImage = cv2.imread(splitPath[0] + "/" + pedestriansLines[i].rstrip("\r\n"), 0)
# positives.append(pedestrianImage)

# positives = np.array(positives)
#
# name = splitPath[1].rsplit('.', 1)[0]
# positives.dump("mit_" + name + "_64x128.dat")
#
# print str(len(positives)) + " positives were loaded."
