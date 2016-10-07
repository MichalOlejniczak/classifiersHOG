import numpy as np
import cv2
import argparse

# Path /home/michalo/Phd/DataBases/CVC03-Virtual-Pedestrian/train/pedestrians/pedestriansCvc03.txt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the file with image list")
args = vars(ap.parse_args())

path = args["images"]

pedestriansFile = open(path)
pedestriansLines = pedestriansFile.readlines()

positives = []
splitPath = path.rsplit('/', 1)

for i in range(0, len(pedestriansLines), 1):
    pedestrianImage = cv2.imread(splitPath[0] + "/" + pedestriansLines[i].rstrip("\r\n"), 0)
    positives.append(pedestrianImage)

positives = np.array(positives)
positives.dump('posCVC03.dat')

print str(len(positives)) + " positives were loaded."

