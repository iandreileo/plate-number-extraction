import cv2
import os
import numpy as np
import numpy as np
import cv2
import argparse
import imutils as im

import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.patches as mpatches
from scipy.spatial import distance
from imutils import grab_contours
import pytesseract
from PIL import Image
from canny import Canny


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    canny = Canny()

    plate_number = canny.extract_plate_number(image)

    print(plate_number)

