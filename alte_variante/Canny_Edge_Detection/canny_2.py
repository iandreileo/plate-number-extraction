import numpy as np
import cv2
import argparse
import imutils as im

from sobel import sobel_edge_detection
from gaussian_smoothing import gaussian_blur

import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.patches as mpatches
from scipy.spatial import distance
from imutils import grab_contours

def grayscale(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            (r,g,b) = image[i][j]

            gray = (r * 0.299 + g * 0.587 + b * 0.114)

            image[i][j] = (gray, gray, gray)

    return image

def sobel(image):

    # # Extracting each one of the RGB components
    # r_img, g_img, b_img = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # # The following operation will take weights and parameters to convert the color image to grayscale
    # gamma = 1.400  # a parameter
    # r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
    # grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

    grayscale_image = grayscale(image)
    print(grayscale_image)
    blurred_image = cv2.GaussianBlur(grayscale_image, (3,3), sigmaX=34, sigmaY=36)


    # Here we define the matrices associated with the Sobel filter
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    # rows, columns = blurred_image.shape[0]  # we need to know the shape of the input grayscale image
    sobel_filtered_image = np.zeros(shape=(blurred_image.shape[0] , blurred_image.shape[1] ))  # initialization of the output image array (all elements are 0)

    # Now we "sweep" the image in both x and y directions and compute the output
    for i in range(blurred_image.shape[0]  - 2):
        for j in range(blurred_image.shape[1] - 2):
            gx = np.sum(np.multiply(Gx, blurred_image[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, blurred_image[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

    # Display the original image and the Sobel filtered image
    # fig2 = plt.figure(1)
    # ax2 = fig2.add_subplot()
    # ax2.imshow(sobel_filtered_image)
    # fig2.show()
    # # cv2.waitKey(0)
    # # print(sobel_filtered_image)
    # sobel_filtered_image = sobel_filtered_image.astype(np.uint8)
    return sobel_filtered_image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    # grayscale(image)

    # cv2.imshow("Resulting_image", image)
    # cv2.waitKey(0)


    # De modificat aici sa fie alt blur
    # blur_img = cv2.GaussianBlur(image, (3,3), sigmaX=34, sigmaY=36)
    
    # cv2.imshow("Blur Image", blur_img)
    # cv2.waitKey(0)

    sobel_img = sobel(image)
    sobel_img = sobel_img.astype(np.uint8)


    cv2.imshow("Sobel Image", sobel_img)
    cv2.waitKey(0)