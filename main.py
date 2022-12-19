from PIL import Image
from PIL import ImageFilter

import cv2
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image, kernel, average=False):

    # Extragem datele de care avem nevoie

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # Iteram prin toti pixelii

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    # Returnam imaginea modificata
    return output


def sobel_edge_detection(image, xfilter, yfilter):
    # Aplicam convolutie pe imagine orizontal
    new_image_x = convolution(image, xfilter)

    # Aplicam convolutie pe imagine vertical (transpus)
    new_image_y = convolution(image, yfilter)

    # Calculam gradientul

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # Returnam magnitudinea gradientului
    return gradient_magnitude

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel, average=True)

def main():
    # Citim poza
    img = cv2.imread('test3.jpg')

    # Aplicam grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Declaram filtru
    xfilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    yfilter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    img = gaussian_blur(img, 2)

    # Facem edge detection folosind sobel de la SPG
    img = sobel_edge_detection(img, xfilter, yfilter)

    # Invert
    mask = np.full(img.shape, 255)
    img = mask - img
    img = img.astype(np.uint8)
    
    # Scriem imaginea inapoi in fisier
    cv2.imwrite("test_rezolvat.jpg", img)


if __name__ == "__main__":
    main()