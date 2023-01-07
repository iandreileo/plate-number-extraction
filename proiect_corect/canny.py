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
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
from itertools import product



class Canny:

    def grayscale(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # def grayscale(self, image):
    #     for i in range(image.shape[0]):
    #         for j in range(image.shape[1]):
    #             (r,g,b) = image[i][j]

    #             gray = (r * 0.299 + g * 0.587 + b * 0.114)

    #             image[i][j] = (gray, gray, gray)

    #     image = image.astype('uint8')
    #     return image


    def gen_gaussian_kernel(self, k_size, sigma):
        center = k_size // 2
        x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        return g


    def gaussian_filter(self, image, k_size, sigma):
        height, width = image.shape[0], image.shape[1]
        # dst image height and width
        dst_height = height - k_size + 1
        dst_width = width - k_size + 1

        # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
        image_array = zeros((dst_height * dst_width, k_size * k_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

        #  turn the kernel into shape(k*k, 1)
        gaussian_kernel = self.gen_gaussian_kernel(k_size, sigma)
        filter_array = ravel(gaussian_kernel)

        # reshape and get the dst image
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        return dst

    def blur(self, image):
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        image = self.gaussian_filter(image, 5, sigma=0.8)

        return image

    def sobel(self, image):
        image = self.blur(self.grayscale(image))
        convolved = np.zeros(image.shape)
        G_x = np.zeros(image.shape)
        G_y = np.zeros(image.shape)
        size = image.shape
        kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
        kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
        
        convolved = np.sqrt(np.square(G_x) + np.square(G_y))
        convolved = np.multiply(convolved, 255.0 / convolved.max())

        angles = np.rad2deg(np.arctan2(G_y, G_x))
        angles[angles < 0] += 180
        convolved = convolved.astype('uint8')
        return convolved, angles

    def nms(self, image, angles):
        size = image.shape
        suppressed = np.zeros(size)
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                # print(angles[i,j])
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(image[i, j - 1], image[i, j + 1])
                elif (22.5 <= angles[i, j] < 67.5):
                    value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
                elif (67.5 <= angles[i, j] < 112.5):
                    value_to_compare = max(image[i - 1, j], image[i + 1, j])
                else:
                    value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
                
                if image[i, j] >= value_to_compare:
                    suppressed[i, j] = image[i, j]
        suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
        return suppressed

    def double_threshold_hysteresis(self, image, low, high):
        weak = 50
        strong = 255
        size = image.shape
        result = np.zeros(size)
        weak_x, weak_y = np.where((image > low) & (image <= high))
        strong_x, strong_y = np.where(image >= high)
        result[strong_x, strong_y] = strong
        result[weak_x, weak_y] = weak
        dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
        dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
        size = image.shape
        
        while len(strong_x):
            x = strong_x[0]
            y = strong_y[0]
            strong_x = np.delete(strong_x, 0)
            strong_y = np.delete(strong_y, 0)
            for direction in range(len(dx)):
                new_x = x + dx[direction]
                new_y = y + dy[direction]
                if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                    result[new_x, new_y] = strong
                    np.append(strong_x, new_x)
                    np.append(strong_y, new_y)
        result[result != strong] = 0
        return result

    def canny(self, image, low, high):
        image, angles = self.sobel(image)
        image = self.nms(image, angles)
        gradient = np.copy(image)
        image = self.double_threshold_hysteresis(image, low, high)
        return image

    def ocr(self, ROI):

        nr_inmatriculare = pytesseract.image_to_string(ROI)
        print(nr_inmatriculare)
        return nr_inmatriculare

    def find_contours(self, image, new_image):
        imagecontours, _ = cv2.findContours(new_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imagecontours = sorted(imagecontours, key=cv2.contourArea, reverse=True)[:20]

        nr_inmatriculare = []
        #for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
        for count in imagecontours:
            epsilon = 0.01 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            #the name of the detected shapes are written on the image
            i, j = approximations[0][0] 
            if len(approximations) == 4:
                x,y,w,h = cv2.boundingRect(count)
                ROI = image[y:y+h, x:x+w]

                # TODO: De pus o conditie de ratia rectangulara

                cv2.imshow('INMATRICULARE',ROI)
                cv2.waitKey()

                # extragem folosind ocr
                current_plate_number = self.ocr(ROI)
                nr_inmatriculare.append(current_plate_number)

                cv2.drawContours(image, [approximations], 0, (0,255,0), 2)
                cv2.putText(image, "NR. INMATRICULARE: " + current_plate_number, (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        #displaying the resulting image as the output on the screen
        cv2.imshow("Resulting_image", image)
        cv2.waitKey(0)

        return nr_inmatriculare


    def extract_plate_number(self, image):

        new_image = self.canny(image, 0, 50)

        new_image = new_image.astype(np.uint8)

        cv2.imshow("Sobel Image", new_image)
        cv2.waitKey(0)

        return self.find_contours(image, new_image)