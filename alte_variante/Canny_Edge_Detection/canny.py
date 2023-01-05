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

def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()

    return output


def threshold(image, low, high, weak, verbose=False):
    # Create a new image with the same size as the original
    thresholded_image = np.zeros(image.shape)

    # Iterate through each pixel in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Get the pixel value
            pixel = image[x, y]

            # If the pixel value is above the threshold, set it to white (255)
            # otherwise, set it to black (0)
            if pixel > low:
                thresholded_image[x, y] = 255
            else:
                thresholded_image[x, y] = 0

    return thresholded_image
    # output = np.zeros(image.shape)

    # strong = 255

    # strong_row, strong_col = np.where(image >= high)
    # weak_row, weak_col = np.where((image <= high) & (image >= low))

    # output[strong_row, strong_col] = strong
    # output[weak_row, weak_col] = weak

    # if verbose:
    #     plt.imshow(output, cmap='gray')
    #     plt.title("threshold")
    #     plt.show()

    # return output


def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    # Resize the image - change width to 500
    # newwidth = 500
    # image = im.resize(image, width=newwidth)

    blurred_image = gaussian_blur(image, kernel_size=9, verbose=False)

    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, edge_filter, convert_to_degree=True, verbose=args["verbose"])

    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])

    weak = 50

    new_image = threshold(new_image, 20, 255, weak=weak, verbose=args["verbose"])

    new_image = hysteresis(new_image, weak)

    new_image = new_image.astype(np.uint8)
    cv2.imshow("Resulting_image", new_image)
    cv2.waitKey(0)
    # plt.imshow(new_image, cmap='gray')
    # plt.title("Canny Edge Detector")
    # plt.show()

    imagecontours, _ = cv2.findContours(new_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagecontours = sorted(imagecontours, key=cv2.contourArea, reverse=True)[:20]
    #for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
    for count in imagecontours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        #the name of the detected shapes are written on the image
        i, j = approximations[0][0] 
        if len(approximations) == 4:
            cv2.drawContours(image, [approximations], 0, (0,255,0), 1)
            cv2.putText(image, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
    #displaying the resulting image as the output on the screen
    cv2.imshow("Resulting_image", image)
    cv2.waitKey(0)

    # key_points =cv2.findContours(new_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # #defining contours from keypoints
    # contours = grab_contours(key_points)

    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # plate_location = None


    # for cnt in contours:
    #     x ,y, w, h = cv2.boundingRect(cnt)
    #     aspectRatio = float(w)/h
    #     print (w,h, aspectRatio)
    #     if aspectRatio >= 3:  
    #         sqaure_approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #         if len(sqaure_approx) == 4:
    #             plate_location = sqaure_approx
    #             break

    # print(plate_location)
    # x1, x2 = min(plate_location[:,0][:,1]), max(plate_location[:,0][:,1])
    # y1, y2 = min(plate_location[:,0][:,0]), max(plate_location[:,0][:,0])
    # cropped_image = image[x1:x2, y1:y2]

    # plt.imshow(cropped_image, cmap='gray')
    # plt.title("Canny Edge Detector")
    # plt.show()

    # cv2.drawContours(new_image,[plate_location],-1,(0,255,0),3)
    # cv2.imshow("img1",image)
    # cv2.waitKey(0)

    # contours, hier = cv2.findContours(new_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # img_cpy = image.copy()

    # width=0 
    # height=0

    # start_x=0 
    # start_y=0
    # end_x=0 
    # end_y=0

    # for i in range(len(contours)):
        
    #     if hier[0][i][2] == -1:
    #         continue
            
    #     x ,y, w, h = cv2.boundingRect(contours[i])
    #     a=w*h    
    #     aspectRatio = float(w)/h
    #     if aspectRatio >= 2:          
    #         approx = cv2.approxPolyDP(contours[i], 0.1 * cv2.arcLength(contours[i], True), True)
    #         if len(approx) == 4 and x > 15 :
    #             width=w
    #             height=h   
    #             start_x=x
    #             start_y=y
    #             end_x=start_x+width
    #             end_y=start_y+height      
    #             cv2.rectangle(img_cpy, (start_x,start_y), (end_x,end_y), (0,0,255),3)
    #             cv2.putText(img_cpy, "rectangle "+str(x)+" , " +str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            
    # plt.imshow(img_cpy)
    # plt.show()

    # print("start",start_x,start_y)
    # print("end", end_x,end_y)



    # # find contours from the edged image and keep only the largest
    # # ones, and initialize our screen contour
    # cnts,new = cv2.findContours(new_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # img1=image.copy()
    # cv2.drawContours(img1,cnts,-1,(0,255,0),3)
    # cv2.imshow("img1",img1)
    # cv2.waitKey(0)

    # #sorts contours based on minimum area 30 and ignores the ones below that
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    # screenCnt = None #will store the number plate contour
    # img2 = image.copy()
    # cv2.drawContours(img2,cnts,-1,(0,255,0),3) 
    # cv2.imshow("img2",img2) #top 30 contours
    # cv2.waitKey(0)
    # count=0

    # idx=7
    # # loop over contours
    # for c in cnts:
    # # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    #     if len(approx) == 4: #chooses contours with 4 corners
    #         screenCnt = approx
    #         x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
    #         new_img=image[y:y+h,x:x+w]
    #         cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
    #         idx+=1
    #         break

    # #draws the selected contour on original image        
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    # cv2.imshow("Final image with plate detected",image)
    # cv2.waitKey(0)

    # contours, hierarchy = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)

    # filtered_countours = []

    # for cnt in contours:
    #     rect = cv2.minAreaRect(cnt)       #I have used min Area rect for better result
    #     width = rect[1][0]
    #     height = rect[1][1]
    #     # if (width<widthmax) and (height <heightmax) and (width >= widthMin) and (height > heightMin):
    #     filtered_countours.append(cnt)
    #     cv2.drawContours(image, filtered_countours, -1, (0, 255, 0), 3)

    #     cv2.imshow('Contours', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # if height > 1.5*width:
    #     #     filtered_countours.append(cnt)
    #     #     print(width, height)

    # # cv2.drawContours(image, filtered_countours, -1, (0, 255, 0), 3)

    # # cv2.imshow('Contours', image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()