import numpy as np
import cv2
import argparse

from sobel import sobel_edge_detection
from gaussian_smoothing import gaussian_blur

import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from scipy.spatial.distance import euclidean
import imutils


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
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()

    return output


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
    image = imutils.resize(image, width=300 )

    blurred_image = gaussian_blur(image, kernel_size=9, verbose=False)

    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, edge_filter, convert_to_degree=True, verbose=args["verbose"])

    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])

    weak = 50

    new_image = threshold(new_image, 5, 20, weak=weak, verbose=args["verbose"])

    new_image = hysteresis(new_image, weak)
    sobel = new_image.astype(np.uint8)


    # cnts,new = cv2.findContours(sobel.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # image1=image.copy()
    # cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    # cv2.imshow("contours",image1)
    # cv2.waitKey(0)


    label_image = label(sobel, connectivity=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(sobel, cmap="gray")

    text_like_regions = []
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        
        asr = w/h
        
        region_area = w*h
        
        wid,hei = sobel.shape
        img_area = wid*hei
        
        # The aspect ratio is less than 1 to eliminate highly elongated regions
        # The size of the region should be greater than 15 pixels but smaller than 1/5th of the image
        # dimension to be considered for further processing
        if region_area > 15 and region_area < (0.2 * img_area) and asr < 1 and h > w:
            #print(w, h, i, region.area, region.bbox)
            text_like_regions.append(region)

    all_points = []
    for region in text_like_regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        circ = mpatches.Circle((minc, minr), radius=5)
        ax.add_patch(circ)
        all_points.append([minc, minr, maxc, maxr])
        
    plt.tight_layout()
    plt.show()        

    def angle_between_three_points(pointA, pointB, pointC):
        BA = pointA - pointB
        BC = pointC - pointB

        try:
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.arccos(cosine_angle)
        except:
            print("exc")
            raise Exception('invalid cosine')

        return np.degrees(angle)

    all_points = np.array(all_points)

    all_points = all_points[all_points[:,1].argsort()]
    height, width = sobel.shape
    groups = []
    for p in all_points:
        cluster = [p]
        lines_found = False
        for q in all_points:
            pmin = np.array([p[0],p[1]])
            qmin = np.array([q[0],q[1]])
            if p[1] < q[1] and euclidean(pmin,qmin) < 0.1 * width:
                # first check if q is already added, if not add.
                point_already_added = False
                for cpoints in cluster:
                    if cpoints[0] == q[0] and cpoints[1] == q[1]:
                        point_already_added = True
                        break
                if not point_already_added:
                    cluster.append(q)
                    
                for r in all_points:
                    rmin = np.array([r[0],r[1]])
                    last_cluster_point = np.array([cluster[-1][0],cluster[-1][1]])
                    if q[1] < r[0] and euclidean(last_cluster_point,rmin) < 0.1 * width:
                        angle = angle_between_three_points(pmin,qmin,rmin)
                        if angle > 170 and angle < 190:
                            lines_found = True
                            cluster.append(r)
                            
        if lines_found:
            groups.append(np.array(cluster))


    # plot the longest found line on the image
    longest = -1
    longest_index = -1
    for index, cluster in enumerate(groups):
        if len(cluster) > longest:
            longest_index = index
            longest = len(cluster)

    print("coordinates of licence plate\n")
    print(groups[longest_index])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(sobel, cmap="gray")
    for region in groups[longest_index]:
        minc = region[0]
        minr = region[1] 
        maxc = region[2]
        maxr = region[3]
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()  

