#import packages
import cv2
import imutils as im

# Read the image file
input = 'images/test.jpg'
image = cv2.imread(input)

# Resize the image - change width to 500
# newwidth = 500
# image = im.resize(image, width=newwidth)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
d, sigmaColor, sigmaSpace = 11,17,17
filtered_img = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

# Find Edges of the grayscale image
lower, upper = 170, 200
edged = cv2.Canny(filtered_img, lower, upper)
cv2.imshow("Resulting_image", edged)
cv2.waitKey(0)
# Find contours based on Edges
# cnts,hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

imagecontours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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



# # Display the original image
# cv2.imshow("Input Image", image)
# # Display Grayscale image
# cv2.imshow("Gray scale Image", gray)
# # Display Filtered image
# cv2.imshow("After Applying Bilateral Filter", filtered_img)
# # Display Canny Image
# cv2.imshow("After Canny Edges", edged)
# # Drawing the selected contour on the original image
# cv2.drawContours(image, [NumberPlateCnt], -1, (255,0,0) , 2)
# cv2.imshow("Output", image)

# cv2.waitKey(0) #Wait for user input before closing the images displayed