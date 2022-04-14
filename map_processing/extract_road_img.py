import cv2
import numpy as np

img = cv2.imread('onemap.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold of orange in HSV space
lower_orange= np.array([0, 140, 240])
upper_orange = np.array([30, 180, 250])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((2,2), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=10)
mask = cv2.erode(mask, kernel, iterations=5)

# result = cv2.bitwise_and(img, img, mask = mask)

cv2.imwrite("road1.png", mask)

print(hsv[978][760])

# Threshold of blue in HSV space
lower_y= np.array([0, 70, 240])
upper_y = np.array([40, 90, 255])

mask = cv2.inRange(hsv, lower_y, upper_y)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=6)
mask = cv2.erode(mask, kernel, iterations=5)

cv2.imwrite("road2.png", mask)


dest = cv2.cornerHarris(mask, 2, 5, 0.07)
#dest = cv2.dilate(dest, None)

temp = np.array(img)
temp[dest > 0.01 * dest.max()] = [0, 0, 255]


cv2.imwrite("road3.png", temp)




