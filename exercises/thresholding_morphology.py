import cv2 as cv
import numpy as np



img = cv.imread("./images/sunflower.jpg")


# 1. Global thresholding, applies same threshold to all pixels
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
threshold_used, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

cv.imshow("Global threshold", binary)


# 2. Otsu's thresholding, uses images intensity histogram to find a threshold value
threshold_used, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Otsu's thresholding", otsu)

# 3. Morphology

#erosion thickens darker areas
kernel = np.ones((7,7), np.uint8)
eroded =  cv.erode(otsu, kernel, iterations=1)
cv.imshow('Eroded otsu', eroded)

#dilation thickens white areas 
dilated = cv.dilate(otsu, kernel,iterations = 1)
cv.imshow("Dilated otsu", dilated)


cv.waitKey(0)