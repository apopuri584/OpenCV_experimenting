import cv2 as cv 
import numpy as np

#Find contours in a thresholded image, draw boxes around those above a minimum area, print each area.

img = cv.imread("./images/sunflower.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
areas = [cv.contourArea(contour) for contour in contours]

for contour in contours:
    area = cv.contourArea(contour)
    if 111718.5> area > 100: 
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)


cv.imshow("Bounding boxes", img)
cv.waitKey(0)

print(f'{len(contours)} contours found!')


