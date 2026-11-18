import cv2 as cv
import numpy as np


blank = np.zeros(img.shape[:2], dtype = 'uint8')
cv.imshow('blank', blank)


img = cv.imread('images/sunflower.jpg')

#blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
#cv.imshow('Blur', img)
#Border types: cv.BORDER_CONSTANT, cv.BORDER_REFLECT, cv.BORDER_REPLICATE, cv.BORDER_WRAP
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray) 

#canny = cv.Canny(gray, 175, 175)
#cv.imshow('Canny Edges', canny)

# contours: contours are curves that join all continuous points along a boundary that share the same color or intensity
#contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#print(f'{len(contours)} contours found!')
#RETR_EXTERNAL - external contours
#RETR_LIST - all contours
#RETR_TREE - all contours and heirarchy
 
#CHAIN_APPROX_NONE - all points of contours
#CHAIN_APPROX_SIMPLE - compresses the contours



#thresholing
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found!')



cv.waitKey(0)


