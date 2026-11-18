import cv2 as cv
img = cv.imread('images/sunflower.jpg')


#Converting to grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#canny edge detection 
canny = cv.Canny(img, 175, 175)
cv.imshow('Canny Edges',  canny)

#Dilating the image
dilated = cv.dilate(canny, (175,175), iterations=3)
cv.imshow('Dilated', dilated)


#eroding the image
eroded = cv.erode(dilated, (175,175), iterations = 3)
cv.imshow('Eroded', eroded)

cv.waitKey(0)
