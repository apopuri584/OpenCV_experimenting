import cv2 as cv
import numpy as np

img = cv.imread("./images/sunflower.jpg")

blurred_img = np.empty_like(img)

h, w = img.shape[:2]
y = 0
x = 0


while y < h - 1 :
    while x < w - 1:
        patch = img[y:y+3, x:x+3]
        average_pixel = np.mean(patch, axis = (0,1))

        blurred_img[y,x] = average_pixel
        x += 1
    x = 0
    y += 1



cv_blur = cv.blur(img, (3,3))
cv.imshow("cv_blurred image", cv_blur)
cv.imshow("Blurred image", blurred_img)
cv.waitKey(0)

