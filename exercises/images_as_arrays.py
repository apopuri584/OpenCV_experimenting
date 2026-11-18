import cv2 as cv
import numpy as np

img = cv.imread("./images/sunflower.jpg")
cv.imshow("Original img", img)

#printed image shape
print(f"Image shape: {img.shape}")

#split channels and showed them
b,g,r = cv.split(img)
cv.imshow("Blue chanel", b)
cv.imshow("Green chanel", g)
cv.imshow("Red chanel", r)


#convert from BGR to RGB
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB image", rgb_img)

#cropped img
cropped_img = img[0:200, 0:300]
cv.imshow("Cropped Image", cropped_img)



cv.waitKey(0)
