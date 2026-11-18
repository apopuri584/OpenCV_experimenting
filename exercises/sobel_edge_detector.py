import cv2 as cv
import numpy as np

img = cv.imread("./images/sunflower.jpg")

blurred = cv.blur(img, (3,3), cv.BORDER_DEFAULT)
gray_img = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

def blur_image(img, kernel_size):
    blurred_img = np.empty_like(img)

    h, w = img.shape[:2]
    y = 0
    x = 0


    while y < h - 1 :
        while x < w - 1:
            patch = img[y:y+kernel_size, x:x+kernel_size]
            average_pixel = np.mean(patch, axis = (0,1))

            blurred_img[y,x] = average_pixel
            x += 1
        x = 0
        y += 1

    return blurred_img, h, w



def sobel_edge_detector(img,threshold):

    
    blurred_img, h, w = blur_image(img, 3)

    sobel_img = np.empty_like(img)


    


    gray_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)

    sobel_horizontal = np.array(([-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]), dtype = np.float32)
    sobel_vertical = np.array(([-1,  0,  1], [-2,  0,  2], [-1,  0,  1]), dtype = np.float32)
    y = 0
    x = 0
    for y in range(h - 3 - 1):
        for x in range(w - 3 + 1):
            patch = gray_img[y:y+3, x:x+3]
            grad_x = np.sum(patch * sobel_horizontal)

            grad_y = np.sum(patch * sobel_vertical)

            magnitude = np.hypot(grad_x, grad_y)

            if magnitude > threshold:
                sobel_img[y,x] = [255, 255, 255]
            else:
                sobel_img[y,x] = [0, 0, 0]

            x += 1
        y +=1 

    return sobel_img



sobel_img = sobel_edge_detector(img,100)


sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)


gradient_magnitude = cv.magnitude(sobelx, sobely)

sobel_combined = cv.convertScaleAbs(gradient_magnitude)

cv.imshow("Sobel Edge Detection", sobel_combined)

cv.imshow("Sobel image", sobel_img)

cv.waitKey(0)













            










