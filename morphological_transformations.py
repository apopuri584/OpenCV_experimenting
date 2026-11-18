import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import os


def morphTrans():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/sunflower.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    maxValue = 255
    blocksize = 7
    offset = 3
    plt.subplot(241)

    imgGause = cv.adaptiveThreshold(imgGray, maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, blocksize, offset)
    imgGause = cv.GaussianBlur(imgGause, (7,7), cv.BORDER_DEFAULT)
    plt.imshow(imgGause, cmap='gray')
    plt.title('Adaptive Gaussian Thresholding')
  
    kernel = np.ones((7,7), np.uint8)
    imgErode = cv.erode(imgGause, kernel, iterations=1)
    plt.subplot(245)
    plt.imshow(imgErode, cmap='gray')
    plt.title('Erosion')

    imgDilate = cv.dilate(imgGause, kernel, iterations=1)
    plt.subplot(248)
    plt.imshow(imgDilate, cmap='gray')
    plt.title('Dilation')


    morphTypes = [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT]

    morphNames = ['Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat']


    for i in range(len(morphTypes)):
        plt.subplot(2,4,i+4)
        plt.imshow(cv.morphologyEx(imgGause, morphTypes[i], kernel), cmap='gray')
        plt.title(morphNames[i])


    plt.show()



if __name__ == "__main__":
    morphTrans()
