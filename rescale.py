import cv2 as cv 


img = cv.imread('images/sunflower.jpg')

cv.imshow('Sunflower', img)


def rescaleFrame(frame, scale=0.75):
    #images, videos, and live videos
    width = int(frame.shape[1] * scale) 
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #Live video 
    capture.set(3, width) #3 references width
    capture.set(4, height) #4 references height 




cv.imshow('Sunflower Resized', rescaleFrame(img))

capture = cv.VideoCapture('videos/wide_clip.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale = 0.2)
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()


