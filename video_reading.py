import cv2 as cv

capture = cv.VideoCapture('videos/wide_clip.mp4')

"""
if not capture.isOpened():
    print("Could not open video. Check the path.")
    exit()
"""

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()