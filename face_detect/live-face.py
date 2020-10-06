import cv2

from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

""" Uses combination of code from face-detect.py and tutorial provided by PaktPub: 
https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam"""

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

clf = CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bounding_boxes = clf.detectMultiScale(frame)
    for box in bounding_boxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()