import cv2

from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import rectangle
from cv2 import circle

# face detection with mtcnn on a photograph
from mtcnn.mtcnn import MTCNN

""" Uses combination of code from deep-learn.py, face_detect/live-face.py and tutorial provided by PaktPub: 
https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam"""

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# create the detector, using default weights
detector = MTCNN()
# detect faces in the image


while True:
    ret, frame = cap.read()
    # detect faces in the image
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    faces = detector.detect_faces(frame)
    for box in faces:
        # extract
        x, y, width, height = box['box']
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
        for key, value in box['keypoints'].items():
            # create and draw dot
            circle(frame, value, radius=2, color=(0,0,255))
    cv2.imshow('Input', frame)


    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()