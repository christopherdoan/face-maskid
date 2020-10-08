import cv2

from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

""" Implementation of tutorial provided by MachineLearningMastery article written by Jason Brownlee: 
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/"""

#load the pretrained model 
clf = CascadeClassifier('haarcascade_frontalface_default.xml')

pixels = imread("../images/test1.jpg")

#create bounding boxes around face
bounding_boxes = clf.detectMultiScale(pixels)
for box in bounding_boxes:
    # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)

# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()
