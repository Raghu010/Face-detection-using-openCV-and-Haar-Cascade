import cv2
import sys

imgPath = 'image.jpg'
path = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(path)
image = cv2.imread(imgPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


face = faceCascade.detectMultiScale(gray, scaleFactor = 1.9,
                                    minNeighbors = 1,
                                    minSize = (50, 50),
                                    flags = cv2.CASCADE_SCALE_IMAGE)

print('Found {0} faces!'.format(len(face)))

for(x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Faces Found', image)
cv2.waitKey(0)
