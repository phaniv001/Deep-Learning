import cv2 as cv
import numpy as np

#Reading the image
img = cv.imread("D:\Data Science\Deep Learning\OpenCV\Photos\shutterstock_648907024.jpg")
cv.imshow("Image", img)

#Converting the image to Gray.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Image in Gray", gray)

#Loading the haarcascade Xml
haar_cascade = cv.CascadeClassifier('Haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 1)
print("No.of Faces : ", len(faces_rect))

# Face Detection on Image.
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,256,0), 1)

cv.imshow("Detected Faces", img)



cv.waitKey(0)
cv.destroyAllWindows()
