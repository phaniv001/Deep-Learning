import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier("Haar_face.xml")
people = ["Lakshman", "Imran"]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

#img = cv.imread("D:\Data Science\Deep Learning\OpenCV\IMG_20180626_112121.jpg")
img = cv.imread("D:\Data Science\Deep Learning\OpenCV\IMG_20180804_225453.jpg")

img = cv.resize(img, (550,550))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Picture", gray)

face_rect = haar_cascade.detectMultiScale(gray, 1.1, 2)
for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (x,y+w+20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,255,0), thickness = 1)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
    

cv.imshow("Detected Image", img)

cv.waitKey(0)
cv.destroyAllWindows()
