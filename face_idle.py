import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);

while(cam.isOpened()):
    ret, im = cam.read();
    if ret:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w, y+h),(0,255,0),2)
        cv2.imshow("face1", im);
    if(cv2.waitKey(1) == ord('q')):
        break;
cam.release()
cv2.destroyAllwindows()

        

    
