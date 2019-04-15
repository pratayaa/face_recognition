import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);

id = raw_input('enter user id')
sampleNum = 0;
while(cam.isOpened()):
    ret, im = cam.read();
    if ret:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        for(x,y,w,h) in faces:
            sampleNum = sampleNum + 1;
            cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x,y),(x+w, y+h),(0,255,0),2)
            cv2.waitKey(100);
        cv2.imshow("face1", im);
        cv2.waitKey(1);
        if(sampleNum > 20):
            break
cam.release()
cv2.destroyAllwindows()

        

    
